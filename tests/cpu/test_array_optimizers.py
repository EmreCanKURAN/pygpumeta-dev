"""PSO and DE on the CPU: NumPy arrays, reference selection in place of the kernels."""

from __future__ import annotations

import numpy as np
import pytest
from isolation import run_without_gpu_libraries
from test_array_optimizers_shared import ReferenceSelector

import pygpumeta
from pygpumeta import Constraint, TerminationReason
from pygpumeta.algorithms import _array
from pygpumeta.space import Categorical, Integer, Real

ALGORITHMS = ("pso", "de")


@pytest.fixture
def reference_selection(monkeypatch):
    ReferenceSelector.calls = []
    monkeypatch.setattr(_array, "DeviceSelector", ReferenceSelector)
    return ReferenceSelector


def sphere(x):
    return (x * x).sum(axis=1)


def shifted(x):
    return ((x - 1.5) ** 2).sum(axis=1)


def rosenbrock(x):
    return (100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2).sum(axis=1)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_native_host_runs_import_no_gpu_library(algorithm):
    out = run_without_gpu_libraries(
        "import pygpumeta\n"
        "result = pygpumeta.minimize(lambda x: (x * x).sum(axis=1), (-5.0, 5.0), dim=5,\n"
        f"    algorithm={algorithm!r}, selection='native', pop_size=40, max_iter=50, mode='host')\n"
        "print(result.nfev, result.fun < 1.0)\n"
    )
    assert out.split() == ["2040", "True"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_native_host_runs(algorithm):
    result = pygpumeta.minimize(
        sphere, (-5.0, 5.0), dim=5, algorithm=algorithm, selection="native", pop_size=40,
        max_iter=200, mode="host", seed=1,
    )  # fmt: skip
    # Tolerance: both algorithms contract geometrically on a five-dimensional
    # sphere; 200 iterations leave values many orders below 1e-8.
    assert result.fun <= 1e-8 and np.abs(result.x).max() <= 1e-3
    assert result.nfev == 40 * 201 and result.nit == 200
    assert result.termination is TerminationReason.MAX_ITER
    assert result.history.shape == (201,) and (np.diff(result.history) <= 0).all()
    assert result.history[-1] == result.fun == sphere(result.x[None, :])[0]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    ("objective", "bounds", "dim", "iterations", "optimum", "minimiser", "tolerance"),
    [
        # Tolerances, with 60 individuals. The shifted sphere is solved to
        # rounding error within 400 iterations. Both algorithms creep along
        # Rosenbrock's curved valley; after 2000 iterations PSO is at about
        # 1e-5 and DE lower, and 1e-3 in value bounds the distance to the
        # minimiser by about 0.05.
        (shifted, (-5.0, 5.0), 6, 400, 0.0, 1.5, 1e-10),
        (rosenbrock, (-2.0, 2.0), 4, 2000, 0.0, 1.0, 1e-3),
    ],
)
def test_known_optima(algorithm, objective, bounds, dim, iterations, optimum, minimiser, tolerance):
    result = pygpumeta.minimize(
        objective, bounds, dim=dim, algorithm=algorithm, selection="native", pop_size=60,
        max_iter=iterations, mode="host", seed=3,
    )  # fmt: skip
    assert result.fun - optimum <= tolerance
    assert np.abs(result.x - minimiser).max() <= 0.1


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("name", ["tournament", "ballot_elite", "migration"])
def test_selection_operators_drive_the_search(reference_selection, algorithm, name):
    common = {
        "dim": 4, "algorithm": algorithm, "selection": name, "pop_size": 70, "max_iter": 150,
        "mode": "host",
    }  # fmt: skip
    result = pygpumeta.minimize(shifted, (-5.0, 5.0), seed=5, **common)
    calls = list(reference_selection.calls)
    # One selection per iteration, with the seed schedule of the kernel algorithms.
    assert [seed for _, seed in calls] == [5 + 3000 + it for it in range(150)]
    assert result.selection.startswith(name)
    # Tolerance: selection inside warps of 32 slows the spread of the best
    # position, so the bar is lower than for the native rules.
    assert result.fun <= 1e-4
    again = pygpumeta.minimize(shifted, (-5.0, 5.0), seed=5, **common)
    other = pygpumeta.minimize(shifted, (-5.0, 5.0), seed=6, **common)
    assert again.fun == result.fun and np.array_equal(again.x, result.x)
    assert not np.array_equal(other.x, result.x)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_selection_changes_the_run(reference_selection, algorithm):
    common = {"dim": 4, "algorithm": algorithm, "pop_size": 70, "max_iter": 20, "mode": "host"}
    native = pygpumeta.minimize(shifted, (-5.0, 5.0), selection="native", **common)
    selected = pygpumeta.minimize(shifted, (-5.0, 5.0), selection="tournament", **common)
    assert not np.array_equal(native.x, selected.x)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_bounds_maximise_constraints_and_decoding(algorithm):
    seen = []

    def objective(x):
        seen.append(x.copy())
        return x[:, 0] + x[:, 1] - 0.1 * (x[:, 2] - 3.0) ** 2 - (x[:, 3] == 1.0)

    result = pygpumeta.minimize(
        objective,
        [Real(0.0, 2.0), (-1.0, 4.0), Integer(0, 6), Categorical(["no", "yes", "maybe"])],
        algorithm=algorithm,
        selection="native",
        pop_size=60,
        max_iter=200,
        mode="host",
        maximize=True,
        constraints=[Constraint(lambda x: x[:, 0] + x[:, 1] - 3.0)],  # x0 + x1 <= 3
        penalty=1e3,
    )
    points = np.concatenate(seen)
    assert (points[:, 0] >= 0.0).all() and (points[:, 0] <= 2.0).all()
    assert (points[:, 1] >= -1.0).all() and (points[:, 1] <= 4.0).all()
    assert set(np.unique(points[:, 2])) <= set(map(float, range(7)))
    assert set(np.unique(points[:, 3])) <= {0.0, 1.0, 2.0}
    # The maximum is 3 on the constraint x0 + x1 = 3, with the integer at 3
    # and the category that is not rewarded with a deduction.
    # Tolerance: the penalty of 1e3 lets the optimum overshoot the
    # constraint by about 5e-4, and the same in value.
    assert result.fun == pytest.approx(3.0, abs=5e-3)
    assert result.constraint_violation <= 5e-3
    assert result.values[2] == 3 and result.values[3] in ("no", "maybe")
    assert result.x[0] + result.x[1] == pytest.approx(3.0, abs=5e-3)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_row_by_row_objective_and_stopping_rules(algorithm):
    calls = []

    def objective(row):
        calls.append(row.shape)
        return float((row**2).sum())

    result = pygpumeta.minimize(
        objective, (-1.0, 1.0), dim=3, algorithm=algorithm, selection="native", pop_size=10,
        max_iter=500, mode="host", vectorized=False, target=1e-3,
    )  # fmt: skip
    assert set(calls) == {(3,)} and len(calls) == result.nfev
    assert result.termination is TerminationReason.TARGET and result.fun <= 1e-3


def test_step_wise_control():
    opt = pygpumeta.PSO(sphere, (-1.0, 1.0), dim=2, pop_size=8, max_iter=5, mode="host",
                        selection="native")  # fmt: skip
    assert opt.best_value == np.inf and np.isnan(opt.best_x).all() and opt.nfev == 0
    opt.step()
    assert opt.iteration == 1 and opt.nfev == 8 and np.isfinite(opt.best_value)
    result = opt.run()
    assert result.nit == 5 and result.nfev == 48
    with pytest.raises(RuntimeError, match="this run has ended"):
        opt.step()


@pytest.mark.parametrize(
    ("algorithm", "options", "error", "message"),
    [
        ("pso", {"inertia": -0.1}, ValueError, "inertia must be a finite number of at least 0"),
        ("pso", {"social": "big"}, TypeError, "social must be a real number"),
        ("pso", {"limit": 3}, TypeError, "algorithm 'pso' has no option 'limit'"),
        ("de", {"differential_weight": 0.0}, ValueError, "differential_weight must be a finite"),
        ("de", {"differential_weight": 2.5}, ValueError, r"must lie in \(0, 2\], got 2.5"),
        ("de", {"crossover": 1.5}, ValueError, r"crossover must lie in \[0, 1\]"),
    ],
)
def test_invalid_parameters(algorithm, options, error, message):
    with pytest.raises(error, match=message):
        pygpumeta.minimize(sphere, (-1.0, 1.0), dim=2, algorithm=algorithm, options=options)


def test_de_needs_four_individuals():
    with pytest.raises(ValueError, match="pop_size must be at least 4, got 3"):
        pygpumeta.DE(sphere, (-1.0, 1.0), dim=2, pop_size=3)


def test_de_partners_differ_from_the_individual(monkeypatch):
    opt = pygpumeta.DE(sphere, (-1.0, 1.0), dim=2, pop_size=4, max_iter=1, mode="host",
                       selection="native")  # fmt: skip
    opt._ensure_started()
    for _ in range(200):
        offsets = 1 + (opt._rng.random_sample((3, 4)) * 3).astype(np.int64)
        assert offsets.min() >= 1 and offsets.max() <= 3
