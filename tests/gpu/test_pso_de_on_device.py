"""PSO and DE on a device, and the selection operators on CuPy arrays."""

from __future__ import annotations

import numpy as np
import pytest
from gpu_common import SIZES, population

import pygpumeta
from pygpumeta import Constraint, selection
from pygpumeta import reference as ref
from pygpumeta.algorithms import _array

ALGORITHMS = ("pso", "de")
SELECTIONS = ("native", "tournament", "ballot_elite", "migration")
SPHERE_SOURCE = """
double evaluate(const double* x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += (x[i] - 1.5) * (x[i] - 1.5); return s;
}
"""


def shifted(x):
    return ((x - 1.5) ** 2).sum(axis=1)


@pytest.mark.parametrize("n", SIZES)
def test_selection_on_cupy_arrays_matches_the_reference(n):
    import cupy as cp

    values, seed = population(n, 1)
    d_values = cp.asarray(values)
    got = selection.tournament(d_values, size=4, seed=seed)
    assert isinstance(got, cp.ndarray) and got.dtype == np.int32
    assert np.array_equal(cp.asnumpy(got), ref.tournament(values, size=4, seed=seed))
    out = cp.zeros(n, np.int32)
    assert selection.migration(d_values, rate=0.5, seed=seed, out=out) is out
    assert np.array_equal(cp.asnumpy(out), ref.migration(values, rate=0.5, seed=seed))
    got = cp.asnumpy(selection.ballot_elite(d_values, alpha=1.1, seed=seed))
    if n <= 256:
        assert np.array_equal(got, ref.ballot_elite(values, alpha=1.1, seed=seed))
    else:
        assert got.min() >= 0 and got.max() < n


def test_selection_rejects_strided_cupy_arrays():
    import cupy as cp

    with pytest.raises(ValueError, match="values must be contiguous"):
        selection.tournament(cp.zeros(64)[::2], seed=0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("name", SELECTIONS)
@pytest.mark.parametrize("mode", ["gpu", "host"])
def test_every_selection_in_both_array_modes(algorithm, name, mode):
    common = {
        "dim": 4, "algorithm": algorithm, "selection": name, "pop_size": 200, "max_iter": 150,
        "mode": mode, "seed": 5,
    }  # fmt: skip
    result = pygpumeta.minimize(shifted, (-5.0, 5.0), **common)
    again = pygpumeta.minimize(shifted, (-5.0, 5.0), **common)
    assert result.fun == again.fun and np.array_equal(result.x, again.x)
    assert result.nfev == 200 * 151 and result.history.shape == (151,)
    assert (np.diff(result.history) <= 0).all() and result.history[-1] == result.fun
    assert float(shifted(result.x[None, :])[0]) == pytest.approx(result.fun, rel=1e-9, abs=1e-15)
    # Tolerance: as on the CPU, where the same configuration reaches 1e-4
    # with the smaller population of 70.
    assert result.fun <= 1e-4


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("name", ["tournament", "ballot_elite", "migration"])
def test_host_mode_with_kernels_equals_host_mode_with_the_reference(monkeypatch, algorithm, name):
    from test_array_optimizers_shared import ReferenceSelector

    common = {
        "dim": 4, "algorithm": algorithm, "selection": name, "pop_size": 200, "max_iter": 40,
        "mode": "host", "seed": 2,
    }  # fmt: skip
    on_device = pygpumeta.minimize(shifted, (-5.0, 5.0), **common)
    monkeypatch.setattr(_array, "DeviceSelector", ReferenceSelector)
    on_host = pygpumeta.minimize(shifted, (-5.0, 5.0), **common)
    assert on_device.fun == on_host.fun and np.array_equal(on_device.x, on_host.x)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind", ["builtin", "cuda", "cuda-host"])
def test_kernel_objectives(algorithm, kind):
    common = {"algorithm": algorithm, "selection": "native", "pop_size": 200, "max_iter": 300}
    if kind == "builtin":
        result = pygpumeta.minimize("sphere", (-5.0, 5.0), dim=5, **common)
        target = 0.0
    else:
        # Constraints may also run on the host while the objective stays in
        # the kernels; the constraint here is inactive at the optimum.
        extra = {}
        if kind == "cuda-host":
            extra = {"mode": "host", "constraints": [Constraint(lambda x: x[:, 0] - 4.0)]}
        result = pygpumeta.minimize(
            pygpumeta.CudaObjective(SPHERE_SOURCE), (-5.0, 5.0), dim=5, **common, **extra
        )
        target = 1.5
        assert result.feasible
    # Tolerance: a five-dimensional quadratic, 300 iterations; both
    # algorithms are far below 1e-6 by then.
    assert result.fun <= 1e-6 and np.abs(result.x - target).max() <= 1e-2
    assert result.nfev == 200 * 301


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_maximise_constraints_and_decoding_on_the_device(algorithm):
    import cupy as cp

    from pygpumeta.space import Categorical, Integer, Real

    def objective(x):
        assert isinstance(x, cp.ndarray)
        return x[:, 0] + x[:, 1] - 0.1 * (x[:, 2] - 3.0) ** 2 - (x[:, 3] == 1.0)

    result = pygpumeta.minimize(
        objective,
        [Real(0.0, 2.0), (-1.0, 4.0), Integer(0, 6), Categorical(["no", "yes", "maybe"])],
        algorithm=algorithm,
        selection="tournament",
        pop_size=256,
        max_iter=300,
        maximize=True,
        constraints=[Constraint(lambda x: x[:, 0] + x[:, 1] - 3.0)],
        penalty=1e3,
    )
    # Tolerance: see the CPU test of the same problem; the tournament
    # converges more slowly, hence the wider margin on the value.
    assert result.fun == pytest.approx(3.0, abs=2e-2)
    assert result.constraint_violation <= 5e-3
    assert result.values[2] == 3 and result.values[3] in ("no", "maybe")


def test_gpu_mode_is_the_default_and_uses_cupy_arrays():
    import cupy as cp

    seen = set()

    def objective(x):
        seen.add(type(x))
        return shifted(x)

    pygpumeta.minimize(objective, (-5.0, 5.0), dim=3, algorithm="pso", pop_size=64, max_iter=3)
    assert seen == {cp.ndarray}
