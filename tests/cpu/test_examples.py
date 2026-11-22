"""The examples, run on the CPU with small budgets.

They are checked against the same reference values that they print. The
algorithms used here are those that need no GPU (PSO and DE with their
native rules); the examples default to other settings on a device.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
pytest.importorskip("scipy")


@pytest.fixture(scope="module", autouse=True)
def examples_on_path():
    sys.path.insert(0, str(EXAMPLES))
    yield
    sys.path.remove(str(EXAMPLES))


def run(name, **options):
    module = importlib.import_module(name)
    options = {"mode": "host", "algorithm": "de", "selection": "native", "seed": 1, **options}
    return module, module.run(**options)


def test_antenna_array_approaches_the_dolph_chebyshev_bound():
    from scipy.signal.windows import chebwin

    module, result = run("01_antenna_array", pop_size=120, max_iter=1500)
    with pytest.warns(UserWarning):
        weights = chebwin(20, at=30)[10:][None, :]
    # The Dolph-Chebyshev weights give the design level on the grid used.
    assert module.sidelobe_level_db(weights)[0] == pytest.approx(-30.0, abs=1e-3)
    # Tolerance: -30 dB is a lower bound; with two nulls enforced the
    # optimum is slightly above it, and 1 dB allows for the small budget.
    assert result.feasible and -30.0 <= result.fun <= -29.0
    assert (module.null_excess_db(result.x[None, :]) <= 0.0).all()


def test_solar_cell_reaches_the_published_error():
    module, result = run("02_solar_cell", pop_size=100, max_iter=3000)
    published = np.array([[0.76077553, 0.32302, 0.03637709, 53.71852, 1.48118359]])
    assert module.rmse(published)[0] == pytest.approx(module.REFERENCE_RMSE, rel=1e-4)
    # Tolerance: DE converges to the published optimum; 0.1 percent covers
    # the last digits of the reference value.
    assert result.fun == pytest.approx(module.REFERENCE_RMSE, rel=1e-3)


def test_truss_reaches_the_published_weight():
    module, result = run("03_truss", pop_size=100, max_iter=3000)
    published = np.array([[30.52, 0.1, 23.20, 15.22, 0.1, 0.55, 7.46, 21.04, 21.53, 0.1]])
    # The published areas, rounded to two decimals, weigh 5061.04 lb and
    # exceed no limit by more than the rounding.
    assert module.weight(published)[0] == pytest.approx(module.REFERENCE_WEIGHT, abs=0.5)
    assert module.limits(published).max() == pytest.approx(0.0, abs=1e-3)
    # Tolerance: one pound in five thousand; the feasibility rule keeps the
    # result feasible exactly.
    assert result.feasible and result.fun == pytest.approx(module.REFERENCE_WEIGHT, abs=1.0)


def test_load_dispatch_cost_function_and_result():
    module, result = run("04_load_dispatch", pop_size=256, max_iter=3000)
    published = np.array(
        [[628.3185, 222.7493, 149.5995, 109.8666, 109.8665, 109.8665, 109.8666, 60.0, 109.8666,
          40.0, 40.0, 55.0, 55.0]]
    )  # fmt: skip
    assert module.fuel_cost(published)[0] == pytest.approx(module.REFERENCE_COST, abs=0.01)
    assert module.cost(published[:, 1:])[0] == pytest.approx(module.REFERENCE_COST, abs=0.01)
    power = module.dispatch(result.x[None, :])[0]
    # The first unit supplies the remainder, so the demand is met exactly
    # (to rounding) and all units are within their limits.
    assert power.sum() == pytest.approx(module.DEMAND, abs=1e-9) and result.feasible
    assert (power >= module.P_MIN - 1e-9).all() and (power <= module.P_MAX + 1e-9).all()
    # Tolerance: four seeds with this budget gave 17969 to 18036 $/h; the
    # reference is the best value known and is not undercut.
    assert module.REFERENCE_COST - 0.01 <= result.fun <= module.REFERENCE_COST + 100.0


def test_thresholding_finds_the_enumerated_optimum():
    module, result = run("05_image_thresholding", pop_size=100, max_iter=200)
    optimum, thresholds = module.exhaustive()
    assert result.fun == pytest.approx(optimum, rel=1e-12)
    assert tuple(sorted(result.values)) == thresholds


def test_fir_filter_matches_the_equiripple_design():
    module, result = run("06_fir_filter", pop_size=150, max_iter=4000)
    # Tolerance: on the frequency grid the two designs agree to a percent.
    assert result.fun == pytest.approx(module.equiripple(), rel=0.01)


def test_svm_tuning_runs_and_is_not_worse_than_the_defaults():
    pytest.importorskip("sklearn")
    module, result = run("07_svm_tuning", algorithm="pso", pop_size=8, max_iter=3)
    assert result.values[2] in module.KERNELS and result.nfev == 8 * 4
    # Tolerance: 32 evaluations are a random search; two percentage points
    # below the defaults would still pass, a broken objective would not.
    assert 1.0 - result.fun >= 1.0 - module.cv_error(None, None, "rbf") - 0.02


def test_custom_algorithm_solves_its_test_function():
    module = importlib.import_module("08_custom_algorithm")
    best, position, history = module.evolve("host", pop_size=1024, max_iter=600, seed=0)
    # Tolerance: four seeds gave values between 1e-9 and 6e-9.
    assert best <= 1e-6 and np.abs(position).max() <= 1e-3
    assert [v for _, v in history] == sorted((v for _, v in history), reverse=True)


def test_examples_run_as_scripts():
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / "05_image_thresholding.py"), "--mode", "host", "--quick",
         "--algorithm", "de", "--selection", "native"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "== Image thresholding" in done.stdout and "exact optimum by enumeration" in done.stdout
