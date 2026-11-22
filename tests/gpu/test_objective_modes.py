"""The four ways of supplying an objective, on a device."""

from __future__ import annotations

import numpy as np
import pytest

import pygpumeta
from pygpumeta import TerminationReason
from pygpumeta._backend.module import CompilationError

ALGORITHMS = ("abc", "gwo", "woa", "sca", "ssa", "hho")
SELECTIONS = ("native", "tournament", "ballot_elite", "migration")

# The built-in sphere, written out again as a user objective.
SPHERE_SOURCE = """
double evaluate(const double* x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += x[i] * x[i]; return s;
}
"""
SHIFTED_SOURCE = """
__device__ double square(double v) { return v * v; }
double evaluate(const double* x, int d) {
    double s = 0.0;
    for (int i = 0; i < d; i++) s += square(x[i] - 1.0);
    return s;
}
"""


def shifted(x):
    return ((x - 1.0) ** 2).sum(axis=1)


@pytest.mark.parametrize("algorithm", ["abc", "hho"])
@pytest.mark.parametrize("selection", SELECTIONS)
@pytest.mark.parametrize("name", ["sphere", "ackley", "levy"])
@pytest.mark.parametrize("ps", [100, 1024])
def test_separated_evaluation_gives_the_run_of_the_fused_kernels(algorithm, selection, name, ps):
    common = {
        "dim": 30, "algorithm": algorithm, "selection": selection, "pop_size": ps,
        "max_iter": 60, "seed": 11,
    }  # fmt: skip
    fused = pygpumeta.minimize(name, options={"evaluation": "fused"}, **common)
    separated = pygpumeta.minimize(name, options={"evaluation": "separated"}, **common)
    assert separated.fun == fused.fun
    assert np.array_equal(separated.x, fused.x)
    assert np.array_equal(separated.history, fused.history)
    assert separated.nfev >= fused.nfev


def test_evaluation_counts_of_separated_updates():
    common = {"dim": 10, "pop_size": 100, "max_iter": 25, "options": {"evaluation": "separated"}}
    assert pygpumeta.minimize("sphere", algorithm="abc", **common).nfev == 100 + 25 * 201
    assert pygpumeta.minimize("sphere", algorithm="hho", **common).nfev == 100 * 26 + 25 * 200


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("selection", ["native", "tournament"])
def test_cuda_objective_equal_to_a_builtin_gives_the_same_run(algorithm, selection):
    common = {
        "dim": 20, "algorithm": algorithm, "selection": selection, "pop_size": 500,
        "max_iter": 40, "seed": 3,
    }  # fmt: skip
    builtin = pygpumeta.minimize("sphere", **common)
    custom = pygpumeta.minimize(pygpumeta.CudaObjective(SPHERE_SOURCE), (-100.0, 100.0), **common)
    assert custom.fun == builtin.fun and np.array_equal(custom.x, builtin.x)
    assert custom.nfev == builtin.nfev


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind", ["gpu", "host", "cuda"])
def test_every_algorithm_in_every_custom_mode(algorithm, kind):
    objective = pygpumeta.CudaObjective(SHIFTED_SOURCE) if kind == "cuda" else shifted
    kwargs = {"mode": kind} if kind != "cuda" else {}
    common = {
        "dim": 5, "algorithm": algorithm, "selection": "native", "pop_size": 256,
        "max_iter": 150, "seed": 1, **kwargs,
    }  # fmt: skip
    result = pygpumeta.minimize(objective, (-5.0, 5.0), **common)
    again = pygpumeta.minimize(objective, (-5.0, 5.0), **common)
    assert result.termination is TerminationReason.MAX_ITER
    assert result.fun == again.fun and np.array_equal(result.x, again.x)
    assert shifted(result.x[None, :])[0] == pytest.approx(result.fun, rel=1e-9, abs=1e-12)
    # Tolerance: this is a smoke test of the plumbing, not of solution
    # quality. A uniformly random point has a value below 1 with probability
    # 5e-5, so 256 random points almost never contain one, while every
    # algorithm that follows the objective gets there within 150 iterations.
    assert result.fun <= 1.0


def test_gpu_and_host_mode_agree_closely():
    common = {"dim": 5, "algorithm": "gwo", "pop_size": 256, "max_iter": 30, "seed": 2}
    on_gpu = pygpumeta.minimize(shifted, (-5.0, 5.0), **common)
    on_host = pygpumeta.minimize(shifted, (-5.0, 5.0), mode="host", **common)
    by_row = pygpumeta.minimize(
        lambda row: float(((row - 1.0) ** 2).sum()),
        (-5.0, 5.0),
        mode="host",
        vectorized=False,
        **common,
    )
    assert on_gpu.nfev == on_host.nfev == by_row.nfev == 256 * 31
    # CuPy and NumPy may sum five terms in a different order, which can
    # change the last bit of a value and, through a comparison, the run.
    # Tolerance: both runs are far from converged after 30 iterations and
    # stay within the same order of magnitude.
    assert on_host.fun == pytest.approx(on_gpu.fun, rel=10.0)
    assert by_row.fun == pytest.approx(on_host.fun, rel=10.0)


def test_cupy_and_pycuda_share_one_context(runtime):
    import cupy as cp

    buffer = runtime.gpuarray.to_gpu(np.arange(12, dtype=np.float64))
    view = cp.asarray(buffer)
    assert view.data.ptr == int(buffer.gpudata)
    view *= 2.0
    assert np.array_equal(buffer.get(), 2.0 * np.arange(12))
    buffer.fill(3.0)
    assert float(view.sum()) == 36.0
    assert cp.cuda.Device().id == 0


def test_gpu_objective_receives_cupy_arrays_on_the_device():
    import cupy as cp

    seen = []

    def objective(x):
        seen.append((type(x), x.shape, x.dtype))
        return shifted(x)

    pygpumeta.minimize(objective, (-5.0, 5.0), dim=4, algorithm="abc", pop_size=50, max_iter=2)
    assert {entry[0] for entry in seen} == {cp.ndarray}
    assert {entry[1] for entry in seen} == {(50, 4), (1, 4)}
    assert {entry[2] for entry in seen} == {np.dtype(np.float64)}


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_non_finite_objective_values_do_not_break_a_run(algorithm):
    import cupy as cp

    def mostly_invalid(x):
        values = shifted(x)
        return cp.where(x[:, 0] > 0.0, values, cp.nan)

    def never_valid(x):
        return cp.full(x.shape[0], cp.inf)

    result = pygpumeta.minimize(
        mostly_invalid, (-5.0, 5.0), dim=3, algorithm=algorithm, pop_size=100, max_iter=20
    )
    assert np.isfinite(result.fun) and result.fun < 1e299 and result.x[0] > 0.0
    result = pygpumeta.minimize(
        never_valid, (-5.0, 5.0), dim=3, algorithm=algorithm, pop_size=100, max_iter=5
    )
    assert result.fun == 1e299 and np.isfinite(result.x).all()


def test_cuda_objective_returning_nan_is_bounded():
    objective = pygpumeta.CudaObjective(
        "double evaluate(const double* x, int d) { return sqrt(x[0]); }"
    )
    result = pygpumeta.minimize(
        objective, (-4.0, 4.0), dim=2, pop_size=64, max_iter=20, selection="native"
    )
    assert 0.0 <= result.fun < 1e299 and result.x[0] >= 0.0


def test_error_paths():
    import cupy as cp

    with pytest.raises(
        TypeError, match=r"must return a cupy\.ndarray, got ndarray; use mode='host'"
    ):
        pygpumeta.minimize(lambda x: np.zeros(len(x)), (-1.0, 1.0), dim=2, pop_size=8, max_iter=1)
    with pytest.raises(ValueError, match=r"an array of shape \(8,\), got shape \(8, 2\)"):
        pygpumeta.minimize(lambda x: x, (-1.0, 1.0), dim=2, pop_size=8, max_iter=1)
    with pytest.raises(ValueError, match=r"got shape \(\)"):
        pygpumeta.minimize(lambda x: cp.sum(x), (-1.0, 1.0), dim=2, pop_size=8, max_iter=1)
    with pytest.raises(ZeroDivisionError):
        pygpumeta.minimize(lambda x: 1 / 0, (-1.0, 1.0), dim=2, pop_size=8, max_iter=1)
    broken = pygpumeta.CudaObjective(
        "double evaluate(const double* x, int d) { return undefined_name; }"
    )
    with pytest.raises(CompilationError, match="the CUDA compiler rejected the kernel source"):
        pygpumeta.minimize(broken, (-1.0, 1.0), dim=2, pop_size=8, max_iter=1)
    # A failed compilation leaves the library usable.
    assert np.isfinite(pygpumeta.minimize("sphere", dim=2, pop_size=8, max_iter=1).fun)
