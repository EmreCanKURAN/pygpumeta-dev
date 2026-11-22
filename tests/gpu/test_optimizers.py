"""Optimizer classes and minimize on a device."""

from __future__ import annotations

import numpy as np
import pytest
from gpu_common import PAPER_TWO_PI, SIZES, population

import pygpumeta
from pygpumeta import TerminationReason, benchmarks, selection
from pygpumeta import reference as ref

CLASSES = {
    "ABC": pygpumeta.ABC,
    "GWO": pygpumeta.GWO,
    "WOA": pygpumeta.WOA,
    "SCA": pygpumeta.SCA,
    "SSA": pygpumeta.SSA,
    "HHO": pygpumeta.HHO,
}
# Library selection and the method name of the paper's runners.
SELECTIONS = {
    "native": "naive",
    "tournament": "tournament_safe",
    "ballot_elite": "ballot_safe",
    "migration": "mig_0.25",
}


@pytest.mark.parametrize("alg", CLASSES)
@pytest.mark.parametrize("sel", SELECTIONS)
@pytest.mark.parametrize("name", ["sphere", "ackley", "levy"])
@pytest.mark.parametrize("ps", [100, 1024])
def test_best_value_equals_the_paper_runner(paper, alg, sel, name, ps):
    info = benchmarks.info(name)
    with paper.use_kernels(paper.exact_constants_handles()):
        expected = paper.GPU_FUNCS[alg](
            ps, 60, info._fid, info.bounds, dim=30, method=SELECTIONS[sel], seed=11
        )["best"]
    opt = CLASSES[alg](
        name, dim=30, pop_size=ps, max_iter=60, selection=sel, seed=11, _constants=PAPER_TWO_PI
    )
    assert opt.run().fun == expected


@pytest.mark.parametrize("alg", CLASSES)
@pytest.mark.parametrize("sel", SELECTIONS)
def test_every_algorithm_with_every_selection(alg, sel):
    result = pygpumeta.minimize(
        "rastrigin",
        dim=20,
        algorithm=alg.lower(),
        selection=sel,
        pop_size=1000,
        max_iter=40,
        seed=5,
    )
    again = pygpumeta.minimize(
        "rastrigin",
        dim=20,
        algorithm=alg.lower(),
        selection=sel,
        pop_size=1000,
        max_iter=40,
        seed=5,
    )
    other = pygpumeta.minimize(
        "rastrigin",
        dim=20,
        algorithm=alg.lower(),
        selection=sel,
        pop_size=1000,
        max_iter=40,
        seed=6,
    )
    assert np.isfinite(result.fun) and result.nit == 40
    assert result.termination is TerminationReason.MAX_ITER
    assert result.algorithm == alg.lower() and result.selection.startswith(sel)
    # Same seed, same result; another seed, another run.
    assert result.fun == again.fun and np.array_equal(result.x, again.x)
    assert np.array_equal(result.history, again.history)
    assert not np.array_equal(result.x, other.x)
    # The reported position reproduces the reported value. Tolerance: device
    # and NumPy sum 20 terms in different orders and use different libm.
    lo, hi = benchmarks.info("rastrigin").bounds
    assert result.x.shape == (20,) and (result.x >= lo).all() and (result.x <= hi).all()
    assert benchmarks.evaluate("rastrigin", result.x)[0] == pytest.approx(
        result.fun, rel=1e-9, abs=1e-12
    )
    # One history entry per iteration plus the final one, never increasing.
    assert result.history.shape == (41,) and result.history[-1] == result.fun
    assert (np.diff(result.history) <= 0).all()


@pytest.mark.parametrize("alg", ["GWO", "WOA", "SCA", "SSA"])
def test_evaluation_count_of_population_based_updates(alg):
    result = CLASSES[alg]("sphere", dim=10, pop_size=100, max_iter=25).run()
    assert result.nfev == 100 * 26


def test_evaluation_count_of_abc_and_hho():
    abc = pygpumeta.ABC("sphere", dim=10, pop_size=100, max_iter=25).run()
    # Initialisation, employed and onlooker phase, at most one scout per iteration.
    assert 100 + 25 * 200 <= abc.nfev <= 100 + 25 * 201
    hho = pygpumeta.HHO("sphere", dim=10, pop_size=100, max_iter=25).run()
    # One evaluation per hawk and iteration plus up to two per rapid dive.
    assert 100 * 26 <= hho.nfev <= 100 * 26 + 2 * 100 * 25
    print(f"nfev: abc {abc.nfev}, hho {hho.nfev}")


def test_interleaved_optimizers_keep_separate_evaluation_counts():
    alone = pygpumeta.HHO("sphere", dim=10, pop_size=100, max_iter=20, seed=1).run().nfev
    first = pygpumeta.HHO("sphere", dim=10, pop_size=100, max_iter=20, seed=1)
    second = pygpumeta.HHO("sphere", dim=10, pop_size=100, max_iter=20, seed=1)
    for _ in range(20):
        first.step()
        second.step()
    assert first.result().nfev == alone and second.result().nfev == alone


@pytest.mark.parametrize("name", benchmarks.names())
@pytest.mark.parametrize("dim", [1, 2, 30, 200])
def test_device_objective_matches_numpy(runtime, name, dim):
    from pygpumeta._backend.selection import load_kernels

    info = benchmarks.info(name)
    points = np.random.RandomState(3).uniform(*info.bounds, (256, dim))
    d_pop = runtime.gpuarray.to_gpu(points.ravel())
    d_out = runtime.gpuarray.zeros(256, np.float64)
    load_kernels().algorithms.kernel("eval_pop")(
        d_pop,
        d_out,
        np.int32(256),
        np.int32(dim),
        np.int32(info._fid),
        block=(256, 1, 1),
        grid=(1, 1),
    )
    # Tolerance: different summation order and libm over up to 200 terms;
    # the absolute part covers values that cancel to nearly zero.
    assert d_out.get() == pytest.approx(benchmarks.evaluate(name, points), rel=1e-10, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "alg", "tolerance"),
    [
        # Unimodal functions on which these algorithms contract geometrically;
        # 300 iterations with 1024 individuals leave values far below 1e-6.
        ("sphere", "gwo", 1e-6),
        ("sphere", "hho", 1e-6),
        ("sum_squares", "woa", 1e-6),
        ("rastrigin", "gwo", 1e-6),
        # Ackley cannot go below about 4e-16 in double precision.
        ("ackley", "hho", 1e-9),
    ],
)
def test_known_optima_are_found(name, alg, tolerance):
    result = pygpumeta.minimize(
        name, dim=10, algorithm=alg, selection="native", pop_size=1024, max_iter=300, seed=2
    )
    info = benchmarks.info(name)
    assert result.fun - info.optimum(10) <= tolerance
    # Tolerance on the position: f <= 1e-6 bounds each coordinate of these
    # functions by 1e-3 or better.
    assert np.abs(result.x - info.minimizer).max() <= 1e-2


def test_step_wise_control_equals_run():
    whole = pygpumeta.GWO("ackley", dim=8, pop_size=200, max_iter=30, seed=9).run()
    opt = pygpumeta.GWO("ackley", dim=8, pop_size=200, max_iter=30, seed=9)
    values = []
    while not opt.finished:
        opt.step()
        values.append(opt.best_value)
    assert opt.iteration == 30 and opt.result().fun == whole.fun
    assert values == sorted(values, reverse=True)
    with pytest.raises(RuntimeError, match="this run has ended"):
        opt.step()


def test_stopping_rules():
    kwargs = {"dim": 8, "pop_size": 256, "max_iter": 500, "seed": 4}
    by_target = pygpumeta.minimize("sphere", target=1e-3, **kwargs)
    assert by_target.termination is TerminationReason.TARGET
    assert by_target.fun <= 1e-3 and by_target.nit < 500
    assert by_target.history[-1] == by_target.fun

    stalled = pygpumeta.minimize("sphere", patience=5, tol=1e300, **kwargs)
    assert stalled.termination is TerminationReason.STALLED and stalled.nit == 6

    seen = []
    by_callback = pygpumeta.minimize(
        "sphere",
        callback=lambda opt: seen.append(opt.iteration) or len(seen) == 3,
        history_every=10,
        **kwargs,
    )
    assert by_callback.termination is TerminationReason.CALLBACK
    assert seen == [10, 20, 30] and by_callback.nit == 30

    timed = pygpumeta.minimize("sphere", max_time=1e-9, **kwargs)
    assert timed.termination is TerminationReason.MAX_TIME and timed.nit == 1


def test_history_interval():
    result = pygpumeta.minimize("sphere", dim=5, pop_size=64, max_iter=25, history_every=10)
    # Iterations 0, 10 and 20, plus the final value.
    assert result.history.shape == (4,)


@pytest.mark.parametrize("n", SIZES)
def test_standalone_operators_match_the_reference(runtime, n):
    values, seed = population(n, 1)
    d_values = runtime.gpuarray.to_gpu(values)
    got = selection.tournament(d_values, size=4, seed=seed)
    assert got.dtype == np.int32
    assert np.array_equal(got.get(), ref.tournament(values, size=4, seed=seed))
    out = runtime.gpuarray.zeros(n, np.int32)
    assert selection.migration(d_values, rate=0.5, seed=seed, out=out) is out
    assert np.array_equal(out.get(), ref.migration(values, rate=0.5, seed=seed))
    got = selection.ballot_elite(d_values, alpha=1.1, seed=seed).get()
    if n <= 256:
        assert np.array_equal(got, ref.ballot_elite(values, alpha=1.1, seed=seed))
    else:
        assert got.min() >= 0 and got.max() < n and np.array_equal(got // 32, np.arange(n) // 32)


def test_standalone_operators_reject_wrong_device_arrays(runtime):
    values = runtime.gpuarray.zeros(8, np.float32)
    with pytest.raises(
        ValueError, match="values must be a non-empty one-dimensional array of float64"
    ):
        selection.tournament(values, seed=0)
    values = runtime.gpuarray.zeros(8, np.float64)
    with pytest.raises(ValueError, match="out must have length 8, got 4"):
        selection.tournament(values, seed=0, out=runtime.gpuarray.zeros(4, np.int32))


def test_doctor_reports_a_working_installation():
    from pygpumeta import doctor

    lines: list[str] = []
    code = doctor.main(write=lines.append)
    print("\n".join(lines))
    assert code == 0 and not [line for line in lines if line.startswith("[FAIL]")]
    assert any("Optimisation run (compiled kernels)" in line and "[ ok ]" in line for line in lines)
