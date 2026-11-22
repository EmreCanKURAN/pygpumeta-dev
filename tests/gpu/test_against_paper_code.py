"""The library kernels against the unmodified paper implementation.

Both are compiled and run in the same process, so the comparison does not
depend on the GPU model or the toolkit version. Complete runs are compared
in test_optimizers.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from gpu_common import SIZES, population

from pygpumeta._backend.selection import load_kernels

FUNCTIONS = {
    "sphere": (0, (-100.0, 100.0)),
    "rastrigin": (1, (-5.12, 5.12)),
    "rosenbrock": (2, (-30.0, 30.0)),
    "ackley": (3, (-32.768, 32.768)),
    "griewank": (4, (-600.0, 600.0)),
    "schwefel": (5, (-500.0, 500.0)),
    "levy": (6, (-10.0, 10.0)),
    "zakharov": (7, (-5.0, 10.0)),
    "sum_squares": (8, (-10.0, 10.0)),
    "michalewicz": (9, (0.0, 3.141592653589793)),
}


def test_library_defines_every_kernel_the_paper_runners_use(paper):
    module = load_kernels().algorithms
    assert set(paper.HANDLE_MAP.values()) <= module.kernel_names
    assert set(paper.SELECTION_KERNELS) <= load_kernels().selection.kernel_names


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("dim", [2, 30, 100])
def test_objective_values_match_paper_kernels_with_exact_constants(runtime, paper, name, dim):
    fid, (lb, ub) = FUNCTIONS[name]
    points = np.random.RandomState(7).uniform(lb, ub, (256, dim))
    d_pop = runtime.gpuarray.to_gpu(points.ravel())
    d_out = runtime.gpuarray.zeros(256, np.float64)
    results = []
    for kernel in (
        paper.exact_constants_handles()["k_eval_pop"],
        load_kernels().algorithms.kernel("eval_pop"),
    ):
        kernel(
            d_pop,
            d_out,
            np.int32(256),
            np.int32(dim),
            np.int32(fid),
            block=(256, 1, 1),
            grid=(1, 1),
        )
        results.append(d_out.get())
    assert np.array_equal(results[0], results[1])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("rep", [0, 1])
def test_selection_matches_paper_dispatcher(paper, selection, buffers, n, rep):
    values, seed = population(n, rep)
    b = buffers(values)

    def old(method):
        b.sel.fill(np.int32(-7))
        paper.fill_selection(method, b.obj, b.fit, b.sel, b.mean, n, 3, 1.2, seed)
        return b.selected()

    def new(call):
        b.sel.fill(np.int32(-7))
        call()
        return b.selected()

    pairs = {
        "tournament_safe": lambda: selection.tournament(b.obj, b.sel, n, 3, seed),
        "ballot_local_safe": lambda: selection.ballot(
            b.obj, b.fit, b.sel, b.mean, n, 3, 1.2, seed, "local"
        ),
        "elitist_dev": lambda: selection.elitist(b.obj, b.sel, n, 3, seed),
        "gbest": lambda: selection.global_best(b.obj, b.sel, n),
        "global3": lambda: selection.global_tournament(b.obj, b.sel, n, 3, seed),
        "mig_0.25": lambda: selection.migration(b.obj, b.sel, n, 3, 0.25, seed),
    }
    if n <= 256:
        # With several blocks the mean, and through it the elite set, may
        # differ in the last bit between two launches of the same kernel.
        pairs["ballot_safe"] = lambda: selection.ballot(
            b.obj, b.fit, b.sel, b.mean, n, 3, 1.2, seed
        )
        pairs["ballot_noswitch_safe"] = lambda: selection.ballot(
            b.obj, b.fit, b.sel, b.mean, n, 3, 1.2, seed, "noswitch"
        )
    for method, call in pairs.items():
        assert np.array_equal(old(method), new(call)), method


def test_runtime_shares_the_context_of_the_paper_code(runtime, paper):
    assert runtime.driver.Context.get_current() is not None
    assert runtime.device_name == paper.GPU_NAME
