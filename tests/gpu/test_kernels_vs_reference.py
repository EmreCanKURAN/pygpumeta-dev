"""Selection kernels against the CPU reference, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
from gpu_common import SIZES, population

from pygpumeta import reference as ref

REPS = (0, 1, 2)
TOURNAMENT_SIZES = (1, 3, 6)


def cases():
    return [(n, rep, ts) for n in SIZES for rep in REPS for ts in TOURNAMENT_SIZES]


def invalidate(b):
    b.sel.fill(np.int32(-7))


@pytest.mark.parametrize(("n", "rep", "ts"), cases())
def test_tournament(selection, buffers, n, rep, ts):
    values, seed = population(n, rep)
    b = buffers(values)
    invalidate(b)
    selection.tournament(b.obj, b.sel, n, ts, seed)
    assert np.array_equal(b.selected(), ref.tournament(values, size=ts, seed=seed))


@pytest.mark.parametrize("rate", [0.0, 1 / 32, 0.25, 1.0])
@pytest.mark.parametrize(("n", "rep", "ts"), cases())
def test_migration(selection, buffers, n, rep, ts, rate):
    values, seed = population(n, rep)
    b = buffers(values)
    invalidate(b)
    selection.migration(b.obj, b.sel, n, ts, rate, seed)
    assert np.array_equal(b.selected(), ref.migration(values, rate=rate, size=ts, seed=seed))


@pytest.mark.parametrize("variant", ["elite", "noswitch", "local"])
@pytest.mark.parametrize(("n", "rep", "ts"), cases())
def test_ballot(selection, buffers, n, rep, ts, variant):
    values, seed = population(n, rep)
    b = buffers(values)
    invalidate(b)
    selection.ballot(b.obj, b.fit, b.sel, b.mean, n, ts, 1.2, seed, variant)
    fit = ref.fitness(values)
    assert np.array_equal(b.fit.get(), fit)
    # Above one block the kernel adds the block sums in an unspecified order,
    # so the reference is given the mean the device computed.
    device_mean = float(b.mean.get()[0])
    if variant != "local":
        if n <= 256:
            assert device_mean == ref._block_mean(fit)
        else:
            # Tolerance: sums of n positive terms in different orders differ
            # by at most a few rounding steps per addition level.
            assert device_mean == pytest.approx(ref._block_mean(fit), rel=1e-13)
    expected = ref._ballot(fit, ts, 1.2, seed, mean=device_mean, variant=variant)
    assert np.array_equal(b.selected(), expected)


@pytest.mark.parametrize(("n", "rep", "ts"), cases())
def test_elitist_global_best_and_global_tournament(selection, buffers, n, rep, ts):
    values, seed = population(n, rep)
    b = buffers(values)
    invalidate(b)
    selection.elitist(b.obj, b.sel, n, ts, seed)
    assert np.array_equal(b.selected(), ref._elitist(values, ts, seed))
    invalidate(b)
    selection.global_best(b.obj, b.sel, n)
    assert np.array_equal(b.selected(), ref._global_best(values))
    invalidate(b)
    selection.global_tournament(b.obj, b.sel, n, ts, seed)
    assert np.array_equal(b.selected(), ref._global_tournament(values, ts, seed))


@pytest.mark.parametrize("n", SIZES)
def test_same_seed_gives_same_selection(selection, buffers, n):
    values, seed = population(n, 0)
    b = buffers(values)
    runs = []
    for _ in range(3):
        invalidate(b)
        selection.migration(b.obj, b.sel, n, 3, 0.25, seed)
        runs.append(b.selected())
    assert np.array_equal(runs[0], runs[1]) and np.array_equal(runs[0], runs[2])
