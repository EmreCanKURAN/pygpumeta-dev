from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pygpumeta import reference as ref

values = hnp.arrays(
    np.float64,
    st.integers(1, 200),
    elements=st.one_of(
        st.floats(-1e6, 1e6, allow_nan=False),
        st.integers(-3, 3).map(float),  # frequent ties
    ),
)
sizes = st.integers(1, 6)
seeds = st.integers(0, 2**32 - 1)
operators = st.sampled_from(["tournament", "migration", "ballot_elite"])


def run(name: str, v: np.ndarray, size: int, seed: int) -> np.ndarray:
    return getattr(ref, name)(v, size=size, seed=seed)


@given(operators, values, sizes, seeds)
def test_indices_are_valid_and_reproducible(name, v, size, seed):
    sel = run(name, v, size, seed)
    assert sel.shape == v.shape and sel.dtype == np.int64
    assert sel.min() >= 0 and sel.max() < len(v)
    assert np.array_equal(sel, run(name, v, size, seed))


@given(values, sizes, seeds)
def test_tournament_winner_is_in_the_warp_and_not_worse(v, size, seed):
    sel = ref.tournament(v, size=size, seed=seed)
    own = np.arange(len(v))
    assert np.array_equal(sel // 32, own // 32)
    assert np.all(v[sel] <= v)


@given(values, sizes, seeds)
def test_tournament_depends_only_on_the_order_of_the_values(v, size, seed):
    ranks = np.unique(v, return_inverse=True)[1].astype(np.float64)
    assert np.array_equal(
        ref.tournament(v, size=size, seed=seed), ref.tournament(ranks, size=size, seed=seed)
    )


@given(values, sizes, seeds)
def test_a_larger_tournament_never_selects_a_worse_individual(v, size, seed):
    # Competitor t is drawn from fixed bits of the state, so the first
    # competitors of a larger tournament are those of the smaller one.
    if size < 6:
        small = ref.tournament(v, size=size, seed=seed)
        large = ref.tournament(v, size=size + 1, seed=seed)
        assert np.all(v[large] <= v[small])


@given(values, sizes, seeds)
def test_migration_limits(v, size, seed):
    plain = ref.tournament(v, size=size, seed=seed)
    assert np.array_equal(ref.migration(v, rate=0.0, size=size, seed=seed), plain)
    full = ref.migration(v, rate=1.0, size=size, seed=seed)
    assert np.all(v[full] == v.min())
    assert np.array_equal(full, ref._elitist(v, size, seed))


@given(values, sizes, seeds, st.floats(0.0, 1.0))
def test_migration_selects_the_tournament_winner_or_the_global_best(v, size, seed, rate):
    sel = ref.migration(v, rate=rate, size=size, seed=seed)
    plain = ref.tournament(v, size=size, seed=seed)
    assert np.all((sel == plain) | (sel == int(np.argmin(v))))


@given(values, sizes, seeds, st.floats(0.5, 3.0))
def test_ballot_stays_in_the_warp(v, size, seed, alpha):
    sel = ref.ballot_elite(v, size=size, alpha=alpha, seed=seed)
    assert np.array_equal(sel // 32, np.arange(len(v)) // 32)


@given(values, sizes, seeds)
def test_ballot_with_one_elite_per_warp_selects_it(v, size, seed):
    fit = ref.fitness(v)
    # A threshold between the two largest fitness values of the population
    # leaves exactly one elite, in one warp, unless the maximum is tied.
    top = np.sort(fit)[::-1]
    if len(top) < 2 or top[0] == top[1]:
        return
    mean = (top[0] + top[1]) / 2
    if not top[1] <= mean < top[0]:
        return
    sel = ref.ballot_elite(v, size=size, alpha=1.0, seed=seed, mean=mean)
    best = int(np.argmax(fit))
    warp = np.arange(len(v)) // 32 == best // 32
    assert np.all(sel[warp] == best)


@given(values)
def test_fitness_reverses_the_order(v):
    fit = ref.fitness(v)
    assert np.all(fit > 0)
    order = np.argsort(v, kind="stable")
    assert np.all(np.diff(fit[order]) <= 0)


@settings(max_examples=25)
@given(st.integers(1, 3000), seeds)
def test_block_mean_is_close_to_the_plain_mean(n, seed):
    fit = np.random.RandomState(seed % 2**31).uniform(0.0, 2.0, n)
    # Tolerance: both are sums of n terms of similar size, so they agree to
    # a few rounding steps times log2(n).
    assert abs(ref._block_mean(fit) - fit.mean()) <= 1e-13 * max(1.0, abs(fit.mean()))
