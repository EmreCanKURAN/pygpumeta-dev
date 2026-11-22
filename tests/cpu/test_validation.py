from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pygpumeta import _validation as v
from pygpumeta import reference as ref

GOOD = np.array([3.0, 1.0, 2.0])


@given(st.integers().filter(lambda s: not 1 <= s <= 6))
def test_tournament_size_out_of_range(size):
    with pytest.raises(ValueError, match="size must be between 1 and 6, got"):
        ref.tournament(GOOD, size=size, seed=0)


@given(st.integers().filter(lambda s: not 0 <= s < 2**32))
def test_seed_out_of_range(seed):
    with pytest.raises(ValueError, match=r"0 <= seed < 2\*\*32, got"):
        ref.tournament(GOOD, seed=seed)


@given(st.one_of(st.floats(), st.text(), st.none(), st.booleans()))
def test_non_integer_arguments(bad):
    with pytest.raises(TypeError, match="must be an integer"):
        ref.tournament(GOOD, size=bad, seed=0)
    with pytest.raises(TypeError, match="must be an integer"):
        ref.tournament(GOOD, seed=bad)


@given(st.floats(allow_nan=True).filter(lambda r: not 0.0 <= r <= 1.0))
def test_rate_out_of_range(rate):
    with pytest.raises(ValueError, match=r"rate must lie in \[0, 1\]"):
        ref.migration(GOOD, rate=rate, seed=0)


@given(st.one_of(st.floats(max_value=0.0), st.just(float("inf")), st.just(float("nan"))))
def test_alpha_must_be_positive_and_finite(alpha):
    with pytest.raises(ValueError, match="alpha must be a finite number greater than 0"):
        ref.ballot_elite(GOOD, alpha=alpha, seed=0)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([], ValueError, "at least one element"),
        ([[1.0, 2.0]], ValueError, r"one-dimensional, got an array of shape \(1, 2\)"),
        ([1.0, float("nan")], ValueError, "must not contain NaN"),
        (["a", "b"], TypeError, "convertible to a float64 array"),
    ],
)
def test_values(values, error, message):
    for operator in (ref.tournament, ref.migration, ref.ballot_elite, ref.fitness):
        with pytest.raises(error, match=message):
            operator(values) if operator is ref.fitness else operator(values, seed=0)


def test_numpy_integers_are_accepted_and_infinite_values_allowed():
    sel = ref.tournament([np.inf, 1.0, -np.inf], size=np.int64(2), seed=np.uint32(7))
    assert sel.tolist() == ref.tournament([np.inf, 1.0, -np.inf], size=2, seed=7).tolist()
    assert v.check_seed(2**32 - 1) == 2**32 - 1
