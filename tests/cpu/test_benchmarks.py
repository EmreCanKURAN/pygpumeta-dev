from __future__ import annotations

import numpy as np
import pytest

from pygpumeta import benchmarks


def test_names_and_info():
    assert benchmarks.names() == (
        "sphere",
        "rastrigin",
        "rosenbrock",
        "ackley",
        "griewank",
        "schwefel",
        "levy",
        "zakharov",
        "sum_squares",
        "michalewicz",
    )
    with pytest.raises(ValueError, match="unknown test function 'spere'; available: sphere"):
        benchmarks.info("spere")


@pytest.mark.parametrize("name", benchmarks.names())
@pytest.mark.parametrize("dim", [1, 2, 30])
def test_value_at_the_known_minimiser(name, dim):
    info = benchmarks.info(name)
    if info.minimizer is None:
        assert info.optimum(dim) is None
        return
    value = benchmarks.evaluate(name, np.full(dim, info.minimizer))[0]
    # Tolerance: the optima are sums of at most 30 terms of order one that
    # cancel to rounding error; Ackley alone loses about 4e-16 in exp/log.
    assert value == pytest.approx(info.optimum(dim), abs=1e-11)


@pytest.mark.parametrize("name", benchmarks.names())
def test_no_sampled_point_is_below_the_optimum(name):
    info = benchmarks.info(name)
    points = np.random.RandomState(0).uniform(*info.bounds, (20000, 5))
    values = benchmarks.evaluate(name, points)
    assert values.shape == (20000,) and np.isfinite(values).all()
    if info.optimum(5) is not None:
        assert values.min() >= info.optimum(5)


def test_schwefel_optimum_is_the_residual_of_the_rounded_constant():
    grid = np.linspace(420.9, 421.0, 200001)[:, None]
    values = benchmarks.evaluate("schwefel", grid)
    # Tolerance: grid spacing 5e-7 around a quadratic minimum.
    assert values.min() == pytest.approx(benchmarks.info("schwefel").optimum(1), abs=1e-12)
    assert benchmarks.info("schwefel").optimum(10) == pytest.approx(1.2727566e-4, rel=1e-6)


def test_rastrigin_equals_the_usual_form():
    x = np.random.RandomState(1).uniform(-5.12, 5.12, (100, 7))
    usual = 10 * 7 + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)
    # Tolerance: the usual form cancels 70 against a sum of cosines.
    assert benchmarks.evaluate("rastrigin", x) == pytest.approx(usual, abs=1e-10)


def test_griewank_matches_the_textbook_function_where_all_cosines_are_positive():
    x = np.random.RandomState(2).uniform(-1.5, 1.5, (100, 6))
    textbook = (
        np.sum(x * x, axis=1) / 4000 + 1 - np.prod(np.cos(x / np.sqrt(np.arange(1, 7))), axis=1)
    )
    assert benchmarks.evaluate("griewank", x) == pytest.approx(textbook, abs=1e-12)
    # Documented deviation: a non-positive cosine factor zeroes the product.
    assert benchmarks.evaluate("griewank", [4.0, 0.0])[0] == pytest.approx(1.004)


def test_evaluate_accepts_one_point_and_rejects_bad_shapes():
    assert benchmarks.evaluate("sphere", [3.0, 4.0]).tolist() == [25.0]
    with pytest.raises(ValueError, match=r"shape \(dim,\) or \(n, dim\)"):
        benchmarks.evaluate("sphere", np.zeros((2, 2, 2)))
