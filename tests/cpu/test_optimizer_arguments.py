"""Argument validation happens before any device is touched."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from isolation import run_without_gpu_libraries

import pygpumeta
from pygpumeta import selection
from pygpumeta._result import OptimizeResult, TerminationReason

ALGORITHMS = [
    pygpumeta.ABC,
    pygpumeta.GWO,
    pygpumeta.WOA,
    pygpumeta.SCA,
    pygpumeta.SSA,
    pygpumeta.HHO,
]


@pytest.mark.parametrize("cls", ALGORITHMS)
def test_construction_touches_no_device(cls):
    opt = cls("sphere", dim=5, pop_size=33, max_iter=7, seed=3)
    assert (opt.dim, opt.pop_size, opt.max_iter, opt.iteration, opt.finished) == (
        5,
        33,
        7,
        0,
        False,
    )
    assert opt.best_value == float("inf") and np.isnan(opt.best_x).all() and opt.nfev == 0
    assert repr(opt).startswith(f"{cls.__name__}('sphere', dim=5, pop_size=33")


def test_construction_imports_no_gpu_library():
    out = run_without_gpu_libraries(
        "import pygpumeta\n"
        "for name in ('ABC', 'GWO', 'WOA', 'SCA', 'SSA', 'HHO', 'PSO', 'DE'):\n"
        "    opt = getattr(pygpumeta, name)('sphere', dim=5, pop_size=33, max_iter=7)\n"
        "    assert opt.nfev == 0 and not opt.finished\n"
        "    print(name)\n"
    )
    assert out.split() == ["ABC", "GWO", "WOA", "SCA", "SSA", "HHO", "PSO", "DE"]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"objective": "spere", "dim": 3}, ValueError, "unknown test function 'spere'"),
        (
            {"objective": 3, "dim": 3},
            TypeError,
            "a callable or a CudaObjective, got int",
        ),
        (
            {"objective": lambda x: x, "dim": 3},
            ValueError,
            "bounds is required for objectives other than built-in",
        ),
        (
            {"objective": "sphere", "dim": 3, "mode": "cpu"},
            ValueError,
            "mode must be one of 'auto', 'gpu', 'host', got 'cpu'",
        ),
        (
            {"objective": "sphere", "dim": 3, "mode": "host"},
            ValueError,
            "mode='host' applies to callable objectives and constraints only",
        ),
        (
            {"objective": "sphere", "dim": 3, "vectorized": False},
            ValueError,
            "vectorized=False is only available with mode='host'",
        ),
        (
            {"objective": "sphere", "dim": 3, "vectorized": 1},
            TypeError,
            "vectorized must be True or False",
        ),
        ({"objective": "sphere"}, ValueError, "dim is required"),
        ({"objective": "sphere", "dim": 0}, ValueError, "dim must be between 1 and 1024, got 0"),
        (
            {"objective": "sphere", "dim": 1025},
            ValueError,
            "dim must be between 1 and 1024, got 1025",
        ),
        ({"objective": "sphere", "dim": 2.0}, TypeError, "dim must be an integer, got float"),
        (
            {"objective": "sphere", "dim": 3, "bounds": (1.0, 1.0)},
            ValueError,
            "lower bound must be smaller",
        ),
        (
            {"objective": "sphere", "dim": 3, "bounds": (0.0, float("inf"))},
            ValueError,
            "bounds must be finite",
        ),
        ({"objective": "sphere", "dim": 3, "bounds": (0, 1, 2)}, ValueError, r"got shape \(3,\)"),
        ({"objective": "sphere", "dim": 3, "bounds": "ab"}, TypeError, r"\(lower, upper\) pair"),
        (
            {"objective": "sphere", "dim": 3, "bounds": [(0, 1), (0, 1)]},
            ValueError,
            "dim is 3 but bounds has 2 rows",
        ),
        (
            {"objective": "sphere", "dim": 3, "pop_size": 1},
            ValueError,
            "pop_size must be at least 2, got 1",
        ),
        (
            {"objective": "sphere", "dim": 1024, "pop_size": 2**21},
            ValueError,
            r"pop_size \* dim must be below 2\*\*31",
        ),
        (
            {"objective": "sphere", "dim": 3, "max_iter": 0},
            ValueError,
            "max_iter must be at least 1, got 0",
        ),
        (
            {"objective": "sphere", "dim": 3, "seed": -1},
            ValueError,
            "seed must satisfy 0 <= seed <= 2147483646",
        ),
        ({"objective": "sphere", "dim": 3, "seed": 1.5}, TypeError, "seed must be an integer"),
        (
            {"objective": "sphere", "dim": 3, "history_every": 0},
            ValueError,
            "history_every must be at least 1",
        ),
        (
            {"objective": "sphere", "dim": 3, "selection": "roulette"},
            ValueError,
            "selection must be one of 'auto', 'native'",
        ),
    ],
)
def test_invalid_arguments(kwargs, error, message):
    kwargs = dict(kwargs)
    objective = kwargs.pop("objective")
    with pytest.raises(error, match=message):
        pygpumeta.GWO(objective, **kwargs)
    with pytest.raises(error, match=message):
        pygpumeta.minimize(objective, **kwargs)


def test_bounds_with_one_row_per_dimension_define_dim():
    opt = pygpumeta.GWO("sphere", [(-1.0, 2.0), (0.0, 5.0), (-3.0, -1.0), (0.0, 1.0)])
    assert opt.dim == 4
    assert opt._problem.lower.tolist() == [-1.0, 0.0, -3.0, 0.0]
    assert opt._problem.upper.tolist() == [2.0, 5.0, -1.0, 1.0]


def test_minimize_rejects_unknown_algorithms_and_options():
    with pytest.raises(
        ValueError, match="algorithm must be one of abc, gwo, woa, sca, ssa, hho, pso, de, got 'ga'"
    ):
        pygpumeta.minimize("sphere", dim=3, algorithm="ga")
    with pytest.raises(TypeError, match="algorithm 'gwo' has no option 'limit'"):
        pygpumeta.minimize("sphere", dim=3, options={"limit": 5})
    with pytest.raises(TypeError, match="options must be a mapping"):
        pygpumeta.minimize("sphere", dim=3, options=[("limit", 5)])
    with pytest.raises(ValueError, match="limit must be at least 1"):
        pygpumeta.minimize("sphere", dim=3, algorithm="abc", options={"limit": 0})


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"patience": 0}, ValueError, "patience must be at least 1"),
        ({"tol": -1.0}, ValueError, "tol must be a non-negative number"),
        ({"max_time": 0.0}, ValueError, "max_time must be positive"),
        ({"callback": 3}, TypeError, "callback must be callable"),
    ],
)
def test_invalid_run_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        pygpumeta.GWO("sphere", dim=3).run(**kwargs)


@given(st.integers(-5, 12), st.floats(-1, 3, allow_nan=False), st.floats(-1, 2, allow_nan=False))
def test_selection_parameters_are_validated(size, alpha, rate):
    for make, valid in (
        (lambda: selection.Tournament(size), 1 <= size <= 6),
        (lambda: selection.BallotElite(size, alpha), 1 <= size <= 6 and alpha > 0),
        (lambda: selection.Migration(rate, size), 1 <= size <= 6 and 0 <= rate <= 1),
    ):
        if valid:
            assert selection.resolve(make()) == make()
        else:
            with pytest.raises(ValueError):
                make()


def test_selection_names_and_descriptions():
    assert selection.resolve("native") is None
    assert selection.describe(None) == "native"
    assert selection.describe(selection.resolve("tournament")) == "tournament(size=3)"
    assert (
        selection.describe(selection.resolve("ballot_elite")) == "ballot_elite(size=3, alpha=1.2)"
    )
    assert selection.describe(selection.Migration(0.03125, 4)) == "migration(rate=0.03125, size=4)"


def test_standalone_operators_explain_host_arrays():
    with pytest.raises(
        TypeError,
        match=r"values must be a pycuda\.gpuarray\.GPUArray or a cupy\.ndarray, got ndarray",
    ):
        selection.tournament(np.zeros(4), seed=0)
    with pytest.raises(ValueError, match="size must be between 1 and 6"):
        selection.tournament(np.zeros(4), size=9, seed=0)


def test_result_object():
    result = OptimizeResult(
        x=np.zeros(3),
        fun=1.5,
        history=np.array([2.0, 1.5]),
        nfev=20,
        nit=1,
        runtime=0.25,
        compile_time=0.0,
        termination=TerminationReason.TARGET,
        seed=4,
        algorithm="gwo",
        selection="native",
    )
    assert result.message == "the target value was reached"
    assert repr(result) == (
        "OptimizeResult(fun=1.5, nit=1, nfev=20, runtime=0.250s, termination=TARGET, "
        "algorithm='gwo', selection='native', dim=3)"
    )
    with pytest.raises(AttributeError):
        result.fun = 0.0


@pytest.mark.parametrize("cls", [pygpumeta.ABC, pygpumeta.HHO])
def test_evaluation_option(cls):
    assert cls("sphere", dim=3)._fused and not cls("sphere", dim=3, evaluation="separated")._fused
    assert not cls(lambda x: x.sum(axis=1), (-1, 1), dim=3)._fused
    with pytest.raises(
        ValueError, match="evaluation must be one of 'auto', 'fused', 'separated', got 'split'"
    ):
        cls("sphere", dim=3, evaluation="split")
    with pytest.raises(
        ValueError, match="evaluation='fused' needs a built-in function or a CudaObjective"
    ):
        cls(lambda x: x.sum(axis=1), (-1, 1), dim=3, evaluation="fused")
    with pytest.raises(TypeError, match="algorithm 'gwo' has no option 'evaluation'"):
        pygpumeta.minimize("sphere", dim=3, options={"evaluation": "fused"})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "double f(const double* x, int d) { return 0; }",
            "exactly one function 'double evaluate.*found 0",
        ),
        ("double evaluate(double* x, int d) { return 0; }", "found 0"),
        ("float evaluate(const double* x, int d) { return 0; }", "found 0"),
        (
            "double evaluate(const double* x, int d) { return 0; }\n"
            "double evaluate(const double* y, int n) { return 1; }",
            "found 2",
        ),
        (
            "double evaluate(const double* x, int d) { return pgm_user_bounded(x, d); }",
            "must not use names starting",
        ),
    ],
)
def test_cuda_objective_source_is_validated(source, message):
    with pytest.raises(ValueError, match=message):
        pygpumeta.CudaObjective(source)
    with pytest.raises(TypeError, match="must be a string"):
        pygpumeta.CudaObjective(3)


def test_cuda_objective_accepts_free_formatting():
    objective = pygpumeta.CudaObjective(
        "double\nevaluate ( const double *x , int n ) { return x[0]; }"
    )
    opt = pygpumeta.GWO(objective, (-1.0, 1.0), dim=2)
    assert opt._problem.kind == "cuda" and opt._problem.function_id == 1000


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (pygpumeta.ABC, "tournament(size=3)"),
        (pygpumeta.GWO, "tournament(size=3)"),
        (pygpumeta.WOA, "tournament(size=3)"),
        (pygpumeta.HHO, "tournament(size=3)"),
        (pygpumeta.SCA, "native"),
        (pygpumeta.SSA, "native"),
        (pygpumeta.PSO, "native"),
        (pygpumeta.DE, "native"),
    ],
)
def test_default_selection_per_algorithm(cls, expected):
    assert selection.describe(cls("sphere", dim=3)._selection) == expected
    assert selection.describe(cls("sphere", dim=3, selection="auto")._selection) == expected
    explicit = cls("sphere", dim=3, selection="migration")
    assert selection.describe(explicit._selection) == "migration(rate=0.25, size=3)"
