"""Bounds per dimension, maximisation, decoding and constraints, without a GPU."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_kernel_calls import device  # noqa: F401 - fixture

import pygpumeta
from pygpumeta import Constraint, _problem
from pygpumeta._objective import Scorer
from pygpumeta.space import Categorical, Integer, Real


def problem(objective=lambda x: x.sum(axis=1), bounds=(-1.0, 1.0), dim=None, **kwargs):
    kwargs.setdefault("mode", "host")
    return _problem.resolve_problem(objective, bounds, dim, **kwargs)


# ------------------------------------------------------------------- space


def test_variables_define_the_search_box():
    p = problem(bounds=[Real(-2, 3), Integer(1, 4), Categorical(["rbf", "linear", "poly"]), (0, 1)])
    assert p.dim == 4
    assert p.lower.tolist() == [-2.0, 0.5, -0.5, 0.0]
    assert p.upper.tolist() == [3.0, 4.5, 2.5, 1.0]
    assert p.discrete_index.tolist() == [1, 2]
    assert p.discrete_low.tolist() == [1.0, 0.0] and p.discrete_high.tolist() == [4.0, 2.0]


@given(st.lists(st.floats(-10, 10), min_size=4, max_size=4))
def test_decoding_rounds_into_the_admissible_range(raw):
    p = problem(bounds=[Real(-2, 3), Integer(1, 4), Categorical(["a", "b", "c"]), (0, 1)])
    decoded = p.decode(np.array(raw))
    assert decoded[0] == raw[0] and decoded[3] == raw[3]
    assert decoded[1] in (1.0, 2.0, 3.0, 4.0) and decoded[2] in (0.0, 1.0, 2.0)
    values = p.values(decoded)
    assert isinstance(values[1], int) and values[2] in ("a", "b", "c")
    assert isinstance(values[0], float)


def test_every_integer_owns_an_interval_of_equal_width():
    p = problem(bounds=[Integer(3, 6)])
    grid = np.linspace(p.lower[0], p.upper[0], 40001)[1:-1]
    decoded = np.array([p.decode(np.array([g]))[0] for g in grid])
    counts = [np.sum(decoded == k) for k in (3, 4, 5, 6)]
    assert max(counts) - min(counts) <= 2


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (lambda: Real(1, 1), ValueError, "lower must be smaller than upper"),
        (lambda: Real(0, float("inf")), ValueError, "upper must be finite"),
        (lambda: Real("a", 1), TypeError, "lower must be a real number"),
        (lambda: Integer(5, 4), ValueError, "lower must not exceed upper"),
        (lambda: Integer(0.5, 4), TypeError, "lower must be an integer"),
        (lambda: Categorical(["only"]), ValueError, "at least two entries, got 1"),
        (lambda: Categorical("abc"), TypeError, "not a single string"),
        (lambda: Categorical(3), TypeError, "choices must be a sequence"),
    ],
)
def test_invalid_variables(make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_discrete_variables_need_a_callable_objective():
    with pytest.raises(ValueError, match="integer and categorical variables need a callable"):
        pygpumeta.GWO("sphere", [Integer(0, 3), Real(0, 1)])
    with pytest.raises(ValueError, match=r"bounds\[1\] must be a Real, Integer or Categorical"):
        problem(bounds=[Integer(0, 3), (1, 2, 3)])
    with pytest.raises(ValueError, match=r"got \(2\.0, 1\.0\) for variable 1"):
        problem(bounds=[(0, 1), (2, 1)])


# -------------------------------------------------------------- constraints


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"constraints": [lambda x: x]},
            TypeError,
            r"wrap functions as pygpumeta\.Constraint\(fun\)",
        ),
        ({"constraints": 5}, TypeError, "constraints must be a sequence of Constraint objects"),
        ({"constraint_handling": "barrier"}, ValueError, "must be 'penalty' or 'feasibility'"),
        ({"penalty": 0.0}, ValueError, "penalty must be a finite number greater than 0"),
        ({"maximize": 1}, TypeError, "maximize must be True or False"),
    ],
)
def test_invalid_constraint_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        pygpumeta.GWO("sphere", dim=3, **kwargs)


def test_invalid_constraints():
    with pytest.raises(TypeError, match="fun must be callable"):
        Constraint(3)
    with pytest.raises(ValueError, match="kind must be 'ineq' or 'eq', got 'le'"):
        Constraint(len, kind="le")
    with pytest.raises(ValueError, match="tol must be a finite number greater than 0"):
        Constraint(len, kind="eq", tol=0.0)


def test_violation_of_inequalities_and_equalities():
    constraints = [
        Constraint(lambda x: x[:, 0] - 1.0),  # x0 <= 1
        Constraint(lambda x: np.stack([-x[:, 1], x[:, 1] - 2.0], axis=1)),  # 0 <= x1 <= 2
        Constraint(lambda x: x[:, 0] + x[:, 1] - 3.0, kind="eq", tol=0.5),
    ]
    scorer = Scorer(
        np, problem(bounds=(-10.0, 10.0), dim=2, constraints=constraints, penalty=100.0)
    )
    points = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, -1.0], [0.0, 2.0]])
    #            feasible   x0 by 2, eq by 1.5   x1 by 1, eq by 2.5   eq by 0.5
    violation = scorer.violation(points)
    assert violation.tolist() == [0.0, 3.5, 3.5, 0.5]
    assert scorer(points).tolist() == [3.0, 5.0 + 350.0, 0.0 + 350.0, 2.0 + 50.0]


def test_scores_when_maximising_with_constraints():
    scorer = Scorer(
        np,
        problem(
            bounds=(-10.0, 10.0),
            dim=2,
            maximize=True,
            penalty=10.0,
            constraints=[Constraint(lambda x: x[:, 0] - 1.0)],
        ),
    )
    # Minimising -f + penalty * violation maximises f among feasible points.
    assert scorer(np.array([[1.0, 2.0], [4.0, 2.0]])).tolist() == [-3.0, -6.0 + 30.0]


def test_non_finite_values_are_bounded_and_nan_constraints_count_as_violated():
    def objective(x):
        return np.array([np.nan, np.inf, -np.inf, 1.0])

    scorer = Scorer(np, problem(objective, dim=1))
    assert scorer(np.zeros((4, 1))).tolist() == [1e299, 1e299, -1e299, 1.0]
    scorer = Scorer(np, problem(dim=1, constraints=[Constraint(lambda x: np.full(len(x), np.nan))]))
    assert (scorer.violation(np.zeros((2, 1))) == 1e299).all()


@pytest.mark.parametrize(
    ("fun", "error", "message"),
    [
        (
            lambda x: np.zeros(3),
            ValueError,
            r"constraint 0 must return an array of shape \(4,\) or \(4, m\)",
        ),
        (lambda x: np.zeros((2, 4)), ValueError, r"got shape \(2, 4\)"),
        (lambda x: "no", TypeError, "constraint 0 must return real numbers"),
    ],
)
def test_constraint_return_values_are_checked(fun, error, message):
    scorer = Scorer(np, problem(dim=2, constraints=[Constraint(fun)]))
    with pytest.raises(error, match=message):
        scorer.violation(np.zeros((4, 2)))


def test_row_by_row_objective_and_constraints():
    seen = []

    def objective(row):
        seen.append(row.shape)
        return float(row.sum())

    scorer = Scorer(
        np,
        problem(
            objective,
            dim=2,
            vectorized=False,
            penalty=1.0,
            constraints=[Constraint(lambda row: [row[0], -row[1]])],
        ),
    )
    assert scorer(np.array([[1.0, 2.0], [-1.0, -3.0]])).tolist() == [3.0 + 1.0, -4.0 + 3.0]
    assert seen == [(2,), (2,)]


# ------------------------------------------------- whole runs on the fake device


@pytest.mark.parametrize("algorithm", ["abc", "gwo", "woa", "sca", "ssa", "hho"])
def test_decoded_positions_reach_objective_and_constraints(device, algorithm):  # noqa: F811
    seen = []

    def objective(x):
        seen.append(x.copy())
        return x.sum(axis=1)

    result = pygpumeta.minimize(
        objective,
        [Real(-1, 1), Integer(2, 5), Categorical(["a", "b"])],
        algorithm=algorithm,
        pop_size=12,
        max_iter=2,
        mode="host",
        constraints=[Constraint(lambda x: x[:, 0])],
    )
    # The fake device leaves the population at zero, which decodes to the
    # nearest admissible values.
    for x in seen:
        assert set(np.unique(x[:, 1])) <= {2.0} and set(np.unique(x[:, 2])) <= {0.0}
    assert result.x.tolist() == [0.0, 2.0, 0.0]
    assert result.values == (0.0, 2, "a") and result.feasible
    assert result.constraint_violation == 0.0


def test_constraints_force_separated_evaluation_of_builtin_objectives(device):  # noqa: F811
    with pytest.raises(ValueError, match=r"evaluation='fused' needs .* without constraints"):
        pygpumeta.HHO("sphere", dim=3, constraints=[Constraint(len)], evaluation="fused")
    calls = []

    def constraint(x):
        calls.append(x.shape)
        return x[:, 0] - 5.0

    result = pygpumeta.minimize(
        "sphere", dim=3, algorithm="abc", pop_size=10, max_iter=2, mode="host",
        constraints=[Constraint(constraint)],
    )  # fmt: skip
    assert "abc_propose_employed" in device.log and "abc_employed" not in device.log
    # The initial population, then two phases and one scout row per iteration.
    assert device.log.count("eval_pop") == 1 + 2 * 3
    assert calls.count((10, 3)) == 5 and calls.count((1, 3)) == 2
    assert result.feasible and result.constraint_violation == 0.0


def test_maximising_a_builtin_compiles_the_negating_kernels(device):  # noqa: F811
    result = pygpumeta.minimize("sphere", dim=3, pop_size=10, max_iter=2, maximize=True)
    assert "return -evaluate_raw(x, d, fid);" in device.modules[0].text
    # The fake device never lowers the initial infinity of the minimised
    # value, which is minus infinity in the user's terms.
    assert result.fun == -np.inf
    opt = pygpumeta.GWO("sphere", dim=3, maximize=True)
    assert opt.best_value == -np.inf


def test_bounds_are_passed_as_device_arrays(device):  # noqa: F811
    opt = pygpumeta.SSA("sphere", [(-1.0, 2.0), (0.0, 5.0)], pop_size=10, max_iter=1)
    opt.run()
    assert opt._lb.get().tolist() == [-1.0, 0.0] and opt._ub.get().tolist() == [2.0, 5.0]


# --------------------------------------------------------- feasibility rule


def feasibility_scorer(maximize=False):
    return Scorer(
        np,
        problem(
            lambda x: x[:, 0],
            bounds=(-1e200, 1e200),
            dim=2,
            maximize=maximize,
            constraint_handling="feasibility",
            constraints=[Constraint(lambda x: x[:, 1])],
        ),
    )


@given(
    st.lists(
        st.tuples(
            st.floats(-1e140, 1e140),
            st.one_of(st.just(0.0), st.floats(-5.0, 0.0), st.floats(1e-12, 1e90)),
        ),
        min_size=2,
        max_size=30,
    ),
    st.booleans(),
)
def test_feasibility_scores_order_positions_by_the_rule(pairs, maximize):
    points = np.array(pairs)  # column 0: objective, column 1: constraint value
    scores = feasibility_scorer(maximize)(points)
    objective = -points[:, 0] if maximize else points[:, 0]
    violation = np.maximum(points[:, 1], 0.0)
    assert np.isfinite(scores).all() and (np.abs(scores) < 1e300).all()
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if violation[i] == 0.0 and violation[j] > 0.0:
                assert scores[i] < scores[j]
            elif violation[i] == 0.0 and violation[j] == 0.0:
                assert (scores[i] < scores[j]) == (objective[i] < objective[j])
            elif violation[i] > 0.0 and violation[j] > 0.0 and scores[i] < scores[j]:
                # Violations that differ by less than one part in 1e15 may
                # share a score; a strict order must still be the right one.
                assert violation[i] < violation[j]


def test_feasibility_scores_resolve_small_violations_and_bound_large_values():
    scorer = feasibility_scorer()
    points = np.array([[5.0, 1e-9], [-7.0, 2e-9], [1e200, 0.0], [-1e200, 0.0], [0.0, 1e300]])
    scores = scorer(points)
    assert scores[0] < scores[1]  # the objective of an infeasible position is ignored
    assert scores[2] == 1e149 and scores[3] == -1e149
    assert scores[2] < scores[0] and scores[4] == pytest.approx(1e250)


@pytest.mark.parametrize("algorithm", ["pso", "de"])
def test_feasibility_rule_returns_a_feasible_optimum(algorithm):
    # Minimise x0^2 + x1^2 inside the unit disc around (1, 1): the optimum
    # (sqrt(2) - 1)^2 = 0.171573 lies on the boundary.
    result = pygpumeta.minimize(
        lambda x: (x * x).sum(axis=1),
        (-3.0, 3.0),
        dim=2,
        algorithm=algorithm,
        selection="native",
        pop_size=60,
        max_iter=300,
        mode="host",
        constraints=[Constraint(lambda x: ((x - 1.0) ** 2).sum(axis=1) - 1.0)],
        constraint_handling="feasibility",
    )
    # Unlike the penalty, the rule never accepts a violation, so the result
    # is feasible exactly and approaches the optimum from inside.
    # Tolerance: 300 iterations in two dimensions; both algorithms settle on
    # the boundary to well within 1e-4.
    assert result.feasible and result.constraint_violation == 0.0
    assert 0.171572 <= result.fun <= 0.171573 + 1e-4
    assert result.history[-1] == result.fun


def test_feasibility_rule_without_a_feasible_point_reports_the_violation():
    result = pygpumeta.minimize(
        lambda x: x.sum(axis=1),
        (-1.0, 1.0),
        dim=2,
        algorithm="de",
        selection="native",
        pop_size=20,
        max_iter=60,
        mode="host",
        constraints=[Constraint(lambda x: 5.0 - x[:, 0])],  # x0 >= 5: impossible
        constraint_handling="feasibility",
    )
    # The search minimises the violation, here 5 - x0 with x0 at most 1.
    assert not result.feasible
    assert result.constraint_violation == pytest.approx(4.0, abs=1e-6)
    assert result.history[-1] == pytest.approx(1e150 * 5.0, rel=1e-6)
    assert result.fun == pytest.approx(result.x.sum())
