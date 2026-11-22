"""Bounds per dimension, maximisation, decoding and constraints on a device."""

from __future__ import annotations

import numpy as np
import pytest

import pygpumeta
from pygpumeta import Constraint
from pygpumeta.space import Categorical, Integer, Real

ALGORITHMS = ("abc", "gwo", "woa", "sca", "ssa", "hho")
BOX = [(-3.0, -1.0), (0.5, 4.0), (10.0, 20.0), (-0.1, 0.1)]
CENTRE = np.array([-2.0, 2.0, 15.0, 0.0])


def off_centre(x):
    import cupy as cp

    return ((x - cp.asarray(CENTRE)) ** 2).sum(axis=1)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("selection", ["native", "tournament"])
def test_every_position_respects_its_own_bounds(algorithm, selection):
    import cupy as cp

    lower, upper = (cp.asarray(np.array(BOX)[:, i]) for i in (0, 1))
    outside = []

    def objective(x):
        outside.append(int(((x < lower) | (x > upper)).sum()))
        return off_centre(x)

    result = pygpumeta.minimize(
        objective, BOX, algorithm=algorithm, selection=selection, pop_size=200, max_iter=100, seed=3
    )
    assert sum(outside) == 0 and len(outside) > 100
    assert (result.x >= np.array(BOX)[:, 0]).all() and (result.x <= np.array(BOX)[:, 1]).all()
    if selection == "native":
        # Tolerance: a quadratic whose minimiser lies inside a box with
        # sides between 0.2 and 10; a search that used one variable's range
        # for another would start and stay far away. Random points have a
        # value below 0.5 with probability under one percent.
        assert result.fun <= 0.5


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_bounds_given_per_dimension_give_the_same_run(algorithm):
    common = {"algorithm": algorithm, "pop_size": 300, "max_iter": 30, "seed": 8}
    pair = pygpumeta.minimize("ackley", (-5.0, 5.0), dim=6, **common)
    rows = pygpumeta.minimize("ackley", [(-5.0, 5.0)] * 6, **common)
    assert pair.fun == rows.fun and np.array_equal(pair.x, rows.x)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind", ["builtin", "cuda", "gpu", "host"])
def test_maximising_is_minimising_the_negative(algorithm, kind):
    source = (
        "double evaluate(const double* x, int d) {"
        " double s = 0; for (int i = 0; i < d; i++) s += x[i] * x[i]; return %s; }"
    )
    common = {"dim": 4, "algorithm": algorithm, "pop_size": 200, "max_iter": 25, "seed": 5}
    if kind == "builtin":
        # No built-in is the negative of another; check the sign handling only.
        result = pygpumeta.minimize("sphere", maximize=True, **common)
        assert result.fun > 0.0 and (np.diff(result.history) >= 0).all()
        assert result.history[-1] == result.fun
        assert (result.x**2).sum() == pytest.approx(result.fun, rel=1e-12)
        return
    if kind == "cuda":
        up = pygpumeta.CudaObjective(source % "s")
        down = pygpumeta.CudaObjective(source % "-s")
        extra = {}
    else:
        up, down = (lambda x: (x * x).sum(axis=1)), (lambda x: -(x * x).sum(axis=1))
        extra = {"mode": kind}
    maximised = pygpumeta.minimize(up, (-2.0, 2.0), maximize=True, **common, **extra)
    minimised = pygpumeta.minimize(down, (-2.0, 2.0), **common, **extra)
    assert maximised.fun == -minimised.fun and np.array_equal(maximised.x, minimised.x)
    assert np.array_equal(maximised.history, -minimised.history)
    # The largest value of the sphere on the box is 16, at a corner.
    assert 0.0 < maximised.fun <= 16.0


def test_target_when_maximising():
    result = pygpumeta.minimize(
        "sphere", (-2.0, 2.0), dim=4, maximize=True, target=8.0, pop_size=200, max_iter=500
    )
    assert result.termination is pygpumeta.TerminationReason.TARGET and result.fun >= 8.0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mode", ["gpu", "host"])
def test_integer_and_categorical_variables(algorithm, mode):
    seen = set()
    weights = np.array([0.0, 3.0, 1.0])  # cost of the three categories

    def objective(x):
        xp = np if isinstance(x, np.ndarray) else __import__("cupy")
        host = x if xp is np else xp.asnumpy(x)
        seen.update(map(tuple, np.unique(host[:, 1:], axis=0)))
        return (
            (x[:, 0] - 0.25) ** 2 + (x[:, 1] - 7.0) ** 2 + xp.asarray(weights)[x[:, 2].astype(int)]
        )

    result = pygpumeta.minimize(
        objective,
        [Real(-1.0, 1.0), Integer(0, 10), Categorical(["cheap", "dear", "middling"])],
        algorithm=algorithm,
        selection="native",
        pop_size=200,
        max_iter=60,
        seed=2,
        mode=mode,
    )
    # The objective only ever saw admissible integers and category indices.
    assert {v[0] for v in seen} <= set(map(float, range(11)))
    assert {v[1] for v in seen} <= {0.0, 1.0, 2.0}
    assert result.x[1] == round(result.x[1]) and result.x[2] in (0.0, 1.0, 2.0)
    assert isinstance(result.values[1], int) and result.values[2] in ("cheap", "dear", "middling")
    # Tolerance: every other combination of the discrete variables costs at
    # least 1, and the initial 200 individuals already contain the optimal
    # one several times, so a value above 0.5 would be a decoding error.
    assert result.values[1:] == (7, "cheap") and result.fun <= 0.5


def disc(x):
    # Stay inside the unit disc around (1, 1): g(x) <= 0.
    return ((x - 1.0) ** 2).sum(axis=1) - 1.0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind", ["builtin", "cuda", "gpu", "host"])
def test_inequality_constraint(algorithm, kind):
    # Minimise x0^2 + x1^2 inside the unit disc around (1, 1). The solution
    # is the point of the disc closest to the origin, (1 - 1/sqrt(2)) twice,
    # with value (sqrt(2) - 1)^2 = 0.171573.
    objective = {
        "builtin": "sphere",
        "cuda": pygpumeta.CudaObjective(
            "double evaluate(const double* x, int d) { return x[0] * x[0] + x[1] * x[1]; }"
        ),
    }.get(kind, lambda x: (x * x).sum(axis=1))
    result = pygpumeta.minimize(
        objective,
        (-3.0, 3.0),
        dim=2,
        algorithm=algorithm,
        selection="native",
        pop_size=512,
        max_iter=200,
        seed=4,
        mode="host" if kind == "host" else "auto",
        constraints=[Constraint(disc)],
        penalty=1e3,
    )
    # Tolerance: with penalty 1e3 the optimum of the penalised problem lies
    # about 1e-3 outside the disc, so a violation up to 5e-3 is the method,
    # not an error; the value tolerance of 0.1 allows for algorithms that
    # approach the boundary slowly but excludes the unconstrained optimum 0
    # and the centre of the disc (value 2).
    assert result.constraint_violation <= 5e-3
    assert result.fun == pytest.approx(0.171573, abs=0.1)
    assert float(disc(result.x[None, :])[0]) <= 5e-3
    # fun is the objective without the penalty.
    assert (result.x**2).sum() == pytest.approx(result.fun, rel=1e-9)


def test_equality_constraint_and_feasibility_flag():
    # Minimise x0 + x1 on the unit circle: -sqrt(2) at (-1/sqrt(2), -1/sqrt(2)).
    circle = Constraint(lambda x: (x * x).sum(axis=1) - 1.0, kind="eq", tol=1e-3)
    result = pygpumeta.minimize(
        lambda x: x.sum(axis=1), (-2.0, 2.0), dim=2, algorithm="gwo", selection="native",
        pop_size=1024, max_iter=300, seed=6, constraints=[circle], penalty=1e3,
    )  # fmt: skip
    # Tolerance: the band of the circle and the finite penalty together move
    # the optimum by about 2e-3; 0.05 leaves a margin for convergence.
    assert result.fun == pytest.approx(-np.sqrt(2.0), abs=0.05)
    assert result.constraint_violation <= 5e-3
    assert result.feasible == (result.constraint_violation == 0.0)

    infeasible = pygpumeta.minimize(
        "sphere", (-1.0, 1.0), dim=2, pop_size=64, max_iter=20,
        constraints=[Constraint(lambda x: 5.0 - x[:, 0])],
    )  # fmt: skip
    # x0 >= 5 cannot hold on the box; the best point is the closest one.
    assert not infeasible.feasible and 4.0 <= infeasible.constraint_violation <= 4.2
    assert infeasible.fun == pytest.approx((infeasible.x**2).sum(), rel=1e-9)


@pytest.mark.parametrize("algorithm", [*ALGORITHMS, "pso", "de"])
@pytest.mark.parametrize("kind", ["builtin", "gpu"])
def test_feasibility_rule(algorithm, kind):
    objective = "sphere" if kind == "builtin" else (lambda x: (x * x).sum(axis=1))
    result = pygpumeta.minimize(
        objective,
        (-3.0, 3.0),
        dim=2,
        algorithm=algorithm,
        selection="native",
        pop_size=512,
        max_iter=200,
        seed=4,
        constraints=[Constraint(disc)],
        constraint_handling="feasibility",
    )
    # The rule never accepts a violation: the result is feasible exactly and
    # cannot be below the constrained optimum 0.171573.
    # Tolerance: 0.1 above the optimum allows for algorithms that settle
    # short of the boundary, as in the penalty test.
    assert result.feasible and result.constraint_violation == 0.0
    assert 0.171572 <= result.fun <= 0.171573 + 0.1
    assert (np.diff(result.history) <= 0).all()


@pytest.mark.parametrize("algorithm", ["gwo", "de"])
def test_feasibility_rule_with_an_infeasible_start(algorithm):
    # A random point satisfies "every variable >= 2" with probability
    # (1/6)^6 = 2e-5, so the run starts without a feasible individual and
    # has to find the region by reducing the violation.
    corner = Constraint(lambda x: 2.0 - x.min(axis=1))
    result = pygpumeta.minimize(
        "sphere", (-3.0, 3.0), dim=6, algorithm=algorithm, selection="native", pop_size=256,
        max_iter=300, seed=1, constraints=[corner], constraint_handling="feasibility",
    )  # fmt: skip
    assert result.history[0] >= 1e150
    # Inside the region the objective lies between 24 (inner corner) and 54.
    assert result.feasible and 24.0 <= result.fun <= 54.0
