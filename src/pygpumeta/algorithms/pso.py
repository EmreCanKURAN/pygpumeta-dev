"""Particle Swarm Optimization."""

from __future__ import annotations

from typing import Any

from pygpumeta import _validation as _v
from pygpumeta.algorithms._array import ArrayOptimizer


class PSO(ArrayOptimizer):
    """Particle Swarm Optimization (Kennedy and Eberhart, 1995; Clerc and Kennedy, 2002).

    Every particle is drawn towards its own best position and towards a
    guide. With ``selection="native"`` the guide is the best position of the
    swarm; with a selection operator, particle ``i`` follows the personal
    best of the particle selected for it, which keeps several attractors
    alive.

    The algorithm is written with array operations only; its source is the
    template for adding an algorithm of your own.

    Parameters
    ----------
    objective, bounds
        See :class:`~pygpumeta.algorithms.Optimizer`.
    inertia : float, default 0.7298
        Weight of the previous velocity.
    cognitive : float, default 1.49618
        Attraction to the particle's own best position.
    social : float, default 1.49618
        Attraction to the guide.
    **common
        The other keyword arguments of :class:`~pygpumeta.algorithms.Optimizer`.

    Notes
    -----
    The defaults are the constriction coefficients of Clerc and Kennedy.
    Velocities are limited to the width of the box and positions are
    projected onto it.
    """

    name = "pso"
    _default_selection = "native"

    def __init__(
        self,
        objective: Any,
        bounds: Any = None,
        *,
        inertia: float = 0.7298,
        cognitive: float = 1.49618,
        social: float = 1.49618,
        **common: Any,
    ) -> None:
        super().__init__(objective, bounds, **common)
        self._inertia = _v.check_non_negative(inertia, "inertia")
        self._cognitive = _v.check_non_negative(cognitive, "cognitive")
        self._social = _v.check_non_negative(social, "social")

    def _initialise(self) -> None:
        xp = self._xp
        self._x = self._uniform_population()
        self._v = xp.zeros_like(self._x)
        self._pbest = self._x.copy()
        self._pbest_score = xp.full(self._ps, xp.inf)

    def _iterate(self, it: int) -> None:
        xp = self._xp
        self._evaluate_swarm()
        selected = self._select(self._pbest_score, it)
        guide = self._best_pos if selected is None else self._pbest[selected]
        r1 = self._rng.random_sample(self._x.shape)
        r2 = self._rng.random_sample(self._x.shape)
        velocity = (
            self._inertia * self._v
            + self._cognitive * r1 * (self._pbest - self._x)
            + self._social * r2 * (guide - self._x)
        )
        span = self._upper - self._lower
        self._v = xp.clip(velocity, -span, span)
        self._x = self._clip(self._x + self._v)

    def _finalise(self) -> None:
        self._evaluate_swarm()

    def _evaluate_swarm(self) -> None:
        xp = self._xp
        scores = self._score(self._x)
        improved = scores < self._pbest_score
        self._pbest = xp.where(improved[:, None], self._x, self._pbest)
        self._pbest_score = xp.where(improved, scores, self._pbest_score)
        self._update_best(self._x, scores)
