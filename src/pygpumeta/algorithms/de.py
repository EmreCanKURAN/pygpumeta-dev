"""Differential Evolution."""

from __future__ import annotations

from typing import Any

from pygpumeta import _validation as _v
from pygpumeta.algorithms._array import ArrayOptimizer


class DE(ArrayOptimizer):
    """Differential Evolution with binomial crossover (Storn and Price, 1997).

    A trial vector is the base vector plus a scaled difference of two other
    individuals, crossed with the current individual, and replaces it when
    it is not worse. With ``selection="native"`` the base vector is a random
    individual (DE/rand/1/bin); with a selection operator, the base vector
    of individual ``i`` is the individual selected for it, which moves the
    scheme towards DE/best/1 by as much as the operator's selection
    pressure.

    Parameters
    ----------
    objective, bounds
        See :class:`~pygpumeta.algorithms.Optimizer`.
    differential_weight : float, default 0.8
        Scale ``F`` of the difference vector, in (0, 2]. Small values let
        the population contract before it has reached the optimum.
    crossover : float, default 0.9
        Probability ``CR`` that a variable is taken from the mutant.
    **common
        The other keyword arguments of :class:`~pygpumeta.algorithms.Optimizer`.
        ``pop_size`` must be at least 4.

    Notes
    -----
    The three individuals that form a mutant differ from the current
    individual; the two that form the difference are drawn independently
    and may coincide, in which case the mutant equals its base vector.
    """

    name = "de"
    _default_selection = "native"
    _min_pop_size = 4

    def __init__(
        self,
        objective: Any,
        bounds: Any = None,
        *,
        differential_weight: float = 0.8,
        crossover: float = 0.9,
        **common: Any,
    ) -> None:
        super().__init__(objective, bounds, **common)
        weight = _v.check_positive(differential_weight, "differential_weight")
        if weight > 2.0:
            raise ValueError(f"differential_weight must lie in (0, 2], got {weight!r}")
        self._weight = weight
        self._crossover = _v.check_unit_interval(crossover, "crossover")

    def _initialise(self) -> None:
        self._x = self._uniform_population()
        self._scores = self._score(self._x)
        self._update_best(self._x, self._scores)

    def _iterate(self, it: int) -> None:
        xp, ps, dim = self._xp, self._ps, self.dim
        own = xp.arange(ps)
        selected = self._select(self._scores, it)
        # Offsets in [1, ps - 1] give partners different from the individual.
        offsets = 1 + (self._rng.random_sample((3, ps)) * (ps - 1)).astype(xp.int64)
        partners = (own + offsets) % ps
        base = self._x[partners[0]] if selected is None else self._x[selected]
        mutant = base + self._weight * (self._x[partners[1]] - self._x[partners[2]])
        take = self._rng.random_sample((ps, dim)) < self._crossover
        # At least one variable comes from the mutant.
        forced = (self._rng.random_sample(ps) * dim).astype(xp.int64)
        take[own, forced] = True
        trial = self._clip(xp.where(take, mutant, self._x))
        trial_scores = self._score(trial)
        accept = trial_scores <= self._scores
        self._x = xp.where(accept[:, None], trial, self._x)
        self._scores = xp.where(accept, trial_scores, self._scores)
        self._update_best(self._x, self._scores)
