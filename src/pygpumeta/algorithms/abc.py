"""Artificial Bee Colony."""

from __future__ import annotations

from typing import Any

import numpy as np

from pygpumeta import _problem
from pygpumeta._backend.runtime import Runtime
from pygpumeta.algorithms._kernel import KernelOptimizer, _Device


class ABC(KernelOptimizer):
    """Artificial Bee Colony (Karaboga, 2005).

    With a selection operator onlooker ``i`` exploits the food source
    selected for it; ``selection="native"`` draws the source by roulette
    wheel on the fitness. One scout per iteration replaces the source with
    the most unsuccessful trials once they exceed ``limit``.

    Parameters
    ----------
    objective, bounds
        See :class:`~pygpumeta.algorithms.Optimizer`.
    limit : int, optional
        Number of unsuccessful trials after which a source is abandoned.
        Defaults to ``min(pop_size * dim, 100 * pop_size)``.
    evaluation : {"auto", "fused", "separated"}, default "auto"
        ``"fused"`` evaluates candidates inside the kernels that create
        them; it needs a built-in function or a
        :class:`~pygpumeta.CudaObjective`. ``"separated"`` creates all
        candidates of a phase, evaluates them outside the kernels and then
        accepts or rejects them; it is the only form available to Python
        callables and evaluates one extra position per iteration for the
        scout. ``"auto"`` picks ``"fused"`` where possible.
    **common
        The other keyword arguments of :class:`~pygpumeta.algorithms.Optimizer`.
    """

    name = "abc"

    def __init__(
        self,
        objective: Any,
        bounds: Any = None,
        *,
        limit: int | None = None,
        evaluation: str = "auto",
        **common: Any,
    ) -> None:
        super().__init__(objective, bounds, **common)
        self._fused = _problem.resolve_evaluation(evaluation, self._problem)
        self._counts_on_device = self._fused
        if limit is None:
            limit = min(self._ps * self.dim, 100 * self._ps)
        self._limit = _problem.check_count(limit, "limit", 1)

    def _extra_buffers(self, rt: Runtime) -> None:
        self._trial = rt.zeros(self._ps, np.int32)
        self._cumulative = rt.zeros(self._ps, np.float64)
        self._scout = rt.zeros(2, np.int32)
        if not self._fused:
            self._cand_obj = rt.zeros(self._ps, np.float64)
            self._scout_row = rt.zeros(self._problem.dim, np.float64)
            self._scout_obj = rt.zeros(1, np.float64)

    def _box(self) -> tuple[Any, ...]:
        return (self._lb, self._ub, np.int32(self._ps), np.int32(self._problem.dim))

    def _initialise(self, dev: _Device) -> None:
        sources = (dev.pop, dev.obj, dev.fit, self._trial)
        if self._fused:
            fid = np.int32(self._problem.function_id)
            self._wide("abc_init", *sources, *self._box(), fid, self._seed_for(1))
            return
        self._wide("abc_init_positions", dev.pop, *self._box(), self._seed_for(1))
        self._evaluate(dev.pop, dev.obj, self._ps)
        self._wide("compute_fit", dev.obj, dev.fit, np.int32(self._ps))

    def _iterate(self, dev: _Device, it: int) -> None:
        if self._fused:
            self._iterate_fused(dev, it)
        else:
            self._iterate_separated(dev, it)
        self._merge_best(dev)

    def _iterate_fused(self, dev: _Device, it: int) -> None:
        p = self._problem
        fid = np.int32(p.function_id)
        sources = (dev.pop, dev.obj, dev.fit, self._trial)
        self._wide("abc_employed", *sources, *self._box(), fid, self._seed_for(2000 + it))
        if self._select(dev, it):
            self._wide(
                "abc_onlooker_selected",
                *sources,
                dev.sel,
                *self._box(),
                fid,
                self._seed_for(5000 + it),
            )
        else:
            self._single("sel_naive", dev.fit, self._cumulative, np.int32(self._ps))
            self._wide(
                "abc_onlooker_roulette",
                *sources,
                self._cumulative,
                *self._box(),
                fid,
                self._seed_for(5000 + it),
            )
        self._single("abc_find_max_trial", self._trial, self._scout, np.int32(self._ps))
        self._single(
            "abc_scout",
            *sources,
            self._scout,
            self._lb,
            self._ub,
            np.int32(p.dim),
            fid,
            np.int32(self._limit),
            self._seed_for(9000 + it),
        )

    def _iterate_separated(self, dev: _Device, it: int) -> None:
        dim, limit = np.int32(self._problem.dim), np.int32(self._limit)
        sources = (dev.pop, dev.obj, dev.fit, self._trial)
        # ABC updates its sources in place, so the second population buffer
        # is free to hold the candidates.
        cand = dev.pop_next
        self._wide("abc_propose_employed", dev.pop, cand, *self._box(), self._seed_for(2000 + it))
        self._accept(dev, cand)
        if self._select(dev, it):
            self._wide(
                "abc_propose_onlooker_selected",
                dev.pop,
                cand,
                dev.sel,
                *self._box(),
                self._seed_for(5000 + it),
            )
        else:
            self._single("sel_naive", dev.fit, self._cumulative, np.int32(self._ps))
            self._wide(
                "abc_propose_onlooker_roulette",
                dev.pop,
                cand,
                self._cumulative,
                *self._box(),
                self._seed_for(5000 + it),
            )
        self._accept(dev, cand)
        self._single("abc_find_max_trial", self._trial, self._scout, np.int32(self._ps))
        self._single(
            "abc_propose_scout",
            dev.pop,
            self._scout,
            self._scout_row,
            self._lb,
            self._ub,
            dim,
            limit,
            self._seed_for(9000 + it),
        )
        self._evaluate(self._scout_row, self._scout_obj, 1)
        self._single(
            "abc_accept_scout", *sources, self._scout, self._scout_row, self._scout_obj, dim, limit
        )

    def _accept(self, dev: _Device, cand: Any) -> None:
        self._evaluate(cand, self._cand_obj, self._ps)
        self._wide(
            "abc_accept",
            dev.pop,
            dev.obj,
            dev.fit,
            self._trial,
            cand,
            self._cand_obj,
            np.int32(self._ps),
            np.int32(self._problem.dim),
        )
