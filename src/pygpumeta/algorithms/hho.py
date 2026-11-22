"""Harris Hawks Optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from pygpumeta import _problem
from pygpumeta._backend.runtime import Runtime
from pygpumeta.algorithms._kernel import GuidedOptimizer, _Device


class HHO(GuidedOptimizer):
    """Harris Hawks Optimization (Heidari et al., 2019).

    With a selection operator the rabbit of a hawk is its selected
    individual instead of the best position found so far. The rapid dives
    test up to two trial positions per hawk.

    Parameters
    ----------
    objective, bounds
        See :class:`~pygpumeta.algorithms.Optimizer`.
    evaluation : {"auto", "fused", "separated"}, default "auto"
        ``"fused"`` evaluates the trial positions inside the update kernel,
        for the diving hawks only; it needs a built-in function or a
        :class:`~pygpumeta.CudaObjective`. ``"separated"`` evaluates outside
        the kernels, which costs three evaluations of the whole population
        per iteration, and is the only form available to Python callables.
        Both give the same run. ``"auto"`` picks ``"fused"`` where possible.
    **common
        The other keyword arguments of :class:`~pygpumeta.algorithms.Optimizer`.
    """

    name = "hho"
    _needs_mean = True

    def __init__(
        self, objective: Any, bounds: Any = None, *, evaluation: str = "auto", **common: Any
    ) -> None:
        super().__init__(objective, bounds, **common)
        self._fused = _problem.resolve_evaluation(evaluation, self._problem)
        self._counts_on_device = self._fused

    def _extra_buffers(self, rt: Runtime) -> None:
        super()._extra_buffers(rt)
        if not self._fused:
            ps = self._ps
            self._cand = rt.zeros(ps * self._problem.dim, np.float64)
            self._cand_obj = rt.zeros(ps, np.float64)
            self._dive_mode = rt.zeros(ps, np.int32)
            self._state = rt.zeros(ps, np.uint32)
            self._energy = rt.zeros(ps, np.float64)
            self._jump = rt.zeros(ps, np.float64)

    def _move(self, dev: _Device, it: int, selected: bool, seed: np.uint32) -> None:
        ps, dim = np.int32(self._ps), np.int32(self._problem.dim)
        e1 = np.float64(2.0 * (1.0 - it / self._max_iter))
        box = (self._lb, self._ub, ps, dim)
        if self._fused:
            self._wide(
                "hho_update_sel" if selected else "hho_update",
                dev.pop,
                dev.pop_next,
                dev.obj,
                dev.sel if selected else self._guide,
                self._mean,
                *box,
                e1,
                np.int32(self._problem.function_id),
                seed,
            )
            return
        pending = (self._dive_mode, self._state, self._energy, self._jump)
        self._wide(
            "hho_propose_sel" if selected else "hho_propose",
            dev.pop,
            dev.pop_next,
            self._cand,
            dev.sel if selected else self._guide,
            self._mean,
            *box,
            e1,
            seed,
            *pending,
        )
        self._evaluate(self._cand, self._cand_obj, self._ps)
        self._wide(
            "hho_dive",
            dev.pop,
            dev.pop_next,
            self._cand,
            dev.obj,
            self._cand_obj,
            self._guide,
            dev.sel,
            np.int32(selected),
            self._mean,
            *box,
            *pending,
        )
        self._evaluate(self._cand, self._cand_obj, self._ps)
        self._wide(
            "hho_dive_accept",
            dev.pop,
            dev.pop_next,
            self._cand,
            dev.obj,
            self._cand_obj,
            ps,
            dim,
            self._dive_mode,
        )
