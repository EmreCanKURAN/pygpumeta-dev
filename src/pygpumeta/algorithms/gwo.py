"""Grey Wolf Optimizer."""

from __future__ import annotations

import numpy as np

from pygpumeta._backend.runtime import Runtime
from pygpumeta.algorithms._kernel import KernelOptimizer, _Device


class GWO(KernelOptimizer):
    """Grey Wolf Optimizer (Mirjalili, Mirjalili and Lewis, 2014).

    With a selection operator the three leaders are the best distinct
    members of the selected individuals; ``selection="native"`` takes the
    three best individuals of the population.
    """

    name = "gwo"

    def _extra_buffers(self, rt: Runtime) -> None:
        self._leaders = rt.zeros(3, np.int32)
        self._leader_pos = rt.zeros(3 * self._problem.dim, np.float64)

    def _initialise(self, dev: _Device) -> None:
        self._wide(
            "pop_init",
            dev.pop,
            self._lb,
            self._ub,
            np.int32(self._ps),
            np.int32(self._problem.dim),
            self._seed_for(1),
        )

    def _iterate(self, dev: _Device, it: int) -> None:
        ps, dim = np.int32(self._ps), np.int32(self._problem.dim)
        self._evaluate_population(dev)
        if self._select(dev, it):
            self._single("gwo_find_leaders_sel", dev.obj, dev.sel, self._leaders, ps)
        else:
            self._single("gwo_find_leaders", dev.obj, self._leaders, ps)
        self._wide(
            "gwo_copy_leaders",
            dev.pop,
            self._leaders,
            self._leader_pos,
            dim,
            count=self._problem.dim,
        )
        a = 2.0 - 2.0 * it / self._max_iter
        self._wide(
            "gwo_update",
            dev.pop,
            dev.pop_next,
            self._leader_pos,
            self._lb,
            self._ub,
            ps,
            dim,
            np.float64(a),
            self._seed_for(7000 + it),
        )
        dev.swap()

    def _finalise(self, dev: _Device) -> None:
        self._evaluate_population(dev)
