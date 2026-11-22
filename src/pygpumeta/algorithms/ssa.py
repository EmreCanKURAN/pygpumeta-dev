"""Salp Swarm Algorithm."""

from __future__ import annotations

import math

import numpy as np

from pygpumeta.algorithms._kernel import GuidedOptimizer, _Device


class SSA(GuidedOptimizer):
    """Salp Swarm Algorithm (Mirjalili et al., 2017).

    The first half of the population are leaders that move around the food
    source, the second half follow their predecessor. With a selection
    operator the food source of a leader is its selected individual instead
    of the best position found so far.
    """

    name = "ssa"
    _default_selection = "native"

    def _move(self, dev: _Device, it: int, selected: bool, seed: np.uint32) -> None:
        c1 = 2.0 * math.exp(-((4.0 * it / self._max_iter) ** 2))
        self._wide(
            "ssa_update_sel" if selected else "ssa_update",
            dev.pop,
            dev.pop_next,
            dev.sel if selected else self._guide,
            self._lb,
            self._ub,
            np.int32(self._ps),
            np.int32(self._problem.dim),
            np.float64(c1),
            seed,
        )
