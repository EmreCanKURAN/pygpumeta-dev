"""Whale Optimization Algorithm."""

from __future__ import annotations

import numpy as np

from pygpumeta.algorithms._kernel import GuidedOptimizer, _Device


class WOA(GuidedOptimizer):
    """Whale Optimization Algorithm (Mirjalili and Lewis, 2016).

    With a selection operator every whale follows its selected individual
    instead of the best position found so far.
    """

    name = "woa"

    def _move(self, dev: _Device, it: int, selected: bool, seed: np.uint32) -> None:
        progress = it / self._max_iter
        self._wide(
            "woa_update_sel" if selected else "woa_update",
            dev.pop,
            dev.pop_next,
            dev.sel if selected else self._guide,
            self._lb,
            self._ub,
            np.int32(self._ps),
            np.int32(self._problem.dim),
            np.float64(2.0 - 2.0 * progress),
            np.float64(-1.0 - progress),
            seed,
        )
