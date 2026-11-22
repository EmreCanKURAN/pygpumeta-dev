"""Sine Cosine Algorithm."""

from __future__ import annotations

import numpy as np

from pygpumeta.algorithms._kernel import GuidedOptimizer, _Device


class SCA(GuidedOptimizer):
    """Sine Cosine Algorithm (Mirjalili, 2016).

    With a selection operator every individual oscillates around its
    selected individual instead of the best position found so far.
    """

    name = "sca"
    _default_selection = "native"

    def _move(self, dev: _Device, it: int, selected: bool, seed: np.uint32) -> None:
        self._wide(
            "sca_update_sel" if selected else "sca_update",
            dev.pop,
            dev.pop_next,
            dev.sel if selected else self._guide,
            self._lb,
            self._ub,
            np.int32(self._ps),
            np.int32(self._problem.dim),
            np.float64(2.0 - 2.0 * (it + 1) / self._max_iter),
            seed,
        )
