"""A selection backend for tests that runs the CPU reference."""

from __future__ import annotations

from typing import ClassVar

from pygpumeta import reference as ref
from pygpumeta import selection


class ReferenceSelector:
    """Stands in for the selection kernels; the two are bit-identical."""

    calls: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, xp, operator, count):
        self.operator = operator

    def __call__(self, scores, seed):
        op = self.operator
        type(self).calls.append((type(op).__name__, seed))
        if isinstance(op, selection.Tournament):
            return ref.tournament(scores, size=op.size, seed=seed)
        if isinstance(op, selection.BallotElite):
            return ref.ballot_elite(scores, size=op.size, alpha=op.alpha, seed=seed)
        return ref.migration(scores, rate=op.rate, size=op.size, seed=seed)
