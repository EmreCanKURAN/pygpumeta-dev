"""Constraints of an optimisation problem."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pygpumeta import _validation as _v

__all__ = ["Constraint"]

KINDS = ("ineq", "eq")


@dataclass(frozen=True)
class Constraint:
    """An inequality or equality constraint.

    Parameters
    ----------
    fun : callable
        Receives the positions exactly as the objective does (an array of
        shape ``(n, dim)``, CuPy or NumPy depending on the objective mode)
        and returns an array of shape ``(n,)``, or ``(n, m)`` for ``m``
        constraints at once.
    kind : {"ineq", "eq"}, default "ineq"
        ``"ineq"`` requires ``fun(x) <= 0``; ``"eq"`` requires
        ``|fun(x)| <= tol``.
    tol : float, default 1e-6
        Tolerance of an equality constraint.

    Notes
    -----
    The violation of a position is ``max(0, fun(x))`` for an inequality and
    ``max(0, |fun(x)| - tol)`` for an equality, summed over all constraints.
    Scale the constraint functions so that violations of different
    constraints are comparable.
    """

    fun: Callable[[Any], Any]
    kind: str = "ineq"
    tol: float = 1e-6

    def __post_init__(self) -> None:
        if not callable(self.fun):
            raise TypeError(f"fun must be callable, got {type(self.fun).__name__}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be 'ineq' or 'eq', got {self.kind!r}")
        object.__setattr__(self, "tol", _v.check_positive(self.tol, "tol"))
