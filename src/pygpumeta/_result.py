"""Result of an optimisation run."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["OptimizeResult", "TerminationReason"]


class TerminationReason(enum.Enum):
    """Why a run ended."""

    MAX_ITER = "the maximum number of iterations was reached"
    TARGET = "the target value was reached"
    STALLED = "the best value stopped improving"
    MAX_TIME = "the time limit was reached"
    CALLBACK = "the callback requested a stop"
    IN_PROGRESS = "the run has not finished"


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of an optimisation run.

    Attributes
    ----------
    x : numpy.ndarray, shape (dim,)
        Best position found.
    fun : float
        Objective value at ``x``.
    history : numpy.ndarray
        Best value found so far, recorded every ``history_every`` iterations
        and once more when the run ends.
    nfev : int
        Number of objective evaluations.
    nit : int
        Number of iterations carried out.
    runtime : float
        Seconds spent iterating, without kernel compilation.
    compile_time : float
        Seconds spent compiling kernels for this run; zero when they were
        already compiled.
    termination : TerminationReason
        Why the run ended.
    seed : int
        Seed of the run.
    algorithm : str
        Name of the algorithm.
    selection : str
        Description of the selection operator.
    constraint_violation : float
        Total constraint violation at ``x``; zero without constraints.
    feasible : bool
        Whether ``x`` satisfies all constraints.
    values : tuple
        ``x`` as Python values: floats, ints for integer variables and the
        chosen entry for categorical variables.

    Notes
    -----
    With constraints, ``fun`` is the objective at ``x`` alone, while
    ``history`` records the value the optimizer minimised: objective plus
    penalty, or, with the feasibility rule, ``1e150 * (1 + violation)`` for
    as long as the best position is infeasible.
    """

    x: NDArray[np.float64]
    fun: float
    history: NDArray[np.float64]
    nfev: int
    nit: int
    runtime: float
    compile_time: float
    termination: TerminationReason
    seed: int
    algorithm: str
    selection: str
    constraint_violation: float = 0.0
    feasible: bool = True
    values: tuple[Any, ...] = ()

    @property
    def message(self) -> str:
        """Human-readable reason for the end of the run."""
        return self.termination.value

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(fun={self.fun!r}, nit={self.nit}, nfev={self.nfev}, "
            f"runtime={self.runtime:.3f}s, termination={self.termination.name}, "
            f"algorithm={self.algorithm!r}, selection={self.selection!r}, dim={self.x.size})"
        )
