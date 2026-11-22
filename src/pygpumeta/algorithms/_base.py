"""The optimizer interface: construction, stepping, stopping rules and results."""

from __future__ import annotations

import abc
import math
import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pygpumeta import _problem
from pygpumeta import selection as _selection
from pygpumeta._result import OptimizeResult, TerminationReason


class Optimizer(abc.ABC):
    """Base class of the optimizers.

    Parameters
    ----------
    objective : callable, str or CudaObjective
        A callable receives the positions to evaluate as a CuPy array of
        shape ``(n, dim)`` and returns their objective values as a CuPy
        array of shape ``(n,)``; ``n`` is usually, but not always, the
        population size. With ``mode="host"`` it receives and returns NumPy
        arrays instead. A string names a built-in test function, see
        :func:`pygpumeta.benchmarks.names`. A
        :class:`~pygpumeta.CudaObjective` is compiled into the kernels.
    bounds : (lower, upper), array_like of shape (dim, 2) or sequence of variables
        Search box: one pair for all variables, one pair per variable, or
        one :class:`~pygpumeta.space.Real`, :class:`~pygpumeta.space.Integer`
        or :class:`~pygpumeta.space.Categorical` per variable. Required
        except for built-in test functions, which default to their customary
        domain. Integer and categorical variables are rounded before the
        objective sees them and need a callable objective.
    dim : int, optional
        Number of variables; required when ``bounds`` is a single pair. At
        most 1024.
    pop_size : int, default 1024
        Number of individuals. Any size of at least 2 is accepted.
    max_iter : int, default 1000
        Number of iterations. It is fixed at construction because the update
        rules are scheduled over the whole run.
    selection : str or selection operator, default "auto"
        ``"tournament"``, ``"ballot_elite"``, ``"migration"``, an instance
        of :class:`~pygpumeta.selection.Tournament`,
        :class:`~pygpumeta.selection.BallotElite` or
        :class:`~pygpumeta.selection.Migration`, or ``"native"`` for the
        rule of the original algorithm. ``"auto"`` picks per algorithm: the
        warp tournament for ABC, GWO, WOA and HHO, where it was found to
        help or do no harm, and the native rule for SCA, SSA, PSO and DE.
    seed : int, default 0
        Seed of the run. Runs with equal arguments and seeds give equal
        results.
    history_every : int, default 1
        Interval, in iterations, at which the best value is recorded and at
        which :meth:`run` evaluates its stopping rules.
    mode : {"auto", "gpu", "host"}, default "auto"
        Where a callable objective runs: ``"gpu"`` (also chosen by
        ``"auto"``) on CuPy arrays on the device, ``"host"`` on NumPy arrays
        on the CPU, which copies the population back and forth every
        evaluation.
    vectorized : bool, default True
        With ``mode="host"`` only: False calls the objective once per
        position with a one-dimensional array.
    constraints : sequence of Constraint, optional
        Constraints, see :class:`~pygpumeta.Constraint`. They receive the
        same arrays as a callable objective. With a built-in function or a
        :class:`~pygpumeta.CudaObjective` they receive CuPy arrays, or NumPy
        arrays with ``mode="host"``.
    constraint_handling : {"penalty", "feasibility"}, default "penalty"
        ``"penalty"`` minimises ``objective + penalty * violation``.
        ``"feasibility"`` applies the feasibility rule of Deb (2000): a
        feasible position beats an infeasible one, feasible positions are
        compared by objective and infeasible ones by violation. It needs no
        tuning and never trades a violation for a better objective, but the
        search cannot approach the optimum through the infeasible side.
    penalty : float, default 1e6
        Penalty factor of ``"penalty"``. It should exceed the range of the
        objective divided by the smallest violation that matters.
    maximize : bool, default False
        Maximise the objective instead of minimising it.

    Notes
    -----
    Objective values that are NaN or infinite are replaced by plus or minus
    1e299. With the feasibility rule, an infeasible position is recorded as
    ``1e150 * (1 + violation)``; this is what ``best_value`` and the history
    show until a feasible position has been found, and feasible objective
    values must stay below 1e149 in magnitude.
    """

    name: ClassVar[str]
    # What selection="auto" means for the algorithm.
    _default_selection: ClassVar[str] = "tournament"

    def __init__(
        self,
        objective: Any,
        bounds: Any = None,
        *,
        dim: int | None = None,
        pop_size: int = 1024,
        max_iter: int = 1000,
        selection: Any = "auto",
        seed: int = 0,
        history_every: int = 1,
        mode: str = "auto",
        vectorized: bool = True,
        constraints: Any = (),
        constraint_handling: str = "penalty",
        penalty: float = 1e6,
        maximize: bool = False,
        _constants: Mapping[str, str] | None = None,
    ) -> None:
        self._problem = _problem.resolve_problem(
            objective,
            bounds,
            dim,
            mode,
            vectorized,
            constraints,
            constraint_handling,
            penalty,
            maximize,
        )
        self._sign = -1.0 if maximize else 1.0
        self._ps = _problem.check_count(pop_size, "pop_size", 2)
        if self._ps * self._problem.dim >= 2**31:
            raise ValueError(
                f"pop_size * dim must be below 2**31, got {self._ps} * {self._problem.dim}"
            )
        self._max_iter = _problem.check_count(max_iter, "max_iter", 1)
        self._selection = _selection.resolve(selection, self._default_selection)
        self._seed = _problem.check_run_seed(seed)
        self._history_every = _problem.check_count(history_every, "history_every", 1)
        self._constants = _constants
        self._iteration = 0
        self._rows_evaluated = 0
        self._elapsed = 0.0
        self._compile_time = 0.0
        self._termination = TerminationReason.IN_PROGRESS
        self._started = False

    # ------------------------------------------------------------------ state

    @property
    def dim(self) -> int:
        """Number of variables."""
        return self._problem.dim

    @property
    def pop_size(self) -> int:
        """Number of individuals."""
        return self._ps

    @property
    def max_iter(self) -> int:
        """Number of iterations of a complete run."""
        return self._max_iter

    @property
    def iteration(self) -> int:
        """Number of iterations carried out so far."""
        return self._iteration

    @property
    def finished(self) -> bool:
        """Whether the run has ended."""
        return self._termination is not TerminationReason.IN_PROGRESS

    @property
    def best_value(self) -> float:
        """Best value found so far, including any constraint penalty.

        Infinite before the first step (minus infinity when maximising).
        Reading it waits for the device.
        """
        return self._sign * (self._best_score() if self._started else math.inf)

    @property
    def best_x(self) -> NDArray[np.float64]:
        """Position of :attr:`best_value`, with integer and categorical variables decoded.

        Reading it waits for the device.
        """
        if not self._started:
            return np.full(self.dim, np.nan)
        return self._problem.decode(self._best_position())

    @property
    def nfev(self) -> int:
        """Number of objective evaluations so far."""
        return self._rows_evaluated

    # ---------------------------------------------------------------- running

    def step(self) -> None:
        """Carry out one iteration.

        Raises
        ------
        RuntimeError
            If the run has already ended.
        """
        if self.finished:
            raise RuntimeError(
                f"this run has ended ({self._termination.value}); create a new optimizer"
            )
        self._ensure_started()
        start = time.perf_counter()
        self._advance(self._iteration)
        if self._iteration % self._history_every == 0:
            self._record()
        self._iteration += 1
        self._elapsed += time.perf_counter() - start
        if self._iteration == self._max_iter:
            self._finish(TerminationReason.MAX_ITER)

    def run(
        self,
        *,
        callback: Callable[[Optimizer], bool | None] | None = None,
        target: float | None = None,
        patience: int | None = None,
        tol: float = 0.0,
        max_time: float | None = None,
    ) -> OptimizeResult:
        """Iterate until ``max_iter`` or until a stopping rule applies.

        The stopping rules and the callback are evaluated every
        ``history_every`` iterations. Each evaluation reads the best value
        from the device, which waits for the queued kernels; without rules
        and callback the loop never waits.

        Parameters
        ----------
        callback : callable, optional
            Called with the optimizer; a return value of True ends the run.
        target : float, optional
            End the run when the best value is at most ``target`` (at least
            ``target`` when maximising).
        patience : int, optional
            End the run when the best value has not improved by more than
            ``tol`` for this many iterations.
        tol : float, default 0.0
            Smallest improvement that resets ``patience``.
        max_time : float, optional
            End the run after this many seconds.

        Returns
        -------
        OptimizeResult
        """
        if patience is not None:
            patience = _problem.check_count(patience, "patience", 1)
        if not (isinstance(tol, int | float) and tol >= 0.0):
            raise ValueError(f"tol must be a non-negative number, got {tol!r}")
        if max_time is not None and not max_time > 0.0:
            raise ValueError(f"max_time must be positive, got {max_time!r}")
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback).__name__}")
        watch = callback is not None or target is not None or patience is not None
        started = time.perf_counter()
        reference_value, reference_iteration = math.inf, self._iteration
        while not self.finished:
            self.step()
            if self.finished or self._iteration % self._history_every:
                continue
            if max_time is not None and time.perf_counter() - started >= max_time:
                self._finish(TerminationReason.MAX_TIME)
            elif watch:
                best = self._best_score()
                if target is not None and best <= self._sign * target:
                    self._finish(TerminationReason.TARGET)
                    continue
                if best < reference_value - tol:
                    reference_value, reference_iteration = best, self._iteration
                elif patience is not None and self._iteration - reference_iteration >= patience:
                    self._finish(TerminationReason.STALLED)
                    continue
                if callback is not None and callback(self) is True:
                    self._finish(TerminationReason.CALLBACK)
        return self.result()

    def result(self) -> OptimizeResult:
        """Return the outcome of the run so far."""
        self._ensure_started()
        self._wait()
        x = self.best_x
        fun, violation = self.best_value, 0.0
        if self._problem.constraints and math.isfinite(self._best_score()):
            # Only the penalised value is tracked during the run; objective
            # and violation are evaluated once more at the best position.
            fun, violation = self._detail()
            self._wait()
        return OptimizeResult(
            x=x,
            fun=fun,
            history=self._sign * np.asarray(self._history_scores(), dtype=np.float64),
            nfev=self.nfev,
            nit=self._iteration,
            runtime=self._elapsed,
            compile_time=self._compile_time,
            termination=self._termination,
            seed=self._seed,
            algorithm=self.name,
            selection=_selection.describe(self._selection),
            constraint_violation=violation,
            feasible=violation == 0.0,
            values=self._problem.values(x),
        )

    def _ensure_started(self) -> None:
        if not self._started:
            self._start()
            self._started = True

    def _finish(self, reason: TerminationReason) -> None:
        self._ensure_started()
        start = time.perf_counter()
        self._end()
        self._record()
        self._wait()
        self._elapsed += time.perf_counter() - start
        self._termination = reason

    # --------------------------------------------------------- for subclasses

    @abc.abstractmethod
    def _start(self) -> None:
        """Allocate the state of the run and create the initial population."""

    @abc.abstractmethod
    def _advance(self, it: int) -> None:
        """Carry out iteration ``it``."""

    def _end(self) -> None:  # noqa: B027 - optional hook
        """Account for the population left by the last iteration."""

    @abc.abstractmethod
    def _record(self) -> None:
        """Append the best value so far to the history."""

    @abc.abstractmethod
    def _wait(self) -> None:
        """Block until all queued device work has finished."""

    @abc.abstractmethod
    def _best_score(self) -> float:
        """Return the minimised value at the best position so far."""

    @abc.abstractmethod
    def _best_position(self) -> NDArray[np.float64]:
        """Return the best position so far, before decoding."""

    @abc.abstractmethod
    def _history_scores(self) -> NDArray[np.float64]:
        """Return the recorded minimised values."""

    @abc.abstractmethod
    def _detail(self) -> tuple[float, float]:
        """Return objective and constraint violation at the best position."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._problem.name!r}, dim={self.dim}, "
            f"pop_size={self._ps}, max_iter={self._max_iter}, "
            f"selection={_selection.describe(self._selection)!r}, seed={self._seed})"
        )
