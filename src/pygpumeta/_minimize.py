"""One-call entry point."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pygpumeta._result import OptimizeResult
from pygpumeta.algorithms import BY_NAME, Optimizer


def minimize(
    objective: Any,
    bounds: Any = None,
    *,
    dim: int | None = None,
    algorithm: str = "gwo",
    selection: Any = "auto",
    pop_size: int = 1024,
    max_iter: int = 1000,
    seed: int = 0,
    mode: str = "auto",
    vectorized: bool = True,
    constraints: Any = (),
    constraint_handling: str = "penalty",
    penalty: float = 1e6,
    maximize: bool = False,
    target: float | None = None,
    patience: int | None = None,
    tol: float = 0.0,
    max_time: float | None = None,
    callback: Callable[[Optimizer], bool | None] | None = None,
    history_every: int = 1,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """Minimise an objective over a box with a GPU metaheuristic.

    Parameters
    ----------
    objective : callable, str or CudaObjective
        A callable receives the positions to evaluate as a CuPy array of
        shape ``(n, dim)`` on the GPU and returns their objective values as
        a CuPy array of shape ``(n,)``. With ``mode="host"`` it receives and
        returns NumPy arrays and runs on the CPU. A string names a built-in
        test function, see :func:`pygpumeta.benchmarks.names`. A
        :class:`~pygpumeta.CudaObjective` is compiled into the kernels.
    bounds : (lower, upper), array_like of shape (dim, 2) or sequence of variables
        Search box: one pair for all variables, one pair per variable, or
        one :class:`~pygpumeta.space.Real`, :class:`~pygpumeta.space.Integer`
        or :class:`~pygpumeta.space.Categorical` per variable. Required
        except for built-in test functions. Integer and categorical
        variables are rounded before the objective sees them and need a
        callable objective.
    dim : int, optional
        Number of variables; required when ``bounds`` is a single pair.
    algorithm : str, default "gwo"
        One of ``"abc"``, ``"gwo"``, ``"woa"``, ``"sca"``, ``"ssa"``, ``"hho"``,
        ``"pso"``, ``"de"``.
    selection : str or selection operator, default "auto"
        ``"tournament"``, ``"ballot_elite"``, ``"migration"``, an instance
        of the classes in :mod:`pygpumeta.selection`, or ``"native"`` for
        the rule of the original algorithm. ``"auto"`` uses the warp
        tournament for ABC, GWO, WOA and HHO and the native rule for SCA,
        SSA, PSO and DE.
    pop_size : int, default 1024
        Number of individuals.
    max_iter : int, default 1000
        Number of iterations.
    seed : int, default 0
        Seed of the run; equal arguments and seeds give equal results.
    mode : {"auto", "gpu", "host"}, default "auto"
        Where a callable objective runs: on CuPy arrays on the device
        (``"gpu"``, also chosen by ``"auto"``) or on NumPy arrays on the CPU
        (``"host"``). The host mode copies the population to the CPU and the
        values back at every evaluation; only the selection and the update
        rules then run on the GPU.
    vectorized : bool, default True
        With ``mode="host"`` only: False calls the objective once per
        position with a one-dimensional array.
    constraints : sequence of Constraint, optional
        Constraints, see :class:`~pygpumeta.Constraint`.
    constraint_handling : {"penalty", "feasibility"}, default "penalty"
        ``"penalty"`` minimises ``objective + penalty * violation``;
        ``"feasibility"`` ranks feasible positions before infeasible ones
        and needs no penalty factor.
    penalty : float, default 1e6
        Penalty factor of ``"penalty"``.
    maximize : bool, default False
        Maximise the objective instead of minimising it.
    target : float, optional
        Stop when the best value is at most ``target`` (at least ``target``
        when maximising).
    patience : int, optional
        Stop when the best value has not improved by more than ``tol`` for
        this many iterations.
    tol : float, default 0.0
        Smallest improvement that resets ``patience``.
    max_time : float, optional
        Stop after this many seconds.
    callback : callable, optional
        Called with the optimizer every ``history_every`` iterations; a
        return value of True stops the run.
    history_every : int, default 1
        Interval at which the best value is recorded and the stopping rules
        are evaluated.
    options : mapping, optional
        Parameters of the algorithm, for example ``{"limit": 200}`` for ABC
        or ``{"evaluation": "separated"}`` for ABC and HHO.

    Returns
    -------
    OptimizeResult

    Examples
    --------
    >>> import pygpumeta
    >>> def shifted_sphere(x):  # x is a cupy.ndarray of shape (n, 30)
    ...     return ((x - 1.0) ** 2).sum(axis=1)
    >>> result = pygpumeta.minimize(shifted_sphere, (-5.0, 5.0), dim=30)  # doctest: +SKIP
    """
    if not isinstance(algorithm, str) or algorithm not in BY_NAME:
        raise ValueError(f"algorithm must be one of {', '.join(BY_NAME)}, got {algorithm!r}")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError(f"options must be a mapping, got {type(options).__name__}")
    cls = BY_NAME[algorithm]
    try:
        optimizer = cls(
            objective,
            bounds,
            dim=dim,
            pop_size=pop_size,
            max_iter=max_iter,
            selection=selection,
            seed=seed,
            history_every=history_every,
            mode=mode,
            vectorized=vectorized,
            constraints=constraints,
            constraint_handling=constraint_handling,
            penalty=penalty,
            maximize=maximize,
            **dict(options or {}),
        )
    except TypeError as exc:
        unknown = [key for key in (options or {}) if f"'{key}'" in str(exc)]
        if unknown:
            raise TypeError(f"algorithm {algorithm!r} has no option {unknown[0]!r}") from None
        raise
    return optimizer.run(
        callback=callback, target=target, patience=patience, tol=tol, max_time=max_time
    )
