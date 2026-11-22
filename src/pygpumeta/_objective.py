"""The four ways of supplying an objective."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pygpumeta._backend import source as _source
from pygpumeta._backend.arrays import kernel_argument
from pygpumeta._backend.runtime import Runtime
from pygpumeta._backend.selection import BLOCK, KernelLibrary, grid_for

__all__ = ["CudaObjective"]

_LARGEST = _source.LARGEST_VALUE
# Feasibility rule: feasible values are kept inside +-FEASIBLE_LIMIT and an
# infeasible position scores INFEASIBLE_BASE * (1 + violation), which stays
# below the 1e300 that the leader search of GWO treats as "no value".
FEASIBLE_LIMIT = 1e149
INFEASIBLE_BASE = 1e150
VIOLATION_LIMIT = 1e100


@dataclass(frozen=True)
class CudaObjective:
    """An objective written in CUDA C and compiled into the kernels.

    Parameters
    ----------
    source : str
        CUDA C code defining ``double evaluate(const double* x, int d)``,
        which returns the objective value of the ``d`` variables at ``x``.
        Helper functions in the same source must be declared ``__device__``.

    Notes
    -----
    Non-finite return values are replaced by plus or minus 1e299. The
    kernels are compiled once per distinct source and process.

    Examples
    --------
    >>> objective = CudaObjective('''
    ... double evaluate(const double* x, int d) {
    ...     double s = 0.0;
    ...     for (int i = 0; i < d; i++) s += (x[i] - 1.0) * (x[i] - 1.0);
    ...     return s;
    ... }''')
    """

    source: str

    def __post_init__(self) -> None:
        _source.prepare_user_objective(self.source)


class Evaluator(Protocol):
    """Evaluates rows of a device array of positions."""

    def __call__(self, d_points: Any, d_values: Any, count: int) -> None:
        """Write the objective of the first ``count`` rows of ``d_points`` to ``d_values``."""


class KernelEvaluator:
    """Built-in function or CUDA objective, evaluated by the ``eval_pop`` kernel."""

    def __init__(self, kernels: KernelLibrary, function_id: int, dim: int) -> None:
        self._kernel = kernels.algorithms.kernel("eval_pop")
        self._fid = np.int32(function_id)
        self._dim = np.int32(dim)

    def __call__(self, d_points: Any, d_values: Any, count: int) -> None:
        self._kernel(
            kernel_argument(d_points),
            kernel_argument(d_values),
            np.int32(count),
            self._dim,
            self._fid,
            block=(BLOCK, 1, 1),
            grid=grid_for(count),
        )


class Scorer:
    """Decoding, constraints and sign: positions to the value the kernels minimise.

    Parameters
    ----------
    xp : module
        Array namespace of the positions, ``numpy`` or ``cupy``.
    problem : Problem
        The validated problem.
    """

    def __init__(self, xp: Any, problem: Any) -> None:
        self._xp = xp
        self._p = problem
        self._sign = -1.0 if problem.maximize else 1.0
        self._columns = xp.asarray(problem.discrete_index)
        self._low = xp.asarray(problem.discrete_low)
        self._high = xp.asarray(problem.discrete_high)

    def decode(self, points: Any) -> Any:
        """Round the discrete variables; the continuous ones pass through."""
        if self._p.discrete_index.size == 0:
            return points
        decoded = points.copy()
        decoded[:, self._columns] = self._xp.clip(
            self._xp.rint(decoded[:, self._columns]), self._low, self._high
        )
        return decoded

    def objective(self, decoded: Any) -> Any:
        """Objective values as the user defined them, shape ``(n,)``."""
        count = decoded.shape[0]
        return self._column(self._call(self._p.function, decoded), count, "objective")

    def violation(self, decoded: Any) -> Any:
        """Total constraint violation, shape ``(n,)``."""
        xp, count = self._xp, decoded.shape[0]
        total = xp.zeros(count, dtype=xp.float64)
        for number, constraint in enumerate(self._p.constraints):
            raw = self._call(constraint.fun, decoded)
            g = self._matrix(raw, count, f"constraint {number}")
            # NaN compares false in maximum(), which would hide it.
            g = xp.nan_to_num(g, nan=_LARGEST, posinf=_LARGEST, neginf=-_LARGEST)
            if constraint.kind == "eq":
                g = xp.abs(g) - constraint.tol
            total = total + xp.maximum(g, 0.0).sum(axis=1)
        return total

    def score(self, signed_objective: Any, violation: Any | None) -> Any:
        """Combine a (sign-corrected) objective with the constraints and bound it."""
        xp = self._xp
        value = signed_objective
        if violation is not None and self._p.constraint_handling == "penalty":
            value = value + self._p.penalty * violation
        value = xp.nan_to_num(value, nan=_LARGEST, posinf=_LARGEST, neginf=-_LARGEST)
        if violation is not None and self._p.constraint_handling == "feasibility":
            # One number that orders positions by the feasibility rule: every
            # infeasible position lies above every feasible one and
            # infeasible positions are ranked by violation alone. The factor
            # form keeps the violation resolved to about 1e-16.
            feasible = xp.clip(value, -FEASIBLE_LIMIT, FEASIBLE_LIMIT)
            infeasible = INFEASIBLE_BASE * (1.0 + xp.minimum(violation, VIOLATION_LIMIT))
            value = xp.where(violation > 0.0, infeasible, feasible)
        return value

    def __call__(self, points: Any) -> Any:
        """Return the value to minimise for every row of ``points``."""
        decoded = self.decode(points)
        violation = self.violation(decoded) if self._p.constraints else None
        return self.score(self._sign * self.objective(decoded), violation)

    def _call(self, function: Callable[[Any], Any], decoded: Any) -> Any:
        if self._p.vectorized:
            return function(decoded)
        return [function(row) for row in decoded]

    def _array(self, raw: Any, what: str) -> Any:
        xp = self._xp
        if xp is np:
            try:
                return np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError):
                raise TypeError(
                    f"the {what} must return real numbers, one row per row of its argument"
                ) from None
        if not isinstance(raw, xp.ndarray):
            raise TypeError(
                f"the {what} must return a cupy.ndarray, got {type(raw).__name__}; "
                "use mode='host' for functions that work on NumPy arrays"
            )
        return raw.astype(xp.float64, copy=False)

    def _column(self, raw: Any, count: int, what: str) -> Any:
        values = self._array(raw, what)
        if values.shape != (count,):
            raise ValueError(
                f"the {what} must return one value per row, an array of shape ({count},), "
                f"got shape {values.shape}"
            )
        return values

    def _matrix(self, raw: Any, count: int, what: str) -> Any:
        values = self._array(raw, what)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != count:
            raise ValueError(
                f"the {what} must return an array of shape ({count},) or ({count}, m), "
                f"got shape {self._array(raw, what).shape}"
            )
        return values


class CupyAdapter:
    """Zero-copy CuPy access to PyCUDA buffers."""

    def __init__(self) -> None:
        self.xp = import_cupy()
        # Views are created once per buffer; the buffers are kept alive
        # alongside so their ids stay unique.
        self._views: dict[int, tuple[Any, Any]] = {}

    def _view(self, buffer: Any) -> Any:
        entry = self._views.get(id(buffer))
        if entry is None:
            entry = self._views[id(buffer)] = (buffer, self.xp.asarray(buffer))
        return entry[1]

    def points(self, d_points: Any, count: int, dim: int) -> Any:
        """Return the first ``count`` positions as an array of shape ``(count, dim)``."""
        return self._view(d_points)[: count * dim].reshape(count, dim)

    def values(self, d_values: Any, count: int) -> Any:
        """Return the first ``count`` values."""
        return self._view(d_values)[:count]

    def write(self, d_values: Any, values: Any, count: int) -> None:
        """Store ``values`` in the first ``count`` elements of ``d_values``."""
        self._view(d_values)[:count] = values


class HostAdapter:
    """NumPy access to PyCUDA buffers through host copies."""

    xp = np

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    def points(self, d_points: Any, count: int, dim: int) -> Any:
        """Return the first ``count`` positions as an array of shape ``(count, dim)``."""
        return d_points.get()[: count * dim].reshape(count, dim)

    def values(self, d_values: Any, count: int) -> Any:
        """Return the first ``count`` values."""
        return d_values.get()[:count]

    def write(self, d_values: Any, values: Any, count: int) -> None:
        """Store ``values`` in the first ``count`` elements of ``d_values``."""
        data = np.ascontiguousarray(values, dtype=np.float64)
        self._rt.driver.memcpy_htod(d_values.gpudata, data)


class PythonEvaluator:
    """Evaluation that involves Python code: a callable objective, constraints or both.

    Parameters
    ----------
    adapter : CupyAdapter or HostAdapter
        Access to the device buffers in the array type the user code expects.
    scorer : Scorer
        Decoding, constraint and sign handling.
    dim : int
        Number of variables.
    kernel : KernelEvaluator, optional
        Evaluates the objective when it is a built-in function or a CUDA
        source; the Python side then only adds the constraint penalty.
    """

    def __init__(
        self, adapter: Any, scorer: Scorer, dim: int, kernel: KernelEvaluator | None = None
    ) -> None:
        self._adapter = adapter
        self._scorer = scorer
        self._dim = dim
        self._kernel = kernel

    def __call__(self, d_points: Any, d_values: Any, count: int) -> None:
        points = self._adapter.points(d_points, count, self._dim)
        if self._kernel is None:
            scores = self._scorer(points)
        else:
            # The kernel already returns the sign-corrected objective.
            self._kernel(d_points, d_values, count)
            violation = self._scorer.violation(self._scorer.decode(points))
            scores = self._scorer.score(self._adapter.values(d_values, count), violation)
        self._adapter.write(d_values, scores, count)

    def detail(self, d_point: Any, d_scratch: Any) -> tuple[float, float]:
        """Return the true objective and the violation of one position."""
        decoded = self._scorer.decode(self._adapter.points(d_point, 1, self._dim))
        violation = float(self._scorer.violation(decoded)[0])
        if self._kernel is None:
            return float(self._scorer.objective(decoded)[0]), violation
        self._kernel(d_point, d_scratch, 1)
        signed = float(self._adapter.values(d_scratch, 1)[0])
        return signed * self._scorer._sign, violation


def import_cupy() -> Any:
    """Import CuPy, explaining what to do when it is missing."""
    try:
        import cupy
    except Exception as exc:
        raise ImportError(
            "objectives and constraints that work on GPU arrays need CuPy, which could not "
            f"be imported ({type(exc).__name__}: {exc}). Install the CuPy build for your CUDA "
            "version, 'pip install cupy-cuda12x' for CUDA 12 or 'pip install cupy-cuda13x' "
            "for CUDA 13, or pass mode='host' to evaluate on the CPU."
        ) from exc
    return cupy
