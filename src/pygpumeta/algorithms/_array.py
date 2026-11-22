"""Optimizers written with array operations instead of CUDA kernels.

The population is an ordinary array, a CuPy array on the device or a NumPy
array on the host, and an algorithm is a few lines of array code. The
selection operators are the only part that runs in hand-written kernels.
"""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pygpumeta import _problem
from pygpumeta import selection as _selection
from pygpumeta._objective import KernelEvaluator, Scorer, import_cupy
from pygpumeta.algorithms._base import Optimizer


class DeviceSelector:
    """Runs a selection operator on scores held in a CuPy or NumPy array."""

    def __init__(self, xp: Any, operator: _selection.Selection, count: int) -> None:
        from pygpumeta._backend.runtime import get_runtime
        from pygpumeta._backend.selection import shared_selection_kernels

        rt = get_runtime()
        gpuarray = rt.gpuarray
        self._xp = xp
        self._operator = operator
        self._count = count
        self._kernels = shared_selection_kernels()
        self._fit = gpuarray.empty(count, np.float64)
        self._mean = rt.zeros(1, np.float64)
        if xp is np:
            self._scores = gpuarray.empty(count, np.float64)
            self._selected = gpuarray.empty(count, np.int32)

    def __call__(self, scores: Any, seed: int) -> Any:
        """Return the selected index of every individual."""
        if self._xp is np:
            self._scores.set(np.ascontiguousarray(scores, dtype=np.float64))
            d_scores, d_selected = self._scores, self._selected
        else:
            d_scores = self._xp.ascontiguousarray(scores, dtype=np.float64)
            d_selected = self._xp.empty(self._count, np.int32)
        _selection.apply(
            self._operator,
            self._kernels,
            d_scores,
            self._fit,
            d_selected,
            self._mean,
            self._count,
            seed,
        )
        return d_selected.get().astype(np.int64) if self._xp is np else d_selected


class ArrayOptimizer(Optimizer):
    """Base class of the optimizers written with array operations.

    A subclass implements :meth:`_initialise` and :meth:`_iterate` with the
    helpers below. ``self._xp`` is the array module (``cupy`` or ``numpy``),
    ``self._rng`` a seeded ``RandomState`` of that module, and
    ``self._lower`` and ``self._upper`` the bounds as arrays.

    - ``self._score(positions)`` returns the value to minimise for every row
      (objective with sign, decoding and constraint penalty applied);
    - ``self._update_best(positions, scores)`` merges a set of evaluated
      positions into the best-so-far;
    - ``self._select(scores, it)`` returns the index selected for every
      individual, or None when the user asked for the native rule;
    - ``self._clip(positions)`` projects positions onto the box.
    """

    _min_pop_size: ClassVar[int] = 2
    _best: Any
    _best_pos: Any

    def __init__(self, objective: Any, bounds: Any = None, **common: Any) -> None:
        super().__init__(objective, bounds, **common)
        _problem.check_count(self._ps, "pop_size", self._min_pop_size)

    # --------------------------------------------------------- for subclasses

    @abc.abstractmethod
    def _initialise(self) -> None:
        """Create the initial population."""

    @abc.abstractmethod
    def _iterate(self, it: int) -> None:
        """Carry out iteration ``it``."""

    def _finalise(self) -> None:
        """Evaluate whatever the last iteration left unevaluated."""

    def _score(self, positions: Any) -> Any:
        """Return the value to minimise for every row of ``positions``."""
        self._rows_evaluated += int(positions.shape[0])
        if self._kernel_objective is None:
            return self._scorer(positions)
        p = self._problem
        values = self._kernel_values(positions)
        violation = (
            self._scorer.violation(self._scorer.decode(positions)) if p.constraints else None
        )
        return self._scorer.score(values, violation)

    def _update_best(self, positions: Any, scores: Any) -> None:
        """Merge evaluated positions into the best-so-far, without waiting for the device."""
        xp = self._xp
        index = xp.argmin(scores)
        better = scores[index] < self._best
        self._best = xp.where(better, scores[index], self._best)
        self._best_pos = xp.where(better, positions[index], self._best_pos)

    def _select(self, scores: Any, it: int) -> Any | None:
        """Return the selected index per individual, or None for the native rule."""
        if self._selector is None:
            return None
        return self._selector(scores, (self._seed + 3000 + it) & 0xFFFFFFFF)

    def _clip(self, positions: Any) -> Any:
        """Project positions onto the box."""
        return self._xp.clip(positions, self._lower, self._upper)

    def _uniform_population(self) -> Any:
        """Return ``pop_size`` positions drawn uniformly from the box."""
        u = self._rng.random_sample((self._ps, self.dim))
        return self._lower + u * (self._upper - self._lower)

    # ------------------------------------------------- the optimizer interface

    def _start(self) -> None:
        p = self._problem
        on_device = p.arrays == "gpu"
        self._kernel_objective: KernelEvaluator | None = None
        before = time.perf_counter()
        if on_device or p.in_kernel:
            from pygpumeta._backend.runtime import get_runtime

            self._rt = get_runtime()
        if p.in_kernel:
            from pygpumeta._backend.selection import load_kernels

            kernels = load_kernels(self._constants, user_objective=p.cuda_source, negate=p.maximize)
            self._kernel_objective = KernelEvaluator(kernels, p.function_id, p.dim)
        self._compile_time = time.perf_counter() - before
        xp = self._xp = import_cupy() if on_device else np
        self._scorer = Scorer(xp, p)
        self._selector = self._make_selector()
        self._rng = xp.random.RandomState(self._seed)
        self._lower = xp.asarray(p.lower)
        self._upper = xp.asarray(p.upper)
        self._best = xp.asarray(np.inf)
        self._best_pos = xp.zeros(p.dim)
        self._history = xp.zeros(self._max_iter // self._history_every + 2)
        self._history_count = 0
        start = time.perf_counter()
        self._initialise()
        self._elapsed += time.perf_counter() - start

    def _make_selector(self) -> Any | None:
        if self._selection is None:
            return None
        return DeviceSelector(self._xp, self._selection, self._ps)

    def _advance(self, it: int) -> None:
        self._iterate(it)

    def _end(self) -> None:
        self._finalise()

    def _record(self) -> None:
        self._history[self._history_count] = self._best
        self._history_count += 1

    def _wait(self) -> None:
        if self._xp is not np:
            self._xp.cuda.Stream.null.synchronize()

    def _best_score(self) -> float:
        return float(self._best)

    def _best_position(self) -> NDArray[np.float64]:
        return self._to_host(self._best_pos)

    def _history_scores(self) -> NDArray[np.float64]:
        return self._to_host(self._history[: self._history_count])

    def _detail(self) -> tuple[float, float]:
        position = self._best_pos.reshape(1, -1)
        decoded = self._scorer.decode(position)
        violation = float(self._scorer.violation(decoded)[0])
        if self._kernel_objective is None:
            return float(self._scorer.objective(decoded)[0]), violation
        return self._sign * float(self._kernel_values(position)[0]), violation

    # --------------------------------------------------------------- internal

    def _kernel_values(self, positions: Any) -> Any:
        """Evaluate a built-in or CUDA objective on array positions."""
        assert self._kernel_objective is not None
        count = int(positions.shape[0])
        if self._xp is np:
            gpuarray = self._rt.gpuarray
            d_points = gpuarray.to_gpu(np.ascontiguousarray(positions, dtype=np.float64).ravel())
            d_values = gpuarray.empty(count, np.float64)
            self._kernel_objective(d_points, d_values, count)
            return d_values.get()
        points = self._xp.ascontiguousarray(positions, dtype=np.float64)
        values = self._xp.empty(count, np.float64)
        self._kernel_objective(points, values, count)
        return values

    def _to_host(self, array: Any) -> NDArray[np.float64]:
        if self._xp is np:
            return np.array(array, dtype=np.float64)
        return np.asarray(self._xp.asnumpy(array), dtype=np.float64)
