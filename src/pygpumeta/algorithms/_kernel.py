"""Common machinery of the optimizers written as CUDA kernels."""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pygpumeta import selection as _selection
from pygpumeta._backend import source as _source
from pygpumeta._backend.runtime import Runtime, get_runtime
from pygpumeta._backend.selection import (
    BLOCK,
    KernelLibrary,
    SelectionKernels,
    grid_for,
    load_kernels,
)
from pygpumeta._objective import (
    CupyAdapter,
    Evaluator,
    HostAdapter,
    KernelEvaluator,
    PythonEvaluator,
    Scorer,
)
from pygpumeta.algorithms._base import Optimizer

_ONE_THREAD: Mapping[str, tuple[int, ...]] = {"block": (1, 1, 1), "grid": (1, 1)}

# The evaluation counter is a global of the compiled module, so it serves one
# optimizer at a time; this records which one, per module.
_counter_owner: dict[str, KernelOptimizer] = {}


class _Device:
    """Device buffers of one run."""

    def __init__(self, rt: Runtime, ps: int, dim: int, history_slots: int) -> None:
        ga = rt.gpuarray
        self.pop = rt.zeros(ps * dim, np.float64)
        self.pop_next = rt.zeros(ps * dim, np.float64)
        self.obj = rt.zeros(ps, np.float64)
        self.fit = rt.zeros(ps, np.float64)
        self.sel = rt.zeros(ps, np.int32)
        self.mean_fit = rt.zeros(1, np.float64)
        self.best_index = rt.zeros(1, np.int32)
        self.best_now = rt.zeros(1, np.float64)
        self.best = ga.to_gpu(np.array([np.inf], dtype=np.float64))
        self.best_pos = rt.zeros(dim, np.float64)
        self.history = rt.zeros(history_slots, np.float64)
        self.history_count = 0

    def swap(self) -> None:
        self.pop, self.pop_next = self.pop_next, self.pop


class KernelOptimizer(Optimizer):
    """An optimizer whose population lives in device buffers updated by kernels."""

    # Set by algorithms that evaluate inside their update kernels, where the
    # number of evaluations is only known to the device.
    _counts_on_device: bool = False
    _counted = 0
    _dev: _Device | None = None

    @property
    def nfev(self) -> int:
        """Number of objective evaluations so far."""
        if (
            self._counts_on_device
            and self._dev is not None
            and _counter_owner.get(self._kernels.algorithms.source_hash) is self
        ):
            self._collect_count()
        return self._rows_evaluated + self._counted

    # ------------------------------------------------- the optimizer interface

    def _start(self) -> None:
        self._device()

    def _advance(self, it: int) -> None:
        self._claim_counter()
        self._iterate(self._device(), it)

    def _end(self) -> None:
        self._claim_counter()
        self._finalise(self._device())

    def _record(self) -> None:
        self._store_history(self._device())

    def _wait(self) -> None:
        self._rt.synchronize()

    def _best_score(self) -> float:
        return float(self._device().best.get()[0])

    def _best_position(self) -> NDArray[np.float64]:
        return np.asarray(self._device().best_pos.get(), dtype=np.float64)

    def _history_scores(self) -> NDArray[np.float64]:
        dev = self._device()
        return np.asarray(dev.history.get()[: dev.history_count], dtype=np.float64)

    def _detail(self) -> tuple[float, float]:
        dev = self._device()
        assert isinstance(self._evaluator, PythonEvaluator)
        return self._evaluator.detail(dev.best_pos, dev.best_now)

    # --------------------------------------------------------- for subclasses

    @abc.abstractmethod
    def _initialise(self, dev: _Device) -> None:
        """Create the initial population."""

    @abc.abstractmethod
    def _iterate(self, dev: _Device, it: int) -> None:
        """Queue the kernels of iteration ``it``."""

    def _finalise(self, dev: _Device) -> None:
        """Account for the population left by the last iteration."""

    def _extra_buffers(self, rt: Runtime) -> None:
        """Allocate buffers specific to the algorithm."""

    def _wide(self, name: str, *args: Any, count: int | None = None) -> None:
        """Launch a kernel with one thread per individual (or per ``count``)."""
        self._kernels.algorithms.kernel(name)(
            *args, block=(BLOCK, 1, 1), grid=grid_for(self._ps if count is None else count)
        )

    def _single(self, name: str, *args: Any) -> None:
        """Launch a kernel that does its work in one thread."""
        self._kernels.algorithms.kernel(name)(*args, **_ONE_THREAD)

    def _seed_for(self, offset: int) -> np.uint32:
        return np.uint32((self._seed + offset) & 0xFFFFFFFF)

    def _select(self, dev: _Device, it: int) -> bool:
        """Fill ``dev.sel``; returns False when the native rule applies."""
        if self._selection is None:
            return False
        _selection.apply(
            self._selection,
            self._select_kernels,
            dev.obj,
            dev.fit,
            dev.sel,
            dev.mean_fit,
            self._ps,
            (self._seed + 3000 + it) & 0xFFFFFFFF,
        )
        return True

    def _evaluate_population(self, dev: _Device) -> None:
        """Clamp, evaluate and merge the population into the best-so-far."""
        ps, dim = np.int32(self._ps), np.int32(self._problem.dim)
        self._wide("boundary_clamp", dev.pop, self._lb, self._ub, ps, dim)
        self._evaluate(dev.pop, dev.obj, self._ps)
        self._merge_best(dev)

    def _evaluate(self, d_points: Any, d_values: Any, count: int) -> None:
        """Evaluate ``count`` positions outside the update kernels."""
        self._evaluator(d_points, d_values, count)
        if not self._counts_on_device:
            self._rows_evaluated += count

    def _merge_best(self, dev: _Device) -> None:
        self._single("find_best_idx", dev.obj, dev.best_index, dev.best_now, np.int32(self._ps))
        self._kernels.selection.kernel("update_best_ever_pos")(
            dev.best_now,
            dev.best,
            dev.pop,
            dev.best_index,
            dev.best_pos,
            np.int32(self._problem.dim),
            **_ONE_THREAD,
        )

    # --------------------------------------------------------------- internal

    def _device(self) -> _Device:
        if self._dev is None:
            self._rt = get_runtime()
            before = time.perf_counter()
            self._kernels: KernelLibrary = load_kernels(
                self._constants,
                count_evaluations=self._counts_on_device,
                user_objective=self._problem.cuda_source,
                negate=self._problem.maximize and self._problem.in_kernel,
            )
            self._evaluator = self._make_evaluator()
            self._select_kernels = SelectionKernels(self._kernels, self._rt)
            self._compile_time = time.perf_counter() - before
            self._lb = self._rt.gpuarray.to_gpu(self._problem.lower)
            self._ub = self._rt.gpuarray.to_gpu(self._problem.upper)
            slots = self._max_iter // self._history_every + 2
            dev = _Device(self._rt, self._ps, self._problem.dim, slots)
            self._extra_buffers(self._rt)
            self._rt.synchronize()
            start = time.perf_counter()
            self._claim_counter()
            self._initialise(dev)
            self._elapsed += time.perf_counter() - start
            self._dev = dev
        return self._dev

    def _make_evaluator(self) -> Evaluator:
        p = self._problem
        kernel = KernelEvaluator(self._kernels, p.function_id, p.dim) if p.in_kernel else None
        if kernel is not None and not p.constraints:
            return kernel
        adapter = HostAdapter(self._rt) if p.arrays == "host" else CupyAdapter()
        return PythonEvaluator(adapter, Scorer(adapter.xp, p), p.dim, kernel)

    def _store_history(self, dev: _Device) -> None:
        self._kernels.selection.kernel("store_history")(
            dev.best, dev.history, np.int32(dev.history_count), **_ONE_THREAD
        )
        dev.history_count += 1

    def _claim_counter(self) -> None:
        if not self._counts_on_device:
            return
        key = self._kernels.algorithms.source_hash
        owner = _counter_owner.get(key)
        if owner is self:
            return
        if owner is not None:
            owner._collect_count()
        else:
            self._write_counter(0)
        _counter_owner[key] = self

    def _collect_count(self) -> None:
        self._rt.synchronize()
        value = np.zeros(1, np.uint64)
        self._rt.driver.memcpy_dtoh(value, self._counter_pointer())
        self._counted += int(value[0])
        self._write_counter(0)

    def _write_counter(self, value: int) -> None:
        self._rt.synchronize()
        self._rt.driver.memcpy_htod(self._counter_pointer(), np.array([value], np.uint64))

    def _counter_pointer(self) -> Any:
        return self._kernels.algorithms.global_pointer(_source.EVALUATION_COUNTER)


class GuidedOptimizer(KernelOptimizer):
    """Algorithms that move the population towards the best position so far.

    Every iteration evaluates the population, updates the guide and writes
    the moved population to the second buffer.
    """

    _needs_mean: ClassVar[bool] = False

    def _extra_buffers(self, rt: Runtime) -> None:
        dim = self._problem.dim
        self._guide = rt.zeros(dim, np.float64)
        self._guide_score = rt.gpuarray.to_gpu(np.array([np.inf], dtype=np.float64))
        self._mean = rt.zeros(dim, np.float64) if self._needs_mean else None

    def _initialise(self, dev: _Device) -> None:
        self._wide(
            "pop_init",
            dev.pop,
            self._lb,
            self._ub,
            np.int32(self._ps),
            np.int32(self._problem.dim),
            self._seed_for(1),
        )

    def _iterate(self, dev: _Device, it: int) -> None:
        ps, dim = np.int32(self._ps), np.int32(self._problem.dim)
        self._evaluate_population(dev)
        self._single(
            "update_guide_if_better",
            dev.pop,
            dev.best_index,
            dev.best_now,
            self._guide,
            self._guide_score,
            dim,
        )
        if self._needs_mean:
            self._wide("compute_mean_pos", dev.pop, self._mean, ps, dim, count=self._problem.dim)
        self._move(dev, it, self._select(dev, it), self._seed_for(7000 + it))
        dev.swap()

    def _finalise(self, dev: _Device) -> None:
        self._evaluate_population(dev)

    @abc.abstractmethod
    def _move(self, dev: _Device, it: int, selected: bool, seed: np.uint32) -> None:
        """Queue the update kernel; ``selected`` says whether ``dev.sel`` is filled."""
