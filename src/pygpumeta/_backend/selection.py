"""Launch code of the selection kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pygpumeta._backend import source
from pygpumeta._backend.arrays import kernel_argument
from pygpumeta._backend.module import CompiledModule, Compiler, compile_cached
from pygpumeta._backend.runtime import Runtime, get_runtime

BLOCK = 256
_ARGMIN_BLOCK = 1024

BALLOT_KERNELS: Mapping[str, str] = {
    "elite": "sel_ballot_elite_safe",
    "noswitch": "sel_ballot_noswitch_safe",
    "local": "sel_ballot_local_safe",
}


def grid_for(count: int, block: int = BLOCK) -> tuple[int, int]:
    """Return the grid that covers ``count`` threads with blocks of ``block``."""
    return ((count + block - 1) // block, 1)


@dataclass(frozen=True)
class KernelLibrary:
    """The two compiled modules of the package."""

    algorithms: CompiledModule
    selection: CompiledModule


def load_kernels(
    constants: Mapping[str, str] | None = None,
    *,
    count_evaluations: bool = False,
    user_objective: str | None = None,
    negate: bool = False,
    compiler: Compiler | None = None,
) -> KernelLibrary:
    """Compile the packaged kernels, reusing modules compiled earlier."""
    return KernelLibrary(
        compile_cached(
            source.algorithms_source(
                constants,
                count_evaluations=count_evaluations,
                user_objective=user_objective,
                negate=negate,
            ),
            label="algorithms",
            compiler=compiler,
        ),
        compile_cached(source.selection_source(), label="selection", compiler=compiler),
    )


class SelectionKernels:
    """Selection operators on device arrays.

    All methods write the selected index of every individual to ``d_sel``
    (int32) and expect ``d_obj`` to hold float64 objective values.

    Parameters
    ----------
    kernels : KernelLibrary, optional
        Compiled modules; compiled on demand by default.
    runtime : Runtime, optional
        CUDA runtime; the shared one by default.
    """

    def __init__(self, kernels: KernelLibrary | None = None, runtime: Runtime | None = None):
        self._rt = runtime or get_runtime()
        self._k = kernels or load_kernels()
        self._best_index = self._rt.zeros(1, np.int32)
        self._best_value = self._rt.zeros(1, np.float64)

    def fitness(self, d_obj: Any, d_fit: Any, n: int) -> None:
        """Write the fitness transform of ``d_obj`` to ``d_fit``."""
        self._alg("compute_fit")(d_obj, d_fit, np.int32(n), block=(BLOCK, 1, 1), grid=grid_for(n))

    def mean_fitness(self, d_fit: Any, d_mean: Any, n: int) -> None:
        """Write the mean of ``d_fit`` to the one-element array ``d_mean``."""
        one = {"block": (1, 1, 1), "grid": (1, 1)}
        self._alg("reset_avg")(d_mean, **one)
        self._alg("compute_avg_fitness")(
            d_fit, d_mean, np.int32(n), block=(BLOCK, 1, 1), grid=grid_for(n), shared=BLOCK * 8
        )
        self._alg("finalize_avg_fitness")(d_mean, np.int32(n), **one)

    def argmin(self, d_obj: Any, n: int) -> Any:
        """Return a one-element device array with the index of the minimum."""
        self._sel("argmin_block")(
            d_obj,
            np.int32(n),
            self._best_index,
            self._best_value,
            block=(_ARGMIN_BLOCK, 1, 1),
            grid=(1, 1),
        )
        return self._best_index

    def tournament(self, d_obj: Any, d_sel: Any, n: int, size: int, seed: int) -> None:
        """Warp tournament."""
        self._launch(
            "sel_tournament_safe", n, d_obj, d_sel, np.int32(n), np.int32(size), _u32(seed)
        )

    def ballot(
        self,
        d_obj: Any,
        d_fit: Any,
        d_sel: Any,
        d_mean: Any,
        n: int,
        size: int,
        alpha: float,
        seed: int,
        variant: str = "elite",
    ) -> None:
        """Ballot selection; ``d_fit`` and ``d_mean`` are work buffers."""
        name = BALLOT_KERNELS[variant]
        self.fitness(d_obj, d_fit, n)
        common = (np.int32(n), np.int32(size), np.float64(alpha), _u32(seed))
        if variant == "local":
            self._launch(name, n, d_fit, d_sel, *common)
        else:
            self.mean_fitness(d_fit, d_mean, n)
            self._launch(name, n, d_fit, d_sel, d_mean, *common)

    def migration(self, d_obj: Any, d_sel: Any, n: int, size: int, rate: float, seed: int) -> None:
        """Warp tournament with global-best migration."""
        best = self.argmin(d_obj, n)
        self._launch(
            "sel_tournament_migrate",
            n,
            d_obj,
            d_sel,
            np.int32(n),
            np.int32(size),
            best,
            np.float64(rate),
            _u32(seed),
        )

    def elitist(self, d_obj: Any, d_sel: Any, n: int, size: int, seed: int) -> None:
        """Warp tournament whose winner is always compared with the global best."""
        best = self.argmin(d_obj, n)
        self._launch(
            "sel_tournament_elitist_dev",
            n,
            d_obj,
            d_sel,
            np.int32(n),
            np.int32(size),
            best,
            _u32(seed),
        )

    def global_best(self, d_obj: Any, d_sel: Any, n: int) -> None:
        """Select the global best for every individual."""
        best = self.argmin(d_obj, n)
        self._launch("sel_globalbest", n, d_sel, np.int32(n), best)

    def global_tournament(self, d_obj: Any, d_sel: Any, n: int, size: int, seed: int) -> None:
        """Tournament with competitors drawn from the whole population."""
        self._alg("sel_tournament_global")(
            d_obj,
            d_sel,
            np.int32(n),
            np.int32(size),
            _u32(seed),
            block=(BLOCK, 1, 1),
            grid=grid_for(n),
        )

    def _launch(self, name: str, n: int, *args: Any) -> None:
        self._sel(name)(*args, block=(BLOCK, 1, 1), grid=grid_for(n))

    def _alg(self, name: str) -> Any:
        return _accepting_cupy(self._k.algorithms.kernel(name))

    def _sel(self, name: str) -> Any:
        return _accepting_cupy(self._k.selection.kernel(name))


def _accepting_cupy(kernel: Any) -> Any:
    def launch(*args: Any, **how: Any) -> None:
        kernel(*(kernel_argument(a) for a in args), **how)

    return launch


_shared: SelectionKernels | None = None


def shared_selection_kernels() -> SelectionKernels:
    """Return the process-wide selection kernels on the default modules."""
    global _shared
    if _shared is None:
        _shared = SelectionKernels()
    return _shared


def _u32(seed: int) -> np.uint32:
    return np.uint32(int(seed) & 0xFFFFFFFF)
