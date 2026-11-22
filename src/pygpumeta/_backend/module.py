"""Compiled CUDA modules with explicit kernel handles."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

from pygpumeta._backend import msvc
from pygpumeta._backend.source import kernel_names

Compiler = Callable[[str], Any]

_cache: dict[str, CompiledModule] = {}
_cache_lock = threading.Lock()


class CompilationError(RuntimeError):
    """Raised when the CUDA compiler rejects a source."""


class CompiledModule:
    """A compiled CUDA source and its kernels.

    Parameters
    ----------
    source : str
        Complete CUDA C source.
    label : str
        Short name used in error messages.
    compiler : callable, optional
        Function turning a source into an object with ``get_function`` and
        ``get_global``. Defaults to PyCUDA's ``SourceModule`` on the shared
        runtime.
    """

    def __init__(self, source: str, *, label: str, compiler: Compiler | None = None) -> None:
        self.label = label
        self.source = source
        self.source_hash = source_digest(source)
        self.kernel_names = kernel_names(source)
        self._handles: dict[str, Any] = {}
        start = time.perf_counter()
        self._module = (compiler or _pycuda_compiler)(source)
        self.compile_time = time.perf_counter() - start

    def kernel(self, name: str) -> Any:
        """Return the handle of a kernel.

        Raises
        ------
        KeyError
            If the source defines no kernel of that name.
        """
        handle = self._handles.get(name)
        if handle is None:
            if name not in self.kernel_names:
                raise KeyError(
                    f"module '{self.label}' has no kernel named {name!r}; "
                    f"available: {', '.join(sorted(self.kernel_names))}"
                )
            handle = self._handles[name] = self._module.get_function(name)
        return handle

    def global_pointer(self, name: str) -> Any:
        """Return the device address of a global variable of the module."""
        return self._module.get_global(name)[0]

    def __repr__(self) -> str:
        return f"CompiledModule({self.label!r}, {len(self.kernel_names)} kernels)"


def source_digest(source: str) -> str:
    """Return the cache key of a source."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compile_cached(source: str, *, label: str, compiler: Compiler | None = None) -> CompiledModule:
    """Compile a source once per process and return the shared module."""
    key = source_digest(source)
    with _cache_lock:
        module = _cache.get(key)
        if module is None:
            module = _cache[key] = CompiledModule(source, label=label, compiler=compiler)
        return module


def clear_cache() -> None:
    """Forget all compiled modules."""
    with _cache_lock:
        _cache.clear()


def _pycuda_compiler(source: str) -> Any:
    from pygpumeta._backend.runtime import get_runtime

    get_runtime()
    from pycuda.compiler import SourceModule

    try:
        return SourceModule(source)
    except Exception as exc:
        compiler = msvc.ensure_host_compiler()
        if not compiler.available:
            raise CompilationError(
                f"the kernels could not be compiled. {msvc.describe_missing(compiler)}\n{exc}"
            ) from exc
        # Kernel names are exported with C linkage and looked up right after
        # compilation, so compiler and linker messages both end up here.
        raise CompilationError(
            f"the CUDA compiler rejected the kernel source. If you passed a CudaObjective, "
            f"check its code; helper functions must be declared __device__.\n{exc}"
        ) from exc
