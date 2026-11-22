"""Lazy creation of the CUDA context."""

from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from pygpumeta._backend import msvc

# HHO and the Levy test function keep arrays of MAX_DIM doubles on the stack.
STACK_SIZE = 65536

_lock = threading.Lock()
_runtime: Runtime | None = None


class CudaUnavailableError(RuntimeError):
    """Raised when PyCUDA or a usable CUDA device is missing."""


@dataclass(frozen=True)
class Runtime:
    """Handles of the initialised CUDA backend.

    Attributes
    ----------
    driver : module
        ``pycuda.driver``.
    gpuarray : module
        ``pycuda.gpuarray``.
    context : pycuda.driver.Context
        Primary context of the device, current on the creating thread.
    device_name : str
        Name reported by the device.
    compute_capability : tuple of int
        Major and minor compute capability.
    """

    driver: Any
    gpuarray: Any
    context: Any
    device_name: str
    compute_capability: tuple[int, int]

    def zeros(self, count: int, dtype: Any) -> Any:
        """Return a zero-filled device array.

        PyCUDA's own ``zeros`` compiles a fill kernel on first use, outside
        the error handling of this package; copying host zeros needs no
        compilation.
        """
        return self.gpuarray.to_gpu(np.zeros(count, dtype))

    def synchronize(self) -> None:
        """Block until all work queued on the context has finished."""
        self.context.synchronize()


def get_runtime() -> Runtime:
    """Return the CUDA runtime, initialising it on first use.

    Raises
    ------
    CudaUnavailableError
        If PyCUDA cannot be imported or no CUDA device can be initialised.
    """
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _initialise()
        return _runtime


def _initialise() -> Runtime:
    try:
        import pycuda.driver as driver
        import pycuda.gpuarray as gpuarray
    except Exception as exc:  # ImportError, or OSError when libcuda is missing
        raise CudaUnavailableError(
            "PyCUDA could not be imported "
            f"({type(exc).__name__}: {exc}). Install PyCUDA and the CUDA toolkit; "
            "'python -m pygpumeta.doctor' lists what is missing."
        ) from exc
    # Before anything is compiled: nvcc needs the host compiler on the PATH.
    msvc.ensure_host_compiler()
    try:
        driver.init()
        device = driver.Device(0)
        # The primary context is the one CuPy uses, so both libraries can
        # exchange device pointers.
        context = device.retain_primary_context()
        context.push()
    except Exception as exc:
        raise CudaUnavailableError(
            f"no usable CUDA device ({type(exc).__name__}: {exc}). "
            "Check the NVIDIA driver with 'nvidia-smi'."
        ) from exc
    atexit.register(context.pop)
    driver.Context.get_current().set_limit(driver.limit.STACK_SIZE, STACK_SIZE)
    major, minor = device.compute_capability()
    return Runtime(driver, gpuarray, context, str(device.name()), (int(major), int(minor)))
