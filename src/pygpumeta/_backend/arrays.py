"""Passing arrays of either GPU library to the kernels."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_cupy_array(array: Any) -> bool:
    """Return whether ``array`` is a CuPy array, without importing CuPy."""
    return type(array).__module__.split(".")[0] == "cupy" and hasattr(array, "data")


def kernel_argument(value: Any) -> Any:
    """Return ``value`` in the form a PyCUDA kernel launch accepts.

    PyCUDA arrays and NumPy scalars pass through. A CuPy array is replaced
    by its device address; both libraries work in the same CUDA context, so
    the address is valid for the kernels.
    """
    if is_cupy_array(value):
        return np.uintp(value.data.ptr)
    return value
