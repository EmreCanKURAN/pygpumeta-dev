"""Importable stand-in for CuPy on machines without CUDA; see ../README.md.

Like the real library given something that is not a device array, it
rejects whatever it is asked to convert.
"""

import numpy as _np

__version__ = "0 (test stand-in)"
float64 = _np.float64
int32 = _np.int32


class ndarray:
    pass


def asarray(obj, dtype=None):
    raise ValueError("Unsupported dtype object")


def arange(*args, **kwargs):
    raise RuntimeError("stand-in CuPy: no CUDA device")
