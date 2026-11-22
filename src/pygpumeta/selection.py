"""Warp-level selection operators on device arrays.

Each operator takes the objective values of a population (smaller is better)
and returns, for every individual, the index of a selected individual.
Competition takes place inside warps: an individual only meets individuals
whose index shares ``index // 32`` with its own, except for the global best
in :func:`migration`. The values are exchanged in registers, without
shared-memory or global-memory traffic.

The functions work for any population size and are bit-identical to their
counterparts in :mod:`pygpumeta.reference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pygpumeta import _validation as _v
from pygpumeta._backend.arrays import is_cupy_array

__all__ = [
    "BallotElite",
    "Migration",
    "Tournament",
    "ballot_elite",
    "migration",
    "tournament",
]


@dataclass(frozen=True)
class Tournament:
    """Warp tournament.

    Parameters
    ----------
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    """

    size: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _v.check_tournament_size(self.size))


@dataclass(frozen=True)
class BallotElite:
    """Ballot-Elite selection.

    Parameters
    ----------
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    alpha : float, default 1.2
        Elite threshold as a multiple of the mean fitness.
    """

    size: int = 3
    alpha: float = 1.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _v.check_tournament_size(self.size))
        object.__setattr__(self, "alpha", _v.check_positive(self.alpha, "alpha"))


@dataclass(frozen=True)
class Migration:
    """Warp tournament with global-best migration.

    Parameters
    ----------
    rate : float, default 0.25
        Probability that an individual also compares its winner with the
        best individual of the population.
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    """

    rate: float = 0.25
    size: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate", _v.check_unit_interval(self.rate, "rate"))
        object.__setattr__(self, "size", _v.check_tournament_size(self.size))


Selection = Tournament | BallotElite | Migration

_BY_NAME: dict[str, Selection | None] = {
    "native": None,
    "tournament": Tournament(),
    "ballot_elite": BallotElite(),
    "migration": Migration(),
}


def resolve(selection: Any, default: str = "tournament") -> Selection | None:
    """Turn a selection argument into an operator; None is the native rule.

    ``"auto"`` stands for ``default``, the choice of the algorithm.
    """
    if isinstance(selection, Tournament | BallotElite | Migration):
        return selection
    if selection == "auto":
        return _BY_NAME[default]
    if isinstance(selection, str) and selection in _BY_NAME:
        return _BY_NAME[selection]
    names = ", ".join(repr(k) for k in ("auto", *_BY_NAME))
    raise ValueError(
        f"selection must be one of {names} or a Tournament, "
        f"BallotElite or Migration instance, got {selection!r}"
    )


def describe(selection: Selection | None) -> str:
    """Return a short description such as ``'tournament(size=3)'``."""
    if selection is None:
        return "native"
    if isinstance(selection, Tournament):
        return f"tournament(size={selection.size})"
    if isinstance(selection, BallotElite):
        return f"ballot_elite(size={selection.size}, alpha={selection.alpha:g})"
    return f"migration(rate={selection.rate:g}, size={selection.size})"


def tournament(values: Any, *, size: int = 3, seed: int, out: Any = None) -> Any:
    """Warp tournament selection.

    Parameters
    ----------
    values : pycuda.gpuarray.GPUArray or cupy.ndarray of float64, shape (n,)
        Objective values (smaller is better).
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.
    out : device array of int32, shape (n,), optional
        Array receiving the result; allocated if omitted.

    Returns
    -------
    device array of int32, shape (n,)
        Index of the selected individual for every individual, in an array
        of the same library as ``values``.
    """
    return _run(Tournament(size), values, seed, out)


def ballot_elite(
    values: Any, *, size: int = 3, alpha: float = 1.2, seed: int, out: Any = None
) -> Any:
    """Ballot-Elite selection.

    Individuals whose fitness exceeds ``alpha`` times the mean fitness are
    elite. A warp with at least two elites runs a tournament restricted to
    elite competitors, a warp with one elite selects it for everyone, and a
    warp without elites falls back to the plain tournament.

    Parameters
    ----------
    values : pycuda.gpuarray.GPUArray or cupy.ndarray of float64, shape (n,)
        Objective values (smaller is better).
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    alpha : float, default 1.2
        Elite threshold as a multiple of the mean fitness.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.
    out : device array of int32, shape (n,), optional
        Array receiving the result; allocated if omitted.

    Returns
    -------
    device array of int32, shape (n,)
        Index of the selected individual for every individual, in an array
        of the same library as ``values``.
    """
    return _run(BallotElite(size, alpha), values, seed, out)


def migration(values: Any, *, rate: float = 0.25, size: int = 3, seed: int, out: Any = None) -> Any:
    """Warp tournament with global-best migration.

    Parameters
    ----------
    values : pycuda.gpuarray.GPUArray or cupy.ndarray of float64, shape (n,)
        Objective values (smaller is better).
    rate : float, default 0.25
        Probability that an individual also compares its winner with the
        best individual of the population.
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.
    out : device array of int32, shape (n,), optional
        Array receiving the result; allocated if omitted.

    Returns
    -------
    device array of int32, shape (n,)
        Index of the selected individual for every individual, in an array
        of the same library as ``values``.
    """
    return _run(Migration(rate, size), values, seed, out)


def apply(
    selection: Selection,
    kernels: Any,
    d_obj: Any,
    d_fit: Any,
    d_sel: Any,
    d_mean: Any,
    n: int,
    seed: int,
) -> None:
    """Launch ``selection`` on prepared device buffers."""
    if isinstance(selection, Tournament):
        kernels.tournament(d_obj, d_sel, n, selection.size, seed)
    elif isinstance(selection, BallotElite):
        kernels.ballot(d_obj, d_fit, d_sel, d_mean, n, selection.size, selection.alpha, seed)
    else:
        kernels.migration(d_obj, d_sel, n, selection.size, selection.rate, seed)


def check_device_vector(array: Any, name: str, dtype: type, length: int | None = None) -> int:
    """Validate a one-dimensional device array and return its length."""
    if not (hasattr(array, "gpudata") or is_cupy_array(array)):
        raise TypeError(
            f"{name} must be a pycuda.gpuarray.GPUArray or a cupy.ndarray, got "
            f"{type(array).__name__}; copy host data to the device first, or use "
            "pygpumeta.reference on the CPU"
        )
    if array.dtype != np.dtype(dtype) or len(array.shape) != 1 or array.shape[0] == 0:
        raise ValueError(
            f"{name} must be a non-empty one-dimensional array of {np.dtype(dtype).name}, "
            f"got dtype {array.dtype} and shape {array.shape}"
        )
    if is_cupy_array(array) and not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous; pass cupy.ascontiguousarray({name})")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {array.shape[0]}")
    return int(array.shape[0])


def _run(selection: Selection, values: Any, seed: int, out: Any) -> Any:
    seed = _v.check_seed(seed)
    n = check_device_vector(values, "values", np.float64)
    if out is not None:
        check_device_vector(out, "out", np.int32, n)
    from pygpumeta._backend.runtime import get_runtime
    from pygpumeta._backend.selection import shared_selection_kernels

    rt = get_runtime()
    gpuarray = rt.gpuarray
    kernels = shared_selection_kernels()
    if out is None:
        if is_cupy_array(values):
            import cupy

            out = cupy.empty(n, np.int32)
        else:
            out = gpuarray.empty(n, np.int32)
    d_fit = d_mean = None
    if isinstance(selection, BallotElite):
        d_fit = gpuarray.empty(n, np.float64)
        d_mean = rt.zeros(1, np.float64)
    apply(selection, kernels, values, d_fit, out, d_mean, n, seed)
    return out
