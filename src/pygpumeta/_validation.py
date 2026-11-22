"""Argument checks shared by the public functions."""

from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

# The kernels draw competitor t from bits 5t..5t+4 of a 32-bit state.
MAX_TOURNAMENT_SIZE = 6


def as_int(value: Any, name: str) -> int:
    """Return ``value`` as a Python int, rejecting bools and non-integers."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got the boolean {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def check_tournament_size(size: Any) -> int:
    """Validate the number of competitors drawn per individual."""
    checked = as_int(size, "size")
    if not 1 <= checked <= MAX_TOURNAMENT_SIZE:
        raise ValueError(f"size must be between 1 and {MAX_TOURNAMENT_SIZE}, got {checked}")
    return checked


def check_seed(seed: Any) -> int:
    """Validate a seed of the 32-bit kernel generators."""
    checked = as_int(seed, "seed")
    if not 0 <= checked < 2**32:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {checked}")
    return checked


def check_unit_interval(value: Any, name: str) -> float:
    """Validate a probability."""
    checked = _as_float(value, name)
    if not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {checked!r}")
    return checked


def check_positive(value: Any, name: str) -> float:
    """Validate a finite, strictly positive number."""
    checked = _as_float(value, name)
    if not (math.isfinite(checked) and checked > 0.0):
        raise ValueError(f"{name} must be a finite number greater than 0, got {checked!r}")
    return checked


def check_non_negative(value: Any, name: str) -> float:
    """Validate a finite number that is not negative."""
    checked = _as_float(value, name)
    if not (math.isfinite(checked) and checked >= 0.0):
        raise ValueError(f"{name} must be a finite number of at least 0, got {checked!r}")
    return checked


def check_values(values: Any, name: str = "values") -> NDArray[np.float64]:
    """Validate a vector of objective values and return it as float64."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be convertible to a float64 array") from None
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got an array of shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one element")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN")
    return array


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.number):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)
