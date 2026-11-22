"""Built-in test functions.

The functions are evaluated on the device by the kernels. This module lists
them with their customary search domains and known optima, and provides
NumPy implementations of the same formulas for checking results on the CPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BenchmarkInfo", "evaluate", "info", "names"]

_Array = NDArray[np.float64]

# Largest value of x * sin(sqrt(x)) on the Schwefel domain; the function is
# written with the customary rounded constant 418.9829, which leaves a small
# positive value at the minimiser.
_SCHWEFEL_PEAK = 418.98288727243363
_SCHWEFEL_ARGMIN = 420.9687456269364


@dataclass(frozen=True)
class BenchmarkInfo:
    """Description of a built-in test function.

    Attributes
    ----------
    name : str
        Name accepted by the optimizers.
    bounds : tuple of float
        Customary lower and upper bound, the same for every dimension.
    minimizer : float or None
        Coordinate value of the global minimiser, the same in every
        dimension, or None if it is not known in closed form.
    """

    name: str
    bounds: tuple[float, float]
    minimizer: float | None
    _fid: int
    _optimum: Callable[[int], float] | None

    def optimum(self, dim: int) -> float | None:
        """Return the global minimum for ``dim`` dimensions, or None if unknown."""
        return None if self._optimum is None else self._optimum(dim)


def _zero(dim: int) -> float:
    return 0.0


_INFO = {
    b.name: b
    for b in (
        BenchmarkInfo("sphere", (-100.0, 100.0), 0.0, 0, _zero),
        BenchmarkInfo("rastrigin", (-5.12, 5.12), 0.0, 1, _zero),
        BenchmarkInfo("rosenbrock", (-30.0, 30.0), 1.0, 2, _zero),
        BenchmarkInfo("ackley", (-32.768, 32.768), 0.0, 3, _zero),
        BenchmarkInfo("griewank", (-600.0, 600.0), 0.0, 4, _zero),
        BenchmarkInfo(
            "schwefel",
            (-500.0, 500.0),
            _SCHWEFEL_ARGMIN,
            5,
            lambda d: d * (418.9829 - _SCHWEFEL_PEAK),
        ),
        BenchmarkInfo("levy", (-10.0, 10.0), 1.0, 6, _zero),
        BenchmarkInfo("zakharov", (-5.0, 10.0), 0.0, 7, _zero),
        BenchmarkInfo("sum_squares", (-10.0, 10.0), 0.0, 8, _zero),
        BenchmarkInfo("michalewicz", (0.0, math.pi), None, 9, None),
    )
}


def names() -> tuple[str, ...]:
    """Return the names of the built-in test functions."""
    return tuple(_INFO)


def info(name: str) -> BenchmarkInfo:
    """Return the description of a built-in test function.

    Raises
    ------
    ValueError
        If no function of that name exists.
    """
    try:
        return _INFO[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown test function {name!r}; available: {', '.join(_INFO)}") from None


def evaluate(name: str, x: ArrayLike) -> _Array:
    """Evaluate a built-in test function with NumPy.

    Parameters
    ----------
    name : str
        One of :func:`names`.
    x : array_like, shape (dim,) or (n, dim)
        One point or one point per row.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Objective values; a single point gives an array of length one.

    Notes
    -----
    The formulas are those of the kernels. ``michalewicz`` is shifted by the
    dimension, ``-sum(...) + dim``. ``griewank`` replaces the cosine product
    by zero wherever one of its factors is not positive, which differs from
    the textbook function away from the optimum, mostly in low dimensions.
    """
    points = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError(f"x must have shape (dim,) or (n, dim), got {np.shape(x)}")
    return np.asarray(_FUNCTIONS[info(name).name](points), dtype=np.float64)


def _index(x: _Array) -> _Array:
    return np.arange(1, x.shape[1] + 1, dtype=np.float64)


def _sphere(x: _Array) -> Any:
    return np.sum(x * x, axis=1)


def _rastrigin(x: _Array) -> Any:
    # Equal to the usual 10 d + sum(x^2 - 10 cos(2 pi x)), without the
    # cancellation near the optimum.
    return np.sum(x * x, axis=1) + 20.0 * np.sum(np.sin(np.pi * x) ** 2, axis=1)


def _rosenbrock(x: _Array) -> Any:
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def _ackley(x: _Array) -> Any:
    d = x.shape[1]
    root = np.sqrt(np.sum(x * x, axis=1) / d)
    return (
        -20.0 * np.exp(-0.2 * root)
        - np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / d)
        + 20.0
        + np.e
    )


def _griewank(x: _Array) -> Any:
    theta = x / np.sqrt(_index(x))
    cosine = np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(cosine > 0.9999999999, -theta * theta / 2.0, np.log(cosine))
    log_product = np.where((cosine <= 0.0).any(axis=1), -np.inf, np.sum(terms, axis=1))
    one_minus = np.where(log_product > -1e-10, -log_product, 1.0 - np.exp(log_product))
    return np.sum(x * x, axis=1) / 4000.0 + one_minus


def _schwefel(x: _Array) -> Any:
    return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def _levy(x: _Array) -> Any:
    w = 1.0 + (x - 1.0) / 4.0
    head = np.sin(np.pi * w[:, 0]) ** 2
    tail = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    body = np.sum(
        (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=1
    )
    return head + body + tail


def _zakharov(x: _Array) -> Any:
    s = np.sum(0.5 * _index(x) * x, axis=1)
    return np.sum(x * x, axis=1) + s**2 + s**4


def _sum_squares(x: _Array) -> Any:
    return np.sum(_index(x) * x * x, axis=1)


def _michalewicz(x: _Array) -> Any:
    return -np.sum(np.sin(x) * np.sin(_index(x) * x * x / np.pi) ** 20, axis=1) + x.shape[1]


_FUNCTIONS: dict[str, Callable[[_Array], Any]] = {
    "sphere": _sphere,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "ackley": _ackley,
    "griewank": _griewank,
    "schwefel": _schwefel,
    "levy": _levy,
    "zakharov": _zakharov,
    "sum_squares": _sum_squares,
    "michalewicz": _michalewicz,
}
