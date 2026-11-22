"""Variable types for mixed search spaces.

The algorithms search a continuous box. Integer and categorical variables
are handled by decoding: the optimizer keeps a continuous value for each of
them and rounds it to the nearest admissible integer every time the
objective is evaluated. The objective, the constraints and the result only
ever see the decoded values.

An integer variable from ``a`` to ``b`` is searched on ``[a - 0.5, b + 0.5]``,
so every integer owns an interval of the same width. A categorical variable
with ``k`` choices is an integer variable from 0 to ``k - 1``; the objective
receives the index, and the result maps it back to the choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pygpumeta import _validation as _v

__all__ = ["Categorical", "Integer", "Real"]


@dataclass(frozen=True)
class Real:
    """A continuous variable.

    Parameters
    ----------
    lower, upper : float
        Finite bounds with ``lower < upper``.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower, upper = _finite(self.lower, "lower"), _finite(self.upper, "upper")
        if not lower < upper:
            raise ValueError(f"lower must be smaller than upper, got ({lower}, {upper})")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class Integer:
    """An integer variable.

    Parameters
    ----------
    lower, upper : int
        Smallest and largest admissible value, with ``lower <= upper``.
    """

    lower: int
    upper: int

    def __post_init__(self) -> None:
        lower, upper = _v.as_int(self.lower, "lower"), _v.as_int(self.upper, "upper")
        if lower > upper:
            raise ValueError(f"lower must not exceed upper, got ({lower}, {upper})")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class Categorical:
    """A variable taking one of several unordered choices.

    Parameters
    ----------
    choices : sequence
        At least two choices. The objective receives the index of a choice
        as a float; :attr:`pygpumeta.OptimizeResult.values` holds the choice
        itself.

    Notes
    -----
    The algorithms move through index space, where neighbouring indices are
    close to each other although the choices are unordered. This works for
    a handful of choices; it is not a substitute for a combinatorial method.
    """

    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        if isinstance(self.choices, str | bytes):
            raise TypeError("choices must be a sequence of choices, not a single string")
        try:
            choices = tuple(self.choices)
        except TypeError:
            raise TypeError(
                f"choices must be a sequence, got {type(self.choices).__name__}"
            ) from None
        if len(choices) < 2:
            raise ValueError(f"choices must contain at least two entries, got {len(choices)}")
        object.__setattr__(self, "choices", choices)


Variable = Real | Integer | Categorical


def box_of(variable: Variable) -> tuple[float, float]:
    """Return the continuous interval searched for a variable."""
    if isinstance(variable, Real):
        return variable.lower, variable.upper
    low, high = integer_range(variable)
    return low - 0.5, high + 0.5


def integer_range(variable: Integer | Categorical) -> tuple[int, int]:
    """Return the smallest and largest decoded value of a discrete variable."""
    if isinstance(variable, Integer):
        return variable.lower, variable.upper
    return 0, len(variable.choices) - 1


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)
