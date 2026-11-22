"""Assembly of the CUDA sources shipped with the package."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from functools import cache
from importlib import resources

ALGORITHMS_FILE = "metaheuristics.cu"
SEPARATED_FILE = "separated.cu"
SELECTION_FILE = "selection_safe.cu"

# Function id under which a user-supplied CUDA objective is dispatched.
USER_FUNCTION_ID = 1000
# Objective values are kept inside this magnitude: the leader search of GWO
# treats 1e300 as "no value".
LARGEST_VALUE = 1e299

# repr() of a float is the shortest literal that parses back to the same double.
CONSTANTS: Mapping[str, str] = {
    "PGM_PI": repr(math.pi),
    "PGM_TWO_PI": repr(2.0 * math.pi),
    "PGM_E": repr(math.e),
}

EVALUATION_COUNTER = "d_fe_count"
_EVALUATE_SIGNATURE = "__device__ double evaluate(double* x, int d, int fid) {"


def _wrapped_evaluate(count: bool, negate: bool) -> str:
    # The dispatcher becomes evaluate_raw; the wrapper keeps its name, so
    # every kernel that evaluates goes through it.
    counter = f"atomicAdd(&{EVALUATION_COUNTER}, 1ULL); " if count else ""
    declaration = f"__device__ unsigned long long {EVALUATION_COUNTER} = 0ULL;\n" if count else ""
    return (
        f"{declaration}"
        "__device__ double evaluate_raw(double* x, int d, int fid);\n"
        f"{_EVALUATE_SIGNATURE} {counter}"
        f"return {'-' if negate else ''}evaluate_raw(x, d, fid); }}\n"
        "__device__ double evaluate_raw(double* x, int d, int fid) {"
    )


_USER_SIGNATURE_RE = re.compile(
    r"\bdouble\s+evaluate\s*\((?=\s*const\s+double\s*\*\s*\w+\s*,\s*int\s+\w+\s*\))"
)
# NaN fails both comparisons of the first test and is mapped like +infinity.
_USER_DISPATCH = (
    "__device__ double pgm_user_bounded(double* x, int d) {\n"
    "    double v = pgm_user_objective(x, d);\n"
    f"    if (!(v < {LARGEST_VALUE!r})) return {LARGEST_VALUE!r};\n"
    f"    if (v < -{LARGEST_VALUE!r}) return -{LARGEST_VALUE!r};\n"
    "    return v;\n"
    "}\n"
)

_KERNEL_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(')
_LITERAL_RE = re.compile(r"[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?")


@cache
def read_kernel_file(name: str) -> str:
    """Return the text of a ``.cu`` file of the package."""
    return resources.files("pygpumeta").joinpath("kernels", name).read_text(encoding="utf-8")


def constants_prelude(overrides: Mapping[str, str] | None = None) -> str:
    """Return the ``#define`` lines of the mathematical constants.

    Parameters
    ----------
    overrides : mapping, optional
        Replacement literals by macro name. Only meant for tests that need
        to reproduce sources compiled with other constants.
    """
    values = dict(CONSTANTS)
    for name, literal in (overrides or {}).items():
        if name not in values:
            raise KeyError(f"unknown constant {name!r}; known constants: {sorted(values)}")
        if not _LITERAL_RE.fullmatch(literal):
            raise ValueError(f"{name} must be a decimal floating-point literal, got {literal!r}")
        values[name] = literal
    return "".join(f"#define {name} {literal}\n" for name, literal in values.items())


def algorithms_source(
    constants: Mapping[str, str] | None = None,
    *,
    count_evaluations: bool = False,
    user_objective: str | None = None,
    negate: bool = False,
) -> str:
    """Return the source of the algorithm kernels.

    Parameters
    ----------
    constants : mapping, optional
        See :func:`constants_prelude`.
    count_evaluations : bool, default False
        Wrap the objective so that every call increments the device global
        ``d_fe_count``. Needed by algorithms whose number of evaluations
        depends on the course of the run.
    user_objective : str, optional
        CUDA C source defining ``double evaluate(const double* x, int d)``;
        it becomes the function with id ``USER_FUNCTION_ID``.
    negate : bool, default False
        Return the negated objective, which turns the minimising kernels
        into maximisers.
    """
    text = read_kernel_file(ALGORITHMS_FILE)
    if text.count(_EVALUATE_SIGNATURE) != 1:
        raise RuntimeError("the objective dispatcher was not found exactly once")
    if user_objective is not None:
        text = text.replace(
            _EVALUATE_SIGNATURE,
            prepare_user_objective(user_objective)
            + _USER_DISPATCH
            + _EVALUATE_SIGNATURE
            + f" if (fid == {USER_FUNCTION_ID}) return pgm_user_bounded(x, d);",
            1,
        )
    if count_evaluations or negate:
        text = text.replace(_EVALUATE_SIGNATURE, _wrapped_evaluate(count_evaluations, negate), 1)
    return constants_prelude(constants) + text + read_kernel_file(SEPARATED_FILE)


def prepare_user_objective(snippet: str) -> str:
    """Validate a user objective and rename it for inclusion in the kernels.

    Raises
    ------
    ValueError
        If the snippet does not define the expected function exactly once.
    """
    if not isinstance(snippet, str):
        raise TypeError(f"the CUDA source must be a string, got {type(snippet).__name__}")
    matches = _USER_SIGNATURE_RE.findall(snippet)
    if len(matches) != 1:
        raise ValueError(
            "the CUDA source must define exactly one function "
            "'double evaluate(const double* x, int d)', "
            f"found {len(matches)}"
        )
    if "pgm_user_" in snippet or _EVALUATE_SIGNATURE in snippet:
        raise ValueError("the CUDA source must not use names starting with 'pgm_user_'")
    return _USER_SIGNATURE_RE.sub("__device__ double pgm_user_objective(", snippet, count=1) + "\n"


def selection_source() -> str:
    """Return the source of the boundary-safe selection kernels."""
    return read_kernel_file(SELECTION_FILE)


def kernel_names(source: str) -> frozenset[str]:
    """Return the names of the kernels a source defines."""
    return frozenset(_KERNEL_RE.findall(source))
