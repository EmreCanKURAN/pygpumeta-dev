"""Validation of the problem definition shared by all optimizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pygpumeta import _validation as _v
from pygpumeta import benchmarks, space
from pygpumeta._backend.source import USER_FUNCTION_ID
from pygpumeta._objective import CudaObjective
from pygpumeta.constraints import Constraint

# Several kernels keep one position in a stack array of this length.
MAX_DIM = 1024
# Seeds are offset per iteration and per kernel inside a signed 32-bit range.
MAX_SEED = 2**31 - 2

MODES = ("auto", "gpu", "host")
EVALUATIONS = ("auto", "fused", "separated")
_VARIABLES = (space.Real, space.Integer, space.Categorical)


def _no_index() -> NDArray[np.int64]:
    return np.zeros(0, dtype=np.int64)


def _no_value() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)


@dataclass(frozen=True)
class Problem:
    """A validated problem.

    ``kind`` is ``"builtin"``, ``"cuda"`` (CUDA source compiled into the
    kernels), ``"gpu"`` (callable on CuPy arrays) or ``"host"`` (callable on
    NumPy arrays). ``arrays`` is the array library user code receives.
    """

    name: str
    kind: str
    dim: int
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    arrays: str = "gpu"
    function_id: int = -1
    cuda_source: str | None = None
    function: Callable[[Any], Any] | None = None
    vectorized: bool = True
    maximize: bool = False
    constraints: tuple[Constraint, ...] = ()
    constraint_handling: str = "penalty"
    penalty: float = 1e6
    variables: tuple[Any, ...] | None = None
    discrete_index: NDArray[np.int64] = field(default_factory=_no_index)
    discrete_low: NDArray[np.float64] = field(default_factory=_no_value)
    discrete_high: NDArray[np.float64] = field(default_factory=_no_value)

    @property
    def in_kernel(self) -> bool:
        """Whether the objective itself is evaluated by a kernel."""
        return self.kind in ("builtin", "cuda")

    @property
    def fusable(self) -> bool:
        """Whether the whole evaluation can happen inside the update kernels."""
        return self.in_kernel and not self.constraints

    def decode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Round the discrete variables of one position."""
        decoded = np.array(x, dtype=np.float64)
        if self.discrete_index.size:
            index = self.discrete_index
            decoded[index] = np.clip(np.rint(decoded[index]), self.discrete_low, self.discrete_high)
        return decoded

    def values(self, decoded: NDArray[np.float64]) -> tuple[Any, ...]:
        """Return a decoded position as Python values, with categorical choices."""
        if self.variables is None:
            return tuple(float(v) for v in decoded)
        out: list[Any] = []
        for value, variable in zip(decoded, self.variables, strict=True):
            if isinstance(variable, space.Categorical):
                out.append(variable.choices[int(value)])
            elif isinstance(variable, space.Integer):
                out.append(int(value))
            else:
                out.append(float(value))
        return tuple(out)


def resolve_problem(
    objective: Any,
    bounds: Any,
    dim: Any,
    mode: Any = "auto",
    vectorized: Any = True,
    constraints: Any = (),
    constraint_handling: Any = "penalty",
    penalty: Any = 1e6,
    maximize: Any = False,
) -> Problem:
    """Validate the arguments that define a problem."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(map(repr, MODES))}, got {mode!r}")
    for name, flag in (("vectorized", vectorized), ("maximize", maximize)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be True or False, got {flag!r}")
    checked_constraints = _resolve_constraints(constraints)
    if constraint_handling not in ("penalty", "feasibility"):
        raise ValueError(
            f"constraint_handling must be 'penalty' or 'feasibility', got {constraint_handling!r}"
        )
    penalty = _v.check_positive(penalty, "penalty")

    default_bounds = None
    fields: dict[str, Any]
    if isinstance(objective, str):
        info = benchmarks.info(objective)
        default_bounds = info.bounds
        fields = {"name": info.name, "kind": "builtin", "function_id": info._fid}
    elif isinstance(objective, CudaObjective):
        fields = {
            "name": "cuda objective",
            "kind": "cuda",
            "function_id": USER_FUNCTION_ID,
            "cuda_source": objective.source,
        }
    elif callable(objective):
        name = getattr(objective, "__name__", type(objective).__name__)
        fields = {"name": name, "kind": "host" if mode == "host" else "gpu", "function": objective}
    else:
        raise TypeError(
            "objective must be the name of a built-in function, a callable or a "
            f"CudaObjective, got {type(objective).__name__}"
        )
    in_kernel = fields["kind"] in ("builtin", "cuda")
    if in_kernel and not checked_constraints and mode != "auto":
        raise ValueError(
            f"mode={mode!r} applies to callable objectives and constraints only; "
            f"a {fields['kind']} objective runs in the kernels"
        )
    arrays = "host" if mode == "host" else "gpu"
    if arrays != "host" and vectorized is not True:
        raise ValueError("vectorized=False is only available with mode='host'")

    if bounds is None:
        if default_bounds is None:
            raise ValueError("bounds is required for objectives other than built-in functions")
        bounds = default_bounds
    lower, upper, variables = _resolve_bounds(bounds, dim)
    discrete = [i for i, v in enumerate(variables or ()) if not isinstance(v, space.Real)]
    if discrete and in_kernel:
        raise ValueError(
            "integer and categorical variables need a callable objective; a "
            f"{fields['kind']} objective is evaluated in the kernels without decoding"
        )
    ranges = [space.integer_range(variables[i]) for i in discrete] if variables else []
    return Problem(
        dim=len(lower),
        lower=lower,
        upper=upper,
        arrays=arrays,
        vectorized=vectorized,
        maximize=maximize,
        constraints=checked_constraints,
        constraint_handling=constraint_handling,
        penalty=penalty,
        variables=variables,
        discrete_index=np.array(discrete, dtype=np.int64),
        discrete_low=np.array([r[0] for r in ranges], dtype=np.float64),
        discrete_high=np.array([r[1] for r in ranges], dtype=np.float64),
        **fields,
    )


def _resolve_constraints(constraints: Any) -> tuple[Constraint, ...]:
    if isinstance(constraints, Constraint):
        return (constraints,)
    if not isinstance(constraints, Sequence) or isinstance(constraints, str):
        raise TypeError(
            "constraints must be a sequence of Constraint objects, got "
            f"{type(constraints).__name__}"
        )
    for item in constraints:
        if not isinstance(item, Constraint):
            raise TypeError(
                "constraints must contain Constraint objects, got "
                f"{type(item).__name__}; wrap functions as pygpumeta.Constraint(fun)"
            )
    return tuple(constraints)


def _resolve_bounds(
    bounds: Any, dim: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[Any, ...] | None]:
    """Return per-dimension bounds and, for typed spaces, the variables."""
    if dim is not None:
        dim = _v.as_int(dim, "dim")
        if not 1 <= dim <= MAX_DIM:
            raise ValueError(f"dim must be between 1 and {MAX_DIM}, got {dim}")
    if isinstance(bounds, _VARIABLES):
        bounds = [bounds]
    if isinstance(bounds, Sequence) and any(isinstance(b, _VARIABLES) for b in bounds):
        variables = tuple(_as_variable(b, i) for i, b in enumerate(bounds))
        pairs = np.array([space.box_of(v) for v in variables], dtype=np.float64)
    else:
        variables = None
        try:
            array = np.asarray(bounds, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                "bounds must be a (lower, upper) pair, an array of shape (dim, 2) or a "
                "sequence of Real, Integer and Categorical variables"
            ) from None
        if array.shape == (2,):
            if dim is None:
                raise ValueError(
                    "dim is required when bounds is a single (lower, upper) pair or omitted"
                )
            pairs = np.tile(array, (dim, 1))
        elif array.ndim == 2 and array.shape[1] == 2 and array.shape[0] >= 1:
            pairs = array
        else:
            raise ValueError(
                "bounds must be a (lower, upper) pair or an array of shape (dim, 2), "
                f"got shape {array.shape}"
            )
        if not np.isfinite(pairs).all():
            raise ValueError("bounds must be finite")
        if not (pairs[:, 0] < pairs[:, 1]).all():
            row = int(np.argmin(pairs[:, 0] < pairs[:, 1]))
            raise ValueError(
                "every lower bound must be smaller than its upper bound, "
                f"got {tuple(float(v) for v in pairs[row])} for variable {row}"
            )
    rows = int(pairs.shape[0])
    if dim is not None and dim != rows:
        raise ValueError(f"dim is {dim} but bounds has {rows} rows")
    if rows > MAX_DIM:
        raise ValueError(f"dim must be between 1 and {MAX_DIM}, got {rows}")
    return np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1]), variables


def _as_variable(entry: Any, position: int) -> Any:
    if isinstance(entry, _VARIABLES):
        return entry
    try:
        lower, upper = entry
        return space.Real(lower, upper)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"bounds[{position}] must be a Real, Integer or Categorical variable or a "
            f"(lower, upper) pair: {exc}"
        ) from None


def resolve_evaluation(evaluation: Any, problem: Problem) -> bool:
    """Return whether the objective is evaluated inside the update kernels."""
    if evaluation not in EVALUATIONS:
        raise ValueError(
            f"evaluation must be one of {', '.join(map(repr, EVALUATIONS))}, got {evaluation!r}"
        )
    if evaluation == "fused" and not problem.fusable:
        raise ValueError(
            "evaluation='fused' needs a built-in function or a CudaObjective without "
            "constraints; Python code is always evaluated as a separate step"
        )
    return problem.fusable if evaluation == "auto" else evaluation == "fused"


def check_count(value: Any, name: str, minimum: int) -> int:
    """Validate an integer that must be at least ``minimum``."""
    checked = _v.as_int(value, name)
    if checked < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {checked}")
    return checked


def check_run_seed(seed: Any) -> int:
    """Validate the seed of an optimisation run."""
    checked = _v.as_int(seed, "seed")
    if not 0 <= checked <= MAX_SEED:
        raise ValueError(f"seed must satisfy 0 <= seed <= {MAX_SEED}, got {checked}")
    return checked
