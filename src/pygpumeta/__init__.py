"""GPU metaheuristics with warp-level selection operators.

Importing the package has no side effects: the CUDA context is created and
the kernels are compiled the first time they are needed.
"""

from pygpumeta._minimize import minimize
from pygpumeta._objective import CudaObjective
from pygpumeta._result import OptimizeResult, TerminationReason
from pygpumeta._version import __version__
from pygpumeta.algorithms import ABC, DE, GWO, HHO, PSO, SCA, SSA, WOA, Optimizer
from pygpumeta.constraints import Constraint

__all__ = [
    "ABC",
    "DE",
    "GWO",
    "HHO",
    "PSO",
    "SCA",
    "SSA",
    "WOA",
    "Constraint",
    "CudaObjective",
    "OptimizeResult",
    "Optimizer",
    "TerminationReason",
    "__version__",
    "minimize",
]
