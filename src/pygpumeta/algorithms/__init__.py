"""Optimizer classes for step-wise control."""

from pygpumeta.algorithms._base import Optimizer
from pygpumeta.algorithms.abc import ABC
from pygpumeta.algorithms.de import DE
from pygpumeta.algorithms.gwo import GWO
from pygpumeta.algorithms.hho import HHO
from pygpumeta.algorithms.pso import PSO
from pygpumeta.algorithms.sca import SCA
from pygpumeta.algorithms.ssa import SSA
from pygpumeta.algorithms.woa import WOA

__all__ = ["ABC", "DE", "GWO", "HHO", "PSO", "SCA", "SSA", "WOA", "Optimizer"]

BY_NAME: dict[str, type[Optimizer]] = {
    cls.name: cls for cls in (ABC, GWO, WOA, SCA, SSA, HHO, PSO, DE)
}
