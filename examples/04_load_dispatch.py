"""Economic load dispatch of 13 generating units with valve-point effects.

The output of each unit is chosen so that the total fuel cost

    sum_i  a_i P_i^2 + b_i P_i + c_i + |e_i sin(f_i (Pmin_i - P_i))|

is minimal while the units together deliver the demand of 1800 MW. The
rectified sine models the opening of steam admission valves and makes the
cost non-smooth with many local minima. Data: Sinha, Chakrabarti and
Chattopadhyay, IEEE Trans. Evolutionary Computation 7 (2003).

The power balance is an equality. Searching all 13 outputs and penalising
the mismatch leaves the optimizer on a thin feasible surface that it keeps
falling off. The formulation used here removes the equality instead: twelve
outputs are searched and the first unit supplies the remainder, which turns
the balance into two inequalities, the limits of that unit. They are
handled with the feasibility rule, so the reported dispatch meets the
demand exactly.

Reference: the lowest cost reported for this system is about 17963.83 $/h;
the published dispatch reproduces that value with this cost function. The
problem is hard: the many local minima of the valve-point terms keep most
general-purpose methods some $/h above it.
"""

from __future__ import annotations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta
from pygpumeta import Constraint

DEMAND = 1800.0
REFERENCE_COST = 17963.83
A = np.array([0.00028, 0.00056, 0.00056] + [0.00324] * 6 + [0.00284] * 4)
B = np.array([8.10, 8.10, 8.10] + [7.74] * 6 + [8.60] * 4)
C = np.array([550.0, 309.0, 307.0] + [240.0] * 6 + [126.0] * 4)
E = np.array([300.0, 200.0, 150.0] + [150.0] * 6 + [100.0] * 4)
F = np.array([0.035, 0.042, 0.042] + [0.063] * 6 + [0.084] * 4)
P_MIN = np.array([0.0, 0.0, 0.0] + [60.0] * 6 + [40.0, 40.0, 55.0, 55.0])
P_MAX = np.array([680.0, 360.0, 360.0] + [180.0] * 6 + [120.0] * 4)


def fuel_cost(power):
    """Total cost of a complete dispatch; rows of 13 outputs."""
    xp = array_module(power)
    a, b, c, e, f, p_min = (xp.asarray(v) for v in (A, B, C, E, F, P_MIN))
    smooth = a * power**2 + b * power + c
    return (smooth + xp.abs(e * xp.sin(f * (p_min - power)))).sum(axis=1)


def dispatch(others):
    """Complete the outputs of units 2 to 13 with what unit 1 has to supply."""
    xp = array_module(others)
    first = DEMAND - others.sum(axis=1, keepdims=True)
    return xp.concatenate([first, others], axis=1)


def cost(others):
    xp = array_module(others)
    power = dispatch(others)
    # Outside its limits the first unit is infeasible anyway; clipping only
    # keeps the cost formula inside its range of validity.
    power[:, 0] = xp.clip(power[:, 0], P_MIN[0], P_MAX[0])
    return fuel_cost(power)


def first_unit_limits(others):
    xp = array_module(others)
    first = dispatch(others)[:, 0]
    return xp.stack([P_MIN[0] - first, first - P_MAX[0]], axis=1)


def run(**options):
    options = {
        "algorithm": "de",
        "selection": "native",
        "pop_size": 1024,
        "max_iter": 3000,
        **options,
    }
    return pygpumeta.minimize(
        cost,
        np.column_stack([P_MIN[1:], P_MAX[1:]]),
        constraints=[Constraint(first_unit_limits)],
        constraint_handling="feasibility",
        **options,
    )


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=1024, max_iter=3000))
    power = dispatch(result.x[None, :])[0]
    report(
        "Economic load dispatch: 13 units, 1800 MW, valve-point effects",
        result,
        {
            "fuel cost": f"{result.fun:.2f} $/h",
            "best value in the literature": f"{REFERENCE_COST:.2f} $/h",
            "power delivered": f"{power.sum():.4f} MW (demand {DEMAND:.0f} MW)",
            "feasible": str(result.feasible),
            "unit outputs [MW]": np.array2string(power, precision=2, separator=", "),
        },
    )


if __name__ == "__main__":
    main()
