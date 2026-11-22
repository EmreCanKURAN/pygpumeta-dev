"""Sizing of the 10-bar planar truss under stress and displacement limits.

The cross-sectional areas of ten bars are chosen to minimise the weight of
the cantilever truss of Venkayya (Computers and Structures 1, 1971): two
bays of 360 in, loads of 100 kips downwards at the two lower free nodes,
E = 1e7 psi, density 0.1 lb/in^3. Every bar stress must stay within
25 ksi and every nodal displacement within 2 in; areas range from 0.1 to
35 in^2.

Each candidate needs a finite-element solve. The stiffness matrix is linear
in the areas, so the matrices of the whole population are assembled with one
tensor product and solved as a batch on the device.

Reference: the lowest weight reported for this problem with continuous
areas is about 5060.85 lb, a value the structural optimisation literature
has converged on. The published areas reproduce it in this model, with the
stress and displacement limits active.
"""

from __future__ import annotations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta
from pygpumeta import Constraint

LENGTH, MODULUS, DENSITY, LOAD = 360.0, 1.0e7, 0.1, 1.0e5
STRESS_LIMIT, DISPLACEMENT_LIMIT = 25000.0, 2.0
REFERENCE_WEIGHT = 5060.85
NODES = np.array(
    [[2 * LENGTH, LENGTH], [2 * LENGTH, 0], [LENGTH, LENGTH], [LENGTH, 0], [0, LENGTH], [0, 0]]
)
BARS = [(5, 3), (3, 1), (6, 4), (4, 2), (3, 4), (1, 2), (5, 4), (6, 3), (3, 2), (4, 1)]


def _model():
    """Return unit stiffness matrices, stress recovery rows, bar lengths and the load."""
    stiffness = np.zeros((10, 12, 12))
    recovery = np.zeros((10, 12))
    lengths = np.zeros(10)
    for b, (i, j) in enumerate(BARS):
        delta = NODES[j - 1] - NODES[i - 1]
        lengths[b] = np.hypot(*delta)
        axis = np.concatenate([-delta, delta]) / lengths[b]
        dofs = [2 * i - 2, 2 * i - 1, 2 * j - 2, 2 * j - 1]
        stiffness[b][np.ix_(dofs, dofs)] = MODULUS / lengths[b] * np.outer(axis, axis)
        recovery[b, dofs] = MODULUS / lengths[b] * axis
    free = slice(0, 8)  # nodes 5 and 6 are fixed
    load = np.zeros(8)
    load[[3, 7]] = -LOAD  # downwards at nodes 2 and 4
    return stiffness[:, free, free], recovery[:, free], lengths, load


UNIT_STIFFNESS, RECOVERY, LENGTHS, FORCES = _model()


def weight(areas):
    xp = array_module(areas)
    return DENSITY * (areas @ xp.asarray(LENGTHS))


def limits(areas):
    """Stress and displacement utilisation minus one; feasible when <= 0."""
    xp = array_module(areas)
    stiffness = xp.einsum("nb,bij->nij", areas, xp.asarray(UNIT_STIFFNESS))
    forces = xp.broadcast_to(xp.asarray(FORCES)[:, None], (areas.shape[0], 8, 1))
    displacement = xp.linalg.solve(stiffness, forces)[:, :, 0]
    stress = displacement @ xp.asarray(RECOVERY).T
    return xp.concatenate(
        [xp.abs(stress) / STRESS_LIMIT - 1.0, xp.abs(displacement) / DISPLACEMENT_LIMIT - 1.0],
        axis=1,
    )


def run(**options):
    options = {
        "algorithm": "de",
        "selection": "native",
        "pop_size": 1024,
        "max_iter": 3000,
        **options,
    }
    return pygpumeta.minimize(
        weight,
        (0.1, 35.0),
        dim=10,
        constraints=[Constraint(limits)],
        constraint_handling="feasibility",
        **options,
    )


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=1024, max_iter=3000))
    utilisation = limits(result.x[None, :])[0] + 1.0
    report(
        "10-bar truss: minimum weight",
        result,
        {
            "weight": f"{result.fun:.2f} lb",
            "best value in the literature": f"{REFERENCE_WEIGHT:.2f} lb",
            "largest stress utilisation": f"{utilisation[:10].max():.4f}",
            "largest displacement utilisation": f"{utilisation[10:].max():.4f}",
            "feasible": str(result.feasible),
            "areas [in^2]": np.array2string(result.x, precision=3, separator=", "),
        },
    )


if __name__ == "__main__":
    main()
