"""Linear antenna array synthesis: low sidelobes with prescribed nulls.

A broadside array of 20 isotropic elements, half a wavelength apart and
excited symmetrically, has the array factor

    AF(u) = sum_n a_n cos((2n - 1) pi u / 2),   n = 1..10,   u = sin(theta).

The ten amplitudes a_n are optimised so that the highest sidelobe outside
the main beam is as low as possible, while the pattern is forced below
-60 dB in two directions from which interference is expected.

Reference: for a given width of the main beam, the lowest possible sidelobe
level without null constraints is that of the Dolph-Chebyshev array
(C. L. Dolph, Proc. IRE 34, 1946). The sidelobe region here begins where
the -30 dB Dolph-Chebyshev pattern first falls to its sidelobe level, so
-30 dB is a lower bound for the sidelobe level; the null constraints cost
a little on top of it.
"""

from __future__ import annotations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta
from pygpumeta import Constraint

HALF = 10  # amplitudes; the array has 2 * HALF elements
DESIGN_SLL_DB = 30.0
NULLS_U = (0.50, 0.62)  # sin(theta) of the directions to suppress
NULL_DEPTH_DB = 60.0


def chebyshev_beam_edge() -> float:
    """Where the Dolph-Chebyshev pattern of the design level first reaches that level.

    The pattern is T(x0 cos(pi u / 2)) with the Chebyshev polynomial T of
    degree 2 * HALF - 1; its sidelobe region is x0 cos(pi u / 2) <= 1.
    """
    ratio = 10.0 ** (DESIGN_SLL_DB / 20.0)
    x0 = np.cosh(np.arccosh(ratio) / (2 * HALF - 1))
    return float(2.0 / np.pi * np.arccos(1.0 / x0))


BEAM_EDGE = chebyshev_beam_edge()
SIDELOBE_U = np.linspace(BEAM_EDGE, 1.0, 600)
ORDERS = 2.0 * np.arange(1, HALF + 1) - 1.0


def pattern(amplitudes, u):
    """Array factor, normalised to its value at broadside; shape (n, len(u))."""
    xp = array_module(amplitudes)
    table = xp.cos(xp.asarray(np.outer(ORDERS, u) * np.pi / 2.0))
    return (amplitudes @ table) / amplitudes.sum(axis=1, keepdims=True)


def sidelobe_level_db(amplitudes):
    """Highest sidelobe in dB below the main beam; one value per row."""
    xp = array_module(amplitudes)
    peak = xp.abs(pattern(amplitudes, SIDELOBE_U)).max(axis=1)
    return 20.0 * xp.log10(peak)


def null_excess_db(amplitudes):
    """How far the pattern exceeds the required null depth, in dB; <= 0 when satisfied."""
    xp = array_module(amplitudes)
    level = 20.0 * xp.log10(xp.abs(pattern(amplitudes, np.array(NULLS_U))) + 1e-300)
    return level + NULL_DEPTH_DB


def run(**options):
    options = {
        "algorithm": "gwo",
        "selection": "native",
        "pop_size": 2048,
        "max_iter": 2000,
        **options,
    }
    return pygpumeta.minimize(
        sidelobe_level_db,
        (0.05, 1.0),
        dim=HALF,
        constraints=[Constraint(null_excess_db)],
        constraint_handling="feasibility",
        **options,
    )


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=2048, max_iter=2000))
    amplitudes = result.x[None, :]
    nulls = null_excess_db(amplitudes)[0] - NULL_DEPTH_DB
    report(
        "Antenna array: 20 elements, amplitude-only synthesis",
        result,
        {
            "peak sidelobe level": f"{result.fun:.2f} dB",
            "lower bound without nulls (Dolph-Chebyshev)": f"-{DESIGN_SLL_DB:.2f} dB",
            "pattern at the nulls": ", ".join(f"{v:.1f} dB" for v in nulls),
            "feasible": str(result.feasible),
            "amplitudes (centre outwards)": np.array2string(
                result.x / result.x.max(), precision=3, separator=", "
            ),
        },
    )


if __name__ == "__main__":
    main()
