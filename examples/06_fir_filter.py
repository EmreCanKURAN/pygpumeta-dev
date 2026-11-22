"""Design of a linear-phase FIR low-pass filter in the minimax sense.

A symmetric filter of length 21 has the zero-phase response

    A(w) = h_0 + 2 sum_k h_k cos(k w),   k = 1..10.

Its eleven coefficients are chosen so that the largest deviation from the
ideal response, 1 in the passband (up to 0.2 pi) and 0 in the stopband (from
0.3 pi), is as small as possible.

Reference: this problem has a unique optimum, the equiripple filter, which
the Parks-McClellan algorithm computes directly (Parks and McClellan, IEEE
Trans. Circuit Theory 19, 1972). The example prints the ripple of that
filter, obtained with scipy.signal.remez when SciPy is installed, next to
the ripple reached by the optimizer. Both are measured on the same 400
frequencies. The optimizer only sees those frequencies, so it can end a
fraction of a percent below the equiripple filter there; between them its
response is not controlled. For this task the dedicated algorithm is the
right tool; the example shows that a general optimizer reaches the same
answer.
"""

from __future__ import annotations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta

HALF_ORDER = 10  # the filter has 2 * HALF_ORDER + 1 taps
PASS_EDGE, STOP_EDGE = 0.2 * np.pi, 0.3 * np.pi
GRID = np.concatenate([np.linspace(0.0, PASS_EDGE, 120), np.linspace(STOP_EDGE, np.pi, 280)])
DESIRED = (GRID <= PASS_EDGE).astype(np.float64)
# Row k holds cos(k w); the factor 2 of the symmetric taps is included.
BASIS = (
    np.cos(np.outer(np.arange(HALF_ORDER + 1), GRID))
    * np.r_[1.0, 2.0 * np.ones(HALF_ORDER)][:, None]
)


def ripple(coefficients):
    """Largest deviation from the ideal response; one value per row."""
    xp = array_module(coefficients)
    response = coefficients @ xp.asarray(BASIS)
    return xp.abs(response - xp.asarray(DESIRED)).max(axis=1)


def equiripple() -> float | None:
    """Ripple of the Parks-McClellan filter, or None without SciPy."""
    try:
        from scipy.signal import remez
    except ImportError:
        return None
    taps = remez(2 * HALF_ORDER + 1, [0.0, PASS_EDGE, STOP_EDGE, np.pi], [1.0, 0.0], fs=2 * np.pi)
    return float(ripple(taps[HALF_ORDER:][None, :])[0])


def run(**options):
    options = {
        "algorithm": "de",
        "selection": "native",
        "pop_size": 2048,
        "max_iter": 4000,
        **options,
    }
    return pygpumeta.minimize(ripple, (-0.5, 0.5), dim=HALF_ORDER + 1, **options)


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=2048, max_iter=4000))
    optimum = equiripple()
    report(
        "FIR low-pass filter: 21 taps, minimax design",
        result,
        {
            "ripple": f"{result.fun:.5f} ({20 * np.log10(result.fun):.2f} dB)",
            "equiripple optimum (Parks-McClellan)": "SciPy not installed"
            if optimum is None
            else f"{optimum:.5f} ({20 * np.log10(optimum):.2f} dB)",
            "taps h_0..h_10": np.array2string(result.x, precision=4, separator=", "),
        },
    )


if __name__ == "__main__":
    main()
