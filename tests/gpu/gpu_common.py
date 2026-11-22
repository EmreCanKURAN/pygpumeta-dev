from __future__ import annotations

import numpy as np

SIZES = (1, 31, 32, 33, 100, 1000, 1024, 1025, 8192)

# Compiling the library with this literal reproduces the paper kernels built
# with exact pi and e, whose Box-Muller constant was left truncated.
PAPER_TWO_PI = {"PGM_TWO_PI": "6.283185307"}


def population(n: int, rep: int) -> tuple[np.ndarray, int]:
    """Objective values and a seed; odd repetitions contain ties."""
    rng = np.random.RandomState(1000 * n + rep)
    values = rng.uniform(1.0, 2.0, n)
    if rep % 2 == 1:
        values = np.round(values, 2)
    if rep == 2:
        values = values - 1.5
    return values, int(rng.randint(1, 2**31 - 1))
