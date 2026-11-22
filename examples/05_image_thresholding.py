"""Multilevel image thresholding with Otsu's criterion.

Three thresholds split the grey levels of an image into four classes. Otsu
(IEEE Trans. Systems, Man, and Cybernetics 9, 1979) chooses them to
maximise the between-class variance

    sum_k  w_k (mu_k - mu)^2,

where w_k and mu_k are the share and the mean grey level of class k. The
thresholds are integers, declared here as integer variables: the optimizer
searches continuous values and the objective receives them rounded.

The image is synthetic (four overlapping grey-level populations), so the
example needs no data file; pass your own histogram to ``otsu`` to use it
on a real image.

Reference: with three thresholds the search space has about 2.7 million
combinations, small enough to enumerate. The example prints the exact
optimum found by enumeration next to the result of the optimizer.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta
from pygpumeta.space import Integer

LEVELS = 256
THRESHOLDS = 3


def synthetic_histogram() -> np.ndarray:
    """Normalised grey-level histogram of a synthetic 512 x 512 image."""
    rng = np.random.RandomState(7)
    parts = [(40, 12, 0.30), (95, 18, 0.25), (150, 14, 0.25), (210, 16, 0.20)]
    pixels = np.concatenate(
        [rng.normal(mean, spread, int(share * 512 * 512)) for mean, spread, share in parts]
    )
    counts = np.bincount(np.clip(np.rint(pixels), 0, LEVELS - 1).astype(int), minlength=LEVELS)
    return counts / counts.sum()


HISTOGRAM = synthetic_histogram()
# Cumulative share and cumulative first moment up to each grey level.
CUM_SHARE = np.concatenate([[0.0], np.cumsum(HISTOGRAM)])
CUM_MOMENT = np.concatenate([[0.0], np.cumsum(HISTOGRAM * np.arange(LEVELS))])


def otsu(thresholds):
    """Between-class variance for every row of thresholds (in any order)."""
    xp = array_module(thresholds)
    share, moment = xp.asarray(CUM_SHARE), xp.asarray(CUM_MOMENT)
    cuts = xp.sort(thresholds, axis=1).astype(xp.int64) + 1
    edges = xp.concatenate(
        [xp.zeros_like(cuts[:, :1]), cuts, xp.full_like(cuts[:, :1], LEVELS)], axis=1
    )
    w = share[edges[:, 1:]] - share[edges[:, :-1]]
    m = moment[edges[:, 1:]] - moment[edges[:, :-1]]
    # sum w_k mu_k^2 - mu^2; an empty class contributes nothing.
    return (m * m / xp.maximum(w, 1e-300)).sum(axis=1) - float(CUM_MOMENT[-1]) ** 2


def exhaustive() -> tuple[float, tuple[int, ...]]:
    """Exact optimum by enumeration of all threshold triples."""
    triples = np.array(list(combinations(range(1, LEVELS - 1), THRESHOLDS)), dtype=np.float64)
    values = np.concatenate([otsu(chunk) for chunk in np.array_split(triples, 20)])
    best = int(np.argmax(values))
    return float(values[best]), tuple(int(t) for t in triples[best])


def run(**options):
    options = {
        "algorithm": "pso",
        "selection": "tournament",
        "pop_size": 1024,
        "max_iter": 300,
        **options,
    }
    return pygpumeta.minimize(otsu, [Integer(1, LEVELS - 2)] * THRESHOLDS, maximize=True, **options)


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=1024, max_iter=300))
    optimum, best_triple = exhaustive()
    report(
        "Image thresholding: three Otsu thresholds",
        result,
        {
            "between-class variance": f"{result.fun:.4f}",
            "exact optimum by enumeration": f"{optimum:.4f}",
            "thresholds": str(tuple(sorted(result.values))),
            "optimal thresholds": str(best_triple),
        },
    )


if __name__ == "__main__":
    main()
