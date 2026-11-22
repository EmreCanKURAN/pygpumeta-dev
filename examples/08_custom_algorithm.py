"""A custom algorithm built from the selection operators.

The selection operators work on any device array of objective values, so
they can drive an algorithm that pygpumeta does not ship. This example is a
small evolution strategy in about twenty lines of array code:

1. every individual selects a parent with the warp tournament;
2. a child is the parent with Gaussian noise added to about every fifth
   variable, with a width that shrinks over time;
3. a child replaces the individual in its slot when it is better.

Because the tournament compares an individual with members of its own warp
of 32 only, good solutions spread through the population gradually; the
migration operator adds an occasional comparison with the global best.

On the GPU the population is a CuPy array and the operators come from
``pygpumeta.selection``. With ``--mode host`` the same loop runs on NumPy
arrays with ``pygpumeta.reference``, which returns exactly the same indices.

Reference: the optimum of the test function (Rastrigin in 10 dimensions) is
0 at the origin.
"""

from __future__ import annotations

import time

import numpy as np
from common import parse_arguments

from pygpumeta import reference, selection

DIM, LOWER, UPPER = 10, -5.12, 5.12


def rastrigin(x, xp):
    return (x * x).sum(axis=1) + 20.0 * (xp.sin(xp.pi * x) ** 2).sum(axis=1)


def evolve(mode: str = "gpu", pop_size: int = 1024, max_iter: int = 600, seed: int = 0):
    """Run the evolution strategy; returns best value, best position and history."""
    if mode == "gpu":
        import cupy as xp

        operators = selection
    else:
        xp, operators = np, reference
    rng = xp.random.RandomState(seed)
    population = LOWER + rng.random_sample((pop_size, DIM)) * (UPPER - LOWER)
    values = rastrigin(population, xp)
    history = []
    for it in range(max_iter):
        # Alternate between the plain tournament and one with migration.
        if it % 4 == 3:
            parents = operators.migration(values, rate=0.1, size=3, seed=seed + it)
        else:
            parents = operators.tournament(values, size=3, seed=seed + it)
        width = 0.2 * (UPPER - LOWER) * (1.0 - it / max_iter) ** 3
        # Changing a few variables at a time suits functions like this one,
        # whose variables can be improved independently.
        mutated = rng.random_sample((pop_size, DIM)) < 0.2
        noise = width * rng.standard_normal((pop_size, DIM))
        children = xp.clip(population[parents] + mutated * noise, LOWER, UPPER)
        child_values = rastrigin(children, xp)
        better = child_values < values
        population = xp.where(better[:, None], children, population)
        values = xp.where(better, child_values, values)
        if it % 50 == 0 or it == max_iter - 1:
            history.append((it, float(values.min())))
    best = int(values.argmin())
    return float(values[best]), np.asarray(population[best].tolist()), history


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    size = {"pop_size": 256, "max_iter": 150} if args.quick else {}
    start = time.perf_counter()
    best, position, history = evolve(args.mode, seed=args.seed, **size)
    elapsed = time.perf_counter() - start
    print("== Custom algorithm: evolution strategy with warp tournament and migration")
    print(f"   {elapsed:.2f} s including kernel compilation and CuPy start-up")
    print(f"   best value: {best:.6e} (optimum 0)")
    print(f"   largest coordinate of the best position: {np.abs(position).max():.2e}")
    print("   best value over the iterations: " + ", ".join(f"{i}: {v:.3g}" for i, v in history))


if __name__ == "__main__":
    main()
