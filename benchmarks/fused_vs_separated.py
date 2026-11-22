"""Cost of evaluating the objective as a separate step, and of each objective mode.

Runs ABC, HHO and GWO on one problem with the same seeds and prints the
median runtime of

    fused       objective evaluated inside the update kernels
    separated   objective evaluated by a kernel of its own between the updates
    cuda        the same function supplied as CUDA source (fused)
    gpu         the same function as a Python callable on CuPy arrays
    host        the same function as a Python callable on NumPy arrays

GWO evaluates outside its update in every form, so its fused and separated
rows are one and the same code path and show the noise of the measurement.
The last column says whether the best value equals that of the fused run.

    python benchmarks/fused_vs_separated.py            # about ten minutes on a T4
    python benchmarks/fused_vs_separated.py --quick
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pygpumeta

RASTRIGIN_SOURCE = """
double evaluate(const double* x, int d) {
    double sum_sq = 0.0, sum_sin = 0.0;
    for (int i = 0; i < d; i++) {
        sum_sq += x[i] * x[i];
        double sv = sin(3.141592653589793 * x[i]);
        sum_sin += sv * sv;
    }
    return sum_sq + 20.0 * sum_sin;
}
"""
BOUNDS = (-5.12, 5.12)


def rastrigin_cupy(x):
    import cupy as cp

    return (x * x).sum(axis=1) + 20.0 * (cp.sin(cp.pi * x) ** 2).sum(axis=1)


def rastrigin_numpy(x):
    import numpy as np

    return (x * x).sum(axis=1) + 20.0 * (np.sin(np.pi * x) ** 2).sum(axis=1)


def variants(algorithm: str) -> dict[str, dict]:
    in_kernel = algorithm in ("abc", "hho")
    fused = {"options": {"evaluation": "fused"}} if in_kernel else {}
    separated = {"options": {"evaluation": "separated"}} if in_kernel else {}
    cuda = pygpumeta.CudaObjective(RASTRIGIN_SOURCE)
    return {
        "fused": {"objective": "rastrigin", **fused},
        "separated": {"objective": "rastrigin", **separated},
        "cuda": {"objective": cuda, "bounds": BOUNDS},
        "gpu": {"objective": rastrigin_cupy, "bounds": BOUNDS},
        "host": {"objective": rastrigin_numpy, "bounds": BOUNDS, "mode": "host"},
    }


def paper_runtime(algorithm: str, ps: int, dim: int, iters: int, seeds: list[int]) -> float | None:
    """Median runtime of the published kernels, which carry no evaluation counter."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    try:
        from legacy import gpu as paper
    except Exception:
        return None
    with paper.use_kernels(paper.exact_constants_handles()):
        run = paper.GPU_FUNCS[algorithm.upper()]
        times = [
            run(ps, iters, 1, BOUNDS, dim=dim, method="tournament_safe", seed=s)["time_ms"] / 1000
            for s in seeds
        ]
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small sizes, for a dry run")
    args = parser.parse_args()
    sizes = (
        [(256, 10, 50)] if args.quick else [(1024, 30, 1000), (8192, 30, 1000), (8192, 100, 1000)]
    )
    seeds = [1, 2] if args.quick else [1, 2, 3]

    # Compile everything and let CuPy build its kernels outside the timings.
    for algorithm in ("abc", "hho", "gwo"):
        for kwargs in variants(algorithm).values():
            pygpumeta.minimize(dim=4, algorithm=algorithm, pop_size=64, max_iter=3, **kwargs)

    columns = ["algorithm", "pop", "dim", "iters", "form", "median s", "nfev"]
    columns += ["s per 1e6 evals", "same best as fused"]
    print("| " + " | ".join(columns) + " |")
    print("|" + " --- |" * len(columns))
    for ps, dim, iters in sizes:
        for algorithm in ("abc", "hho", "gwo"):
            reference = None
            for form, kwargs in variants(algorithm).items():
                results = [
                    pygpumeta.minimize(
                        dim=dim,
                        algorithm=algorithm,
                        pop_size=ps,
                        max_iter=iters,
                        seed=seed,
                        **kwargs,
                    )
                    for seed in seeds
                ]
                best = [r.fun for r in results]
                if form == "fused":
                    reference = best
                seconds = statistics.median(r.runtime for r in results)
                nfev = results[0].nfev
                print(
                    f"| {algorithm} | {ps} | {dim} | {iters} | {form} | {seconds:.3f} | {nfev} | "
                    f"{seconds / nfev * 1e6:.3f} | {best == reference} |"
                )
            published = paper_runtime(algorithm, ps, dim, iters, seeds)
            if published is not None:
                row = f"{algorithm} | {ps} | {dim} | {iters} | published kernels | {published:.3f}"
                print(f"| {row} | | | |")


if __name__ == "__main__":
    main()
