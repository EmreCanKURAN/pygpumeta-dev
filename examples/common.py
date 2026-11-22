"""Helpers shared by the examples."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def array_module(x: Any) -> Any:
    """Return numpy for NumPy arrays and cupy for CuPy arrays.

    The objectives of the examples are written once against this module, so
    the same function runs on the GPU (the default) and on the CPU
    (``--mode host``).
    """
    if isinstance(x, np.ndarray):
        return np
    import cupy

    return cupy


def parse_arguments(description: str) -> argparse.Namespace:
    """Parse the command line options common to all examples."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=["gpu", "host"],
        default="gpu",
        help="evaluate the objective on the GPU with CuPy (default) or on the CPU with NumPy",
    )
    parser.add_argument("--algorithm", default=None, help="override the algorithm of the example")
    parser.add_argument("--selection", default=None, help="override the selection operator")
    parser.add_argument("--quick", action="store_true", help="small budget, for a smoke test")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def settings(args: Any, **defaults: Any) -> dict[str, Any]:
    """Combine the defaults of an example with the command line overrides."""
    out = dict(defaults)
    out["mode"] = args.mode
    out["seed"] = args.seed
    if args.algorithm:
        out["algorithm"] = args.algorithm
    if args.selection:
        out["selection"] = args.selection
    if args.quick:
        out["pop_size"] = min(out["pop_size"], 64)
        out["max_iter"] = min(out["max_iter"], 60)
    return out


def report(title: str, result: Any, lines: dict[str, str]) -> None:
    """Print the outcome of an example in a uniform layout."""
    print(f"== {title}")
    print(f"   algorithm {result.algorithm}, selection {result.selection}, seed {result.seed}")
    print(
        f"   {result.nit} iterations, {result.nfev} evaluations, "
        f"{result.runtime:.2f} s (+ {result.compile_time:.1f} s compilation)"
    )
    for key, value in lines.items():
        print(f"   {key}: {value}")
