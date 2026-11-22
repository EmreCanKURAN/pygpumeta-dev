"""Record digests of the selection reference for fixed inputs.

The digests in reference_golden.json were produced with this script from
the reference implementation that accompanies the paper:

    python tests/cpu/data/make_reference_golden.py path/to/reference.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SIZES = (1, 2, 31, 32, 33, 100, 255, 256, 1000, 1024, 1025, 8192)
RATES = (0.0, 0.03125, 0.25, 1.0)


def inputs(n: int, rep: int) -> tuple[np.ndarray, int]:
    rng = np.random.RandomState(1000 * n + rep)
    values = rng.uniform(1.0, 2.0, n)
    if rep % 2 == 1:
        values = np.round(values, 2)  # ties
    if rep == 2:
        values = values - 1.5  # negative objective values
    return values, int(rng.randint(1, 2**31 - 1))


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def main(path: str) -> None:
    spec = importlib.util.spec_from_file_location("recorded_reference", path)
    assert spec is not None and spec.loader is not None
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    out: dict[str, str] = {}
    for n in SIZES:
        for rep in range(3):
            v, seed = inputs(n, rep)
            fit = ref.emu_fit(v)
            mean = float(fit.mean())
            for ts in (1, 3, 6):
                key = f"n{n}_r{rep}_ts{ts}"
                out[f"tournament_{key}"] = digest(ref.emu_tournament(v, ts, seed))
                out[f"elitist_{key}"] = digest(ref.emu_elitist(v, ts, seed))
                out[f"global_{key}"] = digest(ref.emu_global_tournament(v, ts, seed))
                for rate in RATES:
                    out[f"migrate{rate:g}_{key}"] = digest(ref.emu_migrate(v, ts, seed, rate))
                for variant in ("elite", "noswitch", "local"):
                    for alpha in (1.0, 1.2):
                        out[f"ballot_{variant}_a{alpha:g}_{key}"] = digest(
                            ref.emu_ballot(fit, ts, alpha, seed, avg=mean, variant=variant)
                        )
            out[f"gbest_n{n}_r{rep}"] = digest(ref.emu_gbest(v))
            if n <= 256:
                out[f"blockmean_n{n}_r{rep}"] = float(ref.emu_avg_single_block(fit)).hex()
            sel = ref.emu_tournament(v, 3, seed)
            lead, count = ref.leaders_from_selection(sel, v)
            out[f"leaders_n{n}_r{rep}"] = f"{lead}/{count}"
    target = Path(__file__).with_name("reference_golden.json")
    target.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print(f"{len(out)} digests written to {target}")


if __name__ == "__main__":
    main(sys.argv[1])
