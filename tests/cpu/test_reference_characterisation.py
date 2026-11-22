"""The reference must reproduce the digests recorded for fixed inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from data.make_reference_golden import RATES, SIZES, digest, inputs

from pygpumeta import reference as ref

GOLDEN = json.loads(Path(__file__).parent.joinpath("data", "reference_golden.json").read_text())


def outputs(n: int) -> dict[str, str]:
    got: dict[str, str] = {}
    for rep in range(3):
        v, seed = inputs(n, rep)
        fit = ref.fitness(v)
        mean = float(fit.mean())
        for ts in (1, 3, 6):
            key = f"n{n}_r{rep}_ts{ts}"
            got[f"tournament_{key}"] = digest(ref.tournament(v, size=ts, seed=seed))
            got[f"elitist_{key}"] = digest(ref._elitist(v, ts, seed))
            got[f"global_{key}"] = digest(ref._global_tournament(v, ts, seed))
            for rate in RATES:
                got[f"migrate{rate:g}_{key}"] = digest(
                    ref.migration(v, rate=rate, size=ts, seed=seed)
                )
            for alpha in (1.0, 1.2):
                got[f"ballot_elite_a{alpha:g}_{key}"] = digest(
                    ref.ballot_elite(v, size=ts, alpha=alpha, seed=seed, mean=mean)
                )
                for variant in ("noswitch", "local"):
                    got[f"ballot_{variant}_a{alpha:g}_{key}"] = digest(
                        ref._ballot(fit, ts, alpha, seed, mean=mean, variant=variant)
                    )
        got[f"gbest_n{n}_r{rep}"] = digest(ref._global_best(v))
        if n <= 256:
            got[f"blockmean_n{n}_r{rep}"] = float(ref._block_mean(fit)).hex()
        lead, count = ref._leaders_from_selection(ref.tournament(v, size=3, seed=seed), v)
        got[f"leaders_n{n}_r{rep}"] = f"{lead}/{count}"
    return got


@pytest.mark.parametrize("n", SIZES)
def test_outputs_match_recorded_digests(n: int) -> None:
    got = outputs(n)
    assert got == {k: GOLDEN[k] for k in got}


def test_every_recorded_digest_is_checked() -> None:
    assert {k for n in SIZES for k in outputs(n)} == set(GOLDEN)


def test_block_mean_adds_blocks_in_index_order() -> None:
    fit = np.random.RandomState(3).uniform(0.1, 1.0, 1000)
    blocks = [
        ref._block_mean(fit[i : i + 256]) * len(fit[i : i + 256]) for i in range(0, 1000, 256)
    ]
    total = 0.0
    for b in blocks:
        total += b
    # Tolerance: the per-block means are rescaled by the block length above,
    # which costs at most one rounding step per block.
    assert ref._block_mean(fit) == pytest.approx(total / 1000, rel=1e-15)
    assert ref._block_mean(fit) == pytest.approx(fit.mean(), rel=1e-13)
