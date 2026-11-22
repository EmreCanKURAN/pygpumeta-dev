"""CPU reference of the warp-level selection kernels.

The functions reproduce the boundary-safe CUDA kernels bit for bit: the same
random stream, the same comparisons and the same handling of ties. They are
used to test the kernels and can stand in for them where no GPU is available.

All operators minimise. An individual competes only with individuals of its
own warp, that is with indices sharing ``index // 32``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pygpumeta import _validation as _v

__all__ = ["ballot_elite", "fitness", "migration", "tournament"]

_FloatArray = NDArray[np.float64]
_IndexArray = NDArray[np.int64]
_Better = Callable[[_FloatArray, _FloatArray], NDArray[np.bool_]]

_WARP = 32
_BLOCK = 256
_MASK32 = np.uint64(0xFFFFFFFF)


def fitness(values: ArrayLike) -> _FloatArray:
    """Map objective values to the fitness used by Ballot-Elite selection.

    Parameters
    ----------
    values : array_like, shape (n,)
        Objective values (smaller is better).

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``1 / (1 + v)`` for ``v >= 0`` and ``1 + |v|`` otherwise, so that a
        larger fitness is better.
    """
    return _fitness(_v.check_values(values))


def tournament(values: ArrayLike, *, size: int = 3, seed: int) -> _IndexArray:
    """Warp-level tournament selection.

    Every individual is compared with ``size`` pseudo-randomly chosen
    individuals of its warp and receives the index of the best one seen,
    itself included.

    Parameters
    ----------
    values : array_like, shape (n,)
        Objective values (smaller is better).
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        Index of the selected individual for every individual.
    """
    obj = _v.check_values(values)
    return _warp_tournament(obj, _v.check_tournament_size(size), _v.check_seed(seed), np.less)[0]


def migration(values: ArrayLike, *, rate: float = 0.25, size: int = 3, seed: int) -> _IndexArray:
    """Warp-level tournament with global-best migration.

    After the tournament, an individual additionally compares its winner
    with the best individual of the whole population with probability
    ``rate``. ``rate=0`` is the plain tournament; ``rate=1`` returns an
    individual with the minimum objective value for everyone.

    Parameters
    ----------
    values : array_like, shape (n,)
        Objective values (smaller is better).
    rate : float, default 0.25
        Migration probability in [0, 1].
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        Index of the selected individual for every individual.
    """
    obj = _v.check_values(values)
    rate = _v.check_unit_interval(rate, "rate")
    return _migrate(obj, _v.check_tournament_size(size), _v.check_seed(seed), rate)


def ballot_elite(
    values: ArrayLike,
    *,
    size: int = 3,
    alpha: float = 1.2,
    seed: int,
    mean: float | None = None,
) -> _IndexArray:
    """Warp-level Ballot-Elite selection.

    Individuals whose fitness exceeds ``alpha`` times the mean fitness are
    elite. A warp with at least two elites runs a tournament restricted to
    elite competitors, a warp with one elite selects it for everyone, and a
    warp without elites falls back to the plain tournament.

    Parameters
    ----------
    values : array_like, shape (n,)
        Objective values (smaller is better). They are converted with
        :func:`fitness` first.
    size : int, default 3
        Number of competitors per individual, between 1 and 6.
    alpha : float, default 1.2
        Elite threshold as a multiple of the mean fitness.
    seed : int
        Seed of the random stream, ``0 <= seed < 2**32``.
    mean : float, optional
        Mean fitness to use for the threshold. By default it is computed
        with the summation order of the kernels (pairwise inside blocks of
        256 individuals, blocks added in index order).

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        Index of the selected individual for every individual.
    """
    fit = _fitness(_v.check_values(values))
    size = _v.check_tournament_size(size)
    alpha = _v.check_positive(alpha, "alpha")
    seed = _v.check_seed(seed)
    if mean is None:
        mean = _block_mean(fit)
    return _ballot(fit, size, alpha, seed, mean=float(mean), variant="elite")


def _xorshift(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    # 32-bit xorshift carried out in 64-bit integers.
    x = (x ^ ((x << np.uint64(13)) & _MASK32)) & _MASK32
    x = (x ^ (x >> np.uint64(17))) & _MASK32
    x = (x ^ ((x << np.uint64(5)) & _MASK32)) & _MASK32
    return x


def _state(index: _IndexArray, seed: int) -> NDArray[np.uint64]:
    start = (np.uint64(seed & 0xFFFFFFFF) + index.astype(np.uint64) * np.uint64(54321)) & _MASK32
    return _xorshift(start)


def _offsets(count: int, seed: int, size: int) -> list[_IndexArray]:
    """Lane offsets of the competitors; an offset of 0 is replaced by 1."""
    state = _state(np.arange(count, dtype=np.int64), seed)
    offsets = []
    for t in range(size):
        off = ((state >> np.uint64(5 * t)) & np.uint64(31)).astype(np.int64)
        off[off == 0] = 1
        offsets.append(off)
    return offsets


def _warp_tournament(
    vals: _FloatArray, size: int, seed: int, better: _Better | Any
) -> tuple[_IndexArray, _FloatArray]:
    n = len(vals)
    idx = np.arange(n, dtype=np.int64)
    lane = idx & 31
    best = vals.astype(np.float64).copy()
    win = idx.copy()
    for off in _offsets(n, seed, size):
        ni = idx - lane + (lane ^ off)
        valid = ni < n
        nb = vals[np.minimum(ni, n - 1)]
        upd = valid & better(nb, best)
        best = np.where(upd, nb, best)
        win = np.where(upd, ni, win)
    return win, best


def _elitist(obj: _FloatArray, size: int, seed: int) -> _IndexArray:
    win, best = _warp_tournament(obj, size, seed, np.less)
    gb = int(np.argmin(obj))
    return np.where(obj[gb] < best, gb, win)


def _global_best(obj: _FloatArray) -> _IndexArray:
    return np.full(len(obj), int(np.argmin(obj)), dtype=np.int64)


def _migrate(obj: _FloatArray, size: int, seed: int, rate: float) -> _IndexArray:
    n = len(obj)
    win, best = _warp_tournament(obj, size, seed, np.less)
    idx = np.arange(n, dtype=np.uint64)
    s2 = (
        (np.uint64(seed & 0xFFFFFFFF) ^ np.uint64(0x9E3779B9)) + idx * np.uint64(2654435761)
    ) & _MASK32
    s2 = _xorshift(_xorshift(s2))
    u = (s2 & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0
    gb = int(np.argmin(obj))
    use = (u < rate) if rate < 1.0 else np.ones(n, bool)
    return np.where(use & (obj[gb] < best), gb, win)


def _global_tournament(obj: _FloatArray, size: int, seed: int) -> _IndexArray:
    """Tournament with competitors drawn from the whole population."""
    n = len(obj)
    state = _xorshift(_state(np.arange(n, dtype=np.int64), seed))
    win = (state % np.uint64(n)).astype(np.int64)
    best = obj[win]
    for _ in range(1, size):
        state = _xorshift(state)
        c = (state % np.uint64(n)).astype(np.int64)
        oc = obj[c]
        upd = oc < best
        best = np.where(upd, oc, best)
        win = np.where(upd, c, win)
    return win


def _fitness(obj: _FloatArray) -> _FloatArray:
    # Both branches are evaluated for every element; the reciprocal of the
    # unused branch may divide by zero at obj == -1.
    with np.errstate(divide="ignore"):
        return np.where(obj >= 0.0, 1.0 / (1.0 + obj), 1.0 + np.abs(obj))


def _block_mean(fit: _FloatArray, block: int = _BLOCK) -> float:
    """Mean with the summation order of the kernels.

    Every block of ``block`` individuals is reduced pairwise; the block sums
    are then added in index order.
    """
    n = len(fit)
    total = 0.0
    for start in range(0, n, block):
        s = np.zeros(block, np.float64)
        chunk = fit[start : start + block]
        s[: len(chunk)] = chunk
        step = block // 2
        while step > 0:
            s[:step] = s[:step] + s[step : 2 * step]
            step //= 2
        total = total + float(s[0])
    return total / float(n)


def _ballot(
    fit: _FloatArray, size: int, alpha: float, seed: int, *, mean: float | None, variant: str
) -> _IndexArray:
    """Ballot selection; ``variant`` is ``"elite"``, ``"noswitch"`` or ``"local"``."""
    n = len(fit)
    padded = ((n + _WARP - 1) // _WARP) * _WARP
    index = np.arange(padded, dtype=np.int64)
    lane = index & 31
    base = index - lane
    active = index < n
    raw = np.zeros(padded)
    raw[:n] = fit
    f = np.full(padded, -np.inf)
    f[:n] = fit
    if variant == "local":
        w = raw.copy()
        for off in (16, 8, 4, 2, 1):
            w = w + w[base + (lane ^ off)]
        acount = active.reshape(-1, _WARP).sum(axis=1).repeat(_WARP).astype(np.float64)
        acount[acount == 0] = 1.0
        thr = (w / acount) * alpha
    else:
        if mean is None:
            raise ValueError("mean is required for the 'elite' and 'noswitch' variants")
        thr = np.full(padded, float(mean) * alpha)
    is_elite = active & (f > thr)
    elite = is_elite.reshape(-1, _WARP)
    count = elite.sum(axis=1).repeat(_WARP)
    first = np.where(elite.any(axis=1), elite.argmax(axis=1), -1).repeat(_WARP)
    offsets = _offsets(padded, seed, size)

    def loop(elite_only: bool) -> _IndexArray:
        best = f.copy()
        win = index.copy()
        for off in offsets:
            ni = base + (lane ^ off)
            valid = ni < n
            nf = f[ni]
            upd = valid & (nf > best)
            if elite_only:
                upd &= is_elite[ni]
            best = np.where(upd, nf, best)
            win = np.where(upd, ni, win)
        return win

    w_elite = loop(True)
    w_plain = loop(False)
    if variant == "noswitch":
        winner = np.where(count > 0, w_elite, w_plain)
    else:
        redirect = (w_elite == index) & (~is_elite) & (first >= 0)
        w_elite = np.where(redirect, base + first, w_elite)
        winner = np.where(count >= 2, w_elite, np.where(count == 1, base + first, w_plain))
    return winner[:n]


def _leaders_from_selection(sel: ArrayLike, obj: _FloatArray) -> tuple[list[int], int]:
    """Three best distinct members of a selected multiset.

    Missing slots repeat the best member found. Returns the leaders and the
    number of distinct ones.
    """
    a = b = c = -1
    va = vb = vc = 1e300
    for s in np.asarray(sel).tolist():
        o = obj[s]
        if s in (a, b, c):
            continue
        if o < va:
            vc, c = vb, b
            vb, b = va, a
            va, a = o, s
        elif o < vb:
            vc, c = vb, b
            vb, b = o, s
        elif o < vc:
            vc, c = o, s
    lead = [a, b if b >= 0 else a, c if c >= 0 else (b if b >= 0 else a)]
    return lead, sum(x >= 0 for x in (a, b, c))
