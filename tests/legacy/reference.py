"""CPU reference implementation of the selection kernels.

The functions reproduce the kernels of kernels/selection_safe.cu bit for bit
(same random stream, same comparisons, same tie handling). They serve two
purposes: checking the GPU kernels for arbitrary population sizes, and
providing the selection step of the host-loop CEC experiments, where the
fitness is evaluated on the CPU and the seeds run in parallel processes.
"""
import re

import numpy as np

_M32 = np.uint64(0xFFFFFFFF)


def _emu_xs(x):
    x = (x ^ ((x << np.uint64(13)) & _M32)) & _M32
    x = (x ^ (x >> np.uint64(17))) & _M32
    x = (x ^ ((x << np.uint64(5)) & _M32)) & _M32
    return x


def _emu_state(idx, seed):
    s = (np.uint64(int(seed) & 0xFFFFFFFF) + idx.astype(np.uint64) * np.uint64(54321)) & _M32
    return _emu_xs(s)


def emu_offsets(P, seed, ts):
    idx = np.arange(P, dtype=np.int64)
    st = _emu_state(idx, seed)
    offs = []
    for t in range(ts):
        off = ((st >> np.uint64(5 * t)) & np.uint64(31)).astype(np.int64)
        off[off == 0] = 1
        offs.append(off)
    return offs


def _emu_warp_tournament(vals, ts, seed, better):
    """vals: per-lane comparison values of length n. better(a, b) -> a better than b."""
    n = len(vals)
    idx = np.arange(n, dtype=np.int64)
    lane = idx & 31
    best = vals.astype(np.float64).copy()
    win = idx.copy()
    for off in emu_offsets(n, seed, ts):
        ni = idx - lane + (lane ^ off)
        valid = ni < n
        nb = vals[np.minimum(ni, n - 1)]
        upd = valid & better(nb, best)
        best = np.where(upd, nb, best)
        win = np.where(upd, ni, win)
    return win, best


def emu_tournament(obj, ts, seed):
    return _emu_warp_tournament(np.asarray(obj, np.float64), ts, seed, lambda a, b: a < b)[0]


def emu_elitist(obj, ts, seed):
    obj = np.asarray(obj, np.float64)
    win, best = _emu_warp_tournament(obj, ts, seed, lambda a, b: a < b)
    gb = int(np.argmin(obj))
    return np.where(obj[gb] < best, gb, win)


def emu_gbest(obj):
    return np.full(len(obj), int(np.argmin(obj)), dtype=np.int64)


def emu_migrate(obj, ts, seed, rho):
    obj = np.asarray(obj, np.float64)
    n = len(obj)
    win, best = _emu_warp_tournament(obj, ts, seed, lambda a, b: a < b)
    idx = np.arange(n, dtype=np.uint64)
    s2 = ((np.uint64(int(seed) & 0xFFFFFFFF) ^ np.uint64(0x9E3779B9)) + idx * np.uint64(2654435761)) & _M32
    s2 = _emu_xs(_emu_xs(s2))
    u = (s2 & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0
    gb = int(np.argmin(obj))
    use = (u < rho) if rho < 1.0 else np.ones(n, bool)
    return np.where(use & (obj[gb] < best), gb, win)


def emu_global_tournament(obj, ts, seed):
    obj = np.asarray(obj, np.float64)
    n = len(obj)
    idx = np.arange(n, dtype=np.int64)
    st = _emu_state(idx, seed)
    st = _emu_xs(st)
    win = (st % np.uint64(n)).astype(np.int64)
    best = obj[win]
    for _ in range(1, ts):
        st = _emu_xs(st)
        c = (st % np.uint64(n)).astype(np.int64)
        oc = obj[c]
        upd = oc < best
        best = np.where(upd, oc, best)
        win = np.where(upd, c, win)
    return win


def emu_fit(obj):
    obj = np.asarray(obj, np.float64)
    return np.where(obj >= 0.0, 1.0 / (1.0 + obj), 1.0 + np.abs(obj))


def emu_avg_single_block(fit, block=256):
    """Exact replica of compute_avg_fitness + finalize for n <= block (one block)."""
    n = len(fit)
    assert n <= block, "exact mean emulation needs a single block (n <= 256)"
    s = np.zeros(block, np.float64)
    s[:n] = fit
    step = block // 2
    while step > 0:
        s[:step] = s[:step] + s[step:2 * step]
        step //= 2
    total = 0.0 + s[0]
    return total / float(n)


def emu_ballot(fit, ts, alpha, seed, avg=None, variant='elite'):
    """variant in {'elite', 'noswitch', 'local'}; avg required for elite/noswitch."""
    fit = np.asarray(fit, np.float64)
    n = len(fit)
    P = ((n + 31) // 32) * 32
    L = np.arange(P, dtype=np.int64)
    lane = L & 31
    base = L - lane
    active = L < n
    raw = np.zeros(P); raw[:n] = fit
    f = np.full(P, -np.inf); f[:n] = fit
    if variant == 'local':
        w = raw.copy()
        for off in (16, 8, 4, 2, 1):
            w = w + w[base + (lane ^ off)]
        acount = active.reshape(-1, 32).sum(axis=1).repeat(32).astype(np.float64)
        acount[acount == 0] = 1.0
        thr = (w / acount) * alpha
    else:
        thr = np.full(P, float(avg) * alpha)
    is_elite = active & (f > thr)
    E = is_elite.reshape(-1, 32)
    count = E.sum(axis=1).repeat(32)
    first = np.where(E.any(axis=1), E.argmax(axis=1), -1).repeat(32)
    offs = emu_offsets(P, seed, ts)

    def loop(use_elite):
        best = f.copy(); win = L.copy()
        for off in offs:
            ni = base + (lane ^ off)
            valid = ni < n
            nf = f[ni]
            upd = valid & (nf > best)
            if use_elite:
                upd &= is_elite[ni]
            best = np.where(upd, nf, best)
            win = np.where(upd, ni, win)
        return win

    w_el = loop(True)
    w_pl = loop(False)
    if variant == 'noswitch':
        winner = np.where(count > 0, w_el, w_pl)
    else:
        redirect = (w_el == L) & (~is_elite) & (first >= 0)
        w_el2 = np.where(redirect, base + first, w_el)
        winner = np.where(count >= 2, w_el2, np.where(count == 1, base + first, w_pl))
    return winner[:n]


def emu_select(method, obj, seed, ts=3, alpha=1.2):
    """Emulated counterpart of fill_selection for the host-loop experiments."""
    if method == 'tournament':
        return emu_tournament(obj, ts, seed)
    if method == 'elitist':
        return emu_elitist(obj, ts, seed)
    if method == 'gbest':
        return emu_gbest(obj)
    if method.startswith('mig_'):
        return emu_migrate(obj, ts, seed, float(method[4:]))
    if re.fullmatch(r'global\d+', method):
        return emu_global_tournament(obj, int(method[6:]), seed)
    if method == 'ballot':
        fit = emu_fit(obj)
        return emu_ballot(fit, ts, alpha, seed, avg=emu_avg_single_block(fit), variant='elite')
    if method == 'ballot_local':
        return emu_ballot(emu_fit(obj), ts, alpha, seed, variant='local')
    raise ValueError(method)


def leaders_from_selection(sel, obj):
    """Exact replica of gwo_find_leaders_sel: three best DISTINCT members of the
    selected multiset; missing slots repeat the best one found. Returns
    (leaders, number_of_distinct_leaders)."""
    a = b = c = -1
    va = vb = vc = 1e300
    for s in sel:
        s = int(s)
        o = obj[s]
        if s == a or s == b or s == c:
            continue
        if o < va:
            vc, c = vb, b; vb, b = va, a; va, a = o, s
        elif o < vb:
            vc, c = vb, b; vb, b = o, s
        elif o < vc:
            vc, c = o, s
    lead = [a, b if b >= 0 else a, c if c >= 0 else (b if b >= 0 else a)]
    return lead, int(sum(x >= 0 for x in (a, b, c)))
