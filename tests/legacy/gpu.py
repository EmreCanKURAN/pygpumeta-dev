"""GPU side: CUDA context, kernel compilation, the selection dispatcher and the
runners of the six algorithms.

Importing this module creates the CUDA context and compiles the kernels, so a
CUDA-capable GPU and PyCUDA are required.

Selection methods understood by fill_selection (and therefore by the runners
through their ``method`` argument):

    naive                  algorithm-specific baseline (handled by the runner)
    tournament             GPU Tournament (__shfl_xor_sync), n multiple of 32
    ballot                 GPU Ballot-Elite (__ballot_sync), n multiple of 32
    ballot_local           Ballot-Elite with a warp-local mean
    ballot_noswitch        Ballot-Elite without adaptive strategy switching
    tournament_safe, ballot_safe, ballot_local_safe, ballot_noswitch_safe
                           boundary-safe versions for any population size
    mig_<rho>              warp tournament with global-best migration rate rho
    tournament_elitist     elitist rule, global best found through the host
    elitist_dev            elitist rule, global best kept on the device
    gbest                  direct global-best selection
    tournament_global      tournament over the whole population (global memory)
    global<k>              the same with k contestants
    roulette_scan          roulette wheel with a parallel prefix sum
"""
import atexit
import contextlib
import math
import os
import re
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# The primary context is retained instead of creating a private one, so that
# CuPy (used by the library baseline) and PyCUDA share the same context.
cuda.init()
_context = cuda.Device(0).retain_primary_context()
_context.push()
atexit.register(_context.pop)

device = cuda.Device(0)
GPU_NAME = device.name()
CC = device.compute_capability()

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels')


def _read_source(name):
    with open(os.path.join(_KERNEL_DIR, name)) as f:
        return f.read()


BASE_SRC = _read_source('metaheuristics.cu')
SELECTION_SRC = _read_source('selection_safe.cu')

mod = SourceModule(BASE_SRC)

# HHO (Levy flight) and the device-side CEC functions keep local arrays on the stack.
cuda.Context.get_current().set_limit(cuda.limit.STACK_SIZE, 65536)

k_sel_naive        = mod.get_function("sel_naive")
k_sel_tournament   = mod.get_function("sel_tournament")
k_sel_tourn_global = mod.get_function("sel_tournament_global")
k_sel_tourn_elite  = mod.get_function("sel_tournament_elitist")
k_compute_avg      = mod.get_function("compute_avg_fitness")
k_finalize_avg     = mod.get_function("finalize_avg_fitness")
k_reset_avg        = mod.get_function("reset_avg")
k_sel_ballot       = mod.get_function("sel_ballot_elite")
k_sel_ballot_local = mod.get_function("sel_ballot_local")
k_sel_ballot_nosw  = mod.get_function("sel_ballot_noswitch")
k_gather_obj       = mod.get_function("gather_obj")
k_pop_init         = mod.get_function("pop_init")
k_boundary         = mod.get_function("boundary_clamp")
k_eval_pop         = mod.get_function("eval_pop")
k_compute_fit      = mod.get_function("compute_fit")
k_find_best        = mod.get_function("find_best_idx")
k_copy_indiv       = mod.get_function("copy_individual")
k_mean_pos         = mod.get_function("compute_mean_pos")
k_abc_init         = mod.get_function("abc_init")
k_abc_employed     = mod.get_function("abc_employed")
k_abc_onl_roul     = mod.get_function("abc_onlooker_roulette")
k_abc_onl_sel      = mod.get_function("abc_onlooker_selected")
k_abc_max_trial    = mod.get_function("abc_find_max_trial")
k_abc_scout        = mod.get_function("abc_scout")
k_gwo_leaders      = mod.get_function("gwo_find_leaders")
k_gwo_leaders_sel  = mod.get_function("gwo_find_leaders_sel")
k_gwo_copy_lead    = mod.get_function("gwo_copy_leaders")
k_gwo_update       = mod.get_function("gwo_update")
k_woa_update       = mod.get_function("woa_update")
k_woa_update_sel   = mod.get_function("woa_update_sel")
k_sca_update       = mod.get_function("sca_update")
k_sca_update_sel   = mod.get_function("sca_update_sel")
k_ssa_update       = mod.get_function("ssa_update")
k_ssa_update_sel   = mod.get_function("ssa_update_sel")
k_hho_update       = mod.get_function("hho_update")
k_hho_update_sel   = mod.get_function("hho_update_sel")
k_update_best      = mod.get_function("update_best_ever")
k_update_guide     = mod.get_function("update_guide_if_better")

# Python handle name -> kernel name. The runners look the handles up as module
# globals, which allows use_kernels() to exchange them temporarily.
HANDLE_MAP = {
    'k_sel_naive': 'sel_naive', 'k_sel_tournament': 'sel_tournament',
    'k_sel_tourn_global': 'sel_tournament_global', 'k_sel_tourn_elite': 'sel_tournament_elitist',
    'k_compute_avg': 'compute_avg_fitness', 'k_finalize_avg': 'finalize_avg_fitness',
    'k_reset_avg': 'reset_avg', 'k_sel_ballot': 'sel_ballot_elite', 'k_sel_ballot_local': 'sel_ballot_local',
    'k_sel_ballot_nosw': 'sel_ballot_noswitch', 'k_gather_obj': 'gather_obj', 'k_pop_init': 'pop_init',
    'k_boundary': 'boundary_clamp', 'k_eval_pop': 'eval_pop', 'k_compute_fit': 'compute_fit',
    'k_find_best': 'find_best_idx', 'k_copy_indiv': 'copy_individual', 'k_mean_pos': 'compute_mean_pos',
    'k_abc_init': 'abc_init', 'k_abc_employed': 'abc_employed', 'k_abc_onl_roul': 'abc_onlooker_roulette',
    'k_abc_onl_sel': 'abc_onlooker_selected', 'k_abc_max_trial': 'abc_find_max_trial',
    'k_abc_scout': 'abc_scout', 'k_gwo_leaders': 'gwo_find_leaders', 'k_gwo_leaders_sel': 'gwo_find_leaders_sel',
    'k_gwo_copy_lead': 'gwo_copy_leaders', 'k_gwo_update': 'gwo_update', 'k_woa_update': 'woa_update',
    'k_woa_update_sel': 'woa_update_sel', 'k_sca_update': 'sca_update', 'k_sca_update_sel': 'sca_update_sel',
    'k_ssa_update': 'ssa_update', 'k_ssa_update_sel': 'ssa_update_sel', 'k_hho_update': 'hho_update',
    'k_hho_update_sel': 'hho_update_sel', 'k_update_best': 'update_best_ever',
    'k_update_guide': 'update_guide_if_better',
}
BASE_HANDLES = {name: globals()[name] for name in HANDLE_MAP}

SELECTION_KERNELS = ['sel_tournament_safe', 'sel_ballot_elite_safe', 'sel_ballot_local_safe',
                     'sel_ballot_noswitch_safe', 'sel_tournament_elitist_dev', 'sel_globalbest',
                     'sel_tournament_migrate', 'argmin_block', 'roulette_sample',
                     'sel_tournament_shared', 'update_best_ever_pos']
_selection_mod = SourceModule(SELECTION_SRC)
SELK = {name: _selection_mod.get_function(name) for name in SELECTION_KERNELS}

BS = 256   # threads per block


def grid(ps):
    return ((ps + BS - 1) // BS, 1)


def seed32(s):
    return np.uint32(int(s) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Device globals: fitness-cost multiplier and CEC shift / rotation data
# ---------------------------------------------------------------------------
_cost_ptr = mod.get_global("d_fitness_cost")[0]


def set_fitness_cost(k):
    cuda.memcpy_htod(_cost_ptr, np.int32(k))


_cecg_shift_ptr = mod.get_global("d_cecg_shift")[0]
_cecg_rot_ptr   = mod.get_global("d_cecg_rot")[0]
_cecg_fopt_ptr  = mod.get_global("d_cecg_fopt")[0]

# Device-side function ids and the opfunu CEC2017 classes whose shift vector
# and rotation matrix they use. F1 (bent cigar, fid 200) reproduces opfunu
# exactly; the others combine the official shift and rotation data with a
# standard base function and are used for throughput measurements only.
CECG_FUNCS = {200: ('F12017', 'BentCigar_exact'), 201: ('F52017', 'Rastrigin_SR'),
              202: ('F62017', 'Ackley_SR'), 203: ('F72017', 'Griewank_SR'),
              204: ('F42017', 'Rosenbrock_SR'), 205: ('F32017', 'Zakharov_SR')}


def load_cecg(fid, dim):
    """Copy shift vector, rotation matrix and optimum of a CEC function to the device."""
    from opfunu.cec_based import cec2017
    cls_name = CECG_FUNCS[fid][0]
    fobj = getattr(cec2017, cls_name)(ndim=dim)
    shift = np.asarray(fobj.f_shift, dtype=np.float64).ravel()[:dim]
    rot = np.asarray(fobj.f_matrix, dtype=np.float64)
    if rot.shape[0] != dim:
        rot = rot[:dim, :dim]
    cuda.memcpy_htod(_cecg_shift_ptr, np.ascontiguousarray(shift))
    cuda.memcpy_htod(_cecg_rot_ptr, np.ascontiguousarray(rot.flatten().astype(np.float64)))
    cuda.memcpy_htod(_cecg_fopt_ptr, np.float64(fobj.f_global))
    return float(fobj.f_global)


# ---------------------------------------------------------------------------
# Exchanging kernel handles, source variants
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def use_kernels(handles):
    """Temporarily replace kernel handles of this module (restored on exit)."""
    saved = {k: globals()[k] for k in handles}
    try:
        for k, v in handles.items():
            globals()[k] = v
        yield
    finally:
        for k, v in saved.items():
            globals()[k] = v


def build_module(src, label):
    t0 = time.time()
    mod = SourceModule(src)
    h = {hk: mod.get_function(fn) for hk, fn in HANDLE_MAP.items()}
    print(f"[gpu] module '{label}' compiled in {time.time() - t0:.1f}s")
    return mod, h


def exact_constants_source():
    """metaheuristics.cu writes pi as 3.14159265 and e as 2.71828183; the
    truncated e alone produces an error floor of 1.54e-9 on Ackley. This
    returns the same source with full double-precision constants (objective
    functions and update rules; the Box-Muller constant of the random number
    generator is left as it is)."""
    src = BASE_SRC.replace('3.14159265358979323846', '@@PI@@')
    src, n_pi = re.subn(r'3\.14159265(?!\d)', '3.141592653589793', src)
    src = src.replace('@@PI@@', '3.141592653589793')
    src, n_e = re.subn(r'2\.71828183(?!\d)', '2.718281828459045', src)
    assert n_e >= 1 and n_pi >= 1
    return src


def counting_source(base_src):
    """Source with a device-side counter of objective function evaluations."""
    sig = '__device__ double evaluate(double* x, int d, int fid) {'
    assert base_src.count(sig) == 1, "evaluate() signature not found exactly once"
    patch = ('__device__ unsigned long long d_fe_count = 0ULL;\n'
             '__device__ double evaluate_raw(double* x, int d, int fid);\n'
             + sig + ' atomicAdd(&d_fe_count, 1ULL); return evaluate_raw(x, d, fid); }\n'
             '__device__ double evaluate_raw(double* x, int d, int fid) {')
    return base_src.replace(sig, patch, 1)


def exact_constants_handles():
    """Kernel handles of the module compiled with exact constants (cached)."""
    if 'exact' not in _cache:
        _cache['exact'] = build_module(exact_constants_source(), 'exact constants')[1]
    return _cache['exact']


# ---------------------------------------------------------------------------
# Selection dispatcher
# ---------------------------------------------------------------------------
def _derive_seed(seed, salt=0):
    if seed is None:
        seed = int(time.time() * 1000)
    return np.uint32((int(seed) + salt) % (2**31 - 1))


_cache = {}


def _gbest_buffers():
    if 'gbi' not in _cache:
        _cache['gbi'] = gpuarray.zeros(1, np.int32)
        _cache['gbv'] = gpuarray.zeros(1, np.float64)
    return _cache['gbi'], _cache['gbv']


def _cumsum_buffer(ps):
    key = ('cum', int(ps))
    if key not in _cache:
        _cache[key] = gpuarray.zeros(int(ps), np.float64)
    return _cache[key]


def _scan_kernel():
    if 'scan' not in _cache:
        from pycuda.scan import InclusiveScanKernel
        _cache['scan'] = InclusiveScanKernel(np.float64, "a+b")
    return _cache['scan']


def argmin_device(d_obj, ps):
    gbi, gbv = _gbest_buffers()
    SELK['argmin_block'](d_obj, np.int32(ps), gbi, gbv, block=(1024, 1, 1), grid=(1, 1))
    return gbi


def fill_selection(method, d_obj, d_fit, d_sel, d_avg, ps, ts, alpha, seed):
    """Write the selected index of every individual to d_sel."""
    g = grid(ps)
    s32 = seed32(seed)
    if method == 'tournament':
        k_sel_tournament(d_obj, d_sel, np.int32(ps), np.int32(ts), np.uint32(seed),
                         block=(BS, 1, 1), grid=g)
    elif method == 'tournament_safe':
        SELK['sel_tournament_safe'](d_obj, d_sel, np.int32(ps), np.int32(ts), s32, block=(BS, 1, 1), grid=g)
    elif method in ('ballot', 'ballot_noswitch'):
        assert ps % 32 == 0, "ballot selection requires a population size that is a multiple of 32"
        k_compute_fit(d_obj, d_fit, np.int32(ps), block=(BS, 1, 1), grid=g)
        k_reset_avg(d_avg, block=(1, 1, 1), grid=(1, 1))
        k_compute_avg(d_fit, d_avg, np.int32(ps), block=(BS, 1, 1), grid=g, shared=BS * 8)
        k_finalize_avg(d_avg, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        kern = k_sel_ballot if method == 'ballot' else k_sel_ballot_nosw
        kern(d_fit, d_sel, d_avg, np.int32(ps), np.int32(ts), np.float64(alpha), np.uint32(seed),
             block=(BS, 1, 1), grid=g)
    elif method in ('ballot_safe', 'ballot_noswitch_safe'):
        k_compute_fit(d_obj, d_fit, np.int32(ps), block=(BS, 1, 1), grid=g)
        k_reset_avg(d_avg, block=(1, 1, 1), grid=(1, 1))
        k_compute_avg(d_fit, d_avg, np.int32(ps), block=(BS, 1, 1), grid=g, shared=BS * 8)
        k_finalize_avg(d_avg, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        kern = SELK['sel_ballot_elite_safe'] if method == 'ballot_safe' else SELK['sel_ballot_noswitch_safe']
        kern(d_fit, d_sel, d_avg, np.int32(ps), np.int32(ts), np.float64(alpha), s32, block=(BS, 1, 1), grid=g)
    elif method == 'ballot_local':
        assert ps % 32 == 0, "ballot selection requires a population size that is a multiple of 32"
        k_compute_fit(d_obj, d_fit, np.int32(ps), block=(BS, 1, 1), grid=g)
        k_sel_ballot_local(d_fit, d_sel, np.int32(ps), np.int32(ts), np.float64(alpha), np.uint32(seed),
                           block=(BS, 1, 1), grid=g)
    elif method == 'ballot_local_safe':
        k_compute_fit(d_obj, d_fit, np.int32(ps), block=(BS, 1, 1), grid=g)
        SELK['sel_ballot_local_safe'](d_fit, d_sel, np.int32(ps), np.int32(ts), np.float64(alpha), s32,
                                      block=(BS, 1, 1), grid=g)
    elif method == 'tournament_global':
        k_sel_tourn_global(d_obj, d_sel, np.int32(ps), np.int32(ts), np.uint32(seed),
                           block=(BS, 1, 1), grid=g)
    elif re.fullmatch(r'global\d+', method):
        k = int(method[6:])
        k_sel_tourn_global(d_obj, d_sel, np.int32(ps), np.int32(k), s32, block=(BS, 1, 1), grid=g)
    elif method == 'tournament_elitist':
        # the global-best index goes through the host (one synchronisation per call)
        d_gb_idx = gpuarray.zeros(1, np.int32)
        d_gb_val = gpuarray.zeros(1, np.float64)
        k_find_best(d_obj, d_gb_idx, d_gb_val, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        gb = int(d_gb_idx.get()[0])
        k_sel_tourn_elite(d_obj, d_sel, np.int32(ps), np.int32(ts), np.int32(gb),
                          np.uint32(seed), block=(BS, 1, 1), grid=g)
    elif method == 'elitist_dev':
        gbi = argmin_device(d_obj, ps)
        SELK['sel_tournament_elitist_dev'](d_obj, d_sel, np.int32(ps), np.int32(ts), gbi, s32,
                                           block=(BS, 1, 1), grid=g)
    elif method == 'gbest':
        gbi = argmin_device(d_obj, ps)
        SELK['sel_globalbest'](d_sel, np.int32(ps), gbi, block=(BS, 1, 1), grid=g)
    elif method.startswith('mig_'):
        rho = float(method[4:])
        gbi = argmin_device(d_obj, ps)
        SELK['sel_tournament_migrate'](d_obj, d_sel, np.int32(ps), np.int32(ts), gbi, np.float64(rho), s32,
                                       block=(BS, 1, 1), grid=g)
    elif method == 'roulette_scan':
        k_compute_fit(d_obj, d_fit, np.int32(ps), block=(BS, 1, 1), grid=g)
        cum = _cumsum_buffer(ps)
        cuda.memcpy_dtod(cum.gpudata, d_fit.gpudata, int(ps) * 8)
        _scan_kernel()(cum)
        SELK['roulette_sample'](cum, d_sel, np.int32(ps), s32, block=(BS, 1, 1), grid=g)
    else:
        raise ValueError(f"unknown selection method: {method}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _pop_diversity(d_pop, ps, dim, lb, ub):
    pop = d_pop.get().reshape(ps, dim)
    return float(np.mean(np.std(pop, axis=0)) / (ub - lb))


class _GpuBuffers:
    """Device buffers of one run."""
    def __init__(self, ps, dim):
        self.d_pop  = gpuarray.zeros(ps * dim, np.float64)
        self.d_pop2 = gpuarray.zeros(ps * dim, np.float64)
        self.d_obj  = gpuarray.zeros(ps, np.float64)
        self.d_fit  = gpuarray.zeros(ps, np.float64)
        self.d_sel  = gpuarray.zeros(ps, np.int32)
        self.d_avg  = gpuarray.zeros(1, np.float64)
        self.d_bi   = gpuarray.zeros(1, np.int32)
        self.d_bv   = gpuarray.zeros(1, np.float64)
        self.d_best = gpuarray.to_gpu(np.array([np.inf], dtype=np.float64))
        self.d_guide = gpuarray.zeros(dim, np.float64)
        self.d_gscore = gpuarray.to_gpu(np.array([np.inf], dtype=np.float64))


def _track(b, conv, div, it, every, track_conv, track_div, ps, dim, lb, ub):
    if (track_conv or track_div) and it % every == 0:
        if track_conv:
            conv.append((it, float(b.d_best.get()[0])))
        if track_div:
            div.append((it, _pop_diversity(b.d_pop, ps, dim, lb, ub)))


def run_gwo_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    lb, ub = np.float64(bounds[0]), np.float64(bounds[1])
    s = _derive_seed(seed)
    b = _GpuBuffers(ps, dim)
    d_leaders = gpuarray.zeros(3, np.int32)
    d_lead_pos = gpuarray.zeros(3 * dim, np.float64)
    g = grid(ps); gd = grid(dim)
    conv, div = [], []
    cuda.Context.synchronize()
    t0 = time.time()
    k_pop_init(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), np.uint32(s + 1), block=(BS, 1, 1), grid=g)
    for it in range(iters):
        k_boundary(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), block=(BS, 1, 1), grid=g)
        k_eval_pop(b.d_pop, b.d_obj, np.int32(ps), np.int32(dim), np.int32(fid), block=(BS, 1, 1), grid=g)
        k_find_best(b.d_obj, b.d_bi, b.d_bv, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        k_update_best(b.d_bv, b.d_best, block=(1, 1, 1), grid=(1, 1))
        if method == 'naive':
            k_gwo_leaders(b.d_obj, d_leaders, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        else:
            fill_selection(method, b.d_obj, b.d_fit, b.d_sel, b.d_avg, ps, ts, alpha, s + 3000 + it)
            k_gwo_leaders_sel(b.d_obj, b.d_sel, d_leaders, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        k_gwo_copy_lead(b.d_pop, d_leaders, d_lead_pos, np.int32(dim), block=(BS, 1, 1), grid=gd)
        a = 2.0 - 2.0 * it / iters
        k_gwo_update(b.d_pop, b.d_pop2, d_lead_pos, lb, ub, np.int32(ps), np.int32(dim),
                     np.float64(a), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)
        b.d_pop, b.d_pop2 = b.d_pop2, b.d_pop
        _track(b, conv, div, it, track_every, track_convergence, track_diversity, ps, dim, lb, ub)
    k_boundary(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), block=(BS, 1, 1), grid=g)
    k_eval_pop(b.d_pop, b.d_obj, np.int32(ps), np.int32(dim), np.int32(fid), block=(BS, 1, 1), grid=g)
    k_find_best(b.d_obj, b.d_bi, b.d_bv, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
    k_update_best(b.d_bv, b.d_best, block=(1, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()
    t = (time.time() - t0) * 1000
    return {'best': float(b.d_best.get()[0]), 'time_ms': t, 'convergence': conv, 'diversity': div}


def _run_guide_alg(update_naive, update_sel, ps, iters, fid, bounds, dim, method, ts, alpha,
                   seed, track_convergence, track_diversity, track_every, param_fn,
                   needs_mean=False, needs_obj=False):
    """Common loop of WOA, SCA, SSA and HHO (guide: best position found so far)."""
    lb, ub = np.float64(bounds[0]), np.float64(bounds[1])
    s = _derive_seed(seed)
    b = _GpuBuffers(ps, dim)
    d_mean = gpuarray.zeros(dim, np.float64) if needs_mean else None
    g = grid(ps); gd = grid(dim)
    conv, div = [], []
    cuda.Context.synchronize()
    t0 = time.time()
    k_pop_init(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), np.uint32(s + 1), block=(BS, 1, 1), grid=g)
    for it in range(iters):
        k_boundary(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), block=(BS, 1, 1), grid=g)
        k_eval_pop(b.d_pop, b.d_obj, np.int32(ps), np.int32(dim), np.int32(fid), block=(BS, 1, 1), grid=g)
        k_find_best(b.d_obj, b.d_bi, b.d_bv, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        k_update_best(b.d_bv, b.d_best, block=(1, 1, 1), grid=(1, 1))
        k_update_guide(b.d_pop, b.d_bi, b.d_bv, b.d_guide, b.d_gscore, np.int32(dim),
                       block=(1, 1, 1), grid=(1, 1))
        if needs_mean:
            k_mean_pos(b.d_pop, d_mean, np.int32(ps), np.int32(dim), block=(BS, 1, 1), grid=gd)
        params = param_fn(it, iters, s)
        if method == 'naive':
            update_naive(b, d_mean, lb, ub, ps, dim, fid, params, s, it, g)
        else:
            fill_selection(method, b.d_obj, b.d_fit, b.d_sel, b.d_avg, ps, ts, alpha, s + 3000 + it)
            update_sel(b, d_mean, lb, ub, ps, dim, fid, params, s, it, g)
        b.d_pop, b.d_pop2 = b.d_pop2, b.d_pop
        _track(b, conv, div, it, track_every, track_convergence, track_diversity, ps, dim, lb, ub)
    k_boundary(b.d_pop, lb, ub, np.int32(ps), np.int32(dim), block=(BS, 1, 1), grid=g)
    k_eval_pop(b.d_pop, b.d_obj, np.int32(ps), np.int32(dim), np.int32(fid), block=(BS, 1, 1), grid=g)
    k_find_best(b.d_obj, b.d_bi, b.d_bv, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
    k_update_best(b.d_bv, b.d_best, block=(1, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()
    t = (time.time() - t0) * 1000
    return {'best': float(b.d_best.get()[0]), 'time_ms': t, 'convergence': conv, 'diversity': div}


def run_woa_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    def params(it, iters, s):
        return (2.0 - 2.0 * it / iters, -1.0 - it / iters)

    def upd_naive(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_woa_update(b.d_pop, b.d_pop2, b.d_guide, lb, ub, np.int32(ps), np.int32(dim),
                     np.float64(p[0]), np.float64(p[1]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)

    def upd_sel(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_woa_update_sel(b.d_pop, b.d_pop2, b.d_sel, lb, ub, np.int32(ps), np.int32(dim),
                         np.float64(p[0]), np.float64(p[1]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)
    return _run_guide_alg(upd_naive, upd_sel, ps, iters, fid, bounds, dim, method, ts, alpha,
                          seed, track_convergence, track_diversity, track_every, params)


def run_sca_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    def params(it, iters, s):
        return (2.0 - 2.0 * (it + 1) / iters,)

    def upd_naive(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_sca_update(b.d_pop, b.d_pop2, b.d_guide, lb, ub, np.int32(ps), np.int32(dim),
                     np.float64(p[0]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)

    def upd_sel(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_sca_update_sel(b.d_pop, b.d_pop2, b.d_sel, lb, ub, np.int32(ps), np.int32(dim),
                         np.float64(p[0]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)
    return _run_guide_alg(upd_naive, upd_sel, ps, iters, fid, bounds, dim, method, ts, alpha,
                          seed, track_convergence, track_diversity, track_every, params)


def run_ssa_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    def params(it, iters, s):
        return (2.0 * math.exp(-((4.0 * it / iters) ** 2)),)

    def upd_naive(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_ssa_update(b.d_pop, b.d_pop2, b.d_guide, lb, ub, np.int32(ps), np.int32(dim),
                     np.float64(p[0]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)

    def upd_sel(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_ssa_update_sel(b.d_pop, b.d_pop2, b.d_sel, lb, ub, np.int32(ps), np.int32(dim),
                         np.float64(p[0]), np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)
    return _run_guide_alg(upd_naive, upd_sel, ps, iters, fid, bounds, dim, method, ts, alpha,
                          seed, track_convergence, track_diversity, track_every, params)


def run_hho_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    def params(it, iters, s):
        return (2.0 * (1.0 - it / iters),)

    def upd_naive(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_hho_update(b.d_pop, b.d_pop2, b.d_obj, b.d_guide, d_mean, lb, ub,
                     np.int32(ps), np.int32(dim), np.float64(p[0]), np.int32(fid),
                     np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)

    def upd_sel(b, d_mean, lb, ub, ps, dim, fid, p, s, it, g):
        k_hho_update_sel(b.d_pop, b.d_pop2, b.d_obj, b.d_sel, d_mean, lb, ub,
                         np.int32(ps), np.int32(dim), np.float64(p[0]), np.int32(fid),
                         np.uint32(s + 7000 + it), block=(BS, 1, 1), grid=g)
    return _run_guide_alg(upd_naive, upd_sel, ps, iters, fid, bounds, dim, method, ts, alpha,
                          seed, track_convergence, track_diversity, track_every, params,
                          needs_mean=True, needs_obj=True)


def run_abc_gpu(ps, iters, fid, bounds, dim=30, method='tournament', ts=3, alpha=1.2,
                seed=None, track_convergence=False, track_diversity=False, track_every=10):
    lb, ub = np.float64(bounds[0]), np.float64(bounds[1])
    s = _derive_seed(seed)
    b = _GpuBuffers(ps, dim)
    d_trial = gpuarray.zeros(ps, np.int32)
    d_cum = gpuarray.zeros(ps, np.float64)
    d_sc = gpuarray.zeros(2, np.int32)
    limit = min(ps * dim, 100 * ps)
    g = grid(ps)
    conv, div = [], []
    cuda.Context.synchronize()
    t0 = time.time()
    k_abc_init(b.d_pop, b.d_obj, b.d_fit, d_trial, lb, ub, np.int32(ps), np.int32(dim),
               np.int32(fid), np.uint32(s + 1), block=(BS, 1, 1), grid=g)
    for it in range(iters):
        k_abc_employed(b.d_pop, b.d_obj, b.d_fit, d_trial, lb, ub, np.int32(ps), np.int32(dim),
                       np.int32(fid), np.uint32(s + 2000 + it), block=(BS, 1, 1), grid=g)
        if method == 'naive':
            k_sel_naive(b.d_fit, d_cum, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
            k_abc_onl_roul(b.d_pop, b.d_obj, b.d_fit, d_trial, d_cum, lb, ub,
                           np.int32(ps), np.int32(dim), np.int32(fid),
                           np.uint32(s + 5000 + it), block=(BS, 1, 1), grid=g)
        else:
            fill_selection(method, b.d_obj, b.d_fit, b.d_sel, b.d_avg, ps, ts, alpha, s + 3000 + it)
            k_abc_onl_sel(b.d_pop, b.d_obj, b.d_fit, d_trial, b.d_sel, lb, ub,
                          np.int32(ps), np.int32(dim), np.int32(fid),
                          np.uint32(s + 5000 + it), block=(BS, 1, 1), grid=g)
        k_abc_max_trial(d_trial, d_sc, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        k_abc_scout(b.d_pop, b.d_obj, b.d_fit, d_trial, d_sc, lb, ub, np.int32(dim),
                    np.int32(fid), np.int32(limit), np.uint32(s + 9000 + it), block=(1, 1, 1), grid=(1, 1))
        k_find_best(b.d_obj, b.d_bi, b.d_bv, np.int32(ps), block=(1, 1, 1), grid=(1, 1))
        k_update_best(b.d_bv, b.d_best, block=(1, 1, 1), grid=(1, 1))
        _track(b, conv, div, it, track_every, track_convergence, track_diversity, ps, dim, lb, ub)
    cuda.Context.synchronize()
    t = (time.time() - t0) * 1000
    return {'best': float(b.d_best.get()[0]), 'time_ms': t, 'convergence': conv, 'diversity': div}


GPU_FUNCS = {
    'ABC': run_abc_gpu, 'GWO': run_gwo_gpu, 'WOA': run_woa_gpu,
    'SCA': run_sca_gpu, 'SSA': run_ssa_gpu, 'HHO': run_hho_gpu
}

# ---------------------------------------------------------------------------
# Best-position tracking
# ---------------------------------------------------------------------------
_buffers_init = _GpuBuffers.__init__
_tracking = {'buf': None}


def _buffers_init_tracking(self, ps, dim):
    _buffers_init(self, ps, dim)
    self.dim = int(dim)
    self.d_best_pos = gpuarray.zeros(int(dim), np.float64)
    _tracking['buf'] = self


def _update_best_tracking(d_bv, d_best, block=None, grid=None):
    b = _tracking['buf']
    SELK['update_best_ever_pos'](d_bv, d_best, b.d_pop, b.d_bi, b.d_best_pos, np.int32(b.dim),
                                 block=(1, 1, 1), grid=(1, 1))


@contextlib.contextmanager
def track_best_position():
    """Inside this context the runners also record the best position (needed
    to re-evaluate a result with the CPU reference evaluator)."""
    saved = globals()['k_update_best']
    try:
        _GpuBuffers.__init__ = _buffers_init_tracking
        globals()['k_update_best'] = _update_best_tracking
        yield
    finally:
        _GpuBuffers.__init__ = _buffers_init
        globals()['k_update_best'] = saved
        _tracking['buf'] = None


def run_with_position(alg, ps, iters, fid, bnd, dim, method, seed):
    """Run an algorithm and also return the best position found."""
    r = GPU_FUNCS[alg](ps, iters, fid, bnd, dim=dim, method=method, seed=seed)
    b = _tracking['buf']
    pos = b.d_best_pos.get().copy()
    return r, pos


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def time_events(fn, budget_s=0.5, min_reps=5, max_reps=2000):
    """Mean time of fn() in microseconds, measured with CUDA events."""
    for _ in range(3):
        fn()
    cuda.Context.synchronize()
    s, e = cuda.Event(), cuda.Event()
    s.record(); fn(); e.record(); e.synchronize()
    one = max(s.time_till(e) / 1000.0, 1e-6)
    reps = int(min(max_reps, max(min_reps, budget_s / one)))
    s.record()
    for _ in range(reps):
        fn()
    e.record(); e.synchronize()
    return s.time_till(e) * 1000.0 / reps, reps


_warm = False


def warmup():
    """Launch every runner once so that timings exclude first-call overheads."""
    global _warm
    if _warm:
        return
    _warm = True
    for alg in GPU_FUNCS:
        for m in ['naive', 'tournament', 'ballot']:
            GPU_FUNCS[alg](64, 3, 0, (-100, 100), dim=5, method=m, seed=1)
