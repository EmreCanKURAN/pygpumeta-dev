/*
 * Boundary-safe warp-level selection kernels, the global-best migration
 * kernels and the baseline operators used in the operator benchmarks.
 *
 * Compiled as a separate module with PyCUDA (see gpumeta/gpu.py).
 */
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ double pos_inf() { return __longlong_as_double(0x7ff0000000000000LL); }
__device__ __forceinline__ unsigned int xorshift32(unsigned int x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
__device__ __forceinline__ double randf(unsigned int* st) {
    *st = xorshift32(*st); return (*st & 0x7FFFFFFF) / 2147483647.0;
}

/* ---------------------------------------------------------------------------
   Boundary-safe warp kernels (Algorithms 1 and 2 of the paper). No thread
   returns before a warp collective: inactive lanes (idx >= n) take part in
   every __shfl/__ballot with a neutral value (+inf objective, -inf fitness,
   never elite) and only active lanes store a result. For n % 32 == 0 the
   output is identical to the kernels in metaheuristics.cu (same random
   stream, same comparisons).
   ------------------------------------------------------------------------- */
extern "C" __global__ void sel_tournament_safe(const double* obj, int* sel, int n, int ts, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double myObj = active ? obj[idx] : pos_inf();
    int win = idx; double bestObj = myObj;
    for (int t = 0; t < ts; t++) {
        int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
        double nb = __shfl_xor_sync(FULL_MASK, myObj, off);
        int ni = idx - lane + (lane ^ off);
        if (ni >= 0 && ni < n && nb < bestObj) { bestObj = nb; win = ni; }
    }
    if (active) sel[idx] = win;
}

extern "C" __global__ void sel_ballot_elite_safe(const double* fit, int* sel, const double* avg_fit, int n, int ts,
                                                 double alpha, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double my_fit = active ? fit[idx] : -pos_inf();
    double threshold = avg_fit[0] * alpha;
    int is_elite = (active && my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(FULL_MASK, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx; double best_fit = my_fit;
    if (elite_count >= 2) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(FULL_MASK, is_elite, off);
            if (ni >= 0 && ni < n && ne && nf > best_fit) { best_fit = nf; winner = ni; }
        }
        if (winner == idx && !is_elite) {
            int fel = __ffs(elite_mask) - 1;
            if (fel >= 0) winner = idx - lane + fel;
        }
    } else if (elite_count == 1) {
        int el = __ffs(elite_mask) - 1;
        winner = idx - lane + el;
    } else {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    if (active) sel[idx] = winner;
}

extern "C" __global__ void sel_ballot_local_safe(const double* fit, int* sel, int n, int ts, double alpha,
                                                 unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double raw = active ? fit[idx] : 0.0;            /* neutral for the warp sum */
    unsigned int amask = __ballot_sync(FULL_MASK, active ? 1 : 0);
    int acount = __popc(amask);
    double wsum = raw;
    for (int off = 16; off > 0; off >>= 1)
        wsum += __shfl_xor_sync(FULL_MASK, wsum, off);
    double denom = (acount > 0) ? (double)acount : 1.0;
    double threshold = (wsum / denom) * alpha;
    double my_fit = active ? raw : -pos_inf();
    int is_elite = (active && my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(FULL_MASK, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx; double best_fit = my_fit;
    if (elite_count >= 2) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(FULL_MASK, is_elite, off);
            if (ni >= 0 && ni < n && ne && nf > best_fit) { best_fit = nf; winner = ni; }
        }
        if (winner == idx && !is_elite) {
            int fel = __ffs(elite_mask) - 1;
            if (fel >= 0) winner = idx - lane + fel;
        }
    } else if (elite_count == 1) {
        int el = __ffs(elite_mask) - 1;
        winner = idx - lane + el;
    } else {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    if (active) sel[idx] = winner;
}

extern "C" __global__ void sel_ballot_noswitch_safe(const double* fit, int* sel, const double* avg_fit, int n, int ts,
                                                    double alpha, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double my_fit = active ? fit[idx] : -pos_inf();
    double threshold = avg_fit[0] * alpha;
    int is_elite = (active && my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(FULL_MASK, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx; double best_fit = my_fit;
    if (elite_count > 0) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(FULL_MASK, is_elite, off);
            if (ni >= 0 && ni < n && ne && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    } else {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(FULL_MASK, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    if (active) sel[idx] = winner;
}

/* ---------------------------------------------------------------------------
   Global-best family. The index gb is read from device memory, so no host
   round trip is needed.
   elitist_dev : elitist rule of sel_tournament_elitist with a device-side gb
   globalbest  : direct global-best selection, sel[i] = gb for all i
   migrate     : warp tournament; with probability rho a thread also compares
                 its winner with gb (rho = 0: plain warp tournament,
                 rho = 1: elitist rule, i.e. global-best selection).
   ------------------------------------------------------------------------- */
extern "C" __global__ void sel_tournament_elitist_dev(const double* obj, int* sel, int n, int ts,
                                                      const int* gb_ptr, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double myObj = active ? obj[idx] : pos_inf();
    int win = idx; double bestObj = myObj;
    for (int t = 0; t < ts; t++) {
        int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
        double nb = __shfl_xor_sync(FULL_MASK, myObj, off);
        int ni = idx - lane + (lane ^ off);
        if (ni >= 0 && ni < n && nb < bestObj) { bestObj = nb; win = ni; }
    }
    if (active) {
        int gb = gb_ptr[0];
        if (obj[gb] < bestObj) win = gb;
        sel[idx] = win;
    }
}

extern "C" __global__ void sel_globalbest(int* sel, int n, const int* gb_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) sel[idx] = gb_ptr[0];
}

extern "C" __global__ void sel_tournament_migrate(const double* obj, int* sel, int n, int ts,
                                                  const int* gb_ptr, double rho, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = idx < n;
    unsigned int st = seed + idx * 54321;
    st = xorshift32(st);
    double myObj = active ? obj[idx] : pos_inf();
    int win = idx; double bestObj = myObj;
    for (int t = 0; t < ts; t++) {
        int off = ((st >> (t * 5)) & 31); if (off == 0) off = 1;
        double nb = __shfl_xor_sync(FULL_MASK, myObj, off);
        int ni = idx - lane + (lane ^ off);
        if (ni >= 0 && ni < n && nb < bestObj) { bestObj = nb; win = ni; }
    }
    unsigned int s2 = (seed ^ 0x9E3779B9u) + (unsigned int)idx * 2654435761u;
    s2 = xorshift32(s2);
    s2 = xorshift32(s2);
    double u = (s2 & 0x7FFFFFFF) / 2147483647.0;
    if (active) {
        if (rho >= 1.0 || u < rho) {
            int gb = gb_ptr[0];
            if (obj[gb] < bestObj) win = gb;
        }
        sel[idx] = win;
    }
}

/* Parallel arg-min (single block, 1024 threads, grid-stride). Ties -> lowest
   index, i.e. the same result as the serial find_best_idx kernel. */
extern "C" __global__ void argmin_block(const double* obj, int n, int* out_idx, double* out_val) {
    __shared__ double sv[1024];
    __shared__ int si[1024];
    int t = threadIdx.x;
    double bv = pos_inf(); int bi = 0x7fffffff;
    for (int i = t; i < n; i += blockDim.x) {
        double v = obj[i];
        if (v < bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            double v = sv[t + s]; int i = si[t + s];
            if (v < sv[t] || (v == sv[t] && i < si[t])) { sv[t] = v; si[t] = i; }
        }
        __syncthreads();
    }
    if (t == 0) { out_idx[0] = (si[0] == 0x7fffffff) ? 0 : si[0]; out_val[0] = sv[0]; }
}

/* Roulette sampling by binary search on an (unnormalised) inclusive prefix sum */
extern "C" __global__ void roulette_sample(const double* cum, int* sel, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int st = seed + idx * 98765 + 23;
    st = xorshift32(st);
    double r = randf(&st) * cum[n - 1];
    int l = 0, ri = n - 1;
    while (l < ri) { int m = (l + ri) >> 1; if (cum[m] < r) l = m + 1; else ri = m; }
    sel[idx] = l;
}

/* Conventional block-level tournament through shared memory (baseline) */
extern "C" __global__ void sel_tournament_shared(const double* obj, int* sel, int n, int ts, unsigned int seed) {
    extern __shared__ double sobj[];
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    int idx = base + t;
    sobj[t] = (idx < n) ? obj[idx] : pos_inf();
    __syncthreads();
    if (idx >= n) return;
    int blen = min((int)blockDim.x, n - base);
    unsigned int st = xorshift32(seed + idx * 54321);
    int win = idx; double best = sobj[t];
    for (int r = 0; r < ts; r++) {
        st = xorshift32(st);
        int c = (int)(st % (unsigned int)blen);
        double v = sobj[c];
        if (v < best) { best = v; win = base + c; }
    }
    sel[idx] = win;
}

/* Best-so-far value and position (needed when the result is re-evaluated on the CPU) */
extern "C" __global__ void update_best_ever_pos(const double* cur_val, double* best_ever, const double* pop,
                                                const int* bi, double* best_pos, int dim) {
    if (threadIdx.x || blockIdx.x) return;
    if (cur_val[0] < best_ever[0]) {
        best_ever[0] = cur_val[0];
        long long b = bi[0];
        for (int j = 0; j < dim; j++) best_pos[j] = pop[b * dim + j];
    }
}
