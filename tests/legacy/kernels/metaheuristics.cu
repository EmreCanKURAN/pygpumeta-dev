/*
 * CUDA kernels of the six metaheuristics (ABC, GWO, WOA, SCA, SSA, HHO),
 * the benchmark functions and the selection operators.
 *
 * The file is compiled at run time with PyCUDA (see gpumeta/gpu.py).
 */
#define WARP_SIZE 32
#define MAX_DIM 1024

__device__ int d_fitness_cost = 1;

/* Shift vector and rotation matrix of a CEC function, copied from the host.
   Only used by the on-device CEC throughput test. */
#define CEC_GPU_MAXDIM 128
__device__ double d_cecg_shift[CEC_GPU_MAXDIM];
__device__ double d_cecg_rot[CEC_GPU_MAXDIM * CEC_GPU_MAXDIM];
__device__ double d_cecg_fopt = 0.0;

__device__ unsigned int xorshift(unsigned int* st) {
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

__device__ double randf(unsigned int* st) { return (xorshift(st) & 0x7FFFFFFF) / 2147483647.0; }
__device__ int randi(unsigned int* st, int m) { return xorshift(st) % m; }

__device__ double randn(unsigned int* st) {
    double u1 = randf(st) + 1e-10;
    double u2 = randf(st);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

/* ---------------- classical benchmark functions ---------------- */
__device__ double sphere(double* x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += x[i] * x[i]; return s;
}

__device__ double rastrigin(double* x, int d) {
    double sum_sq = 0.0, sum_sin = 0.0;
    for (int i = 0; i < d; i++) {
        sum_sq += x[i] * x[i];
        double sv = sin(3.14159265358979323846 * x[i]);
        sum_sin += sv * sv;
    }
    return sum_sq + 20.0 * sum_sin;
}

__device__ double rosenbrock(double* x, int d) {
    double s = 0;
    for (int i = 0; i < d-1; i++) {
        double t1 = x[i+1] - x[i]*x[i]; double t2 = x[i] - 1.0;
        s += 100.0*t1*t1 + t2*t2;
    }
    return s;
}

__device__ double ackley(double* x, int d) {
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < d; i++) {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * 3.14159265 * x[i]);
    }
    return -20.0 * exp(-0.2 * sqrt(sum1 / d)) - exp(sum2 / d) + 20.0 + 2.71828183;
}

__device__ double griewank(double* x, int d) {
    double sum_part = 0, log_prod = 0.0;
    for (int i = 0; i < d; i++) {
        sum_part += x[i] * x[i];
        double theta = x[i] / sqrt((double)(i + 1));
        double cv = cos(theta);
        if (cv > 0.9999999999) log_prod += -theta * theta / 2.0;
        else if (cv > 0) log_prod += log(cv);
        else log_prod = -1e100;
    }
    double omp;
    if (log_prod > -1e-10) omp = -log_prod;
    else omp = 1.0 - exp(log_prod);
    return sum_part / 4000.0 + omp;
}

__device__ double schwefel(double* x, int d) {
    double s = 418.9829 * d;
    for (int i = 0; i < d; i++) s -= x[i] * sin(sqrt(fabs(x[i])));
    return s;
}

__device__ double levyf(double* x, int d) {
    double w[MAX_DIM];
    for (int i = 0; i < d; i++) w[i] = 1.0 + (x[i] - 1.0) / 4.0;
    double term1 = sin(3.14159265 * w[0]) * sin(3.14159265 * w[0]);
    double term3 = (w[d-1] - 1.0) * (w[d-1] - 1.0) * (1.0 + sin(2.0 * 3.14159265 * w[d-1]) * sin(2.0 * 3.14159265 * w[d-1]));
    double sp = 0;
    for (int i = 0; i < d - 1; i++)
        sp += (w[i] - 1.0) * (w[i] - 1.0) * (1.0 + 10.0 * sin(3.14159265 * w[i] + 1.0) * sin(3.14159265 * w[i] + 1.0));
    return term1 + sp + term3;
}

__device__ double zakharov(double* x, int d) {
    double s1 = 0, s2 = 0;
    for (int i = 0; i < d; i++) { s1 += x[i] * x[i]; s2 += 0.5 * (i + 1) * x[i]; }
    return s1 + s2 * s2 + s2 * s2 * s2 * s2;
}

__device__ double sum_squares(double* x, int d) {
    double s = 0;
    for (int i = 0; i < d; i++) s += (i + 1) * x[i] * x[i];
    return s;
}

__device__ double michalewicz(double* x, int d) {
    double s = 0;
    for (int i = 0; i < d; i++) {
        double sx = sin(x[i]);
        double inner = (i + 1) * x[i] * x[i] / 3.14159265;
        double si = sin(inner);
        double pw = 1.0;
        for (int j = 0; j < 10; j++) pw *= si * si;
        s += sx * pw;
    }
    return -s + d;
}

/* Sphere whose evaluation cost scales with d_fitness_cost (fid 60) */
__device__ double sphere_expensive(double* x, int d) {
    double s = 0.0;
    for (int r = 0; r < d_fitness_cost; r++) {
        double t = 0.0;
        for (int i = 0; i < d; i++) t += x[i] * x[i] + 1e-15 * sin((double)(r + 1) * x[i]);
        s = t;
    }
    return s;
}

/* Shifted and rotated functions evaluated on the device (fid 200..205):
   z = M (x - o), then a base function, plus the optimum value f_opt. */
__device__ void cecg_transform(double* x, double* z, int d) {
    double y[CEC_GPU_MAXDIM];
    for (int i = 0; i < d; i++) y[i] = x[i] - d_cecg_shift[i];
    for (int i = 0; i < d; i++) {
        double s = 0.0;
        for (int j = 0; j < d; j++) s += d_cecg_rot[i * d + j] * y[j];
        z[i] = s;
    }
}

__device__ double cecg_bent_cigar(double* z, int d) {
    double s = z[0] * z[0];
    for (int i = 1; i < d; i++) s += 1e6 * z[i] * z[i];
    return s;
}

__device__ double evaluate(double* x, int d, int fid) {
    if (fid == 0) return sphere(x, d);
    else if (fid == 1) return rastrigin(x, d);
    else if (fid == 2) return rosenbrock(x, d);
    else if (fid == 3) return ackley(x, d);
    else if (fid == 4) return griewank(x, d);
    else if (fid == 5) return schwefel(x, d);
    else if (fid == 6) return levyf(x, d);
    else if (fid == 7) return zakharov(x, d);
    else if (fid == 8) return sum_squares(x, d);
    else if (fid == 9) return michalewicz(x, d);
    else if (fid == 60) return sphere_expensive(x, d);
    else if (fid >= 200 && fid <= 205) {
        double z[CEC_GPU_MAXDIM];
        cecg_transform(x, z, d);
        double base;
        if (fid == 200) base = cecg_bent_cigar(z, d);           /* CEC2017 F1 */
        else if (fid == 201) { for (int i=0;i<d;i++) z[i]*=0.0512; base = rastrigin(z, d); }  /* F5-like */
        else if (fid == 202) base = ackley(z, d);               /* F6-like  */
        else if (fid == 203) { for (int i=0;i<d;i++) z[i]*=6.0;  base = griewank(z, d); }     /* F7-like */
        else if (fid == 204) { for (int i=0;i<d;i++) z[i]*=0.02048; base = rosenbrock(z, d); } /* F4-like */
        else base = zakharov(z, d);                              /* F3-like */
        return base + d_cecg_fopt;
    }
    else return sphere(x, d);
}

/* ======================= SELECTION MECHANISMS ======================= */

extern "C" __global__ void sel_naive(double* fit, double* cum, int n) {
    if (threadIdx.x || blockIdx.x) return;
    double t = 0; for (int i = 0; i < n; i++) t += fit[i];
    double c = 0; for (int i = 0; i < n; i++) { c += fit[i]/t; cum[i] = c; }
}

__device__ int binary_search_sel(double* cum, int n, double r) {
    int l = 0, ri = n - 1;
    while (l < ri) { int m = (l + ri) >> 1; if (cum[m] < r) l = m + 1; else ri = m; }
    return l;
}

/* GPU Tournament: competitors are exchanged inside the warp with
   __shfl_xor_sync (minimisation on obj). Threads with idx >= n leave early,
   so this version expects n to be a multiple of 32; selection_safe.cu holds
   the boundary-safe version for arbitrary n. */
extern "C" __global__ void sel_tournament(double* obj, int* sel, int n, int ts, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double myObj = obj[idx]; int win = idx; double bestObj = myObj;
    for (int t = 0; t < ts; t++) {
        int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
        double neighborObj = __shfl_xor_sync(0xffffffff, myObj, off);
        int ni = idx - lane + (lane ^ off);
        if (ni >= 0 && ni < n && neighborObj < bestObj) { bestObj = neighborObj; win = ni; }
    }
    sel[idx] = win;
}

/* Reference: classical tournament with contestants drawn from the whole
   population (global-memory gathers). */
extern "C" __global__ void sel_tournament_global(double* obj, int* sel, int n, int ts, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int win = randi(&st, n);
    double best = obj[win];
    for (int t = 1; t < ts; t++) {
        int c = randi(&st, n);
        double oc = obj[c];
        if (oc < best) { best = oc; win = c; }
    }
    sel[idx] = win;
}

extern "C" __global__ void compute_avg_fitness(double* fit, double* avg_fit, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? fit[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(avg_fit, sdata[0]);
}

extern "C" __global__ void finalize_avg_fitness(double* avg_fit, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) avg_fit[0] = avg_fit[0] / (double)n;
}

extern "C" __global__ void reset_avg(double* avg) {
    if (threadIdx.x == 0 && blockIdx.x == 0) avg[0] = 0.0;
}

/* GPU Ballot-Elite: elite mask from __ballot_sync against alpha times the
   global mean fitness, strategy chosen by the elite count of the warp. */
extern "C" __global__ void sel_ballot_elite(double* fit, int* sel, double* avg_fit, int n, int ts, double elite_threshold, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double my_fit = fit[idx];
    double threshold = avg_fit[0] * elite_threshold;
    int is_elite = (my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(0xffffffff, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx;
    double best_fit = my_fit;
    if (elite_count >= 2) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(0xffffffff, is_elite, off);
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
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    sel[idx] = winner;
}

/* Ballot-Local: Ballot-Elite with a warp-local mean. The threshold is computed
   in registers with a shuffle reduction. Requires n to be a multiple of 32. */
extern "C" __global__ void sel_ballot_local(double* fit, int* sel, int n, int ts, double elite_threshold, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double my_fit = fit[idx];
    double wsum = my_fit;
    for (int off = 16; off > 0; off >>= 1)
        wsum += __shfl_xor_sync(0xffffffff, wsum, off);
    double threshold = (wsum / 32.0) * elite_threshold;
    int is_elite = (my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(0xffffffff, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx;
    double best_fit = my_fit;
    if (elite_count >= 2) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(0xffffffff, is_elite, off);
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
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    sel[idx] = winner;
}

/* Ballot-NoSwitch: ballot mask and global mean threshold without the adaptive
   strategy switch (elite-restricted tournament; plain tournament only when
   the warp contains no elite). */
extern "C" __global__ void sel_ballot_noswitch(double* fit, int* sel, double* avg_fit, int n, int ts, double elite_threshold, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double my_fit = fit[idx];
    double threshold = avg_fit[0] * elite_threshold;
    int is_elite = (my_fit > threshold) ? 1 : 0;
    unsigned int elite_mask = __ballot_sync(0xffffffff, is_elite);
    int elite_count = __popc(elite_mask);
    int winner = idx;
    double best_fit = my_fit;
    if (elite_count > 0) {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            int ne = __shfl_xor_sync(0xffffffff, is_elite, off);
            if (ni >= 0 && ni < n && ne && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    } else {
        for (int t = 0; t < ts; t++) {
            int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
            double nf = __shfl_xor_sync(0xffffffff, my_fit, off);
            int ni = idx - lane + (lane ^ off);
            if (ni >= 0 && ni < n && nf > best_fit) { best_fit = nf; winner = ni; }
        }
    }
    sel[idx] = winner;
}

/* Gather: new_obj[i] = obj[sel[i]] (selection-only reproduction, takeover test) */
extern "C" __global__ void gather_obj(double* src, double* dst, int* sel, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[sel[idx]];
}

/* Elitist tournament: the warp-tournament winner is compared with the global
   best (index gb, supplied by the host) and the better one is kept. Since the
   global best beats every warp winner, the result equals direct global-best
   selection; the kernel is kept as a reference variant. */
extern "C" __global__ void sel_tournament_elitist(double* obj, int* sel, int n, int ts,
                                                  int gb, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    unsigned int st = seed + idx * 54321;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double myObj = obj[idx]; int win = idx; double bestObj = myObj;
    for (int t = 0; t < ts; t++) {
        int off = ((st >> (t*5)) & 31); if (off == 0) off = 1;
        double neighborObj = __shfl_xor_sync(0xffffffff, myObj, off);
        int ni = idx - lane + (lane ^ off);
        if (ni >= 0 && ni < n && neighborObj < bestObj) { bestObj = neighborObj; win = ni; }
    }
    /* compare warp-tournament winner against the global best; keep the better */
    if (obj[gb] < bestObj) win = gb;
    sel[idx] = win;
}

/* ======================= GENERIC POPULATION KERNELS ======================= */

extern "C" __global__ void pop_init(double* pop, double lb, double ub, int ps, int dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 12345 + 7;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++)
        pop[idx * dim + j] = lb + randf(&st) * (ub - lb);
}

extern "C" __global__ void boundary_clamp(double* pop, double lb, double ub, int ps, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    for (int j = 0; j < dim; j++) {
        double v = pop[idx * dim + j];
        if (v > ub) v = ub;
        if (v < lb) v = lb;
        pop[idx * dim + j] = v;
    }
}

extern "C" __global__ void eval_pop(double* pop, double* obj, int ps, int dim, int fid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    obj[idx] = evaluate(&pop[idx * dim], dim, fid);
}

extern "C" __global__ void compute_fit(double* obj, double* fit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double o = obj[idx];
    fit[idx] = (o >= 0.0) ? 1.0 / (1.0 + o) : 1.0 + fabs(o);
}

extern "C" __global__ void find_best_idx(double* obj, int* out_idx, double* out_val, int n) {
    if (threadIdx.x || blockIdx.x) return;
    int bi = 0; double bv = obj[0];
    for (int i = 1; i < n; i++) if (obj[i] < bv) { bv = obj[i]; bi = i; }
    out_idx[0] = bi; out_val[0] = bv;
}

extern "C" __global__ void copy_individual(double* pop, int* idx_ptr, double* dst, int dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < dim) dst[j] = pop[idx_ptr[0] * dim + j];
}

extern "C" __global__ void compute_mean_pos(double* pop, double* mean_pos, int ps, int dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    double s = 0.0;
    for (int i = 0; i < ps; i++) s += pop[i * dim + j];
    mean_pos[j] = s / (double)ps;
}

/* ======================= ABC KERNELS ======================= */

extern "C" __global__ void abc_init(double* pop, double* obj, double* fit, int* trial,
                                    double lb, double ub, int ps, int dim, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 12345 + 13;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++)
        pop[idx * dim + j] = lb + randf(&st) * (ub - lb);
    double o = evaluate(&pop[idx * dim], dim, fid);
    obj[idx] = o;
    fit[idx] = (o >= 0.0) ? 1.0 / (1.0 + o) : 1.0 + fabs(o);
    trial[idx] = 0;
}

extern "C" __global__ void abc_employed(double* pop, double* obj, double* fit, int* trial,
                                        double lb, double ub, int ps, int dim, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 54321 + 17;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int j = randi(&st, dim);
    int k = randi(&st, ps);
    if (k == idx) k = (k + 1) % ps;
    double xnew[MAX_DIM];
    for (int d = 0; d < dim; d++) xnew[d] = pop[idx * dim + d];
    double phi = -1.0 + 2.0 * randf(&st);
    xnew[j] = pop[idx * dim + j] + phi * (pop[idx * dim + j] - pop[k * dim + j]);
    if (xnew[j] < lb) xnew[j] = lb;
    if (xnew[j] > ub) xnew[j] = ub;
    double no = evaluate(xnew, dim, fid);
    double nf = (no >= 0.0) ? 1.0 / (1.0 + no) : 1.0 + fabs(no);
    if (nf > fit[idx]) {
        for (int d = 0; d < dim; d++) pop[idx * dim + d] = xnew[d];
        obj[idx] = no; fit[idx] = nf; trial[idx] = 0;
    } else {
        trial[idx] = trial[idx] + 1;
    }
}

/* Onlooker, NAIVE: roulette-wheel source via binary search on cumulative fit */
extern "C" __global__ void abc_onlooker_roulette(double* pop, double* obj, double* fit, int* trial, double* cum,
                                                 double lb, double ub, int ps, int dim, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 98765 + 23;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int s = binary_search_sel(cum, ps, randf(&st));
    int j = randi(&st, dim);
    int k = randi(&st, ps);
    if (k == s) k = (k + 1) % ps;
    double xnew[MAX_DIM];
    for (int d = 0; d < dim; d++) xnew[d] = pop[s * dim + d];
    double phi = -1.0 + 2.0 * randf(&st);
    xnew[j] = pop[s * dim + j] + phi * (pop[s * dim + j] - pop[k * dim + j]);
    if (xnew[j] < lb) xnew[j] = lb;
    if (xnew[j] > ub) xnew[j] = ub;
    double no = evaluate(xnew, dim, fid);
    double nf = (no >= 0.0) ? 1.0 / (1.0 + no) : 1.0 + fabs(no);
    if (nf > fit[idx]) {
        for (int d = 0; d < dim; d++) pop[idx * dim + d] = xnew[d];
        obj[idx] = no; fit[idx] = nf; trial[idx] = 0;
    } else {
        trial[idx] = trial[idx] + 1;
    }
}

/* Onlooker, WARP-SELECTED: source index comes from sel[] (tournament/ballot) */
extern "C" __global__ void abc_onlooker_selected(double* pop, double* obj, double* fit, int* trial, int* sel,
                                                 double lb, double ub, int ps, int dim, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 98765 + 29;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int s = sel[idx];
    int j = randi(&st, dim);
    int k = randi(&st, ps);
    if (k == s) k = (k + 1) % ps;
    double xnew[MAX_DIM];
    for (int d = 0; d < dim; d++) xnew[d] = pop[s * dim + d];
    double phi = -1.0 + 2.0 * randf(&st);
    xnew[j] = pop[s * dim + j] + phi * (pop[s * dim + j] - pop[k * dim + j]);
    if (xnew[j] < lb) xnew[j] = lb;
    if (xnew[j] > ub) xnew[j] = ub;
    double no = evaluate(xnew, dim, fid);
    double nf = (no >= 0.0) ? 1.0 / (1.0 + no) : 1.0 + fabs(no);
    if (nf > fit[idx]) {
        for (int d = 0; d < dim; d++) pop[idx * dim + d] = xnew[d];
        obj[idx] = no; fit[idx] = nf; trial[idx] = 0;
    } else {
        trial[idx] = trial[idx] + 1;
    }
}

extern "C" __global__ void abc_find_max_trial(int* trial, int* out_idx, int ps) {
    if (threadIdx.x || blockIdx.x) return;
    int bi = 0, bv = trial[0];
    for (int i = 1; i < ps; i++) if (trial[i] > bv) { bv = trial[i]; bi = i; }
    out_idx[0] = bi; out_idx[1] = bv;
}

extern "C" __global__ void abc_scout(double* pop, double* obj, double* fit, int* trial, int* scout_idx,
                                     double lb, double ub, int dim, int fid, int limit, unsigned int seed) {
    if (threadIdx.x || blockIdx.x) return;
    if (scout_idx[1] <= limit) return;
    int i = scout_idx[0];
    unsigned int st = seed + i * 31337 + 41;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++)
        pop[i * dim + j] = lb + randf(&st) * (ub - lb);
    double o = evaluate(&pop[i * dim], dim, fid);
    obj[i] = o;
    fit[i] = (o >= 0.0) ? 1.0 / (1.0 + o) : 1.0 + fabs(o);
    trial[i] = 0;
}

/* Running best-so-far merge (single thread; avoids per-iteration host syncs) */
extern "C" __global__ void update_best_ever(double* cur_val, double* best_ever) {
    if (threadIdx.x || blockIdx.x) return;
    if (cur_val[0] < best_ever[0]) best_ever[0] = cur_val[0];
}

/* Historical guide (best-so-far position) update for WOA/SCA/SSA/HHO */
extern "C" __global__ void update_guide_if_better(double* pop, int* bi, double* bv,
                                                  double* guide, double* guide_score, int dim) {
    if (threadIdx.x || blockIdx.x) return;
    if (bv[0] < guide_score[0]) {
        guide_score[0] = bv[0];
        int b = bi[0];
        for (int j = 0; j < dim; j++) guide[j] = pop[b * dim + j];
    }
}

/* ======================= GWO KERNELS ======================= */

extern "C" __global__ void gwo_find_leaders(double* obj, int* leaders, int n) {
    if (threadIdx.x || blockIdx.x) return;
    int a = -1, b = -1, c = -1;
    double va = 1e300, vb = 1e300, vc = 1e300;
    for (int i = 0; i < n; i++) {
        double o = obj[i];
        if (o < va) { vc = vb; c = b; vb = va; b = a; va = o; a = i; }
        else if (o < vb) { vc = vb; c = b; vb = o; b = i; }
        else if (o < vc) { vc = o; c = i; }
    }
    leaders[0] = a; leaders[1] = (b >= 0 ? b : a); leaders[2] = (c >= 0 ? c : (b >= 0 ? b : a));
}

/* Leaders restricted to the SELECTED multiset (warp tournament / ballot output) */
extern "C" __global__ void gwo_find_leaders_sel(double* obj, int* sel, int* leaders, int n) {
    if (threadIdx.x || blockIdx.x) return;
    int a = -1, b = -1, c = -1;
    double va = 1e300, vb = 1e300, vc = 1e300;
    for (int i = 0; i < n; i++) {
        int s = sel[i];
        double o = obj[s];
        if (s == a || s == b || s == c) continue;
        if (o < va) { vc = vb; c = b; vb = va; b = a; va = o; a = s; }
        else if (o < vb) { vc = vb; c = b; vb = o; b = s; }
        else if (o < vc) { vc = o; c = s; }
    }
    leaders[0] = a; leaders[1] = (b >= 0 ? b : a); leaders[2] = (c >= 0 ? c : (b >= 0 ? b : a));
}

extern "C" __global__ void gwo_copy_leaders(double* pop, int* leaders, double* lead_pos, int dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;
    lead_pos[0 * dim + j] = pop[leaders[0] * dim + j];
    lead_pos[1 * dim + j] = pop[leaders[1] * dim + j];
    lead_pos[2 * dim + j] = pop[leaders[2] * dim + j];
}

extern "C" __global__ void gwo_update(double* pop_in, double* pop_out, double* lead_pos,
                                      double lb, double ub, int ps, int dim, double a, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 77777 + 3;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++) {
        double x = pop_in[idx * dim + j];
        double r1 = randf(&st), r2 = randf(&st);
        double A1 = 2.0 * a * r1 - a, C1 = 2.0 * r2;
        double X1 = lead_pos[j] - A1 * fabs(C1 * lead_pos[j] - x);
        r1 = randf(&st); r2 = randf(&st);
        double A2 = 2.0 * a * r1 - a, C2 = 2.0 * r2;
        double X2 = lead_pos[dim + j] - A2 * fabs(C2 * lead_pos[dim + j] - x);
        r1 = randf(&st); r2 = randf(&st);
        double A3 = 2.0 * a * r1 - a, C3 = 2.0 * r2;
        double X3 = lead_pos[2 * dim + j] - A3 * fabs(C3 * lead_pos[2 * dim + j] - x);
        pop_out[idx * dim + j] = (X1 + X2 + X3) / 3.0;
    }
}

/* ======================= WOA KERNELS ======================= */

extern "C" __global__ void woa_update(double* pop_in, double* pop_out, double* leader,
                                      double lb, double ub, int ps, int dim,
                                      double a, double a2, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 88888 + 5;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double r1 = randf(&st), r2 = randf(&st);
    double A = 2.0 * a * r1 - a;
    double C = 2.0 * r2;
    double l = (a2 - 1.0) * randf(&st) + 1.0;
    double p = randf(&st);
    int rand_idx = randi(&st, ps);
    for (int j = 0; j < dim; j++) {
        double x = pop_in[idx * dim + j];
        double out;
        if (p < 0.5) {
            if (fabs(A) >= 1.0) {
                double xr = pop_in[rand_idx * dim + j];
                out = xr - A * fabs(C * xr - x);
            } else {
                out = leader[j] - A * fabs(C * leader[j] - x);
            }
        } else {
            double dist = fabs(leader[j] - x);
            out = dist * exp(l) * cos(l * 2.0 * 3.14159265) + leader[j];
        }
        pop_out[idx * dim + j] = out;
    }
}

extern "C" __global__ void woa_update_sel(double* pop_in, double* pop_out, int* sel,
                                          double lb, double ub, int ps, int dim,
                                          double a, double a2, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 88888 + 5;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int li = sel[idx];
    double r1 = randf(&st), r2 = randf(&st);
    double A = 2.0 * a * r1 - a;
    double C = 2.0 * r2;
    double l = (a2 - 1.0) * randf(&st) + 1.0;
    double p = randf(&st);
    int rand_idx = randi(&st, ps);
    for (int j = 0; j < dim; j++) {
        double x = pop_in[idx * dim + j];
        double lj = pop_in[li * dim + j];
        double out;
        if (p < 0.5) {
            if (fabs(A) >= 1.0) {
                double xr = pop_in[rand_idx * dim + j];
                out = xr - A * fabs(C * xr - x);
            } else {
                out = lj - A * fabs(C * lj - x);
            }
        } else {
            double dist = fabs(lj - x);
            out = dist * exp(l) * cos(l * 2.0 * 3.14159265) + lj;
        }
        pop_out[idx * dim + j] = out;
    }
}

/* ======================= SCA KERNELS ======================= */

extern "C" __global__ void sca_update(double* pop_in, double* pop_out, double* guide,
                                      double lb, double ub, int ps, int dim, double r1, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 99991 + 11;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++) {
        double x = pop_in[idx * dim + j];
        double r2 = 2.0 * 3.14159265 * randf(&st);
        double r3 = 2.0 * randf(&st);
        double r4 = randf(&st);
        double g = guide[j];
        double out;
        if (r4 < 0.5) out = x + r1 * sin(r2) * fabs(r3 * g - x);
        else out = x + r1 * cos(r2) * fabs(r3 * g - x);
        pop_out[idx * dim + j] = out;
    }
}

extern "C" __global__ void sca_update_sel(double* pop_in, double* pop_out, int* sel,
                                          double lb, double ub, int ps, int dim, double r1, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 99991 + 11;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int gi = sel[idx];
    for (int j = 0; j < dim; j++) {
        double x = pop_in[idx * dim + j];
        double r2 = 2.0 * 3.14159265 * randf(&st);
        double r3 = 2.0 * randf(&st);
        double r4 = randf(&st);
        double g = pop_in[gi * dim + j];
        double out;
        if (r4 < 0.5) out = x + r1 * sin(r2) * fabs(r3 * g - x);
        else out = x + r1 * cos(r2) * fabs(r3 * g - x);
        pop_out[idx * dim + j] = out;
    }
}

/* ======================= SSA KERNELS (buffered/Jacobi) ======================= */

extern "C" __global__ void ssa_update(double* pop_in, double* pop_out, double* food,
                                      double lb, double ub, int ps, int dim, double c1, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 13579 + 19;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int half = ps / 2;
    for (int j = 0; j < dim; j++) {
        double out;
        if (idx < half) {
            double c2 = randf(&st), c3 = randf(&st);
            if (c3 < 0.5) out = food[j] + c1 * ((ub - lb) * c2 + lb);
            else out = food[j] - c1 * ((ub - lb) * c2 + lb);
        } else {
            out = (pop_in[idx * dim + j] + pop_in[(idx - 1) * dim + j]) / 2.0;
        }
        pop_out[idx * dim + j] = out;
    }
}

extern "C" __global__ void ssa_update_sel(double* pop_in, double* pop_out, int* sel,
                                          double lb, double ub, int ps, int dim, double c1, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 13579 + 19;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int half = ps / 2;
    int fi = sel[idx];
    for (int j = 0; j < dim; j++) {
        double out;
        if (idx < half) {
            double c2 = randf(&st), c3 = randf(&st);
            double f = pop_in[fi * dim + j];
            if (c3 < 0.5) out = f + c1 * ((ub - lb) * c2 + lb);
            else out = f - c1 * ((ub - lb) * c2 + lb);
        } else {
            out = (pop_in[idx * dim + j] + pop_in[(idx - 1) * dim + j]) / 2.0;
        }
        pop_out[idx * dim + j] = out;
    }
}

/* ======================= HHO KERNELS ======================= */

__device__ void hho_core(double* pop_in, double* pop_out, double* obj_in,
                         double* rabbit, double* mean_pos,
                         double lb, double ub, int ps, int dim,
                         double E1, int fid, unsigned int st, int idx) {
    const double beta = 1.5;
    const double sigma = 0.6966;
    double E0 = 2.0 * randf(&st) - 1.0;
    double E = E1 * E0;
    if (fabs(E) >= 1.0) {
        double q = randf(&st);
        int rand_idx = randi(&st, ps);
        if (q < 0.5) {
            for (int j = 0; j < dim; j++) {
                double xr = pop_in[rand_idx * dim + j];
                pop_out[idx * dim + j] = xr - randf(&st) * fabs(xr - 2.0 * randf(&st) * pop_in[idx * dim + j]);
            }
        } else {
            for (int j = 0; j < dim; j++) {
                pop_out[idx * dim + j] = (rabbit[j] - mean_pos[j]) - randf(&st) * ((ub - lb) * randf(&st) + lb);
            }
        }
        return;
    }
    double r = randf(&st);
    double J = 2.0 * (1.0 - randf(&st));
    if (r >= 0.5 && fabs(E) < 0.5) {
        for (int j = 0; j < dim; j++)
            pop_out[idx * dim + j] = rabbit[j] - E * fabs(rabbit[j] - pop_in[idx * dim + j]);
    } else if (r >= 0.5 && fabs(E) >= 0.5) {
        for (int j = 0; j < dim; j++)
            pop_out[idx * dim + j] = (rabbit[j] - pop_in[idx * dim + j]) - E * fabs(J * rabbit[j] - pop_in[idx * dim + j]);
    } else {
        double X1[MAX_DIM];
        if (fabs(E) >= 0.5) {
            for (int j = 0; j < dim; j++) {
                X1[j] = rabbit[j] - E * fabs(J * rabbit[j] - pop_in[idx * dim + j]);
                if (X1[j] > ub) X1[j] = ub;
                if (X1[j] < lb) X1[j] = lb;
            }
        } else {
            for (int j = 0; j < dim; j++) {
                X1[j] = rabbit[j] - E * fabs(J * rabbit[j] - mean_pos[j]);
                if (X1[j] > ub) X1[j] = ub;
                if (X1[j] < lb) X1[j] = lb;
            }
        }
        double f1 = evaluate(X1, dim, fid);
        if (f1 < obj_in[idx]) {
            for (int j = 0; j < dim; j++) pop_out[idx * dim + j] = X1[j];
        } else {
            double X2[MAX_DIM];
            for (int j = 0; j < dim; j++) {
                double u = randn(&st) * sigma;
                double v = randn(&st);
                double step = u / pow(fabs(v) + 1e-300, 1.0 / beta);
                double base = (fabs(E) >= 0.5) ? pop_in[idx * dim + j] : mean_pos[j];
                X2[j] = rabbit[j] - E * fabs(J * rabbit[j] - base) + randf(&st) * step;
                if (X2[j] > ub) X2[j] = ub;
                if (X2[j] < lb) X2[j] = lb;
            }
            double f2 = evaluate(X2, dim, fid);
            if (f2 < obj_in[idx]) {
                for (int j = 0; j < dim; j++) pop_out[idx * dim + j] = X2[j];
            } else {
                for (int j = 0; j < dim; j++) pop_out[idx * dim + j] = pop_in[idx * dim + j];
            }
        }
    }
}

extern "C" __global__ void hho_update(double* pop_in, double* pop_out, double* obj_in,
                                      double* rabbit, double* mean_pos,
                                      double lb, double ub, int ps, int dim,
                                      double E1, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 24681 + 31;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    hho_core(pop_in, pop_out, obj_in, rabbit, mean_pos, lb, ub, ps, dim, E1, fid, st, idx);
}

extern "C" __global__ void hho_update_sel(double* pop_in, double* pop_out, double* obj_in, int* sel,
                                          double* mean_pos,
                                          double lb, double ub, int ps, int dim,
                                          double E1, int fid, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 24681 + 31;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    double rabbit[MAX_DIM];
    int ri = sel[idx];
    for (int j = 0; j < dim; j++) rabbit[j] = pop_in[ri * dim + j];
    hho_core(pop_in, pop_out, obj_in, rabbit, mean_pos, lb, ub, ps, dim, E1, fid, st, idx);
}
