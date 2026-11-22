/*
 * ABC and HHO with the objective evaluated outside the update: one kernel
 * proposes candidate positions, the caller evaluates them, a second kernel
 * accepts or rejects. Random streams and arithmetic are those of the kernels
 * that evaluate in place, so both forms produce the same run.
 *
 * Appended to metaheuristics.cu, whose helpers it uses.
 */

/* ======================= ABC ======================= */

extern "C" __global__ void abc_init_positions(double* pop, const double* lb, const double* ub, int ps, int dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 12345 + 13;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++)
        pop[idx * dim + j] = lb[j] + randf(&st) * (ub[j] - lb[j]);
}

/* Candidate of thread idx: source s with one coordinate moved relative to a
   random neighbour. */
__device__ void abc_neighbour(const double* pop, double* cand, int idx, int s,
                              const double* lb, const double* ub, int ps, int dim, unsigned int* st) {
    int j = randi(st, dim);
    int k = randi(st, ps);
    if (k == s) k = (k + 1) % ps;
    for (int d = 0; d < dim; d++) cand[idx * dim + d] = pop[s * dim + d];
    double phi = -1.0 + 2.0 * randf(st);
    double v = pop[s * dim + j] + phi * (pop[s * dim + j] - pop[k * dim + j]);
    if (v < lb[j]) v = lb[j];
    if (v > ub[j]) v = ub[j];
    cand[idx * dim + j] = v;
}

extern "C" __global__ void abc_propose_employed(const double* pop, double* cand,
                                                const double* lb, const double* ub, int ps, int dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 54321 + 17;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    abc_neighbour(pop, cand, idx, idx, lb, ub, ps, dim, &st);
}

extern "C" __global__ void abc_propose_onlooker_roulette(const double* pop, double* cand, double* cum,
                                                         const double* lb, const double* ub, int ps, int dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 98765 + 23;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    int s = binary_search_sel(cum, ps, randf(&st));
    abc_neighbour(pop, cand, idx, s, lb, ub, ps, dim, &st);
}

extern "C" __global__ void abc_propose_onlooker_selected(const double* pop, double* cand, const int* sel,
                                                         const double* lb, const double* ub, int ps, int dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 98765 + 29;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    abc_neighbour(pop, cand, idx, sel[idx], lb, ub, ps, dim, &st);
}

/* Greedy replacement of source idx by its evaluated candidate. */
extern "C" __global__ void abc_accept(double* pop, double* obj, double* fit, int* trial,
                                      const double* cand, const double* cand_obj, int ps, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    double no = cand_obj[idx];
    double nf = (no >= 0.0) ? 1.0 / (1.0 + no) : 1.0 + fabs(no);
    if (nf > fit[idx]) {
        for (int d = 0; d < dim; d++) pop[idx * dim + d] = cand[idx * dim + d];
        obj[idx] = no; fit[idx] = nf; trial[idx] = 0;
    } else {
        trial[idx] = trial[idx] + 1;
    }
}

/* The scout row is always filled, with the current position when no source
   is abandoned, so that the caller can evaluate it without asking the host
   whether a scout was sent. */
extern "C" __global__ void abc_propose_scout(const double* pop, const int* scout_idx, double* row,
                                             const double* lb, const double* ub, int dim, int limit, unsigned int seed) {
    if (threadIdx.x || blockIdx.x) return;
    int i = scout_idx[0];
    if (scout_idx[1] <= limit) {
        for (int j = 0; j < dim; j++) row[j] = pop[i * dim + j];
        return;
    }
    unsigned int st = seed + i * 31337 + 41;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    for (int j = 0; j < dim; j++)
        row[j] = lb[j] + randf(&st) * (ub[j] - lb[j]);
}

extern "C" __global__ void abc_accept_scout(double* pop, double* obj, double* fit, int* trial,
                                            const int* scout_idx, const double* row, const double* row_obj,
                                            int dim, int limit) {
    if (threadIdx.x || blockIdx.x) return;
    if (scout_idx[1] <= limit) return;
    int i = scout_idx[0];
    for (int j = 0; j < dim; j++) pop[i * dim + j] = row[j];
    double o = row_obj[0];
    obj[i] = o;
    fit[i] = (o >= 0.0) ? 1.0 / (1.0 + o) : 1.0 + fabs(o);
    trial[i] = 0;
}

/* ======================= HHO ======================= */

/* First stage. Hawks that do not dive get their new position in pop_out and
   mode 0. Diving hawks get their first trial position in cand, mode 1
   (|E| >= 0.5) or 2, and their random state, energy and jump strength are
   kept for the second trial. cand holds the current position for all other
   hawks, so that every row of it is a point inside the bounds. */
__device__ void hho_propose_core(const double* pop_in, double* pop_out, double* cand,
                                 const double* rabbit, const double* mean_pos,
                                 const double* lb, const double* ub, int ps, int dim, double E1, unsigned int st, int idx,
                                 int* mode, unsigned int* state, double* energy, double* jump) {
    for (int j = 0; j < dim; j++) cand[idx * dim + j] = pop_in[idx * dim + j];
    mode[idx] = 0;
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
                pop_out[idx * dim + j] = (rabbit[j] - mean_pos[j]) - randf(&st) * ((ub[j] - lb[j]) * randf(&st) + lb[j]);
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
        if (fabs(E) >= 0.5) {
            for (int j = 0; j < dim; j++) {
                double x = rabbit[j] - E * fabs(J * rabbit[j] - pop_in[idx * dim + j]);
                if (x > ub[j]) x = ub[j];
                if (x < lb[j]) x = lb[j];
                cand[idx * dim + j] = x;
            }
            mode[idx] = 1;
        } else {
            for (int j = 0; j < dim; j++) {
                double x = rabbit[j] - E * fabs(J * rabbit[j] - mean_pos[j]);
                if (x > ub[j]) x = ub[j];
                if (x < lb[j]) x = lb[j];
                cand[idx * dim + j] = x;
            }
            mode[idx] = 2;
        }
        state[idx] = st; energy[idx] = E; jump[idx] = J;
    }
}

extern "C" __global__ void hho_propose(const double* pop_in, double* pop_out, double* cand,
                                       const double* rabbit, const double* mean_pos,
                                       const double* lb, const double* ub, int ps, int dim, double E1, unsigned int seed,
                                       int* mode, unsigned int* state, double* energy, double* jump) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 24681 + 31;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    hho_propose_core(pop_in, pop_out, cand, rabbit, mean_pos, lb, ub, ps, dim, E1, st, idx,
                     mode, state, energy, jump);
}

extern "C" __global__ void hho_propose_sel(const double* pop_in, double* pop_out, double* cand,
                                           const int* sel, const double* mean_pos,
                                           const double* lb, const double* ub, int ps, int dim, double E1, unsigned int seed,
                                           int* mode, unsigned int* state, double* energy, double* jump) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    unsigned int st = seed + idx * 24681 + 31;
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    hho_propose_core(pop_in, pop_out, cand, &pop_in[sel[idx] * dim], mean_pos, lb, ub, ps, dim, E1, st, idx,
                     mode, state, energy, jump);
}

/* Second stage, after cand has been evaluated into cand_obj. A diving hawk
   whose first trial improved takes it; the others put their second trial
   (with the Levy step) in cand. */
extern "C" __global__ void hho_dive(const double* pop_in, double* pop_out, double* cand,
                                    const double* obj_in, const double* cand_obj,
                                    const double* guide, const int* sel, int use_sel, const double* mean_pos,
                                    const double* lb, const double* ub, int ps, int dim,
                                    int* mode, const unsigned int* state, const double* energy, const double* jump) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    int m = mode[idx];
    if (m == 0) return;
    if (cand_obj[idx] < obj_in[idx]) {
        for (int j = 0; j < dim; j++) pop_out[idx * dim + j] = cand[idx * dim + j];
        mode[idx] = 0;
        return;
    }
    const double beta = 1.5;
    const double sigma = 0.6966;
    unsigned int st = state[idx];
    double E = energy[idx];
    double J = jump[idx];
    const double* rabbit = use_sel ? &pop_in[sel[idx] * dim] : guide;
    for (int j = 0; j < dim; j++) {
        double u = randn(&st) * sigma;
        double v = randn(&st);
        double step = u / pow(fabs(v) + 1e-300, 1.0 / beta);
        double base = (m == 1) ? pop_in[idx * dim + j] : mean_pos[j];
        double x = rabbit[j] - E * fabs(J * rabbit[j] - base) + randf(&st) * step;
        if (x > ub[j]) x = ub[j];
        if (x < lb[j]) x = lb[j];
        cand[idx * dim + j] = x;
    }
}

/* Third stage, after the second trials have been evaluated. */
extern "C" __global__ void hho_dive_accept(const double* pop_in, double* pop_out, const double* cand,
                                           const double* obj_in, const double* cand_obj,
                                           int ps, int dim, const int* mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ps) return;
    if (mode[idx] == 0) return;
    const double* src = (cand_obj[idx] < obj_in[idx]) ? &cand[idx * dim] : &pop_in[idx * dim];
    for (int j = 0; j < dim; j++) pop_out[idx * dim + j] = src[j];
}
