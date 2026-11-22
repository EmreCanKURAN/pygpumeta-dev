# Which objective mode should I use?

pygpumeta is built for problems where the population is large and the
objective can be evaluated for the whole population on the GPU. There are
four ways to supply the objective. They differ in how much you have to
write and in what they cost.

| Your objective is ... | Use | You write | Where it runs |
| --- | --- | --- | --- |
| one of the built-in test functions | its name, `"rastrigin"` | nothing | inside the kernels |
| expressible with array operations (sums, products, `sin`, matrix products, batched linear algebra) | a Python function on CuPy arrays (the default for any callable) | NumPy-style code | on the GPU, between the kernels |
| a tight loop that array operations express badly, or needed at the highest speed | `CudaObjective` with CUDA C source | one C function | inside the kernels |
| calling external code: a simulator, scikit-learn, a file-based tool | a Python function with `mode="host"` | ordinary Python | on the CPU |

Start with a Python function on CuPy arrays. It receives the positions as
an array of shape `(n, dim)` and returns `n` values:

```python
import pygpumeta


def objective(x):  # x is a cupy.ndarray on the GPU
    return ((x - 1.0) ** 2).sum(axis=1)


result = pygpumeta.minimize(objective, (-5.0, 5.0), dim=30)
```

Code written with NumPy functions usually works unchanged, because CuPy
arrays have the same methods and `cupy` mirrors the `numpy` namespace.
`n` is the population size in most calls, but not in all: ABC, for example,
evaluates a single position for its scout.

## What each mode costs

Measured on a Tesla T4 (Google Colab, CUDA 13.0, PyCUDA 2026.1, CuPy 14.0)
with `benchmarks/fused_vs_separated.py`: Rastrigin, tournament selection,
1000 iterations, median of three seeds. Times in seconds; the factor is
relative to the built-in function.

| Algorithm | Population x dimension | Built-in | CUDA source | Python on CuPy | Python on the CPU |
| --- | --- | --- | --- | --- | --- |
| GWO | 1024 x 30 | 0.540 | 0.541 (1.0x) | 0.599 (1.1x) | 1.043 (1.9x) |
| GWO | 8192 x 30 | 2.356 | 2.356 (1.0x) | 2.847 (1.2x) | 5.810 (2.5x) |
| GWO | 8192 x 100 | 3.054 | 3.058 (1.0x) | 4.056 (1.3x) | 20.710 (6.8x) |
| HHO | 1024 x 30 | 1.126 | 1.124 (1.0x) | 1.552 (1.4x) | 3.414 (3.0x) |
| HHO | 8192 x 30 | 1.777 | 1.790 (1.0x) | 3.375 (1.9x) | 26.990 (15x) |
| HHO | 8192 x 100 | 4.509 | 4.555 (1.0x) | 7.692 (1.7x) | 61.363 (14x) |
| ABC | 1024 x 30 | 0.217 | 0.212 (1.0x) | 1.756 (8.1x) | 1.688 (7.8x) |
| ABC | 8192 x 30 | 0.692 | 0.685 (1.0x) | 1.794 (2.6x) | 10.069 (15x) |
| ABC | 8192 x 100 | 1.237 | 1.240 (1.0x) | 3.663 (3.0x) | 36.322 (29x) |

What the numbers say:

- A CUDA source objective costs the same as a built-in function.
- A Python function on CuPy arrays costs between 1.1 and 3 times as much
  for this cheap objective at large populations. Each call into CuPy has a
  fixed cost, which is what the ABC row at 1024 individuals shows: its time
  is the same as at 8192. The more work the objective does per call, the
  less this overhead matters.
- A Python function on the CPU gets slower in proportion to the population
  and the dimension, because the population is copied to the host and the
  objective no longer runs in parallel on the GPU. Do not expect a speedup
  over a CPU optimizer in this mode. It exists so that objectives which
  cannot run on the GPU can still be used.

These timings are for one GPU and one cheap objective. Run the script on
your own machine to see where your problem stands.

## Evaluation inside or outside the kernels

ABC and HHO create and test candidate positions in the middle of their
update. With a built-in function or a `CudaObjective` they evaluate inside
the update kernel (`evaluation="fused"`). A Python function cannot run
there, so the update is split into kernels that propose candidates and
kernels that accept them, with the evaluation in between
(`evaluation="separated"`). Both forms produce the same run for the same
seed. On the same GPU and problem as above:

| Algorithm | Population x dimension | Fused (s) | Separated (s) | Evaluations, fused | Evaluations, separated |
| --- | --- | --- | --- | --- | --- |
| ABC | 1024 x 30 | 0.217 | 0.258 | 2 049 024 | 2 050 024 |
| ABC | 8192 x 30 | 0.692 | 0.727 | 16 392 192 | 16 393 192 |
| ABC | 8192 x 100 | 1.237 | 1.552 | 16 392 192 | 16 393 192 |
| HHO | 1024 x 30 | 1.126 | 1.141 | 1 655 769 | 3 073 024 |
| HHO | 8192 x 30 | 1.777 | 1.804 | 13 216 987 | 24 584 192 |
| HHO | 8192 x 100 | 4.509 | 4.652 | 13 180 909 | 24 584 192 |

The fused form is the default wherever it is possible. For HHO the
separated form evaluates the whole population three times per iteration,
against once plus the diving hawks for the fused form, so an expensive
Python objective makes HHO correspondingly more expensive than the other
algorithms.

## Rules that apply to every mode

- Return finite numbers. NaN and infinite values are replaced by plus or
  minus 1e299, which lets a run continue but tells the optimizer nothing
  about how wrong a position is.
- Do not modify the array you receive.
- Constraints receive the same arrays as the objective. With a built-in
  function or a `CudaObjective` they receive CuPy arrays, or NumPy arrays
  with `mode="host"`.
- Integer and categorical variables need a Python objective.
