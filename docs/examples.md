# Examples

The scripts in `examples/` are complete programs. Each one states its
problem, runs an optimizer and prints the result next to a reference value.

```
python examples/01_antenna_array.py           # one example, on the GPU
python examples/run_all.py                    # all of them
python examples/02_solar_cell.py --mode host  # objective on the CPU
```

`--algorithm`, `--selection`, `--seed` and `--quick` are accepted by all of
them. Every objective is written once against an array module, so the same
function runs on CuPy arrays on the GPU and on NumPy arrays on the CPU.

The results below are the output of `python examples/run_all.py` with seed 0
on a Tesla T4 (Google Colab, CUDA 13.0, PyCUDA 2026.1, CuPy 14.0).
Runtimes exclude kernel compilation, which took 0.3 to 0.4 s per example.

| Example | Algorithm, population x iterations | Result | Reference | Runtime |
| --- | --- | --- | --- | --- |
| 1. Antenna array | GWO native, 2048 x 2000 | sidelobe level -29.68 dB, nulls at -60.4 and -61.3 dB | -30.00 dB without nulls (lower bound) | 4.58 s |
| 2. Solar cell | DE, 1024 x 3000 | RMSE 9.860219e-4 A | 9.8602e-4 A | 6.80 s |
| 3. 10-bar truss | DE, 1024 x 3000 | 5060.85 lb, feasible | 5060.85 lb | 10.45 s |
| 4. Load dispatch | DE, 1024 x 3000 | 17969.49 USD/h, demand met exactly | 17963.83 USD/h | 9.76 s |
| 5. Image thresholding | PSO with tournament, 1024 x 300 | variance 3940.7894 at thresholds (68, 122, 180) | 3940.7894 at (68, 122, 180), by enumeration | 1.29 s |
| 6. FIR filter | DE, 2048 x 4000 | ripple 0.05647 (-24.96 dB) | 0.05662 (-24.94 dB), Parks-McClellan | 7.74 s |
| 7. SVM tuning | GWO with tournament, 32 x 12, host mode | accuracy 0.9789 | 0.9719 with default hyperparameters | 15.22 s |
| 8. Custom algorithm | evolution strategy, 1024 x 600 | 2.82e-9 | 0 | 1.68 s in total |

## 1. Linear antenna array synthesis

`01_antenna_array.py`. The ten amplitudes of a symmetric 20-element
broadside array are chosen to minimise the highest sidelobe, with the
pattern forced below -60 dB in two directions. The null requirements are
inequality constraints handled by the feasibility rule.

The Dolph-Chebyshev array gives the lowest possible sidelobe level for a
given main-beam width when there are no nulls to place, so its -30 dB is a
lower bound here. The result, -29.68 dB with both nulls satisfied, is
0.32 dB above that bound.

## 2. Solar cell parameter extraction

`02_solar_cell.py`. The five parameters of the single-diode model are fitted
to the 26 measured points of the RTC France silicon cell. The result
reproduces the best error known for this data set, 9.8602e-4 A, with
Iph = 0.76078 A, I0 = 0.32302 uA, Rs = 0.03638 Ohm, Rsh = 53.719 Ohm and
n = 1.48118.

## 3. Truss sizing

`03_truss.py`. The areas of the classic 10-bar truss are sized for minimum
weight under stress and displacement limits. Every candidate needs a
finite-element solve; the stiffness matrices of the whole population are
assembled with one tensor product and solved as a batch on the device. With
the feasibility rule the result is feasible exactly, with the governing
stress and displacement limits at a utilisation of 1.0000, and its weight
equals the best value known, 5060.85 lb.

## 4. Economic load dispatch

`04_load_dispatch.py`. Thirteen generating units with valve-point effects
supply 1800 MW at minimum fuel cost. The example removes the power balance
equality by letting one unit supply the remainder, which leaves two
inequality constraints on that unit.

The formulation matters. Searching all 13 outputs and penalising the
mismatch of the power balance, with the same algorithm, 2048 individuals
and 3000 iterations, reaches only 18525.69 USD/h on the T4 and misses the
demand by 0.024 MW. With the equality removed the result is 17969.49 USD/h
with the demand met exactly, 5.66 USD/h above the best value known.

## 5. Multilevel image thresholding

`05_image_thresholding.py`. Three Otsu thresholds are declared as integer
variables and the between-class variance is maximised. The search space is
small enough to enumerate, and the optimizer finds the exact optimum.

## 6. FIR filter design

`06_fir_filter.py`. The eleven distinct taps of a 21-tap linear-phase
low-pass filter are chosen to minimise the largest deviation from the ideal
response on 400 frequencies. The result is within 0.3 percent of the
equiripple filter computed by the Parks-McClellan algorithm, on the lower
side: the optimizer only sees the 400 frequencies and can use the gaps
between them. Parks-McClellan is the right tool for this task; the example
shows that a general optimizer arrives at the same design.

## 7. SVM hyperparameter tuning

`07_svm_tuning.py`. The objective trains a support vector machine with
scikit-learn, so it runs on the CPU, one candidate at a time
(`mode="host"`, `vectorized=False`). The kernel is a categorical variable.
After 416 evaluations the cross-validated accuracy is 0.9789, against
0.9719 with the default hyperparameters.

The 15 seconds are spent in scikit-learn. The GPU contributes nothing to
the speed of a problem like this; the example is there to show that such
objectives can be used.

## 8. A custom algorithm

`08_custom_algorithm.py`. An evolution strategy in about twenty lines of
array code: parents are chosen with the warp tournament and, every fourth
iteration, with migration; children are parents with Gaussian noise on some
variables. On Rastrigin in 10 dimensions it reaches 2.82e-9 in 600
iterations. With `--mode host` the same loop runs on NumPy arrays with
`pygpumeta.reference`.
