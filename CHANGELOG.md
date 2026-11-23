# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-23

### Changed

- Citation updated with the DOI of the paper.
- Project logo, `CITATION.cff` and a link to the paper added.

### Fixed

- Links to the documentation in the README work on PyPI.

## [0.1.0] - 2026-11-22

### Added

- `minimize` and the optimizer classes `ABC`, `GWO`, `WOA`, `SCA`, `SSA`
  and `HHO` (CUDA kernels) and `PSO` and `DE` (array operations on CuPy or
  NumPy arrays), with step-wise control, callbacks, target, patience and
  time limits, and a typed `OptimizeResult` holding the best position, the
  convergence history, the number of evaluations and the termination
  reason.
- Four ways to supply an objective: the name of a built-in test function, a
  Python callable on CuPy arrays evaluated on the device, a `CudaObjective`
  source compiled into the kernels, and a Python callable on NumPy arrays
  with `mode="host"`.
- `evaluation` option of `ABC` and `HHO`: the objective is evaluated either
  inside the update kernels or as a separate step, with the same result.
- Bounds per variable, `maximize=True`, and integer and categorical
  variables through `pygpumeta.space.Real`, `Integer` and `Categorical`,
  decoded before the objective is evaluated.
- Inequality and equality constraints (`pygpumeta.Constraint`), handled by
  a penalty or by the feasibility rule of Deb; the result reports the
  objective, the constraint violation and feasibility.
- `pygpumeta.selection`: warp tournament, Ballot-Elite and migration
  selection on PyCUDA and CuPy device arrays. `selection="auto"`, the
  default of the optimizers, is the warp tournament for ABC, GWO, WOA and
  HHO and the native rule for SCA, SSA, PSO and DE.
- `pygpumeta.reference`: CPU reference that reproduces the selection
  operators bit for bit.
- `pygpumeta.benchmarks`: names, domains, known optima and NumPy versions
  of the built-in test functions.
- CUDA kernels shipped as package data, compiled on first use and cached
  per source; importing the package creates no CUDA context. Mathematical
  constants are exact double-precision values. Non-finite objective values
  are replaced by plus or minus 1e299.
- On Windows the MSVC compiler is located through Visual Studio when
  `cl.exe` is not on the PATH, for the running process only;
  `PYGPUMETA_NO_MSVC_DISCOVERY=1` switches this off.
- `python -m pygpumeta.doctor` checks driver, CUDA toolkit, C++ compiler,
  PyCUDA and CuPy, explains what is missing and finishes with a small run.
- Installation guide for Google Colab, Windows and Linux with conda-forge,
  stating which setups have been verified; documentation on objective
  modes with measured timings, on the selection operators and on adding an
  algorithm; eight runnable examples, each compared with a reference value.
- Test suite in two tiers (CPU anywhere, GPU on a CUDA device), CI for
  Python 3.10 to 3.13, and a release workflow with PyPI trusted publishing.
