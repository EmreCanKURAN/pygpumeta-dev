# Reference implementation of the paper

The kernels (`kernels/`), the GPU runners (`gpu.py`) and the CPU reference
of the selection operators (`reference.py`) as they accompany the paper,
unmodified. The GPU test tier compiles them next to the library and
compares results bit for bit; `benchmarks/fused_vs_separated.py` uses them
as a timing baseline.

Importing `gpu.py` creates a CUDA context and compiles the kernels at
import time. Nothing in the library imports this directory.
