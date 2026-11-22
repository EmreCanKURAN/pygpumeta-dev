# Stand-ins for the GPU libraries

Minimal `pycuda` and `cupy` packages that can be imported but have no
device behind them. Putting this directory on `PYTHONPATH` makes a machine
without CUDA look, to the import system, like one where the GPU libraries
are installed:

```
PYTHONPATH=tests/stubs python -m pytest
```

CI runs the test suite this way to catch tests that silently rely on the
GPU libraries being absent. The stubs are never imported otherwise.
