# Installation

pygpumeta compiles its CUDA kernels on your machine the first time they are
used. Besides an NVIDIA GPU with its driver, that needs

- the CUDA toolkit, for its compiler `nvcc`;
- a C++ compiler that `nvcc` can drive (GCC on Linux, MSVC on Windows);
- PyCUDA;
- optionally CuPy, for objectives written as Python functions on GPU arrays.

PyCUDA is distributed on PyPI as source only, so `pip install pycuda`
compiles it and fails without the CUDA toolkit and a C++ compiler.

After any of the routes, run

```
python -m pygpumeta.doctor
```

It checks each component, prints what is missing together with the command
that fixes it, and ends with a small optimisation run.

## What has been verified

"Verified" means that on that setup the doctor reported no problem and the
GPU test tier passed (1250 tests).

| Setup | Status |
| --- | --- |
| Google Colab: Tesla T4, driver 580.82, CUDA 13.0, Python 3.13, PyCUDA 2026.1 built by pip, the CuPy 14.0.1 that Colab preinstalls | verified |
| Windows: GeForce RTX 4060 Laptop, driver 610.62, CUDA Toolkit 13.1, MSVC 14.42 (Visual Studio 2022 Community), Python 3.11 in an existing Anaconda environment, PyCUDA 2026.1, CuPy 14.2.0 built from the `cupy` source package by pip | verified |
| Linux with conda-forge packages | not yet verified |
| Windows with conda-forge packages for PyCUDA, CuPy or the toolkit | not yet verified |
| Any setup with CUDA 12 | not yet verified |
| The prebuilt CuPy wheels `cupy-cuda12x` and `cupy-cuda13x` | not yet verified |

## Choosing the CuPy package

CuPy is published once per CUDA major version. A package for another
version installs and imports but cannot find its libraries, so match it to
the toolkit that `nvcc --version` reports.

| Installed CUDA toolkit | Package | With pygpumeta |
| --- | --- | --- |
| 12.x | `pip install cupy-cuda12x` | `pip install "pygpumeta[cupy12]"` |
| 13.x | `pip install cupy-cuda13x` | `pip install "pygpumeta[cupy13]"` |
| any, built on your machine | `pip install cupy` (compiles for a long time; needs the toolkit and a C++ compiler) | |
| any, in a conda environment | `conda install -c conda-forge cupy` | |

Of these, only the source build (`pip install cupy`, on Windows with CUDA
13.1) and the CuPy preinstalled on Colab (CUDA 13.0) have been verified
with pygpumeta. The two wheels exist for Linux and Windows and are the
route CuPy recommends, but they have not been tried here yet.

## Google Colab

Choose Runtime > Change runtime type > GPU. The toolkit, the compiler and
CuPy are already there; only PyCUDA has to be built, which takes a few
minutes.

```
!pip install pycuda
!pip install pygpumeta
!python -m pygpumeta.doctor
```

Status: verified, see the table above.

## Windows

1. Install the NVIDIA driver and check that `nvidia-smi` works in a
   terminal.
2. Install the CUDA Toolkit from
   https://developer.nvidia.com/cuda-downloads and check `nvcc --version`
   in a new terminal.
3. Install Visual Studio (Community is enough) or "Build Tools for Visual
   Studio", with the workload "Desktop development with C++". This provides
   the MSVC compiler, which is needed to build PyCUDA and again every time
   kernels are compiled.
4. In the Python environment of your choice:

```
pip install pycuda
pip install cupy-cuda13x
pip install pygpumeta
python -m pygpumeta.doctor
```

Use `cupy-cuda12x` with a CUDA 12 toolkit. `pip install cupy`, which builds
CuPy from source against the installed toolkit, is the variant that was
verified.

Any prompt will do, including a plain Anaconda Prompt. `nvcc` needs the
MSVC compiler `cl.exe` on the PATH, where a normal prompt does not have it.
pygpumeta therefore looks for it before it compiles its first kernel: it
asks the Visual Studio installer (`vswhere.exe`) for the latest
installation with the C++ x64 tools, runs that installation's
`vcvars64.bat` in a child process and takes over the resulting environment
variables (PATH, INCLUDE, LIB, LIBPATH and the others the script sets). This
happens once per Python process and changes that process only; your system
and user environment are not touched. The doctor reports where the compiler
was found.

To switch the search off, set the environment variable
`PYGPUMETA_NO_MSVC_DISCOVERY=1`. The compiler then has to be on the PATH
already.

Status: verified with the versions in the table above, from a prompt in
which the Visual Studio environment was loaded. The automatic search for
the compiler is not yet verified on a real machine.

## Linux with conda-forge

Install the NVIDIA driver with your distribution's tools and check that
`nvidia-smi` lists the GPU. Then, with Miniforge or any conda installation:

```
conda create -n pygpumeta -c conda-forge python=3.12 pycuda cupy cuda-nvcc gxx
conda activate pygpumeta
pip install pygpumeta
python -m pygpumeta.doctor
```

`cuda-nvcc` provides the CUDA compiler and `gxx` the C++ compiler inside the
environment, so nothing has to be installed system-wide except the driver.
conda-forge has prebuilt PyCUDA, which avoids compiling it. Leave out `cupy`
if you only use built-in functions, CUDA objectives or objectives evaluated
on the CPU.

Status: not yet verified on a real machine. conda-forge publishes PyCUDA
and CuPy for linux-64, linux-aarch64 and win-64.

## Without a GPU

`pip install --no-deps pygpumeta` together with `pip install numpy` installs
the package without PyCUDA. `pygpumeta.reference` (the CPU reference of the
selection operators), `pygpumeta.benchmarks`, and `PSO` and `DE` with
`mode="host"` and `selection="native"` work in such an environment.
Everything else needs the components above.

## Common problems

| Symptom | Cause and fix |
| --- | --- |
| `pip install pycuda` fails with `cuda.h: No such file` | The CUDA toolkit is not installed or not found. Install it, or use the conda route. |
| `CudaUnavailableError: PyCUDA could not be imported` | PyCUDA is not installed in the active environment. |
| `CompilationError` saying that `cl.exe` was not found, on Windows | No Visual Studio with the C++ tools was found. Install "Build Tools for Visual Studio" with the workload "Desktop development with C++". If it is installed in an unusual place, or the search is switched off with `PYGPUMETA_NO_MSVC_DISCOVERY`, start Python from the "x64 Native Tools Command Prompt for VS", which puts the compiler on the PATH. |
| `ImportError: ... need CuPy` | Install the CuPy package for your CUDA version, or pass `mode="host"`, or use a built-in function or a `CudaObjective`. |
| CuPy imports but fails on first use | The CuPy package does not match the installed CUDA version; see "Choosing the CuPy package". |
| `pytest` stops with collection errors about `hypothesis` | The test dependencies are missing: `pip install -e ".[test]"`. |
