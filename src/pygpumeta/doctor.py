"""Check that this machine can run pygpumeta.

Run it with::

    python -m pygpumeta.doctor

It looks for the NVIDIA driver, the CUDA compiler, the host C++ compiler,
PyCUDA and the optional CuPy, says how to fix what is missing, and finishes
with a small optimisation run. On Windows it also locates the MSVC compiler
of an installed Visual Studio when that is not on the PATH.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pygpumeta._backend import msvc

__all__ = ["Check", "main", "run_checks"]

OK, WARN, FAIL = "ok", "warn", "FAIL"
ISSUES_URL = "https://github.com/EmreCanKURAN/pygpumeta-dev/issues"


@dataclass(frozen=True)
class Check:
    """Outcome of one check.

    Attributes
    ----------
    name : str
        What was checked.
    status : str
        ``"ok"``, ``"warn"`` (optional component missing) or ``"FAIL"``.
    detail : str
        What was found.
    fix : str
        What to do about a warning or failure; empty when all is well.
    """

    name: str
    status: str
    detail: str
    fix: str = ""


def _command_output(command: Sequence[str]) -> str | None:
    """Return the output of a command, or None if it cannot be run."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return (done.stdout or done.stderr).strip()


def check_python() -> Check:
    """Report the interpreter and the operating system."""
    version = platform.python_version()
    return Check("Python", OK, f"Python {version} on {platform.system()} {platform.machine()}")


def check_driver() -> Check:
    """Look for an NVIDIA GPU through ``nvidia-smi``."""
    query = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    output = _command_output(query) if shutil.which("nvidia-smi") else None
    if not output:
        return Check(
            "NVIDIA driver",
            FAIL,
            "nvidia-smi was not found or reported no GPU",
            "Install the NVIDIA driver for your GPU (https://www.nvidia.com/drivers). "
            "On Google Colab choose Runtime > Change runtime type > GPU.",
        )
    return Check("NVIDIA driver", OK, "; ".join(output.splitlines()))


def check_toolkit() -> Check:
    """Look for the CUDA compiler, which compiles the kernels at run time."""
    path = shutil.which("nvcc")
    output = _command_output(["nvcc", "--version"]) if path else None
    if not output:
        return Check(
            "CUDA toolkit (nvcc)",
            FAIL,
            "nvcc is not on the PATH",
            "Install the CUDA toolkit: 'conda install -c conda-forge cuda-nvcc' in a conda "
            "environment, or the installer from https://developer.nvidia.com/cuda-downloads, "
            "and make sure its bin directory is on the PATH.",
        )
    return Check("CUDA toolkit (nvcc)", OK, f"{output.splitlines()[-1].strip()} ({path})")


def cuda_major_version(nvcc_output: str) -> int | None:
    """Return the major CUDA version named in the output of ``nvcc --version``."""
    match = re.search(r"release (\d+)\.\d+|cuda_(\d+)\.\d+", nvcc_output)
    return int(match.group(1) or match.group(2)) if match else None


def cupy_install_hint(cuda_major: int | None) -> str:
    """Name the CuPy package that matches a CUDA version."""
    if cuda_major is None:
        return (
            "'pip install cupy-cuda12x' for CUDA 12 or 'pip install cupy-cuda13x' for CUDA 13 "
            "('nvcc --version' shows which you have)"
        )
    return f"'pip install cupy-cuda{cuda_major}x' for your CUDA {cuda_major}"


def check_host_compiler() -> Check:
    """Look for the C++ compiler that nvcc drives.

    On Windows this also makes the compiler usable by this process when it
    is installed but not on the PATH.
    """
    if platform.system() == "Windows":
        name = "C++ compiler (MSVC cl.exe)"
        status = msvc.ensure_host_compiler()
        if status.how == msvc.ON_PATH:
            return Check(name, OK, str(status.path))
        if status.how == msvc.DISCOVERED:
            return Check(
                name,
                OK,
                f"{status.path} (not on the PATH; found through Visual Studio discovery "
                f"in {status.detail})",
            )
        return Check(name, FAIL, f"cl.exe is not available: {status.detail}", _missing_msvc(status))
    for name in ("g++", "c++", "clang++"):
        path = shutil.which(name)
        if path is not None:
            return Check("C++ compiler", OK, path)
    return Check(
        "C++ compiler",
        FAIL,
        "no g++, c++ or clang++ on the PATH",
        "Install a C++ compiler: 'sudo apt install g++' on Debian and Ubuntu, or "
        "'conda install -c conda-forge gxx' in a conda environment.",
    )


def _missing_msvc(status: msvc.CompilerStatus) -> str:
    if status.how == msvc.DISABLED:
        return (
            f"Unset the environment variable {msvc.OPT_OUT_VARIABLE} so that Visual Studio is "
            "searched for, or start Python from the 'x64 Native Tools Command Prompt for VS'."
        )
    return msvc.INSTALL_HINT


def check_pycuda() -> Check:
    """Import PyCUDA and initialise the device."""
    try:
        import pycuda
        import pycuda.driver as driver
    except Exception as exc:
        return Check(
            "PyCUDA",
            FAIL,
            f"import failed: {type(exc).__name__}: {exc}",
            "Install PyCUDA after the CUDA toolkit: 'conda install -c conda-forge pycuda' "
            "(prebuilt) or 'pip install pycuda' (compiles; needs the toolkit and a C++ "
            "compiler).",
        )
    try:
        driver.init()
        device = driver.Device(0)
        name = device.name()
        capability = ".".join(map(str, device.compute_capability()))
        memory = device.total_memory() // 2**20
    except Exception as exc:
        return Check(
            "PyCUDA",
            FAIL,
            f"PyCUDA {pycuda.VERSION_TEXT} found, but no device: {type(exc).__name__}: {exc}",
            "Check that 'nvidia-smi' lists a GPU and that no other process holds it exclusively.",
        )
    detail = f"PyCUDA {pycuda.VERSION_TEXT}; {name}, compute capability {capability}, {memory} MiB"
    return Check("PyCUDA", OK, detail)


def check_cupy(cuda_major: int | None = None) -> Check:
    """Import the optional CuPy and do a small computation with it.

    ``cuda_major`` is the installed CUDA version, used to name the matching
    CuPy package when CuPy is missing.
    """
    try:
        import cupy

        total = float(cupy.arange(10, dtype=cupy.float64).sum())
    except Exception as exc:
        return Check(
            "CuPy (optional)",
            WARN,
            f"not usable: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
            "Needed for objectives written as Python functions on GPU arrays. Install the "
            f"build for your CUDA version: {cupy_install_hint(cuda_major)}, or 'conda install "
            "-c conda-forge cupy'. A build for another CUDA version imports but cannot find "
            "its libraries. Without CuPy, use built-in functions, CudaObjective or "
            "mode='host'.",
        )
    if total != 45.0:
        return Check("CuPy (optional)", WARN, f"wrong result from a test sum: {total}")
    return Check("CuPy (optional)", OK, f"CuPy {cupy.__version__}")


def check_kernel_run() -> Check:
    """Compile the kernels and minimise a small sphere function."""
    try:
        import pygpumeta

        result = pygpumeta.minimize(
            "sphere", dim=5, algorithm="gwo", selection="native", pop_size=256, max_iter=200
        )
    except Exception as exc:
        return Check(
            "Optimisation run (compiled kernels)",
            FAIL,
            f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
            "Fix the failures and warnings above first. If there are none, open an issue "
            f"with this output at {ISSUES_URL}.",
        )
    detail = (
        f"sphere in 5 dimensions: best value {result.fun:.3e} after {result.nfev} evaluations, "
        f"compiled in {result.compile_time:.1f} s, ran in {result.runtime:.3f} s"
    )
    if not result.fun < 1e-6:
        fix = f"Open an issue with this output at {ISSUES_URL}."
        return Check("Optimisation run (compiled kernels)", FAIL, detail, fix)
    return Check("Optimisation run (compiled kernels)", OK, detail)


def check_cupy_run() -> Check:
    """Minimise a Python objective on GPU arrays."""
    try:
        import pygpumeta

        result = pygpumeta.minimize(
            lambda x: ((x - 1.0) ** 2).sum(axis=1),
            (-5.0, 5.0),
            dim=5,
            algorithm="pso",
            selection="tournament",
            pop_size=256,
            max_iter=200,
        )
    except Exception as exc:
        return Check(
            "Optimisation run (Python objective on GPU arrays)",
            WARN,
            f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
            "PyCUDA and CuPy must use the same CUDA installation.",
        )
    detail = f"shifted sphere in 5 dimensions: best value {result.fun:.3e}"
    status = OK if result.fun < 1e-3 else WARN
    return Check("Optimisation run (Python objective on GPU arrays)", status, detail)


def run_checks() -> list[Check]:
    """Run all checks in order and return their outcomes.

    The optimisation runs are skipped when an earlier check they depend on
    did not pass.
    """
    toolkit = check_toolkit()
    checks = [check_python(), check_driver(), toolkit, check_host_compiler()]
    pycuda = check_pycuda()
    cupy = check_cupy(cuda_major_version(toolkit.detail))
    checks += [pycuda, cupy]
    required_ok = all(c.status != FAIL for c in checks)
    if required_ok:
        checks.append(check_kernel_run())
        if cupy.status == OK and checks[-1].status == OK:
            checks.append(check_cupy_run())
    else:
        checks.append(
            Check("Optimisation run", WARN, "skipped because a required component is missing")
        )
    return checks


def main(argv: Sequence[str] | None = None, write: Callable[[str], object] = print) -> int:
    """Print the report and return 0 if pygpumeta can run here, 1 otherwise."""
    del argv  # no options
    write("pygpumeta doctor")
    checks = run_checks()
    for check in checks:
        write(f"[{check.status:^4}] {check.name}: {check.detail}")
        if check.fix:
            write(f"       -> {check.fix}")
    failed = [c for c in checks if c.status == FAIL]
    skipped = any(c.name == "Optimisation run" for c in checks)
    if failed or skipped:
        write(f"\n{len(failed)} problem(s) found. pygpumeta cannot run its kernels here yet.")
        write("pygpumeta.reference and pygpumeta.benchmarks work without a GPU.")
        return 1
    warned = [c for c in checks if c.status == WARN]
    if warned:
        write("\nThe kernels work. Optional parts are unavailable, see the warnings above.")
    else:
        write("\nEverything works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
