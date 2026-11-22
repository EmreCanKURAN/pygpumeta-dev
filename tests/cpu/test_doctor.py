from __future__ import annotations

import subprocess
import sys
import types

import pytest

from pygpumeta import doctor
from pygpumeta._backend import msvc


def test_report_on_a_machine_without_cuda():
    # This tier runs where there is no GPU; the doctor must say so and how
    # to proceed, and exit with an error code.
    lines: list[str] = []
    code = doctor.main(write=lines.append)
    report = "\n".join(lines)
    if "Everything works." in report or "The kernels work." in report:
        pytest.skip("a CUDA device is available here")
    assert code == 1
    assert lines[0] == "pygpumeta doctor" and "[ ok ] Python: Python 3." in report
    assert "problem(s) found" in report and "skipped because a required component" in report
    assert "pygpumeta.reference and pygpumeta.benchmarks work without a GPU." in report


def test_module_is_runnable():
    done = subprocess.run(
        [sys.executable, "-m", "pygpumeta.doctor"], capture_output=True, text=True, check=False
    )
    assert done.returncode in (0, 1) and done.stdout.startswith("pygpumeta doctor\n")
    assert "Traceback" not in done.stderr


def fake_commands(monkeypatch, available: dict[str, str]):
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name: f"/bin/{name}" if name in available else None
    )
    monkeypatch.setattr(doctor, "_command_output", lambda command: available.get(command[0]))


def test_driver_toolkit_and_compiler_found(monkeypatch):
    fake_commands(
        monkeypatch,
        {
            "nvidia-smi": "Tesla T4, 580.82.07, 15360 MiB",
            "nvcc": "nvcc: NVIDIA (R) Cuda compiler driver\nBuild cuda_13.0.r13.0/compiler.1_0",
            "g++": "",
        },
    )
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    assert doctor.check_driver() == doctor.Check(
        "NVIDIA driver", "ok", "Tesla T4, 580.82.07, 15360 MiB"
    )
    toolkit = doctor.check_toolkit()
    assert (
        toolkit.status == "ok" and "cuda_13.0" in toolkit.detail and "/bin/nvcc" in toolkit.detail
    )
    assert doctor.check_host_compiler() == doctor.Check("C++ compiler", "ok", "/bin/g++")


@pytest.mark.parametrize(
    ("system", "check", "hint"),
    [
        ("Linux", doctor.check_driver, "Install the NVIDIA driver"),
        ("Linux", doctor.check_toolkit, "conda install -c conda-forge cuda-nvcc"),
        ("Linux", doctor.check_host_compiler, "sudo apt install g++"),
    ],
)
def test_missing_components_come_with_a_fix(monkeypatch, system, check, hint):
    fake_commands(monkeypatch, {})
    monkeypatch.setattr(doctor.platform, "system", lambda: system)
    outcome = check()
    assert outcome.status == "FAIL" and hint in outcome.fix


def compiler_status(monkeypatch, status):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(doctor.msvc, "ensure_host_compiler", lambda: status)
    return doctor.check_host_compiler()


def test_windows_compiler_on_the_path(monkeypatch):
    outcome = compiler_status(monkeypatch, msvc.CompilerStatus(msvc.ON_PATH, r"C:\VS\cl.exe"))
    assert outcome == doctor.Check("C++ compiler (MSVC cl.exe)", "ok", r"C:\VS\cl.exe")


def test_windows_compiler_found_through_visual_studio(monkeypatch):
    status = msvc.CompilerStatus(msvc.DISCOVERED, r"C:\VS\VC\bin\cl.exe", r"C:\VS")
    outcome = compiler_status(monkeypatch, status)
    assert outcome.status == "ok" and outcome.fix == ""
    assert outcome.detail == (
        r"C:\VS\VC\bin\cl.exe (not on the PATH; found through Visual Studio discovery in C:\VS)"
    )


def test_windows_compiler_missing_names_the_exact_fix(monkeypatch):
    status = msvc.CompilerStatus(msvc.NOT_FOUND, detail="no Visual Studio installer at X")
    outcome = compiler_status(monkeypatch, status)
    assert outcome.status == "FAIL" and "no Visual Studio installer at X" in outcome.detail
    assert "'Build Tools for Visual Studio'" in outcome.fix
    assert "'Desktop development with C++'" in outcome.fix


def test_windows_compiler_discovery_switched_off(monkeypatch):
    status = msvc.CompilerStatus(msvc.DISABLED, detail="PYGPUMETA_NO_MSVC_DISCOVERY is set")
    outcome = compiler_status(monkeypatch, status)
    assert outcome.status == "FAIL" and "Unset the environment variable" in outcome.fix
    assert "x64 Native Tools Command Prompt" in outcome.fix


@pytest.mark.parametrize(
    ("output", "major"),
    [
        (
            "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.4",
            12,
        ),
        ("Build cuda_13.1.r13.1/compiler.36836380_0 (C:\\CUDA\\v13.1\\bin\\nvcc.EXE)", 13),
        ("something else", None),
    ],
)
def test_cupy_hint_follows_the_cuda_version(monkeypatch, output, major):
    assert doctor.cuda_major_version(output) == major
    monkeypatch.setitem(sys.modules, "cupy", None)
    fix = doctor.check_cupy(major).fix
    if major is None:
        assert "cupy-cuda12x' for CUDA 12" in fix and "cupy-cuda13x' for CUDA 13" in fix
    else:
        assert f"'pip install cupy-cuda{major}x' for your CUDA {major}" in fix
        assert f"cupy-cuda{25 - major}x" not in fix


def test_missing_pycuda_and_cupy_are_explained(monkeypatch):
    # An entry of None makes the import fail whether or not the library is installed.
    monkeypatch.setitem(sys.modules, "pycuda", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    outcome = doctor.check_pycuda()
    assert outcome.status == "FAIL" and "import failed" in outcome.detail
    assert "conda install -c conda-forge pycuda" in outcome.fix
    outcome = doctor.check_cupy()
    assert outcome.status == "warn" and "mode='host'" in outcome.fix


def test_pycuda_without_a_device_is_reported(monkeypatch):
    def no_device():
        raise RuntimeError("no device")

    driver = types.SimpleNamespace(init=no_device)
    monkeypatch.setitem(
        sys.modules, "pycuda", types.SimpleNamespace(VERSION_TEXT="1.2", driver=driver)
    )
    monkeypatch.setitem(sys.modules, "pycuda.driver", driver)
    outcome = doctor.check_pycuda()
    assert outcome.status == "FAIL" and "PyCUDA 1.2 found, but no device" in outcome.detail


def test_runs_are_only_attempted_when_the_components_are_there(monkeypatch):
    ok = doctor.Check("x", "ok", "")
    for name in (
        "check_python",
        "check_driver",
        "check_toolkit",
        "check_host_compiler",
        "check_pycuda",
    ):
        monkeypatch.setattr(doctor, name, lambda ok=ok: ok)
    no_cupy = doctor.Check("CuPy (optional)", "warn", "")
    monkeypatch.setattr(doctor, "check_cupy", lambda major=None: no_cupy)
    monkeypatch.setattr(doctor, "check_kernel_run", lambda: doctor.Check("run", "ok", "fine"))
    monkeypatch.setattr(doctor, "check_cupy_run", lambda: pytest.fail("must not run without CuPy"))
    lines: list[str] = []
    assert doctor.main(write=lines.append) == 0
    assert (
        lines[-1].strip()
        == "The kernels work. Optional parts are unavailable, see the warnings above."
    )
    monkeypatch.setattr(doctor, "check_cupy", lambda major=None: ok)
    monkeypatch.setattr(doctor, "check_cupy_run", lambda: ok)
    lines.clear()
    assert doctor.main(write=lines.append) == 0 and lines[-1].strip() == "Everything works."
    monkeypatch.setattr(doctor, "check_toolkit", lambda: doctor.Check("nvcc", "FAIL", "", "fix"))
    monkeypatch.setattr(doctor, "check_kernel_run", lambda: pytest.fail("must not run"))
    assert doctor.main(write=lines.append) == 1
