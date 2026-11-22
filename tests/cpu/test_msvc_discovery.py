"""Locating the MSVC compiler, with Visual Studio and its tools faked."""

from __future__ import annotations

import pytest

from pygpumeta._backend import module as backend_module
from pygpumeta._backend import msvc

VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
INSTALLATION = r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
VCVARS = INSTALLATION + r"\VC\Auxiliary\Build\vcvars64.bat"
TOOLS = INSTALLATION + r"\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64"
BEFORE = {"Path": r"C:\Windows\system32;C:\env", "TEMP": r"C:\Temp", "ComSpec": "cmd.exe"}
# What "set" prints after vcvars64.bat has run.
AFTER = "\n".join(
    [
        "=C:=C:\\work",
        f"Path={TOOLS};C:\\Windows\\system32;C:\\env",
        r"TEMP=C:\Temp",
        "ComSpec=cmd.exe",
        f"INCLUDE={INSTALLATION}\\VC\\Tools\\MSVC\\14.42.34433\\include",
        f"LIB={INSTALLATION}\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64",
        f"LIBPATH={INSTALLATION}\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64",
        "VSCMD_ARG_TGT_ARCH=x64",
        "VCToolsVersion=14.42.34433",
        "",
    ]
)


class Machine:
    """A faked Windows machine: which files exist and what commands print."""

    def __init__(self, files=(VSWHERE, VCVARS), vswhere=INSTALLATION + "\n", vcvars=AFTER):
        self.files = set(files)
        self.vswhere, self.vcvars = vswhere, vcvars
        self.commands: list[str] = []

    def run(self, command):
        self.commands.append(command)
        return self.vswhere if "vswhere.exe" in command else self.vcvars

    def isfile(self, path):
        return path in self.files

    def which(self, name, path):
        return TOOLS + r"\cl.exe" if name == "cl" and path and TOOLS in path else None

    def discover(self, environ=None):
        environ = BEFORE if environ is None else environ
        return msvc.discover(environ, run=self.run, isfile=self.isfile, which=self.which)


def test_compiler_already_on_the_path_needs_no_search():
    machine = Machine()
    status, variables = machine.discover({"Path": TOOLS + r";C:\Windows"})
    assert status == msvc.CompilerStatus(msvc.ON_PATH, TOOLS + r"\cl.exe")
    assert variables == {} and machine.commands == [] and status.available


def test_compiler_is_found_through_visual_studio():
    machine = Machine()
    status, variables = machine.discover()
    assert status == msvc.CompilerStatus(msvc.DISCOVERED, TOOLS + r"\cl.exe", INSTALLATION)
    assert status.available
    # vswhere is asked for the latest installation with the C++ x64 tools.
    assert machine.commands[0] == (
        f'"{VSWHERE}" -latest -products * -requires '
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"
    )
    assert machine.commands[1] == f'cmd.exe /d /s /c ""{VCVARS}" >nul 2>&1 && set"'
    # Everything vcvars added or changed is taken over; the rest is left alone.
    assert set(variables) == {
        "Path", "INCLUDE", "LIB", "LIBPATH", "VSCMD_ARG_TGT_ARCH", "VCToolsVersion",
    }  # fmt: skip
    assert variables["Path"].startswith(TOOLS) and variables["INCLUDE"].endswith("include")


def test_vswhere_location_follows_the_program_files_variable():
    assert msvc.vswhere_location({}) == VSWHERE
    assert msvc.vswhere_location({"ProgramFiles(x86)": r"D:\PF86"}) == (
        r"D:\PF86\Microsoft Visual Studio\Installer\vswhere.exe"
    )


@pytest.mark.parametrize(
    ("machine", "reason"),
    [
        (Machine(files=()), "no Visual Studio installer at " + VSWHERE),
        (Machine(vswhere=""), "no Visual Studio installation with the C++ x64 tools"),
        (Machine(vswhere=None), "no Visual Studio installation with the C++ x64 tools"),
        (Machine(files=(VSWHERE,)), VCVARS + " is missing"),
        (Machine(vcvars=None), VCVARS + " did not provide cl.exe"),
        (Machine(vcvars="Path=C:\\Windows\n"), VCVARS + " did not provide cl.exe"),
    ],
)
def test_nothing_usable_is_reported_with_its_reason(machine, reason):
    status, variables = machine.discover()
    assert status == msvc.CompilerStatus(msvc.NOT_FOUND, detail=reason)
    assert variables == {} and not status.available
    message = msvc.describe_missing(status)
    assert reason in message and "'Desktop development with C++'" in message


@pytest.mark.parametrize("value", ["1", "yes"])
def test_opt_out_leaves_everything_alone(value):
    machine = Machine()
    status, variables = machine.discover({**BEFORE, msvc.OPT_OUT_VARIABLE: value})
    assert status.how == msvc.DISABLED and not status.available
    assert variables == {} and machine.commands == []
    assert "switched off (PYGPUMETA_NO_MSVC_DISCOVERY is set)" in msvc.describe_missing(status)


@pytest.mark.parametrize("value", ["", "0"])
def test_opt_out_variable_set_to_nothing_or_zero_does_not_opt_out(value):
    status, _ = Machine().discover({**BEFORE, msvc.OPT_OUT_VARIABLE: value})
    assert status.how == msvc.DISCOVERED


def test_environment_parsing_and_comparison():
    parsed = msvc.parse_environment("=C:=C:\\x\nA=1\nB=two=parts\nnonsense\n\nC=")
    assert parsed == {"A": "1", "B": "two=parts", "C": ""}
    changed = msvc.changed_variables(
        {"PATH": "a", "Keep": "k"}, {"Path": "b;a", "KEEP": "k", "New": "n"}
    )
    assert changed == {"Path": "b;a", "New": "n"}


@pytest.fixture
def fresh_search(monkeypatch):
    msvc.reset()
    environ = dict(BEFORE)
    monkeypatch.setattr(msvc, "_environment", lambda: environ)
    yield environ
    msvc.reset()


def test_other_platforms_are_left_alone(fresh_search, monkeypatch):
    monkeypatch.setattr(msvc.sys, "platform", "linux")
    monkeypatch.setattr(msvc, "_run", lambda command: pytest.fail("nothing must be run"))
    status = msvc.ensure_host_compiler()
    assert status.how == msvc.NOT_NEEDED and status.available and fresh_search == BEFORE


def test_windows_search_runs_once_and_changes_this_process_only(fresh_search, monkeypatch):
    machine = Machine()
    monkeypatch.setattr(msvc.sys, "platform", "win32")
    monkeypatch.setattr(msvc, "_run", machine.run)
    monkeypatch.setattr(msvc.os.path, "isfile", machine.isfile)
    monkeypatch.setattr(msvc, "_which", machine.which)
    first = msvc.ensure_host_compiler()
    assert first.how == msvc.DISCOVERED and len(machine.commands) == 2
    assert fresh_search["Path"].startswith(TOOLS) and "INCLUDE" in fresh_search
    assert fresh_search["TEMP"] == BEFORE["TEMP"]
    # The outcome is cached: no further processes are started.
    assert msvc.ensure_host_compiler() is first and len(machine.commands) == 2


def test_windows_opt_out_through_the_environment(fresh_search, monkeypatch):
    machine = Machine()
    fresh_search[msvc.OPT_OUT_VARIABLE] = "1"
    monkeypatch.setattr(msvc.sys, "platform", "win32")
    monkeypatch.setattr(msvc, "_run", machine.run)
    monkeypatch.setattr(msvc, "_which", machine.which)
    assert msvc.ensure_host_compiler().how == msvc.DISABLED
    assert machine.commands == [] and "INCLUDE" not in fresh_search


def test_failed_commands_are_reported_as_no_output():
    # A command that cannot be started, as on a machine without cmd.exe.
    assert msvc._run('"/nonexistent/vswhere.exe" -latest') is None


def test_compile_error_names_the_missing_compiler(monkeypatch):
    import sys
    import types

    def failing_source_module(source):
        raise RuntimeError("nvcc preprocessing of a temporary .cu file failed")

    missing = msvc.CompilerStatus(msvc.NOT_FOUND, detail="no Visual Studio installer at X")
    monkeypatch.setattr("pygpumeta._backend.runtime.get_runtime", lambda: None)
    monkeypatch.setitem(sys.modules, "pycuda", types.ModuleType("pycuda"))
    compiler = types.ModuleType("pycuda.compiler")
    compiler.SourceModule = failing_source_module
    monkeypatch.setitem(sys.modules, "pycuda.compiler", compiler)
    monkeypatch.setattr(msvc, "ensure_host_compiler", lambda: missing)
    with pytest.raises(backend_module.CompilationError) as caught:
        backend_module._pycuda_compiler("source")
    message = str(caught.value)
    assert "The MSVC compiler cl.exe was not found (no Visual Studio installer at X)" in message
    assert "'Build Tools for Visual Studio'" in message and "nvcc preprocessing" in message

    present = msvc.CompilerStatus(msvc.ON_PATH, "cl.exe")
    monkeypatch.setattr(msvc, "ensure_host_compiler", lambda: present)
    with pytest.raises(backend_module.CompilationError, match="rejected the kernel source"):
        backend_module._pycuda_compiler("source")
