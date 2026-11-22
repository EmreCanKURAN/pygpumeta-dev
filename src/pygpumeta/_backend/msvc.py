"""Finding the MSVC compiler on Windows.

``nvcc`` drives ``cl.exe`` whenever a kernel is compiled, and expects it on
the PATH. A normal command prompt does not have it there; only the prompts
that Visual Studio installs, which run ``vcvars64.bat`` first, do. This
module does what those prompts do, for the current process only: it asks
Visual Studio where it is installed, runs ``vcvars64.bat`` in a child
process and copies the environment that results.

Nothing outside the process is changed, and nothing happens at import.
"""

from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

OPT_OUT_VARIABLE = "PYGPUMETA_NO_MSVC_DISCOVERY"
INSTALL_HINT = (
    "Install 'Build Tools for Visual Studio' (or Visual Studio) with the workload "
    "'Desktop development with C++'."
)
_VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

# How the compiler was located.
ON_PATH = "on the PATH"
DISCOVERED = "found through Visual Studio"
DISABLED = "discovery switched off"
NOT_FOUND = "not found"
NOT_NEEDED = "not needed on this platform"


@dataclass(frozen=True)
class CompilerStatus:
    """Outcome of the search for the host compiler.

    Attributes
    ----------
    how : str
        One of the module constants ``ON_PATH``, ``DISCOVERED``,
        ``DISABLED``, ``NOT_FOUND`` and ``NOT_NEEDED``.
    path : str or None
        Location of ``cl.exe`` when it is available.
    detail : str
        For ``DISCOVERED`` the Visual Studio installation; otherwise the
        reason why no compiler is available.
    """

    how: str
    path: str | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        """Whether kernels can be compiled as far as the host compiler goes."""
        return self.how in (ON_PATH, DISCOVERED, NOT_NEEDED)


Runner = Callable[[str], "str | None"]
Which = Callable[[str, "str | None"], "str | None"]


def _lookup(environ: Mapping[str, str], name: str) -> str | None:
    """Read a variable without regard to the case of its name, as Windows does."""
    wanted = name.upper()
    return next((value for key, value in environ.items() if key.upper() == wanted), None)


def vswhere_location(environ: Mapping[str, str]) -> str:
    """Return the fixed location of ``vswhere.exe``."""
    root = _lookup(environ, "ProgramFiles(x86)") or r"C:\Program Files (x86)"
    return ntpath.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe")


def parse_environment(text: str) -> dict[str, str]:
    """Parse the output of the ``set`` command into a mapping."""
    variables = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        # cmd keeps per-drive working directories in names such as "=C:".
        if separator and name:
            variables[name] = value
    return variables


def changed_variables(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, str]:
    """Return the variables of ``after`` that are new or differ from ``before``.

    Names are compared without regard to case, as Windows does.
    """
    known = {name.upper(): value for name, value in before.items()}
    return {name: value for name, value in after.items() if known.get(name.upper()) != value}


def discover(
    environ: Mapping[str, str],
    *,
    run: Runner,
    isfile: Callable[[str], bool],
    which: Which,
) -> tuple[CompilerStatus, dict[str, str]]:
    """Locate ``cl.exe`` and return the variables that make it usable.

    Parameters
    ----------
    environ : mapping
        The current environment.
    run : callable
        Runs a command line and returns its standard output, or None when
        the command fails.
    isfile : callable
        Tells whether a path is an existing file.
    which : callable
        ``which(name, path)`` looks a program up on a search path; None for
        the path means the PATH of ``environ``.

    Returns
    -------
    status : CompilerStatus
    variables : dict
        Environment variables to set in this process; empty unless the
        compiler was found through Visual Studio.
    """
    on_path = which("cl", _lookup(environ, "PATH"))
    if on_path:
        return CompilerStatus(ON_PATH, on_path), {}
    if (_lookup(environ, OPT_OUT_VARIABLE) or "").strip() not in ("", "0"):
        return CompilerStatus(DISABLED, detail=f"{OPT_OUT_VARIABLE} is set"), {}
    vswhere = vswhere_location(environ)
    if not isfile(vswhere):
        return CompilerStatus(NOT_FOUND, detail=f"no Visual Studio installer at {vswhere}"), {}
    listing = run(
        f'"{vswhere}" -latest -products * -requires {_VC_TOOLS_COMPONENT} '
        "-property installationPath"
    )
    installation = next((line.strip() for line in (listing or "").splitlines() if line.strip()), "")
    if not installation:
        reason = "no Visual Studio installation with the C++ x64 tools"
        return CompilerStatus(NOT_FOUND, detail=reason), {}
    vcvars = ntpath.join(installation, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not isfile(vcvars):
        return CompilerStatus(NOT_FOUND, detail=f"{vcvars} is missing"), {}
    # The outer quotes belong to cmd /s /c, the inner ones to the path.
    dump = run(f'cmd.exe /d /s /c ""{vcvars}" >nul 2>&1 && set"')
    variables = changed_variables(environ, parse_environment(dump or ""))
    search_path = _lookup(variables, "PATH")
    compiler = which("cl", search_path) if search_path else None
    if not compiler:
        return CompilerStatus(NOT_FOUND, detail=f"{vcvars} did not provide cl.exe"), {}
    return CompilerStatus(DISCOVERED, compiler, installation), variables


_lock = threading.Lock()
_status: CompilerStatus | None = None


def ensure_host_compiler() -> CompilerStatus:
    """Make the host compiler usable by this process and say how it was found.

    On Windows, when ``cl.exe`` is not on the PATH, the environment of
    ``vcvars64.bat`` is applied to this process. The search runs once per
    process. Set the environment variable ``PYGPUMETA_NO_MSVC_DISCOVERY`` to
    ``1`` to leave the environment alone.
    """
    global _status
    with _lock:
        if _status is None:
            if sys.platform != "win32":
                _status = CompilerStatus(NOT_NEEDED)
            else:
                environ = _environment()
                _status, variables = discover(
                    environ, run=_run, isfile=os.path.isfile, which=_which
                )
                environ.update(variables)
        return _status


def reset() -> None:
    """Forget the outcome of the search; the environment is left as it is."""
    global _status
    with _lock:
        _status = None


def describe_missing(status: CompilerStatus) -> str:
    """Explain a missing compiler and how to get one."""
    if status.how == DISABLED:
        return (
            f"The MSVC compiler cl.exe is not on the PATH and the search for Visual Studio is "
            f"switched off ({status.detail}). Unset that variable, or start Python from the "
            "'x64 Native Tools Command Prompt for VS'."
        )
    return (
        f"The MSVC compiler cl.exe was not found ({status.detail}). {INSTALL_HINT} If it is "
        "installed in an unusual place, start Python from the 'x64 Native Tools Command "
        "Prompt for VS'."
    )


def _environment() -> MutableMapping[str, str]:
    return os.environ


def _run(command: str) -> str | None:
    try:
        done = subprocess.run(
            command, capture_output=True, timeout=120, check=False, encoding="oem", errors="replace"
        )
    except (OSError, subprocess.SubprocessError, LookupError):
        return None
    return done.stdout if done.returncode == 0 else None


def _which(name: str, path: str | None) -> str | None:
    return shutil.which(name, path=path)
