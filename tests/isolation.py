"""Checks that have to run in a fresh interpreter.

Whether a piece of code imports a GPU library can only be observed where no
earlier test has imported it. On a machine with CUDA the test session itself
imports PyCUDA while deciding whether to skip the GPU tier.
"""

from __future__ import annotations

import subprocess
import sys

GPU_LIBRARIES = ("pycuda", "cupy")
_CHECK = (
    "\nimport sys\n"
    f"loaded = sorted(m for m in sys.modules if m.split('.')[0] in {GPU_LIBRARIES!r})\n"
    "assert not loaded, f'GPU libraries were imported: {loaded}'\n"
)


def run_without_gpu_libraries(code: str) -> str:
    """Run ``code`` in a new interpreter and fail if it imported a GPU library.

    Returns what the code printed.
    """
    done = subprocess.run(
        [sys.executable, "-c", code + _CHECK], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout
