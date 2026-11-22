"""Run every example and print its report.

    python examples/run_all.py              # full budgets, GPU
    python examples/run_all.py --quick      # smoke test
    python examples/run_all.py --mode host  # objectives on the CPU

Options are passed on to the examples.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    failed = []
    for script in sorted(here.glob("0*.py")):
        done = subprocess.run([sys.executable, str(script), *sys.argv[1:]], cwd=here, check=False)
        if done.returncode != 0:
            failed.append(script.name)
        print()
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
