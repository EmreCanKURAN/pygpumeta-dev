from __future__ import annotations

import pytest


def _cuda_problem() -> str | None:
    try:
        from pygpumeta._backend.runtime import get_runtime

        get_runtime()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    problem = _cuda_problem()
    if problem is None:
        return
    skip = pytest.mark.skip(reason=f"CUDA unavailable ({problem.splitlines()[0][:120]})")
    for item in gpu_items:
        item.add_marker(skip)
