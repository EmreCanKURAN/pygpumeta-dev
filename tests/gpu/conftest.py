from __future__ import annotations

import numpy as np
import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/gpu/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def runtime():
    from pygpumeta._backend.runtime import get_runtime

    return get_runtime()


@pytest.fixture(scope="session")
def selection(runtime):
    from pygpumeta._backend.selection import SelectionKernels

    return SelectionKernels()


@pytest.fixture(scope="session")
def paper(runtime):
    """The paper implementation; importing it compiles its kernels."""
    from legacy import gpu

    return gpu


class Buffers:
    def __init__(self, runtime, values: np.ndarray) -> None:
        ga = runtime.gpuarray
        self.n = len(values)
        self.obj = ga.to_gpu(np.ascontiguousarray(values, np.float64))
        self.fit = ga.zeros(self.n, np.float64)
        self.sel = ga.zeros(self.n, np.int32)
        self.mean = ga.zeros(1, np.float64)

    def selected(self) -> np.ndarray:
        return self.sel.get().astype(np.int64)


@pytest.fixture
def buffers(runtime):
    return lambda values: Buffers(runtime, values)
