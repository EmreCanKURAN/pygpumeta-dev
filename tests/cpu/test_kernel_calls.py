"""Run the optimizers against a fake device.

The fake checks every kernel launch against the signature in the CUDA
source: the kernel must exist and each argument must have the type of the
parameter it is passed for.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import pygpumeta
from pygpumeta._backend import module as backend_module
from pygpumeta._backend import selection as backend_selection
from pygpumeta._backend.runtime import Runtime
from pygpumeta.algorithms import _kernel as _base

SIGNATURE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)')
SCALARS = {"int": np.int32, "double": np.float64, "unsigned int": np.uint32}


class FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.dtype = data.dtype
        self.shape = data.shape
        self.gpudata = data

    def get(self) -> np.ndarray:
        return self.data.copy()


class FakeGpuArray:
    @staticmethod
    def zeros(n, dtype):
        return FakeArray(np.zeros(n, dtype))

    empty = zeros

    @staticmethod
    def to_gpu(data):
        return FakeArray(np.array(data))


class FakeModule:
    def __init__(self, text: str, log: list[str]) -> None:
        self.text = text
        self.params = {}
        for name, params in SIGNATURE.findall(text):
            self.params[name] = [" ".join(p.split()) for p in params.split(",")]
        self.log = log
        self.counter = np.zeros(1, np.uint64)

    def get_function(self, name: str):
        params = self.params[name]

        def launch(*args, block, grid, shared=0):
            self.log.append(name)
            assert len(args) == len(params), f"{name}: {len(args)} arguments for {params}"
            for arg, param in zip(args, params, strict=True):
                ctype, _, _ = param.rpartition(" ")
                if "*" in param:
                    ctype = param.split("*")[0].replace("const", "").strip()
                    assert isinstance(arg, FakeArray), f"{name}: {param} got {type(arg).__name__}"
                    assert arg.dtype == SCALARS[ctype], f"{name}: {param} got {arg.dtype}"
                else:
                    assert type(arg) is SCALARS[ctype], f"{name}: {param} got {type(arg).__name__}"
            assert len(block) == 3 and len(grid) == 2 and grid[0] >= 1

        return launch

    def get_global(self, name: str):
        assert name == "d_fe_count"
        return (self.counter, 8)


@pytest.fixture
def device(monkeypatch):
    log: list[str] = []
    modules: list[FakeModule] = []

    def compiler(text):
        modules.append(FakeModule(text, log))
        return modules[-1]

    def memcpy_dtoh(dest, pointer):
        dest[:] = pointer

    def memcpy_htod(pointer, src):
        pointer[: len(src)] = src

    runtime = Runtime(
        driver=SimpleNamespace(memcpy_dtoh=memcpy_dtoh, memcpy_htod=memcpy_htod),
        gpuarray=FakeGpuArray,
        context=SimpleNamespace(synchronize=lambda: None),
        device_name="fake",
        compute_capability=(0, 0),
    )
    real_load = backend_selection.load_kernels
    backend_module.clear_cache()
    monkeypatch.setattr(_base, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        _base,
        "load_kernels",
        lambda constants=None, **kw: real_load(constants, compiler=compiler, **kw),
    )
    monkeypatch.setattr(_base, "_counter_owner", {})
    yield SimpleNamespace(log=log, modules=modules)
    backend_module.clear_cache()


@pytest.mark.parametrize("algorithm", ["abc", "gwo", "woa", "sca", "ssa", "hho"])
@pytest.mark.parametrize("selection", ["native", "tournament", "ballot_elite", "migration"])
def test_every_launch_matches_its_kernel_signature(device, algorithm, selection):
    result = pygpumeta.minimize(
        "levy",
        dim=7,
        algorithm=algorithm,
        selection=selection,
        pop_size=300,
        max_iter=4,
        history_every=2,
    )
    assert result.nit == 4 and result.x.shape == (7,)
    assert result.history.shape == (3,)  # iterations 0 and 2, and the final value
    assert result.termination is pygpumeta.TerminationReason.MAX_ITER
    assert device.log.count("store_history") == 3
    assert_selection_launches(device, selection, 4)


def assert_selection_launches(device, selection, iterations):
    expected = {
        "tournament": "sel_tournament_safe",
        "ballot_elite": "sel_ballot_elite_safe",
        "migration": "sel_tournament_migrate",
    }
    for name, kernel in expected.items():
        assert device.log.count(kernel) == (iterations if name == selection else 0)


def test_iteration_order_of_a_guided_algorithm(device):
    pygpumeta.minimize(
        "sphere", dim=3, algorithm="woa", selection="native", pop_size=10, max_iter=2
    )
    one = ["boundary_clamp", "eval_pop", "find_best_idx", "update_best_ever_pos"]
    step = [*one, "update_guide_if_better", "woa_update", "store_history"]
    assert device.log == ["pop_init", *step, *step, *one, "store_history"]


def test_evaluation_counts_without_a_device_counter(device):
    result = pygpumeta.minimize("sphere", dim=3, algorithm="sca", pop_size=10, max_iter=5)
    assert result.nfev == 60


def test_device_counter_is_handed_over_between_optimizers(device):
    first = pygpumeta.HHO("sphere", dim=3, pop_size=10, max_iter=3)
    second = pygpumeta.HHO("sphere", dim=3, pop_size=10, max_iter=3)
    first.step()
    counter = device.modules[0].counter
    counter[0] = 17  # evaluations the kernels of `first` would have counted
    second.step()
    assert counter[0] == 0
    counter[0] = 5
    assert first.nfev == 17 and second.nfev == 5 and counter[0] == 0
    first.step()
    counter[0] = 2
    assert first.result().nfev == 19 and second.nfev == 5


def test_stopping_rules_on_a_fake_device(device):
    seen = []
    result = pygpumeta.GWO("sphere", dim=3, pop_size=10, max_iter=50, history_every=5).run(
        callback=lambda opt: seen.append(opt.iteration) or len(seen) == 2
    )
    assert seen == [5, 10] and result.nit == 10
    assert result.termination is pygpumeta.TerminationReason.CALLBACK
    # The fake never improves on the initial infinity, so the run stalls.
    stalled = pygpumeta.GWO("sphere", dim=3, pop_size=10, max_iter=50).run(patience=4)
    assert stalled.termination is pygpumeta.TerminationReason.STALLED and stalled.nit == 4
    with pytest.raises(RuntimeError, match="this run has ended"):
        opt = pygpumeta.GWO("sphere", dim=3, pop_size=10, max_iter=1)
        opt.run()
        opt.step()


@pytest.mark.parametrize("algorithm", ["abc", "hho"])
@pytest.mark.parametrize("selection", ["native", "tournament", "ballot_elite", "migration"])
def test_separated_evaluation_launches_match_their_signatures(device, algorithm, selection):
    result = pygpumeta.minimize(
        "levy",
        dim=7,
        algorithm=algorithm,
        selection=selection,
        pop_size=300,
        max_iter=4,
        options={"evaluation": "separated"},
    )
    assert_selection_launches(device, selection, 4)
    assert "d_fe_count" not in device.modules[0].text
    if algorithm == "abc":
        # Initial population, two phases per iteration and one scout row.
        assert result.nfev == 300 + 4 * (2 * 300 + 1)
        assert device.log.count("abc_accept") == 8 and device.log.count("abc_accept_scout") == 4
        assert not {"abc_employed", "abc_scout", "abc_init"} & set(device.log)
    else:
        # The population at the start of each iteration and after the last,
        # and both trial positions of the dives.
        assert result.nfev == 300 * 5 + 4 * 2 * 300
        assert device.log.count("hho_dive") == 4 and "hho_update" not in device.log


def test_cuda_objective_is_compiled_into_the_kernels(device):
    objective = pygpumeta.CudaObjective(
        "double evaluate(const double* x, int d) { return x[0] * x[0]; }"
    )
    result = pygpumeta.minimize(
        objective, (-1.0, 1.0), dim=4, algorithm="hho", pop_size=64, max_iter=2
    )
    text = device.modules[0].text
    assert (
        "__device__ double pgm_user_objective(const double* x, int d) { return x[0] * x[0]; }"
        in text
    )
    assert "if (fid == 1000) return pgm_user_bounded(x, d);" in text
    assert "hho_update_sel" in device.log  # fused by default
    assert result.nit == 2


@pytest.mark.parametrize("algorithm", ["abc", "gwo", "woa", "sca", "ssa", "hho"])
@pytest.mark.parametrize("vectorized", [True, False])
def test_host_objective_sees_every_position(device, algorithm, vectorized):
    shapes = []

    def objective(x):
        shapes.append(x.shape)
        return (x**2).sum(axis=-1) + 1.0

    result = pygpumeta.minimize(
        objective,
        (-2.0, 2.0),
        dim=3,
        algorithm=algorithm,
        pop_size=20,
        max_iter=2,
        mode="host",
        vectorized=vectorized,
    )
    rows = sum(1 if len(s) == 1 else s[0] for s in shapes)
    assert result.nfev == rows
    if vectorized:
        assert set(shapes) <= {(20, 3), (1, 3)}
        assert ((1, 3) in shapes) == (algorithm == "abc")
    else:
        assert set(shapes) == {(3,)}
    per_run = {"abc": 20 + 2 * 41, "hho": 20 * 3 + 2 * 40}.get(algorithm, 60)
    assert rows == per_run


def test_host_objective_values_reach_the_device_bounded(device):
    def objective(x):
        values = np.arange(len(x), dtype=np.float64)
        values[:3] = [np.nan, np.inf, -np.inf]
        return values

    opt = pygpumeta.GWO(objective, (-1.0, 1.0), dim=2, pop_size=6, max_iter=1, mode="host")
    opt.step()
    assert opt._dev.obj.get().tolist() == [1e299, 1e299, -1e299, 3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    ("objective", "error", "message"),
    [
        (lambda x: x.sum(axis=1)[:-1], ValueError, r"an array of shape \(6,\), got shape \(5,\)"),
        (lambda x: x, ValueError, r"got shape \(6, 2\)"),
        (lambda x: ["a"] * len(x), TypeError, "must return real numbers"),
    ],
)
def test_host_objective_errors(device, objective, error, message):
    with pytest.raises(error, match=message):
        pygpumeta.minimize(objective, (-1.0, 1.0), dim=2, pop_size=6, max_iter=1, mode="host")


class NumpyAsCupy:
    """Enough of the CuPy namespace to run the GPU-array evaluator on host arrays."""

    ndarray = np.ndarray
    float64 = np.float64
    nan_to_num = staticmethod(np.nan_to_num)

    @staticmethod
    def asarray(buffer):
        return buffer.data


def test_gpu_array_objective_logic(device, monkeypatch):
    from pygpumeta import _objective

    monkeypatch.setattr(_objective, "import_cupy", lambda: NumpyAsCupy)
    seen = []

    def objective(x):
        seen.append(x.shape)
        out = (x**2).sum(axis=1)
        out[0] = np.nan
        return out.astype(np.float32)

    opt = pygpumeta.SCA(objective, (-1.0, 1.0), dim=3, pop_size=10, max_iter=2)
    opt.step()
    values = opt._dev.obj.get()
    assert seen == [(10, 3)] and values[0] == 1e299 and values.dtype == np.float64
    assert opt.run().nfev == 30

    with pytest.raises(TypeError, match=r"must return a cupy\.ndarray, got list; use mode='host'"):
        pygpumeta.minimize(lambda x: [0.0] * len(x), (-1.0, 1.0), dim=3, pop_size=10, max_iter=1)
    with pytest.raises(ValueError, match=r"shape \(10,\), got shape \(10, 3\)"):
        pygpumeta.minimize(lambda x: x, (-1.0, 1.0), dim=3, pop_size=10, max_iter=1)


def test_missing_cupy_is_explained(device, monkeypatch):
    # An entry of None makes "import cupy" fail whether or not it is installed.
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(
        ImportError, match=r"need CuPy, which could not be imported .* or pass mode='host'"
    ):
        pygpumeta.minimize(lambda x: x.sum(axis=1), (-1.0, 1.0), dim=3, pop_size=10, max_iter=1)
