from __future__ import annotations

import re
import subprocess
import sys

import pytest

from pygpumeta._backend import module as mod
from pygpumeta._backend import source
from pygpumeta._backend.selection import grid_for, load_kernels


class FakeModule:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requested: list[str] = []

    def get_function(self, name: str) -> str:
        self.requested.append(name)
        return f"<{name}>"


class FakeCompiler:
    def __init__(self) -> None:
        self.modules: list[FakeModule] = []

    def __call__(self, text: str) -> FakeModule:
        self.modules.append(FakeModule(text))
        return self.modules[-1]


@pytest.fixture(autouse=True)
def fresh_cache():
    mod.clear_cache()
    yield
    mod.clear_cache()


def test_import_has_no_side_effects(tmp_path):
    code = (
        "import os, sys; before = set(os.listdir('.')); import pygpumeta, pygpumeta.reference;"
        "assert not [m for m in sys.modules if m.split('.')[0] in ('pycuda', 'cupy')];"
        "assert set(os.listdir('.')) == before; print(pygpumeta.__version__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+\S*\n", out.stdout)


def test_prelude_uses_exact_double_constants():
    prelude = source.constants_prelude()
    assert "#define PGM_PI 3.141592653589793\n" in prelude
    assert "#define PGM_TWO_PI 6.283185307179586\n" in prelude
    assert "#define PGM_E 2.718281828459045\n" in prelude


def test_prelude_overrides_are_validated():
    assert "#define PGM_TWO_PI 6.283185307\n" in source.constants_prelude(
        {"PGM_TWO_PI": "6.283185307"}
    )
    with pytest.raises(KeyError, match="unknown constant 'PGM_TAU'"):
        source.constants_prelude({"PGM_TAU": "6.28"})
    with pytest.raises(ValueError, match="decimal floating-point literal"):
        source.constants_prelude({"PGM_PI": "3.14; system()"})


def test_sources_contain_no_literal_constants_and_only_known_macros():
    text = source.read_kernel_file(source.ALGORITHMS_FILE)
    assert not re.search(r"3\.14159|2\.71828|6\.28318", text)
    assert set(re.findall(r"PGM_\w+", text)) == set(source.CONSTANTS)
    assert "PGM_" not in source.selection_source()


def test_expected_kernels_are_defined():
    algorithms = source.kernel_names(source.algorithms_source())
    selection = source.kernel_names(source.selection_source())
    assert len(algorithms) == 49
    assert {
        "eval_pop",
        "gwo_update",
        "hho_update_sel",
        "abc_scout",
        "hho_dive",
        "abc_accept",
    } <= algorithms
    assert selection == {
        "sel_tournament_safe",
        "sel_ballot_elite_safe",
        "sel_ballot_local_safe",
        "sel_ballot_noswitch_safe",
        "sel_tournament_elitist_dev",
        "sel_globalbest",
        "sel_tournament_migrate",
        "argmin_block",
        "roulette_sample",
        "sel_tournament_shared",
        "update_best_ever_pos",
        "store_history",
    }


def test_modules_are_compiled_once_per_source():
    compiler = FakeCompiler()
    first = load_kernels(compiler=compiler)
    second = load_kernels(compiler=compiler)
    assert first.algorithms is second.algorithms and first.selection is second.selection
    assert len(compiler.modules) == 2
    other = load_kernels({"PGM_TWO_PI": "6.283185307"}, compiler=compiler)
    assert other.algorithms is not first.algorithms and other.selection is first.selection
    assert len(compiler.modules) == 3


def test_kernel_handles_are_cached_and_unknown_names_explained():
    compiler = FakeCompiler()
    module = mod.CompiledModule(source.selection_source(), label="selection", compiler=compiler)
    assert module.kernel("argmin_block") == "<argmin_block>"
    module.kernel("argmin_block")
    assert compiler.modules[0].requested == ["argmin_block"]
    with pytest.raises(
        KeyError, match="module 'selection' has no kernel named 'nope'; available: argmin_block"
    ):
        module.kernel("nope")
    assert "12 kernels" in repr(module)


@pytest.mark.parametrize(("n", "blocks"), [(1, 1), (256, 1), (257, 2), (8192, 32)])
def test_grid(n, blocks):
    assert grid_for(n) == (blocks, 1)


def test_counting_and_negating_variants_wrap_the_objective_once():
    plain = source.algorithms_source()
    counting = source.algorithms_source(count_evaluations=True)
    negating = source.algorithms_source(negate=True)
    both = source.algorithms_source(count_evaluations=True, negate=True)
    assert "d_fe_count" not in plain and "d_fe_count" not in negating
    assert counting.count("atomicAdd(&d_fe_count, 1ULL); return evaluate_raw(x, d, fid); }") == 1
    assert negating.count("{ return -evaluate_raw(x, d, fid); }") == 1
    assert both.count("atomicAdd(&d_fe_count, 1ULL); return -evaluate_raw(x, d, fid); }") == 1
    for variant in (counting, negating, both):
        assert variant.count("__device__ double evaluate_raw(double* x, int d, int fid) {") == 1
        assert source.kernel_names(variant) == source.kernel_names(plain)
    # Apart from the wrapper the sources are the same text.
    wrapper = source._wrapped_evaluate(True, False)
    assert counting.replace(wrapper, source._EVALUATE_SIGNATURE) == plain
