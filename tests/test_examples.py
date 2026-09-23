"""Smoke suite: executes every example's main() and asserts the PASS contract.

Run: python3 -m pytest tests/ -q   (or `make test`)
"""
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("m*/main.py"))


def test_all_examples_exist():
    assert len(EXAMPLES) >= 28, f"expected 28 examples, found {len(EXAMPLES)}"


def _ids():
    return [p.parent.name for p in EXAMPLES]


def test_ai_core_imports():
    sys.path.insert(0, str(ROOT / "packages"))
    from ai_core import get_llm, accuracy, VectorStore  # noqa
    from ai_core.llm import MockLLM
    assert get_llm().name in ("mock-llm", "openai", "anthropic")
    assert MockLLM().embed("hello world") != MockLLM().embed("different text")


def test_each_example_prints_pass(request):
    for path in EXAMPLES:
        def _make(p=path):
            def _run():
                r = subprocess.run([sys.executable, str(p)], capture_output=True,
                                   text=True, timeout=300, cwd=str(ROOT),
                                   env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
                assert r.returncode == 0, r.stderr[-2000:]
                assert "PASS" in r.stdout, r.stdout[-1000:]
            return _run
        request.node.add_marker  # noqa - placeholder to keep signature simple
        _run = _make()
        try:
            _run()
        except AssertionError as e:
            pytest_fail(path.parent.name, str(e))


def pytest_fail(name, msg):
    import pytest
    pytest.fail(f"{name}: {msg}")
