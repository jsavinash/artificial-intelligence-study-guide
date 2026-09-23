"""Tests for the optional PyTorch accelerator layer (module 27).

These must pass in BOTH configurations — with torch installed and without —
so nothing here assumes torch is importable.
Run: python3 -m pytest tests/test_accelerators.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from ai_core import accelerators as A  # noqa: E402


def test_backend_info_reports_truthfully():
    info = A.backend_info()
    assert info["torch_available"] is A.HAS_TORCH
    assert info["device"] == ("cpu" if A.HAS_TORCH else "numpy")
    assert info["import_cost_s"] > 0


def test_rule_arithmetic_is_backend_independent():
    """break-even = import_cost * s/(s-1): decreasing in speedup, always above
    the import cost itself (asymptote), and > import cost for any finite s."""
    for s in (2, 3, 5, 12, 70):
        be = A.IMPORT_COST * s / (s - 1)
        assert be > A.IMPORT_COST              # approaches C from above
        # a workload just below break-even stays NumPy, just above switches
        assert A.should_accelerate(be * 0.5, s) is False
        if A.HAS_TORCH:
            assert A.should_accelerate(be * 2.0, s) is True
    # large speedups asymptote to the import cost (~1.26s at 70x)
    assert A.IMPORT_COST * 70 / 69 < A.IMPORT_COST * 1.02
    assert A.IMPORT_COST * 5 / 4 == pytest.approx(1.55, abs=0.01)


def test_should_accelerate_gates_on_torch_presence():
    """Without torch there is nothing to accelerate to — must be False."""
    if not A.HAS_TORCH:
        assert A.should_accelerate(999.0, 50) is False
    else:
        assert A.should_accelerate(0.01, 50) is False   # too small to amortize
        assert A.should_accelerate(999.0, 50) is True   # easily amortized


def test_conv_fallback_matches_reference_numerically():
    img, ker = np.random.rand(20, 20), np.random.rand(3, 3)
    ref = A.numpy_conv2d(img, ker)
    got = A.torch_conv2d(img, ker)          # falls back internally if no torch
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4)


def test_attention_fallback_matches_reference_numerically():
    Q, K, V = (np.random.rand(24, 8) for _ in range(3))
    ref, _ = A.numpy_attention(Q, K, V, causal=True)
    got, _ = A.torch_attention(Q, K, V, causal=True)
    assert np.allclose(got, ref, atol=1e-5)


def test_mlp_numpy_backend_learns_without_torch():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=16, n_classes=4,
                               n_informative=8, random_state=0)
    r = A.train_mlp(X, y, epochs=60, backend="numpy")
    assert r["backend"] == "numpy" and r["device"] == "cpu"
    assert r["losses"][-1] < r["losses"][0]          # it learned
    assert r["final_acc"] > 0.5, r["final_acc"]      # better than chance (0.25)
    assert len(r["losses"]) == 60


def test_backends_agree_when_torch_present():
    """Same init distribution + same math => same accuracy (the fairness check)."""
    if not A.HAS_TORCH:
        pytest.skip("torch not installed")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=4000, n_features=32, n_classes=6,
                               n_informative=12, random_state=0)
    np_r = A.train_mlp(X, y, epochs=80, backend="numpy")
    to_r = A.train_mlp(X, y, epochs=80, backend="torch")
    assert abs(np_r["final_acc"] - to_r["final_acc"]) < 0.05


def test_benchmark_report_shape():
    rep = A.benchmark_backends(quiet=True)
    assert rep["torch"] is A.HAS_TORCH
    assert len(rep["results"]) == 3
    for r in rep["results"]:
        assert r["numpy_s"] > 0
        assert isinstance(r["worth_it"], bool)
        if A.HAS_TORCH:
            assert r["torch_s"] is not None and r["speedup"] > 0
