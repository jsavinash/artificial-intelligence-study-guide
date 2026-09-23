"""Filesystem model registry + drift monitor (module 21, MLOps).

Real-life shape: artifacts/registry/<model_id>/{model.json, meta.json},
promotion states (staging -> production) and PSI drift scoring.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "artifacts" / "registry"


class ModelRegistry:
    def __init__(self, base: Path | None = None):
        self.base = Path(base) if base is not None else Path(REGISTRY)
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, model_id: str, payload: dict, metrics: dict, tags: dict | None = None) -> Path:
        d = self.base / model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.json").write_text(json.dumps(payload, indent=2, default=_jsonable))
        meta = {"model_id": model_id, "created_at": time.time(),
                "metrics": metrics, "tags": tags or {}, "stage": "staging"}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return d

    def load(self, model_id: str) -> dict:
        d = self.base / model_id
        return json.loads((d / "model.json").read_text())

    def meta(self, model_id: str) -> dict:
        return json.loads((self.base / model_id / "meta.json").read_text())

    def promote(self, model_id: str, stage: str = "production") -> dict:
        m = self.meta(model_id)
        m["stage"] = stage
        m["promoted_at"] = time.time()
        (self.base / model_id / "meta.json").write_text(json.dumps(m, indent=2))
        return m

    def list(self) -> list[dict]:
        return [self.meta(d.name) for d in sorted(self.base.iterdir())
                if (d / "meta.json").exists()]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — the classic data-drift score.

    PSI < 0.1 stable | 0.1–0.25 investigate | > 0.25 retrain.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    p = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    q = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(baseline: np.ndarray, current: np.ndarray) -> dict:
    score = psi(baseline, current)
    verdict = "stable" if score < 0.1 else ("investigate" if score < 0.25 else "retrain")
    return {"psi": round(score, 4), "verdict": verdict,
            "thresholds": {"stable": "<0.1", "investigate": "0.1-0.25", "retrain": ">0.25"}}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
