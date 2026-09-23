"""m21 — MLOps: model registry with staging->production promotion, PSI drift
monitoring, batch vs online serving patterns, prediction logging.

Proves theory doc 21-mlops-and-production-ml.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core.metrics import accuracy  # noqa: E402
from ai_core.registry import ModelRegistry, drift_report, psi  # noqa: E402


def batch_serve(model, X):
    """Batch pattern: score a whole DataFrame/nightly job in one call."""
    t0 = time.perf_counter()
    preds = model.predict_proba(X)[:, 1]
    return preds, (time.perf_counter() - t0) * 1000


def online_serve(model, row):
    """Online pattern: single-row REST prediction (per-request latency matters)."""
    t0 = time.perf_counter()
    p = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    return p, (time.perf_counter() - t0) * 1000


def main():
    X, y = classification(n=600, d=6, seed=31)
    Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]

    # ---- 1) train + evaluate before registering ----
    model = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    acc = accuracy(yte, model.predict(Xte))
    assert acc > 0.6

    # ---- 2) registry: save with metrics, load, promote staging -> production ----
    reg = ModelRegistry()
    payload = {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
               "n_features": int(X.shape[1])}
    path = reg.save("logreg-v1", payload, metrics={"acc": round(acc, 4)},
                    tags={"algo": "logreg", "data": "synthetic-v1"})
    assert (path / "model.json").exists()
    loaded = reg.load("logreg-v1")
    assert np.allclose(loaded["coef"], model.coef_)
    meta = reg.promote("logreg-v1", "production")
    assert meta["stage"] == "production"
    listing = reg.list()
    assert any(m["model_id"] == "logreg-v1" for m in listing)

    # ---- 3) drift monitoring with PSI thresholds ----
    stable = drift_report(Xte[:, 0], Xte[:, 0] + 0.01 * np.random.default_rng(0)
                          .normal(size=len(Xte)))
    shifted = drift_report(Xtr[:, 0], Xtr[:, 0] + 2.5)   # distribution moved hard
    assert stable["verdict"] == "stable", stable
    assert shifted["verdict"] == "retrain", shifted
    assert psi(Xtr[:, 0], Xtr[:, 0]) < 1e-9             # identical => PSI 0

    # ---- 4) batch vs online serving ----
    batch_preds, batch_ms = batch_serve(model, Xte)
    row_ms, online_ms = [], []
    for i in range(20):
        p, ms = online_serve(model, Xte[i])
        row_ms.append(p); online_ms.append(ms)
    assert len(batch_preds) == len(Xte)
    assert abs(row_ms[0] - batch_preds[0]) < 1e-9       # same model, both paths
    assert batch_ms < sum(online_ms)                    # batch amortizes overhead

    # ---- 5) prediction logging (feeds future training + audits) ----
    log_lines = [f"{i},{p:.4f},{int(yte[i])}" for i, p in enumerate(batch_preds)]
    log_path = Path(__file__).resolve().parents[2] / "artifacts" / "prediction_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text("idx,score,label\n" + "\n".join(log_lines))
    assert log_path.exists() and len(log_path.read_text().splitlines()) == len(Xte) + 1

    print(f"PASS m21 mlops | acc={acc:.3f} registry={meta['stage']} "
          f"drift[stable={stable['verdict']},shifted={shifted['verdict']}] "
          f"batch={batch_ms:.2f}ms vs online_avg={np.mean(online_ms):.2f}ms "
          f"pred_log={len(log_lines)} rows")


if __name__ == "__main__":
    main()
