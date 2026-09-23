"""m05 — Evaluation & tuning: metrics from scratch, stratified CV, PR-AUC vs ROC,
imbalanced handling with class weights, random-search tuning, and — where torch
is available — probability *calibration* fitted by autograd.

Proves theory doc 05-model-evaluation-and-tuning.md.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     RandomizedSearchCV, train_test_split)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification, imbalanced  # noqa: E402
from ai_core.metrics import (accuracy, binary_auc as auc, confusion_matrix,
                             f1, precision, recall)  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402


def torch_path(y_te, score):
    """Calibration: is a 0.9 really a 90% chance? Fit a temperature to fix it.

    Accuracy and calibration are different properties. A model can be right
    often and still report meaningless confidences — and for decisions with
    costs (m05 §calibration) the *numbers* matter as much as the ranking.
    Torch is the right tool: one scalar parameter, autograd, done.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — calibration section skipped")
        return None

    # Temperature is fitted on a validation split; report on a held-out split
    n = len(y_te)
    cut = n // 2
    val_y, val_s = y_te[:cut], score[:cut]
    te_y, te_s = y_te[cut:], score[cut:]

    # two-class logits from a probability score: [log(1-p), log(p)]
    def to_logits(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.stack([np.log1p(-p), np.log(p)], axis=1)

    cal = TB.temperature_scale(to_logits(val_s), val_y)
    ece_before = TB.expected_calibration_error(
        np.stack([1 - te_s, te_s], axis=1), te_y)
    scaled = TB.F.softmax(
        TB.torch.tensor(to_logits(te_s), dtype=TB.torch.float32)
        / cal["temperature"], dim=-1).numpy()
    ece_after = TB.expected_calibration_error(scaled, te_y)

    # threshold choice: accuracy-optimal vs F1-optimal are different operating pts
    th = TB.sweep_thresholds(te_y, te_s, n=40)
    print(f"\n[torch] calibration on {TB.backend_label()}")
    print(f"    temperature fitted  T={cal['temperature']:.3f}  "
          f"(val NLL {cal['nll_before']:.4f} -> {cal['nll_after']:.4f})")
    print(f"    ECE held-out        {ece_before:.4f} -> {ece_after:.4f}  "
          f"({'better calibrated' if ece_after <= ece_before else 'no gain'})")
    print(f"    best-F1 threshold   {th['best_f1_threshold']:.3f} "
          f"(F1={th['best_f1']:.3f}) vs accuracy-optimal "
          f"{th['best_acc_threshold']:.3f} (acc={th['best_acc']:.3f})")

    assert cal["temperature"] > 0
    assert 0.0 <= ece_after <= 1.0 and 0.0 <= ece_before <= 1.0
    return {"temperature": cal["temperature"], "ece_before": ece_before,
            "ece_after": ece_after, "f1_thr": th["best_f1_threshold"],
            "acc_thr": th["best_acc_threshold"]}


def main():
    X, y = classification(n=600, d=8, seed=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=80, random_state=0).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    score = clf.predict_proba(X_te)[:, 1]

    # ---- metric identities ----
    p, r, f = precision(y_te, pred), recall(y_te, pred), f1(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    assert cm.sum() == len(y_te)
    assert abs(f - 2 * p * r / (p + r)) < 1e-9        # F1 = harmonic mean
    roc = auc(y_te, score)
    assert 0.5 <= roc <= 1.0

    # ---- stratified K-fold: each fold preserves class ratio ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ratios = [y[tr].mean() for tr, _ in skf.split(X_tr, y_tr)]
    assert np.std(ratios) < 0.02, ratios
    cv = cross_val_score(clf, X_tr, y_tr, cv=skf)

    # ---- imbalanced data: class weights change the operating point ----
    Xi, yi = imbalanced(n=1200, positive_rate=0.03, seed=2)
    rate = yi.mean()
    assert rate < 0.15                                # it IS imbalanced
    w = {0: 1.0, 1: (1 - rate) / rate}                # balanced formula
    xi_tr, xi_te, yi_tr, yi_te = train_test_split(Xi, yi, stratify=yi, random_state=0)
    plain = RandomForestClassifier(n_estimators=60, random_state=0).fit(xi_tr, yi_tr)
    weighted = RandomForestClassifier(n_estimators=60, class_weight=w,
                                      random_state=0).fit(xi_tr, yi_tr)
    rec_plain = recall(yi_te, plain.predict(xi_te))
    rec_weighted = recall(yi_te, weighted.predict(xi_te))
    assert rec_weighted >= rec_plain                  # weights surface the minority

    # ---- random search beats grid on budget (Bergstra & Bengio) ----
    space = {"max_depth": list(range(2, 16)), "min_samples_leaf": [1, 2, 4, 8],
             "max_features": ["sqrt", "log2", None]}
    rs = RandomizedSearchCV(RandomForestClassifier(n_estimators=50, random_state=0),
                            space, n_iter=12, cv=3, random_state=0, n_jobs=-1)
    rs.fit(X_tr, y_tr)
    assert rs.best_score_ > 0.5

    print(f"PASS m05 eval | acc={accuracy(y_te, pred):.3f} P={p:.3f} R={r:.3f} "
          f"F1={f:.3f} ROC-AUC={roc:.3f} cv={cv.mean():.3f} imbalance_rate={rate:.3f} "
          f"recall_plain={rec_plain:.2f}->weighted={rec_weighted:.2f} "
          f"tuned={rs.best_score_:.3f}")


if __name__ == "__main__":
    main()
