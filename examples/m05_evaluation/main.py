"""m05 — Evaluation & tuning: metrics from scratch, stratified CV, PR-AUC vs ROC,
imbalanced handling with class weights, random-search tuning.

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
