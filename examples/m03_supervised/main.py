"""m03 — Supervised algorithms: from-scratch (GD, logistic, KNN, tree split)
plus sklearn ensembles & boosting comparison.

Proves theory doc 03-supervised-learning-algorithms.md.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402


def torch_path(X, yb):
    """The two gradient-based learners via autograd instead of hand-written math.

    Trees/forests/boosting stay in sklearn on purpose — gradient boosting is not
    a GPU/tensor problem, and the [27] decision table says so.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — scratch + sklearn path only")
        return None
    print(f"\n[torch] autograd learners on {TB.get_device()}")
    out = {}

    yv = X[:, 0] * 2.0 + 1.0                      # a known linear target
    lr = TB.fit_linear_regression(X, yv, epochs=400, lr=0.05, seed=0)
    print(f"      linear (GD)   mse={lr['mse']:.4f} weights_ok="
          f"{lr['agrees_with_closed_form']} ({lr['backend']})")

    lg = TB.train_logistic_regression(X, yb, epochs=300, lr=0.1, l2=0.01, seed=0)
    print(f"      logistic      acc={lg['acc']:.3f} params={lg['n_params']} "
          f"({lg['backend']})")

    out = {"linreg_mse": lr["mse"], "logreg_acc": lg["acc"]}
    assert lr["mse"] < 0.05, f"GD should fit a linear target ({lr['mse']})"
    assert lr["agrees_with_closed_form"], "GD must converge to the OLS solution"
    assert lg["acc"] > 0.75, lg["acc"]
    # the loss curves must actually descend — the signature of a working optimizer
    assert lg["losses"][-1] < lg["losses"][0]
    assert lr["losses"][-1] < lr["losses"][0]
    return out



def linreg_gd(X, y, lr=0.1, iters=300):
    """Linear regression via gradient descent (closed-form alternative)."""
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(X)
    for _ in range(iters):
        pred = X @ w + b
        err = pred - y
        w -= lr * (2 / n) * (X.T @ err)
        b -= lr * (2 / n) * err.sum()
    return w, b


def logistic_gd(X, y, lr=0.5, iters=300):
    """Logistic regression: sigmoid + binary cross-entropy gradients."""
    w = np.zeros(X.shape[1]); b = 0.0
    n = len(X)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g) / n
        b -= lr * g.mean()
    return w, b


def knn_predict(X_tr, y_tr, X_q, k=5):
    """Majority vote of k nearest (Euclidean) — instance-based learning."""
    preds = []
    for q in X_q:
        d = np.linalg.norm(X_tr - q, axis=1)
        idx = np.argsort(d)[:k]
        preds.append(int(np.bincount(y_tr[idx]).argmax()))
    return np.array(preds)


def best_gini_split(x, y):
    """One CART step: feature/threshold minimizing weighted Gini impurity."""
    best = (None, None, 1.0)
    for j in range(x.shape[1]):
        for t in np.quantile(x[:, j], np.linspace(0.1, 0.9, 9)):
            L, R = x[:, j] <= t, x[:, j] > t
            if L.sum() == 0 or R.sum() == 0:
                continue
            def gini(z):
                p = y[z].mean()
                return 1 - p ** 2 - (1 - p) ** 2
            w = (L.mean() * gini(L) + R.mean() * gini(R))
            if w < best[2]:
                best = (j, t, w)
    return best


def main():
    X, y = classification(n=700, d=6, classes=2, seed=11)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=1)

    # from-scratch models
    w, b = linreg_gd(X_tr, y_tr.astype(float))
    lr_acc = accuracy_score(y_te, ((X_te @ w + b) > 0.5).astype(int))
    w2, b2 = logistic_gd(X_tr, y_tr.astype(float))
    log_acc = accuracy_score(y_te, (1 / (1 + np.exp(-(X_te @ w2 + b2))) > 0.5).astype(int))
    knn_acc = accuracy_score(y_te[:100], knn_predict(X_tr, y_tr, X_te[:100], k=5))
    j, t, g = best_gini_split(X_tr, y_tr)
    assert j is not None and 0 < g < 1

    # sklearn ensembles: boosting usually wins on tabular data
    results = {}
    for name, mdl in [
        ("logistic", LogisticRegression(max_iter=500)),
        ("tree", DecisionTreeClassifier(max_depth=6, random_state=0)),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=0)),
        ("gbm", GradientBoostingClassifier(random_state=0)),
    ]:
        mdl.fit(X_tr, y_tr)
        results[name] = accuracy_score(y_te, mdl.predict(X_te))

    assert results["gbm"] > results["tree"], results  # ensembling helps
    assert log_acc > 0.55 and knn_acc > 0.55

    fmt = " ".join(f"{k}={v:.3f}" for k, v in results.items())
    print(f"PASS m03 supervised | scratch: linreg={lr_acc:.3f} logreg={log_acc:.3f} "
          f"knn={knn_acc:.3f} tree_split(j={j},g={g:.3f}) | {fmt}")

    tp = torch_path(X, y)
    if tp:
        print(f"PASS m03 torch | linreg_mse={tp['linreg_mse']:.4f} "
              f"logreg_acc={tp['logreg_acc']:.3f} "
              f"backend={TB.backend_label()}")


if __name__ == "__main__":
    main()
