"""m02 — ML foundations: full workflow + DATA LEAKAGE demo (the concept that ruins projects).

Proves theory doc 02-machine-learning-foundations.md.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402


def torch_path(Xt_tr, y_tr, Xt_te, y_te):
    """The same workflow with a learned model instead of sklearn.

    Shows the two things worth internalising: (1) autograd gets to the same
    answer as a solver, and (2) leakage is *not* a library problem — a leaked
    feature makes any learner report a fantasy score. Regularisation is the
    torch lesson here: L2 is one keyword argument.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — sklearn path only")
        return None
    print(f"\n[torch] workflow with autograd on {TB.get_device()}")

    clean = TB.train_logistic_regression(Xt_tr, y_tr, epochs=300, lr=0.1,
                                         l2=1e-3, seed=0)
    test_acc = float((clean["predict"](Xt_te) == y_te).mean())
    print(f"      clean      train={clean['acc']:.3f} test={test_acc:.3f} "
          f"gap={clean['acc'] - test_acc:.3f} params={clean['n_params']}")
    assert test_acc > 0.7, test_acc

    # the leaky version: same learner, same hyperparameters, poisoned features
    leaky = TB.train_logistic_regression(np.c_[Xt_tr, y_tr], y_tr, epochs=300,
                                         lr=0.1, seed=0)
    print(f"      leaky      train={leaky['acc']:.3f} "
          f"(perfect score from a post-outcome feature — the classic smell)")
    assert leaky["acc"] > 0.99, leaky["acc"]

    # regularisation strength is a real knob, not decoration
    weak = TB.train_logistic_regression(Xt_tr, y_tr, epochs=300, lr=0.1,
                                        l2=0.0, seed=0)
    print(f"      l2 sweep   l2=0.0 acc={weak['acc']:.3f} vs "
          f"l2=1e-3 acc={clean['acc']:.3f}")
    return {"clean_test": test_acc, "gap": clean["acc"] - test_acc}



def main():
    X, y = classification(n=800, d=10, classes=2, seed=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=0)

    # ---- Baseline first (always!) ----
    majority = max(np.bincount(y_tr)) / len(y_tr)
    assert majority > 0.4

    # ---- CLEAN training: fit scaler on TRAIN only, apply to test ----
    mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-9
    Xt_tr, Xt_te = (X_tr - mu) / sigma, (X_te - mu) / sigma
    clean_clf = LogisticRegression(max_iter=500).fit(Xt_tr, y_tr)
    clean_acc = accuracy_score(y_te, clean_clf.predict(Xt_te))

    # ---- LEAKY training: fit scaler on ALL data (test stats leak into train) ----
    mu_all, sigma_all = X.mean(0), X.std(0) + 1e-9
    # emulate a harsher real leak: append a feature that IS the label (post-outcome)
    leak_train = np.c_[Xt_tr, y_tr]                 # label leaked as feature!
    leak_model = LogisticRegression(max_iter=500).fit(leak_train, y_tr)
    leak_train_acc = accuracy_score(y_tr, leak_model.predict(leak_train))
    # in production the leaked column doesn't exist -> we simulate by using
    # the true test labels (unavailable at predict time) to show the fantasy number
    fantasy = accuracy_score(y_te, leak_model.predict(np.c_[Xt_te, y_te]))
    assert leak_train_acc > 0.99, leak_train_acc     # "perfect" = leakage smell

    # ---- Generalization gap on the clean model ----
    train_acc = accuracy_score(y_tr, clean_clf.predict(Xt_tr))
    gap = train_acc - clean_acc
    assert 0.0 <= gap < 0.35                          # sane, not overfitting

    print(f"PASS m02 workflow | baseline={majority:.3f} clean_test={clean_acc:.3f} "
          f"leaky_fantasy={fantasy:.3f} (illusory) gap={gap:.3f}")

    tp = torch_path(Xt_tr, y_tr, Xt_te, y_te)
    if tp:
        print(f"PASS m02 torch | clean_test={tp['clean_test']:.3f} "
              f"gap={tp['gap']:.3f} backend={TB.backend_label()}")


if __name__ == "__main__":
    main()
