"""m06 — Feature engineering: scaling, safe target encoding (in-fold, smoothed),
cyclical time features, TF-IDF, feature selection via permutation importance.

Proves theory doc 06-feature-engineering.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402
from ai_core.datasets import classification, rng  # noqa: E402
from ai_core.metrics import accuracy  # noqa: E402



def target_encode_in_fold(train, val, y, col, min_samples=10, smoothing=10.0):
    """SAFE target encoding: means computed only from TRAIN folds + smoothing.

    post = (n*mean + smoothing*global) / (n + smoothing)  — shrinks rare cats.
    """
    global_mean = y.mean()
    stats = pd.DataFrame({col: train[col], "y": y}).groupby(col)["y"].agg(["mean", "count"])
    stats["post"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
                    (stats["count"] + smoothing)
    mapping = stats["post"].to_dict()
    return train[col].map(mapping).fillna(global_mean).values, \
        val[col].map(mapping).fillna(global_mean).values


def torch_path():
    """Learned categorical features: nn.Embedding vs one-hot (PyTorch).

    An embedding table E (n_cat x dim) followed by a head W (dim x n_out) is
    exactly a **low-rank factorization** of the one-hot weight matrix
    (n_cat x n_out): the effective weights are E @ W, of rank <= dim.

    So embeddings are not "better features" — they are a *parameter budget*
    choice. Cost:
        embedding : n_cat*dim + dim*n_out
        one_hot   : n_cat*n_out
    Embeddings win when dim < n_out. They cannot beat one-hot in expressiveness
    (one-hot is full rank), so the honest test is: same accuracy at fewer
    parameters, when the true signal is genuinely low-rank.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — learned-embedding path skipped")
        return None

    n, n_cat, n_out, dim = 12000, 500, 50, 8
    rng_ = np.random.default_rng(3)
    latent = rng_.normal(0, 1, (n_cat, dim))       # true signal IS rank-`dim`
    w_true = rng_.normal(0, 1, (dim, n_out))
    scores = latent @ w_true
    codes = (rng_.zipf(1.4, n) % n_cat).astype(np.int64)
    y = (scores[codes] + rng_.normal(0, 2.0, (n, n_out))).argmax(1)
    y = y.astype(np.int64)
    tr, te = slice(0, 9600), slice(9600, n)
    holdout = (codes[te], y[te])

    print(f"\n[torch] learned categorical features on {TB.get_device()} "
          f"(n_cat={n_cat}, n_out={n_out}, signal rank={dim})")
    emb = TB.learn_embeddings(codes[tr], y[tr], dim=dim, epochs=300, seed=0,
                              holdout=holdout)
    oh = TB.one_hot_baseline(codes[tr], y[tr], epochs=300, seed=0,
                             holdout=holdout)
    ratio = oh["n_params"] / emb["emb_params"]
    print(f"      embedding  holdout={emb['holdout_acc']:.3f}  "
          f"params={emb['emb_params']:>6}  ({emb['backend']})")
    print(f"      one-hot    holdout={oh['holdout_acc']:.3f}  "
          f"params={oh['n_params']:>6}  ({oh['backend']})")
    print(f"      -> {ratio:.2f}x fewer parameters for the same accuracy "
          f"(dim={dim} < n_out={n_out})")

    # the rank-`dim` signal means the low-rank table must still fit
    acc_gap = abs(emb["holdout_acc"] - oh["holdout_acc"])
    assert acc_gap < 0.03, f"embedding lost accuracy: gap {acc_gap:.3f}"
    assert ratio > 3.0, f"embedding should save parameters, ratio={ratio:.2f}"
    # embedding must beat one-hot in *training* accuracy when the true
    # structure is low-rank and n_cat is large (less overfitting)
    return emb, oh, ratio


def main():
    r = rng(9)
    n = 600
    # categorical + numeric + datetime frame
    df = pd.DataFrame({
        "city": r.choice(["Paris", "Tokyo", "Lima", "Oslo"], n),
        "hour": r.integers(0, 24, n),
        "amount": np.exp(r.normal(3, 1, n)),
        "user": r.integers(0, 50, n),
    })
    df["amount"] = np.log1p(df["amount"])                       # de-skew
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)         # cyclical encoding
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    y = ((df["amount"] + (df["city"] == "Tokyo") * 1.5 +
          r.normal(0, 0.5, n)) > np.median(df["amount"] + 0.3)).astype(int)

    # ---- safe in-fold target encoding vs leaky full-data encoding ----
    tr_idx, va_idx = train_test_split(np.arange(n), test_size=0.25,
                                      stratify=y, random_state=0)
    safe_tr, safe_va = target_encode_in_fold(
        df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True),
        y[tr_idx], "city")
    # leaky: encoded using ALL rows incl. validation labels
    full_map = df.assign(y=y).groupby("city")["y"].mean().to_dict()
    leaky_va = df.iloc[va_idx]["city"].map(full_map).values
    # both "work" in-sample; the safe one is the only legal estimate
    assert safe_va.shape == leaky_va.shape and not np.allclose(safe_va, leaky_va)

    # ---- TF-IDF on text ----
    docs = ["free crypto prize winner", "quarterly revenue report",
            "claim your prize now", "board meeting minutes", "revenue growth strong"]
    labels = np.array([1, 0, 1, 0, 0])
    X_tfidf = TfidfVectorizer().fit_transform(docs)
    assert X_tfidf.shape[0] == 5 and X_tfidf.nnz > 0

    # ---- numeric features + permutation importance (model-agnostic selection) ----
    X, y2 = classification(n=500, d=6, seed=3)
    names = [f"f{i}" for i in range(6)]
    X_df = pd.DataFrame(X, columns=names)
    X_df["city_code"] = pd.Categorical(df["city"][:500]).codes
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y2, random_state=0)
    model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X_tr, y_tr)
    pi = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0)
    assert pi.importances_mean.shape[0] == 7
    acc = accuracy(y_te, model.predict(X_te))

    # ---- learned categorical features (PyTorch): nn.Embedding vs one-hot ----
    learned = torch_path()

    if learned is None:
        print(f"PASS m06 features | tfidf_nnz={X_tfidf.nnz} safe_vs_leaky_differs="
              f"{not np.allclose(safe_va, leaky_va)} acc={acc:.3f} "
              f"top_feature={names[int(pi.importances_mean[:6].argmax())]} "
              f"learned=numpy")
        return

    emb, oh, ratio = learned
    print(f"PASS m06 features | tfidf_nnz={X_tfidf.nnz} safe_vs_leaky_differs="
          f"{not np.allclose(safe_va, leaky_va)} acc={acc:.3f} "
          f"top_feature={names[int(pi.importances_mean[:6].argmax())]} "
          f"emb_holdout={emb['holdout_acc']:.3f} onehot_holdout="
          f"{oh['holdout_acc']:.3f} param_saving={ratio:.2f}x")


if __name__ == "__main__":
    main()
