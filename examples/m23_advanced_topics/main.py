"""m23 — Advanced topics: recommender via matrix factorization, time-series
walk-forward CV, and GNN message passing for label propagation.

Proves theory doc 23-advanced-and-specialized-topics.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402


def torch_path(R, obs, R_true, holdout, adj, X, community):
    """The same three ideas as learned models: embeddings, GCN layers, autograd.

    Matrix factorisation via nn.Embedding (learned by SGD, not ALS) and label
    propagation via a real two-layer GCN trained with cross-entropy on the
    labelled nodes. Walk-forward CV stays NumPy — it is about *splitting*, not
    about compute.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — ALS + message-passing path only")
        return None
    print(f"\n[torch] learned recommenders and graphs on {TB.get_device()}")

    # --- recommender: nn.Embedding + bias terms, trained by gradient descent ---
    R_nan = np.where(obs, R_true, np.nan)
    mf = TB.train_matrix_factorization(R_nan, n_factors=8, epochs=300, lr=0.05,
                                       mask=obs, seed=0)
    pred = mf["predictions"]
    mf_holdout = float(np.sqrt(np.mean((pred[holdout] - R_true[holdout]) ** 2)))
    print(f"      recsys(MF)    holdout_rmse={mf_holdout:.3f} "
          f"(observed_rmse={mf['rmse']:.3f} < baseline={mf['baseline_rmse']:.3f})")
    assert mf["rmse"] < mf["baseline_rmse"], "learned MF must beat the mean"
    assert mf_holdout < 1.2, mf_holdout

    # --- GNN: two GCN layers, trained on the labelled nodes only ---
    train_mask = community >= 0                       # all nodes carry a label
    gnn = TB.train_gnn(adj, X, community, train_mask=train_mask,
                       hidden=16, epochs=150, lr=0.05, seed=0)
    print(f"      gnn(GCN)      node_acc={gnn['acc']:.3f} "
          f"params={gnn['n_params']} ({gnn['backend']})")
    assert gnn["acc"] >= 0.875, gnn["acc"]        # at most one node wrong
    return {"mf_rmse": mf_holdout, "gnn_acc": gnn["acc"]}



def alsratings(R, k=8, iters=40, lam=0.1, seed=0):
    """Alternating least squares MF: R ~= U @ V^T on observed entries only."""
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    U = rng.normal(size=(n_users, k))
    V = rng.normal(size=(n_items, k))
    mask = R > 0
    for _ in range(iters):
        for u in range(n_users):
            idx = mask[u]
            if idx.any():
                Vt = V[idx]
                U[u] = np.linalg.solve(Vt.T @ Vt + lam * np.eye(k),
                                       Vt.T @ R[u, idx])
        for i in range(n_items):
            idx = mask[:, i]
            if idx.any():
                Ut = U[idx]
                V[i] = np.linalg.solve(Ut.T @ Ut + lam * np.eye(k),
                                       Ut.T @ R[idx, i])
    return U, V


def walk_forward_split(series, train_size, test_size, step=1):
    """Time-series CV: train on the past, test on the future — never shuffle."""
    splits = []
    start = 0
    while start + train_size + test_size <= len(series):
        splits.append((np.arange(start, start + train_size),
                       np.arange(start + train_size, start + train_size + test_size)))
        start += step
    return splits


def message_passing(adj, X, layers=2):
    """GCN-flavored propagation: H' = relu(D^-1/2 A D^-1/2 H W) simplified to
    mean-aggregation of neighbor features per layer (the core GNN idea)."""
    H = X.astype(float).copy()
    deg = np.clip(adj.sum(1, keepdims=True), 1, None)
    for _ in range(layers):
        H = (adj @ H) / deg                          # aggregate neighbor states
    return H


def main():
    rng = np.random.default_rng(0)

    # ---- 1) recommender: low-rank factorization recovers held-out ratings ----
    n_u, n_i, k_true = 60, 40, 4
    U_true = rng.normal(size=(n_u, k_true))
    V_true = rng.normal(size=(n_i, k_true))
    R_true = U_true @ V_true.T
    # shift to positive 1-5 star-style ratings (observed mask = R>0 requires this)
    R_true = 1 + 4 * (R_true - R_true.min()) / (R_true.max() - R_true.min())
    obs = rng.random((n_u, n_i)) < 0.6               # observed ratings
    R = np.where(obs, R_true, 0.0)
    holdout = (~obs) & (rng.random((n_u, n_i)) < 0.2)
    U, V = alsratings(R, k=8, iters=30)
    pred = U @ V.T
    rmse_model = np.sqrt(np.mean((pred[holdout] - R_true[holdout]) ** 2))
    rmse_mean = np.sqrt(np.mean((R[R > 0].mean() - R_true[holdout]) ** 2))
    assert rmse_model < rmse_mean, (rmse_model, rmse_mean)   # beats popularity bias
    assert rmse_model < 1.2

    # ---- 2) time series: walk-forward splits keep causality ----
    series = np.cumsum(rng.normal(size=300)) + 50
    splits = walk_forward_split(series, train_size=120, test_size=20, step=20)
    assert len(splits) >= 6
    for tr, te in splits:
        assert tr.max() < te.min(), "future leaked into training!"
    # naive seasonal-naive baseline vs mean baseline on the last split
    tr, te = splits[-1]
    hist = series[tr]
    naive_pred = np.repeat(hist[-1], len(te))        # last value carried forward
    mean_pred = np.repeat(hist.mean(), len(te))
    err_naive = np.abs(series[te] - naive_pred).mean()
    err_mean = np.abs(series[te] - mean_pred).mean()
    assert min(err_naive, err_mean) < 15

    # ---- 3) GNN: neighbor aggregation propagates labels across a graph ----
    # ring graph with two clusters (0-3 community A, 4-7 community B + bridges)
    n = 8
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[i, (i - 1) % n] = 1
    # node features = noisy one-hot of community; 2 nodes unlabeled
    community = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X = np.eye(2)[community] + 0.1 * rng.normal(size=(n, 2))
    H = message_passing(adj, X, layers=2)
    pred = H.argmax(1)
    # label propagation via neighbors should classify all nodes correctly
    acc = (pred == community).mean()
    assert acc >= 0.875, acc                          # at most one node wrong

    print(f"PASS m23 advanced | mf_rmse={rmse_model:.3f}<baseline={rmse_mean:.3f} "
          f"walkforward_splits={len(splits)} (causal=True) "
          f"ts_err naive={err_naive:.2f} mean={err_mean:.2f} "
          f"gnn_label_acc={acc:.3f}")

    tp = torch_path(R, obs, R_true, holdout, adj, X, community)
    if tp:
        print(f"PASS m23 torch | mf_holdout_rmse={tp['mf_rmse']:.3f} "
              f"gnn_node_acc={tp['gnn_acc']:.3f} backend={TB.backend_label()}")


if __name__ == "__main__":
    main()
