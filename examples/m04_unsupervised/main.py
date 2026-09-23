"""m04 — Unsupervised learning: K-Means (from scratch), PCA, DBSCAN, silhouette.

Proves theory doc 04-unsupervised-learning.md.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import moons, rng  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402


def torch_path(X, y):
    """Same three algorithms as tensors — K-Means and PCA in torch, GMM by SGD.

    K-Means and PCA are not gradient methods, so torch is used as a *tensor
    engine* (broadcast distances, `svd`) rather than for autograd. The GMM is
    the interesting case: EM's M-step becomes a real optimizer step on the
    log-likelihood, which is how mixtures scale to high dimensions.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — NumPy/scikit-learn path only")
        return None
    print(f"\n[torch] tensor implementations on {TB.get_device()}")
    out = {}

    km = TB.kmeans(X, k=2, epochs=60, seed=0)
    ari_t = adjusted_rand_score(y, km["labels"])
    print(f"      kmeans      ari={ari_t:.3f} inertia={km['inertia']:.1f} "
          f"iters={km['iterations']} ({km['backend']})")

    pc = TB.pca(X, n_components=1)
    pc1 = float(pc["explained_variance_ratio"][0])
    print(f"      pca         pc1_var_ratio={pc1:.3f} "
          f"cumulative={pc['cumulative']:.3f} ({pc['backend']})")

    gm = TB.train_gmm(X, k=2, epochs=250, lr=0.02, seed=0)
    nll0, nll1 = gm["losses"][0], gm["losses"][-1]
    llh = [float(v) for v in gm["log_likelihood"]]     # mean loglik per sample
    print(f"      gmm (SGD)   nll {nll0:.2f} -> {nll1:.2f}  "
          f"loglik {llh[0]:.2f} -> {llh[-1]:.2f}  "
          f"weights_sum={float(np.sum(gm['weights'])):.3f} ({gm['backend']})")

    out = {"kmeans_ari": ari_t, "pca_ratio": pc1, "gmm_gain": nll0 - nll1}
    assert ari_t > 0.9, f"torch kmeans should also recover the blobs ({ari_t})"
    assert pc1 > 0.4
    # log-likelihood and NLL are negatives of each other — check both move the
    # right way, and that they stay consistent (same fit, two conventions).
    assert nll1 < nll0, "NLL must decrease"
    assert llh[-1] > llh[0], "mean log-likelihood must increase"
    assert abs(llh[-1] + nll1) < 1e-6, "loglik and NLL must be exact negatives"
    assert abs(float(np.sum(gm["weights"])) - 1.0) < 1e-3, "mixture weights sum to 1"
    return out



def kmeans(X, k=2, iters=50, seed=0):
    """Lloyd's algorithm: assign -> update -> repeat (k-means++ style init)."""
    r = np.random.default_rng(seed)
    centers = X[r.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.array([X[labels == c].mean(0) if (labels == c).any()
                                else centers[c] for c in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def main():
    # Gaussian blobs: K-Means' spherical assumption HOLDS here (ARI ~ 1.0)
    r = np.random.default_rng(0)
    X = np.vstack([r.normal(loc=[-2, 0], scale=0.6, size=(200, 2)),
                   r.normal(loc=[2, 1], scale=0.6, size=(200, 2))])
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)

    # ---- K-Means recovers the two blobs ----
    labels, centers = kmeans(X, k=2, seed=0)
    ari = adjusted_rand_score(y, labels)
    sil = silhouette_score(X, labels)
    assert ari > 0.9 and sil > 0.5, (ari, sil)

    # ---- elbow: inertia decreases with k (we pick k=2 by the knee) ----
    inertias = []
    for k in (1, 2, 3, 5):
        lab, _ = kmeans(X, k=k, seed=0)
        c = centers if k == 2 else None
        d = np.linalg.norm(X[:, None, :] -
                           np.array([X[lab == i].mean(0) for i in range(k)])[None], axis=2)
        inertias.append(float((d[np.arange(len(X)), lab] ** 2).sum()))
    assert inertias[0] > inertias[1] > inertias[-1]

    # ---- PCA: 2D -> 1D keeps max variance; explained ratio sorted desc ----
    pca = PCA(n_components=1).fit(X)
    ratio = pca.explained_variance_ratio_[0]
    assert 0.4 < ratio <= 1.0
    X1 = pca.transform(X)
    assert X1.shape == (400, 1)

    # ---- DBSCAN finds non-convex clusters + noise (K-Means would fail here) ----
    Xm, ym = moons(n=400, seed=0)
    db = DBSCAN(eps=0.15, min_samples=5).fit(Xm)
    n_clusters = len(set(db.labels_) - {-1})
    noise = (db.labels_ == -1).mean()
    assert n_clusters >= 1 and 0.0 <= noise < 0.6

    print(f"PASS m04 unsupervised | kmeans_ari={ari:.3f} silhouette={sil:.3f} "
          f"pca_var={ratio:.3f} dbscan_k={n_clusters} noise={noise:.2f} "
          f"elbow={'->'.join(f'{i:.0f}' for i in inertias)}")

    tp = torch_path(X, y)
    if tp:
        print(f"PASS m04 torch | kmeans_ari={tp['kmeans_ari']:.3f} "
              f"pca_pc1={tp['pca_ratio']:.3f} gmm_gain={tp['gmm_gain']:.2f} "
              f"backend={TB.backend_label()}")



if __name__ == "__main__":
    main()
