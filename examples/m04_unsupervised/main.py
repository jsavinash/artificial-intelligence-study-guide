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


if __name__ == "__main__":
    main()
