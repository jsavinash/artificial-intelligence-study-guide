# 04 — Unsupervised Learning

> Back to [index](README.md) · Prev: [03 Supervised Learning](03-supervised-learning-algorithms.md) · Next: [05 Evaluation & Tuning](05-model-evaluation-and-tuning.md)

No labels — the algorithm finds structure. Evaluation is harder: **no ground truth means intrinsic metrics + human judgment.**

## 1. Clustering

### K-Means
- **Algorithm:** pick K centroids → assign points to nearest centroid → recompute centroids → repeat until stable (Lloyd's).
- **Objective:** minimize within-cluster sum of squares (variance).
- **Pros:** simple, scalable O(N·K·iter), good baseline.
- **Cons:** must choose K (use **elbow method** / **silhouette**), assumes spherical equal-size clusters, sensitive to scale & init (use k-means++), can't learn arbitrary shapes, deterministic-ish but init-dependent.

### Hierarchical Agglomerative Clustering (HAC)
- **Bottom-up:** each point starts as a cluster; repeatedly merge closest pairs (linkage: single, complete, average, Ward).
- **Output:** a **dendrogram** — cut at your chosen distance for K clusters. Great when cluster count is unknown or a taxonomy is wanted.
- **Cons:** O(N²)–O(N³), no revisiting merges.

### DBSCAN
- **Idea:** dense regions of points (ε-neighborhood with min_samples) separated by sparse regions; **finds arbitrary shapes and marks outliers as noise**.
- **Pros:** no K needed, detects outliers, non-convex clusters.
- **Cons:** struggles with varying densities; ε and min_samples need tuning; bad in high dimensions (distance concentration).

### Others worth knowing
- **Gaussian Mixture Models (GMM):** soft probabilistic clusters via EM; covariance shapes allowed; gives `P(cluster|point)`.
- **Mean Shift:** density-gradient climbing, automatic K.
- **Spectral clustering:** uses graph Laplacian eigenvectors — non-convex clusters (image segmentation classic).
- **HDBSCAN:** density-based, robust, varying densities — strong modern default.

### Validating clusters
- **Internal:** silhouette (−1..1 cohesion/separation), Davies-Bouldin, inertia (elbow).
- **External (if some labels exist):** adjusted Rand index, normalized mutual information.
- **Ultimately:** do clusters map to actionable segments? (Spike-and-slab human review.)

## 2. Dimensionality Reduction

### PCA (Principal Component Analysis)
- Find orthogonal directions (eigenvectors of covariance) of **max variance**; project onto top-d components. SVD under the hood.
- **Uses:** visualization (2–3 components), denoising, decorrelation, compression, preprocessing (watch leakage!).
- **Assumptions:** linear, variance = importance (can discard low-variance-but-predictive signals).

### t-SNE & UMAP (visualization-focused)
- **t-SNE:** preserves *local* neighborhoods in 2D; great maps, but distances between clusters and perplexity changes are not trustworthy — never read global structure off a t-SNE.
- **UMAP:** faster, better global structure retained than t-SNE, good for big data.
- **Rule:** these are for *seeing*, not for downstream features (usually).

### Others
- **Truncated SVD (LSA):** PCA-like on sparse text matrices.
- **Autoencoders:** neural nonlinear compression ([07](07-deep-learning-fundamentals.md)) — used in anomaly detection & recommendations.
- **Factor analysis / ICA:** latent factors; ICA → independent sources (cocktail party).

## 3. Anomaly/Occusion Detection (outliers)
- Distance/density (LOF, Isolation Forest), statistical thresholds, autoencoder reconstruction error, one-class SVM.
- **Domain:** fraud, manufacturing defects, intrusion detection. Base rates are tiny → precision collapses without care ([05](05-model-evaluation-and-tuning.md)).

## 4. Association Rule Learning
- **Apriori / FP-Growth:** support, confidence, lift → "market baskets" (`bread ⇒ butter`).
- **Domain:** retail merchandising, usage-pattern mining. Largely superseded by embeddings for personalization ([23](23-advanced-and-specialized-topics.md)).

## 5. Topic Modeling
- **LDA (Latent Dirichlet Allocation):** documents = mixtures of word-distribution topics.
- Modern alternative: **BERTopic** (embeddings + UMAP + HDBSCAN) — handles context better than bag-of-words LDA.

## 6. When is unsupervised worth it?
- Labeling is expensive → cluster/embed first, **label only representative points** (active learning).
- EDA & data understanding before any supervised project.
- Pretraining representations (self-supervised → the modern story; [11](11-transformers-and-foundation-models.md)).

## Mastery Checklist

- [ ] Chooses K-Means vs DBSCAN vs GMM vs HAC given data shape & density needs
- [ ] Uses elbow/silhouette to pick K and explains their limits
- [ ] Explains why t-SNE cluster distances lie, and what PCA assumes
- [ ] Runs anomaly detection and understands the low-base-rate precision trap
- [ ] Knows when unsupervised output becomes supervised labels (pseudo-labeling/active learning)
