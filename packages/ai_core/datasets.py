"""Synthetic datasets + a small RAG/text corpus — no downloads required (M1/CPU friendly)."""
from __future__ import annotations

import numpy as np


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def classification(n: int = 600, d: int = 8, classes: int = 2, seed: int = 42):
    """Linearly-separable-ish blobs for supervised examples. -> X, y"""
    r = rng(seed)
    X = r.normal(size=(n, d))
    w = r.normal(size=(d, classes))
    y = (X @ w + 0.3 * r.normal(size=(n, classes))).argmax(axis=1)
    if classes == 2:
        y = (X @ w[:, 0] - X @ w[:, 1] + 0.5 * r.normal(size=n) > 0).astype(int)
    return X.astype(float), y


def regression(n: int = 500, d: int = 6, noise: float = 0.5, seed: int = 7):
    """y = Xw + b + noise. -> X, y"""
    r = rng(seed)
    X = r.normal(size=(n, d))
    w = r.normal(size=d)
    y = X @ w + 2.0 + noise * r.normal(size=n)
    return X, y


def imbalanced(n: int = 1000, positive_rate: float = 0.02, seed: int = 3):
    """Rare-event fraud-style data with an exact positive rate + learnable signal.

    Returns X, y where y.mean() ~= positive_rate.
    """
    r = rng(seed)
    X = r.normal(size=(n, 6))
    n_pos = max(5, int(round(n * positive_rate)))
    idx = r.choice(n, size=n_pos, replace=False)
    y = np.zeros(n, dtype=int)
    y[idx] = 1
    X[idx] += 1.3          # positives shifted -> model can learn something
    return X, y


def moons(n: int = 400, seed: int = 0):
    """Two interleaving half-circles for clustering/nonlinear demos."""
    r = rng(seed)
    n1 = n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n - n1)
    X = np.vstack([
        np.c_[np.cos(t1), np.sin(t1)],
        np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5],
    ]) + 0.08 * r.normal(size=(n, 2))
    y = np.r_[np.zeros(n1), np.ones(n - n1)].astype(int)
    return X, y


def image_grid(size: int = 8, seed: int = 0):
    """Fake 'image' tensor (C,H,W) for CNN/contrastive demos."""
    r = rng(seed)
    img = r.normal(size=(3, size, size))
    img[:, : size // 2] += 1.0        # top half brighter (learnable pattern)
    return img


# ---------------- RAG corpus (used by m17, m18, apps) ----------------

RAG_CORPUS = [
    {"id": "d1", "title": "Retrieval-Augmented Generation",
     "text": "RAG retrieves relevant documents at query time and places them in the prompt so the model answers from grounded context with citations. It fixes knowledge cutoffs and hallucination without retraining."},
    {"id": "d2", "title": "LoRA fine-tuning",
     "text": "LoRA freezes the base weight W and learns a low-rank update dW = B A where rank r is much smaller than the dimension. Adapters can be merged into the base weights for zero inference overhead."},
    {"id": "d3", "title": "Vector databases",
     "text": "A vector database indexes embedding vectors so nearest-neighbor search returns semantically similar chunks. Hybrid search combines dense vectors with BM25 keyword scores using reciprocal rank fusion."},
    {"id": "d4", "title": "Prompt injection attacks",
     "text": "Prompt injection hides instructions inside user input or retrieved documents to override the system prompt. Defenses treat untrusted content as data, restrict tool authority, and scan outputs."},
    {"id": "d5", "title": "Model Context Protocol",
     "text": "MCP is an open protocol where servers expose tools, resources and prompts and clients discover them dynamically. It acts like USB-C for AI tooling and replaces point-to-point integrations."},
    {"id": "d6", "title": "Scaling laws",
     "text": "Language model loss follows a power law in parameters, data and compute. The Chinchilla result shows roughly twenty training tokens per parameter is compute-optimal for decoder-only transformers."},
    {"id": "d7", "title": "Batch normalization vs layer normalization",
     "text": "Batch norm normalizes across the batch dimension and depends on batch size, while layer norm normalizes per sample. Transformers use layer norm or RMS norm because decoding batches are size one."},
    {"id": "d8", "title": "Drift monitoring in production",
     "text": "Data drift measures shift in input distributions using PSI or Kolmogorov-Smirnov tests, while concept drift means the relationship between inputs and labels changed. Drift triggers retraining pipelines."},
]


def rag_corpus():
    return [dict(d) for d in RAG_CORPUS]
