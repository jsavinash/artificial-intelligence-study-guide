"""Vector store: dense cosine search + sparse BM25 + Reciprocal Rank Fusion (module 17)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b / denom)


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Classic Okapi BM25 sparse scores (module 17's hybrid-search half)."""
    tokenized = [d.lower().split() for d in docs]
    n, avgdl = len(docs), np.mean([len(d) for d in tokenized]) or 1.0
    df = Counter(t for d in tokenized for t in set(d))
    scores = []
    q_terms = query.lower().split()
    for dl, doc in zip([len(d) for d in tokenized], tokenized):
        tf = Counter(doc)
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over multiple retrievers -> ordered doc ids."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)
    return sorted(scores, key=scores.get, reverse=True)


class VectorStore:
    """Minimal in-memory store — the concept behind FAISS/chroma/pgvector."""

    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.vecs: list[np.ndarray] = []
        self.meta: list[dict] = []

    def add(self, doc_id: str, text: str, vector, **meta):
        self.ids.append(doc_id)
        self.texts.append(text)
        self.vecs.append(np.asarray(vector, float))
        self.meta.append(meta)

    def search(self, query_vec, k: int = 3) -> list[tuple[str, float, str]]:
        """Returns [(doc_id, cosine_score, text), ...] descending."""
        q = np.asarray(query_vec, float)
        scored = [(_cos(q, v), i) for i, v in enumerate(self.vecs)]
        scored.sort(reverse=True)
        return [(self.ids[i], s, self.texts[i]) for s, i in scored[:k]]

    def hybrid_search(self, query: str, query_vec, k: int = 3) -> list[tuple[str, float]]:
        """Dense cosine + BM25 fused with RRF — the production default."""
        dense = self.search(query_vec, k=len(self.ids) or 1)
        dense_rank = [d for d, _, _ in dense]
        sparse = bm25_scores(query, self.texts)
        sparse_rank = [self.ids[i] for i in np.argsort(sparse)[::-1]]
        fused = rrf([dense_rank, sparse_rank])
        dense_map = {d: s for d, s, _ in dense}
        out = [(i, dense_map.get(i, 0.0) + 0.01 * (sparse_rank.index(i) < len(sparse_rank))
                if i in sparse_rank else dense_map.get(i, 0.0)) for i in fused[:k]]
        return out

    def __len__(self):
        return len(self.ids)
