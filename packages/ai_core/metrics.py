"""Evaluation metrics implemented from scratch (module 05 vocabulary)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, labels=(0, 1)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.array([[int(((y_true == a) & (y_pred == b)).sum()) for b in labels]
                     for a in labels])  # rows=true, cols=pred


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred, pos=1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred, pos=1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    return tp / (tp + fn) if tp + fn else 0.0


def f1(y_true, y_pred, pos=1) -> float:
    p, r = precision(y_true, y_pred, pos), recall(y_true, y_pred, pos)
    return 2 * p * r / (p + r) if p + r else 0.0


def mse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0


def cross_entropy(probs, y_true, eps: float = 1e-12) -> float:
    """H(p,q) = -sum p log q ; y_true are integer labels."""
    probs = np.clip(np.asarray(probs, float), eps, 1 - eps)
    y_true = np.asarray(y_true, int)
    return float(-np.log(probs[np.arange(len(y_true)), y_true]).mean())


def kl_divergence(p, q, eps: float = 1e-12) -> float:
    p, q = np.clip(np.asarray(p, float), eps, 1), np.clip(np.asarray(q, float), eps, 1)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def binary_auc(y_true, scores) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # efficient pairwise comparison via sorting
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for val in np.unique(scores):
        mask = scores == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[y_true == 1].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))
