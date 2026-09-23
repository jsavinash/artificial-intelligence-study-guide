"""m11 — Transformers: scaled dot-product attention from scratch (causal mask),
multi-head split, KV-cache equivalence, and a tiny next-token LM that trains.

Proves theory doc 11-transformers-and-foundation-models.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402


def attention(Q, K, V, causal=False, scale=True):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V  (with optional causal mask)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / (np.sqrt(d_k) if scale else 1.0)
    if causal:
        T = scores.shape[0]
        scores = np.where(np.tril(np.ones((T, T))) == 1, scores, -1e9)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ V, p


def multi_head(X, Wq, Wk, Wv, heads=2):
    """Split d_model into h heads, attend in parallel, concatenate."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d = X.shape[1] // heads
    outs = []
    for h in range(heads):
        o, _ = attention(Q[:, h * d:(h + 1) * d], K[:, h * d:(h + 1) * d],
                         V[:, h * d:(h + 1) * d], causal=True)
        outs.append(o)
    return np.concatenate(outs, axis=1)


def torch_path(data, vocab_size, np_loss):
    """The framework equivalent — a real decoder-only transformer.

    1) one attention head via F.scaled_dot_product_attention (the fused kernel)
    2) a 2-layer GPT-style LM trained with autograd, compared to the NumPy
       bigram-style model above on the SAME corpus.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — NumPy attention/LM path only")
        return None
    print(f"\n[torch] fused attention + transformer LM on {TB.get_device()}")

    # 1) fused attention: same math as our softmax(QK^T/sqrt(d))V, one kernel
    x = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)
    out = TB.attention_output(x, d_head=16, causal=True, seed=0)
    assert out.shape == (8, 16)
    # batched + causal must also work (what real training uses)
    xb = np.random.default_rng(1).normal(size=(4, 6, 16)).astype(np.float32)
    outb = TB.attention_output(xb, d_head=16, causal=True, seed=1)
    assert outb.shape == (4, 6, 16)
    print(f"  SDPA   : head out {out.shape} batched={outb.shape} (is_causal=True)")

    # 2) real transformer LM on the same corpus.
    #    Note device: this model is ~60K params — far too small to amortize GPU
    #    kernel launches, so module 27's rule says run it on the CPU. Same
    #    accuracy, less overhead.
    res = TB.train_tiny_lm(data, vocab=vocab_size, block_size=16, epochs=60,
                           lr=3e-3, d_model=48, n_heads=4, n_layers=2, seed=0,
                           device="cpu")
    print(f"  tinyLM : loss {res['losses'][0]:.3f}->{res['losses'][-1]:.3f} "
          f"next-tok acc={res['acc']:.3f} params={res['n_params']} "
          f"({res['backend']}, {res['seconds']:.2f}s)")
    print(f"  numpy  : loss {np_loss:.3f}  (the hand-written model above)")
    return res



def main():
    r = np.random.default_rng(0)
    T, D = 8, 16
    X = r.normal(size=(T, D))

    # ---- 1) causal attention mechanics ----
    O, P = attention(X, X, X, causal=True)
    assert O.shape == (T, D)
    assert np.allclose(P.sum(axis=1), 1.0)                 # rows are distributions
    upper = P[np.triu(np.ones_like(P), k=1) == 1]
    assert np.all(upper < 1e-9), "future tokens must be masked"
    # position 0 attends ONLY to itself (the definition of causal masking)
    assert np.isclose(P[0, 0], 1.0)

    # ---- 2) sqrt(d_k) scaling keeps softmax from saturating to one-hot ----
    # independent Q,K (same matrix would self-dot to a dominating diagonal)
    Qb = r.normal(size=(T, 64)) * 1.5
    Kb = r.normal(size=(T, 64)) * 1.5
    _, P_unscaled = attention(Qb, Kb, Kb, scale=False)
    _, P_scaled = attention(Qb, Kb, Kb, scale=True)
    ent = lambda P: float(-(P * np.log(P + 1e-12)).sum(axis=1).mean())
    assert ent(P_scaled) > ent(P_unscaled) + 0.01, (ent(P_scaled), ent(P_unscaled))

    # ---- 3) multi-head output shape + KV-cache equivalence ----
    mh = multi_head(X, 0.1 * r.normal(size=(D, D)),
                    0.1 * r.normal(size=(D, D)),
                    0.1 * r.normal(size=(D, D)), heads=4)
    assert mh.shape == (T, D)
    # incremental decode with cached K/V == full batch row t (why KV-cache is exact)
    q_t, K_all, V_all = X[-1], X, X * 1.0
    o_full, _ = attention(X, K_all, V_all, causal=True)
    o_cached, _ = attention(q_t.reshape(1, -1), K_all, V_all, causal=False)
    assert np.allclose(o_full[-1], o_cached[0], atol=1e-9)

    # ---- 4) tiny next-token LM trains (the GPT objective) ----
    corpus = ("machine learning models learn patterns from data. " * 40)
    vocab = sorted(set(corpus))
    stoi = {c: i for i, c in enumerate(vocab)}
    data = np.array([stoi[c] for c in corpus])
    V, E = len(vocab), 12                            # vocab x embedding dim
    Emb = 0.02 * r.normal(size=(V, E))
    W2 = 0.02 * r.normal(size=(E, V))                # embedding -> logits head
    losses = []
    lr = 0.05
    for step in range(400):
        pos = r.integers(0, len(data) - 1)
        x = data[pos:pos + 32]
        target = data[pos + 1:pos + 33]
        h = Emb[x]
        logits = h @ W2
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(len(target)), target] + 1e-12).mean()
        g = p.copy(); g[np.arange(len(target)), target] -= 1
        W2 -= lr * h.T @ g / len(target)
        Emb[x] -= lr * g @ W2.T                         # token embeddings learn
        losses.append(loss)
    uniform = np.log(V)
    assert losses[-1] < uniform * 0.7, (losses[-1], uniform)   # below chance

    print(f"PASS m11 transformers | causal_mask=True rows_sum=1 "
          f"kv_cache_equiv=True mh={mh.shape} "
          f"nxttok_loss {losses[0]:.3f}->{losses[-1]:.3f} < uniform={uniform:.3f}")

    res = torch_path(data, V, losses[-1])
    if res:
        print(f"PASS m11 transformer_torch | backend={TB.backend_label()} "
              f"sdpa_ok=True lm_loss={res['losses'][-1]:.3f} "
              f"lm_nexttok_acc={res['acc']:.3f} lm_params={res['n_params']}")
    else:
        print("SKIP m11 transformer_torch | torch absent — NumPy attention/LM "
              "path above is complete")



if __name__ == "__main__":
    main()
