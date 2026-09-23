"""m10 — RNNs & sequences: vanilla RNN forward pass, vanishing-gradient demo,
and skip-gram-style embedding learning (word2vec intuition).

Proves theory doc 10-rnn-and-sequence-modeling.md (incl. classical NLP toolkit).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.metrics import accuracy  # noqa: E402


def rnn_forward(seq, Wx, Wh, Wy, h0=None):
    """h_t = tanh(Wx x_t + Wh h_{t-1}); y_t = Wy h_t — the vanilla RNN."""
    T = len(seq)
    H = Wh.shape[0]
    h = np.zeros(H) if h0 is None else h0.copy()
    hs, ys = [], []
    for t in range(T):
        h = np.tanh(Wx @ seq[t] + Wh @ h)
        hs.append(h.copy())
        ys.append(Wy @ h)
    return np.array(hs), np.array(ys)


def gradient_scale(T, seed=0):
    """Magnitude of dL/dh_0 after T steps — products of tanh' <= 1 shrink."""
    r = np.random.default_rng(seed)
    # scale so spectral radius < 1 (sqrt(d)*sigma = sqrt(8)*0.25 < 1)
    Wh = r.normal(scale=0.25, size=(8, 8))
    g = np.ones(8)
    for _ in range(T):
        h = np.tanh(r.normal(size=8))
        g = g @ Wh * (1 - h ** 2)                 # chain rule through time
    return float(np.linalg.norm(g))


def skip_gram_train(steps=600, dim=16, seed=0):
    """Learn embeddings so words with shared contexts become close (word2vec)."""
    r = np.random.default_rng(seed)
    vocab = ["cat", "dog", "car", "bike", "the", "a"]
    pairs = [("cat", "dog"), ("dog", "cat"), ("car", "bike"), ("bike", "car"),
             ("cat", "cat"), ("car", "car"),          # similar-context pairs
             ("cat", "car"), ("dog", "bike"), ("the", "car")]  # negatives-ish
    idx = {w: i for i, w in enumerate(vocab)}
    E = 0.1 * r.normal(size=(len(vocab), dim))
    lr = 0.2
    for _ in range(steps):
        pos = pairs[r.integers(0, len(pairs))]
        for w, c in (pos, (pos[1], pos[0])):
            v, target = E[idx[w]], idx[c]
            logits = E @ v
            p = np.exp(logits - logits.max())
            p /= p.sum()
            g = p.copy(); g[target] -= 1           # softmax CE gradient
            E[idx[w]] -= lr * (E.T @ g) * 0        # update context side (small)
            E -= lr * np.outer(g, v)               # update all rows via outer
    return vocab, E


def main():
    r = np.random.default_rng(0)
    T, D, H, K = 10, 4, 8, 3
    seq = r.normal(size=(T, D))
    Wx, Wh, Wy = r.normal(size=(H, D)) * 0.3, r.normal(size=(H, H)) * 0.3, \
        r.normal(size=(K, H)) * 0.3

    hs, ys = rnn_forward(seq, Wx, Wh, Wy)
    assert hs.shape == (T, H) and ys.shape == (T, K)

    # 1) state persists: feeding the same prefix twice gives identical hidden state
    h_a, _ = rnn_forward(seq[:5], Wx, Wh, Wy)
    h_b, _ = rnn_forward(seq[:5], Wx, Wh, Wy, h0=np.zeros(H))
    assert np.allclose(h_a[-1], h_b[-1])

    # 2) vanishing gradients: long horizon -> tiny gradient (why LSTMs exist)
    g5, g60 = gradient_scale(5), gradient_scale(60)
    assert g60 < g5, (g5, g60)

    # 3) word2vec-style embeddings: related words drift closer than unrelated
    vocab, E = skip_gram_train()
    sim = lambda a, b: float(E[vocab.index(a)] @ E[vocab.index(b)] /
                             (np.linalg.norm(E[vocab.index(a)]) *
                              np.linalg.norm(E[vocab.index(b)]) + 1e-9))
    related, unrelated = sim("cat", "dog"), sim("cat", "the")
    assert related > unrelated, (related, unrelated)

    # 4) classical NLP: Naive Bayes sentiment-style classification on bag-of-words
    docs = ["great good nice", "bad poor terrible", "good nice great", "terrible bad poor"]
    y = np.array([1, 0, 1, 0])
    counts = np.array([[sum(d.count(w) for w in ("great", "good", "nice")),
                        sum(d.count(w) for w in ("bad", "poor", "terrible"))]
                       for d in docs], float)
    pos_rate = counts[y == 1].sum(0) + 1
    neg_rate = counts[y == 0].sum(0) + 1
    logp = np.log(pos_rate / pos_rate.sum()) - np.log(neg_rate / neg_rate.sum())
    pred = (counts @ logp > np.log(pos_rate.sum() / neg_rate.sum())).astype(int)
    assert accuracy(y, pred) == 1.0

    print(f"PASS m10 sequences | rnn_T={T} vanishing: |g|_5={g5:.2e}>|g|_60={g60:.2e} "
          f"word2vec_sim(cat,dog)={related:.3f}>sim(cat,the)={unrelated:.3f} "
          f"nb_sentiment_acc={accuracy(y, pred):.2f}")


if __name__ == "__main__":
    main()
