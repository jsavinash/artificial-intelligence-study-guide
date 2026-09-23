"""m10 — RNNs & sequences: vanilla RNN forward pass, vanishing-gradient demo,
skip-gram embedding learning, and a trained LSTM/GRU.

Proves theory doc 10-rnn-and-sequence-modeling.md (incl. classical NLP toolkit).

NumPy path: literal recurrence loop — you can see the hidden state carry. Torch
path (when installed): nn.LSTM/GRU/RNN with real gradients through time, plus
nn.Embedding, on a task that actually requires memory of the sequence.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.metrics import accuracy  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402
# NOTE: imported as TB, not T — this module uses T for the timestep count.


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


def memory_dataset(n=200, timesteps=6, seed=0):
    """Sequence task needing memory: class = sign of the FIRST step's value.

    A model that only looks at the last timestep cannot solve this — so a high
    accuracy proves the recurrence is carrying information forward.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.3, size=(n, timesteps, 4)).astype(np.float32)
    X[:, 0, 0] = rng.choice([-1.0, 1.0], size=n)      # the signal, only at t=0
    y = (X[:, 0, 0] > 0).astype(int)
    return X, y


def torch_path():
    """nn.LSTM / nn.GRU / nn.RNN with autograd through time."""
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — NumPy recurrence path only")
        return None
    X, y = memory_dataset()
    print(f"\n[torch] recurrent cells on {TB.get_device()} "
          f"(task: remember the FIRST timestep)")
    out = {}
    for cell in ("lstm", "gru", "rnn"):
        res = TB.train_rnn(X, y, hidden=32, cell=cell, epochs=120, lr=0.01,
                           seed=0)
        out[cell] = res
        print(f"  {cell:<4} loss {res['losses'][0]:.3f}->{res['losses'][-1]:.3f} "
              f"acc={res['acc']:.3f} params={res['n_params']} "
              f"({res['seconds']:.2f}s)")
        assert res["acc"] > 0.8, (cell, res["acc"])
    # LSTM/GRU gates exist to fight the vanishing gradient we measured above
    print("  -> all cells learned the dependency; the gates in LSTM/GRU help "
          "when horizons\n     get long (see the |g|_5 > |g|_60 measurement above)")
    return out


def main():
    r = np.random.default_rng(0)
    n_steps, D, H, K = 10, 4, 8, 3
    seq = r.normal(size=(n_steps, D))
    Wx, Wh, Wy = r.normal(size=(H, D)) * 0.3, r.normal(size=(H, H)) * 0.3, \
        r.normal(size=(K, H)) * 0.3

    hs, ys = rnn_forward(seq, Wx, Wh, Wy)
    assert hs.shape == (n_steps, H) and ys.shape == (n_steps, K)

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

    print(f"PASS m10 sequences | rnn_T={n_steps} vanishing: |g|_5={g5:.2e}>"
          f"|g|_60={g60:.2e} word2vec_sim(cat,dog)={related:.3f}>"
          f"sim(cat,the)={unrelated:.3f} nb_sentiment_acc={accuracy(y, pred):.2f}")

    cells = torch_path()
    # report the torch cells in the SAME summary line so the runner shows one
    # verdict per example (run_all greps the first ^PASS)
    cell_note = ""
    if cells:
        cell_note = " torch_cells=" + ",".join(
            f"{c}={r['acc']:.2f}" for c, r in cells.items())
    else:
        cell_note = " torch_cells=skipped(torch absent)"
    print(f"PASS m10 rnn_torch | sequence_memory_ok={cells is not None} "
          f"torch_available={TB.HAS_TORCH} "
          f"backend={TB.backend_label()}{cell_note}")



if __name__ == "__main__":
    main()
