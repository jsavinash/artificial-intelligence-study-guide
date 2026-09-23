"""m08 — Training & regularizing networks: Adam vs SGD, dropout, weight decay.

Proves theory doc 08-training-and-regularizing-networks.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core.metrics import accuracy  # noqa: E402


def sgd(params, grads, lr):
    return {k: params[k] - lr * grads[k] for k in params}


def adam(params, grads, state, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    """Adam = momentum + per-parameter adaptive LR + bias correction."""
    out, state = {}, {}
    for k in params:
        g = grads[k]
        m = b1 * state.get(k, (np.zeros_like(g), np.zeros_like(g)))[0] + (1 - b1) * g
        v = b2 * state.get(k, (np.zeros_like(g), np.zeros_like(g)))[1] + (1 - b2) * g ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        out[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)
        state[k] = (m, v)
    return out, state


def train(X, y, *, optimizer, dropout=0.0, wd=0.0, epochs=150, seed=0, lr=None):
    r = np.random.default_rng(seed)
    lr_sgd = lr if (lr is not None and optimizer == "sgd") else 0.2
    lr_adam = lr if (lr is not None and optimizer == "adam") else 0.05
    p = {"W1": 0.1 * r.normal(size=(X.shape[1], 32)), "b1": np.zeros(32),
         "W2": 0.1 * r.normal(size=(32, 2)), "b2": np.zeros(2)}
    state, losses = {}, []
    for t in range(1, epochs + 1):
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        if dropout > 0:                                  # dropout: zero units in TRAIN only
            mask = (r.random(size=a1.shape) > dropout) / (1 - dropout)
            a1 = a1 * mask
        logits = a1 @ p["W2"] + p["b2"]
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        d = probs.copy()
        d[np.arange(len(y)), y] -= 1
        g = {"W2": a1.T @ d / len(y), "b2": d.mean(0)}
        d1 = d @ p["W2"].T * (z1 > 0)
        g["W1"] = X.T @ d1 / len(y) + wd * p["W1"]      # weight decay added to grad
        g["b1"] = d1.mean(0)
        if optimizer == "sgd":
            p = sgd(p, g, lr=lr_sgd)
        else:
            p, state = adam(p, g, state, t, lr=lr_adam)
        losses.append(loss)
    return p, losses


def predict(p, X):
    a1 = np.maximum(X @ p["W1"] + p["b1"], 0)
    return (a1 @ p["W2"] + p["b2"]).argmax(1)


def main():
    X, y = classification(n=600, d=8, seed=21)
    Xtr, ytr = X[:450], y[:450]
    Xte, yte = X[450:], y[450:]

    # 1) both optimizers converge from their standard default LRs
    _, l_sgd = train(Xtr, ytr, optimizer="sgd")
    p_adam, l_adam = train(Xtr, ytr, optimizer="adam")
    assert l_adam[-1] < l_adam[0] * 0.5, (l_adam[0], l_adam[-1])
    assert l_sgd[-1] < l_sgd[0] * 0.5, (l_sgd[0], l_sgd[-1])

    # 2) learning-rate is THE hyperparameter: SGD at lr=30 diverges into instability
    #    (measured: final loss ~2.7 vs ~0.07 at the good lr=0.2; lr=100 -> NaN)
    _, l_sgd_bad = train(Xtr, ytr, optimizer="sgd", lr=30.0)
    assert l_sgd_bad[-1] > l_sgd[-1], (l_sgd_bad[-1], l_sgd[-1])

    # 3) dropout: stochastic in training, deterministic scaled at inference
    p_drop, l_drop = train(Xtr, ytr, optimizer="adam", dropout=0.3, seed=1)
    assert l_drop[-1] < l_drop[0]

    # 3) weight decay shrinks the weight norms (regularization = smaller ||W||)
    p_noreg, _ = train(Xtr, ytr, optimizer="adam", wd=0.0, seed=2)
    p_reg, _ = train(Xtr, ytr, optimizer="adam", wd=0.05, seed=2)
    norm_plain = np.linalg.norm(p_noreg["W1"])
    norm_reg = np.linalg.norm(p_reg["W1"])
    assert norm_reg < norm_plain, (norm_reg, norm_plain)

    # 4) generalization held up on test data
    acc = accuracy(yte, predict(p_adam, Xte))
    assert acc > 0.6, acc

    print(f"PASS m08 training | sgd_final={l_sgd[-1]:.3f} adam_final={l_adam[-1]:.3f} "
          f"sgd_lr30_final={l_sgd_bad[-1]:.3f}(worse) dropout_final={l_drop[-1]:.3f} "
          f"||W||_plain={norm_plain:.3f}>reg={norm_reg:.3f} test_acc={acc:.3f}")


if __name__ == "__main__":
    main()
