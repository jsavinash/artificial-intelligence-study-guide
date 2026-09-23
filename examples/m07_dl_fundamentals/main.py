"""m07 — Deep learning fundamentals: MLP + backpropagation.

Proves theory doc 07-deep-learning-fundamentals.md.

Two paths, both exercised:
  1. NumPy from-scratch  — forward/backward by hand + gradient check. This is
     the part that teaches you what the framework is doing.
  2. PyTorch autograd    — the same model via nn.Module + loss.backward().
     Runs when torch is installed; the example then asserts both agree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core.metrics import accuracy  # noqa: E402
from ai_core import torch_backend as T  # noqa: E402


def relu(z): return np.maximum(z, 0.0)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    W1, b1, W2, b2 = (params[k] for k in ("W1", "b1", "W2", "b2"))
    z1 = X @ W1 + b1
    a1 = relu(z1)                      # nonlinearity: without it depth is pointless
    logits = a1 @ W2 + b2
    return logits, (X, z1, a1)


def backward(params, cache, y, logits):
    X, z1, a1 = cache
    n = len(X)
    probs = softmax(logits)
    probs[np.arange(n), y] -= 1.0       # dCE/dlogits for softmax+xent
    grads = {}
    grads["W2"] = a1.T @ probs / n
    grads["b2"] = probs.sum(0) / n
    d_a1 = probs @ params["W2"].T
    grads["W1"] = X.T @ (d_a1 * (z1 > 0)) / n
    grads["b1"] = (d_a1 * (z1 > 0)).sum(0) / n
    loss = -np.log(softmax(logits)[np.arange(n), y] + 1e-12).mean()
    return grads, loss


def torch_path(X, y3, n_classes, np_acc):
    """The framework equivalent: autograd replaces our hand-written backward().

    This is what you actually ship — the NumPy version above is the explanation.
    """
    res = T.train_classifier(X, y3, hidden=(16,), epochs=400, lr=0.5,
                             optimizer="sgd", seed=0)
    print(f"  torch   : loss {res['losses'][0]:.3f}->{res['losses'][-1]:.3f} "
          f"acc={res['acc']:.3f} params={res['n_params']} "
          f"({res['backend']}/{res['device']}, {res['seconds']:.2f}s)")
    print(f"  numpy   : acc={np_acc:.3f}  -> "
          f"agreement={abs(res['acc'] - np_acc) < 0.15}")
    return res


def main():
    X, y = classification(n=800, d=6, classes=3, seed=17)
    if y.max() > 2:                     # ensure 3-class setup works
        pass
    # relabel as 3 classes using clustering-free mod for demo stability
    y3 = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.6).astype(int)
    n_classes = int(y3.max()) + 1

    r = np.random.default_rng(0)
    params = {"W1": 0.1 * r.normal(size=(6, 16)), "b1": np.zeros(16),
              "W2": 0.1 * r.normal(size=(16, n_classes)), "b2": np.zeros(n_classes)}
    lr = 0.5
    losses = []
    for epoch in range(400):
        logits, cache = forward(params, X)
        grads, loss = backward(params, cache, y3, logits)
        for k in params:
            params[k] -= lr * grads[k]  # gradient DESCENT — the whole idea
        losses.append(loss)

    assert losses[-1] < losses[0] * 0.35, (losses[0], losses[-1])   # it learned
    logits, _ = forward(params, X)
    acc = accuracy(y3, logits.argmax(1))
    assert acc > 0.8, acc

    # gradient sanity: numerical check on W1[0,0]
    eps = 1e-5
    _, cache = forward(params, X)
    g, _ = backward(params, cache, y3, logits)
    p0 = params["W1"][0, 0]
    params["W1"][0, 0] = p0 + eps
    lp = backward(params, cache, y3, forward(params, X)[0])[1]
    params["W1"][0, 0] = p0 - eps
    lm = backward(params, cache, y3, forward(params, X)[0])[1]
    num = (lp - lm) / (2 * eps)
    assert abs(num - g["W1"][0, 0]) < 1e-4, (num, g["W1"][0, 0])

    print(f"PASS m07 dl_fundamentals | numpy loss {losses[0]:.3f}->"
          f"{losses[-1]:.3f} acc={acc:.3f} gradcheck_analytic="
          f"{g['W1'][0,0]:.5f} numeric={num:.5f}")

    if T.HAS_TORCH:
        print(f"\n[torch] autograd path (same model, loss.backward() instead of "
              f"hand-derived grads) on {T.get_device()}")
        res = torch_path(X, y3, n_classes, acc)
        assert res["losses"][-1] < res["losses"][0], "torch path did not learn"
        assert res["acc"] > 0.8, res["acc"]
        assert abs(res["acc"] - acc) < 0.15, (res["acc"], acc)

    print(f"PASS m07 torch | backend={T.backend_label()} "
          f"torch_available={T.HAS_TORCH} device={T.get_device()}")



if __name__ == "__main__":
    main()
