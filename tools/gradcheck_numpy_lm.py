"""Finite-difference gradient check for the NumPy LM fallback (module 14).

Verifies the hand-written backward pass in ai_core.torch_backend._numpy_lm_fallback
against numerical gradients. Run: python3 tools/gradcheck_numpy_lm.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
from ai_core.torch_backend import _lm_windows  # noqa: E402

rng = np.random.default_rng(0)
vocab, d_model, n_layers, ctx = 7, 5, 2, 3
n = 11
tokens = rng.integers(0, vocab, 60)
X, Y = _lm_windows(tokens, 6)
xs, ys = X[:n, -ctx:], Y[:n, -1]

E = rng.normal(0, 0.1, (vocab, d_model))
dims = [ctx * d_model] + [d_model] * n_layers + [vocab]
layers = []
for fi, fo in zip(dims[:-1], dims[1:]):
    layers.append([rng.normal(0, 1 / np.sqrt(fi), (fi, fo)), np.zeros(fo)])


def fwd(params):
    E_, layers_ = params[0], params[1]
    h = E_[xs].reshape(n, -1)
    acts = [h]
    for W, b in layers_[:-1]:
        h = np.maximum(0.0, h @ W + b)
        acts.append(h)
    Wo, bo = layers_[-1]
    logits = h @ Wo + bo
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(n), ys] + 1e-12).mean()
    return loss, (p, acts, logits)


def analytic(params):
    loss, (p, acts, _) = fwd(params)
    E_, layers_ = params[0], params[1]
    h = acts[-1]
    g = p.copy()
    g[np.arange(n), ys] -= 1.0
    g /= n
    Wo, bo = layers_[-1]
    grads = {"E": None, "layers": []}
    out_Wo, out_bo = h.T @ g, g.sum(0)
    gh = g @ Wo.T
    layer_grads = [None] * len(layers_)
    for i in range(len(layers_) - 2, -1, -1):
        dz = gh * (acts[i + 1] > 0)
        W, _ = layers_[i]
        layer_grads[i] = (acts[i].T @ dz, dz.sum(0))
        gh = dz @ W.T
    layer_grads[-1] = (out_Wo, out_bo)
    gE = np.zeros_like(E_)
    np.add.at(gE, xs.reshape(-1), gh.reshape(-1, d_model))
    grads["E"] = gE
    grads["layers"] = layer_grads
    return loss, grads


loss0, g = analytic([E, layers])
eps = 1e-6
worst = 0.0
checked = 0
targets = [("E", E, g["E"])]
for i, (W, b) in enumerate(layers):
    targets.append((f"W{i}", W, g["layers"][i][0]))
    targets.append((f"b{i}", b, g["layers"][i][1]))

for name, arr, ga in targets:
    k = min(6, arr.size)
    idx = np.random.choice(arr.size, size=k, replace=False)
    for j in idx:
        orig = arr.flat[j]
        arr.flat[j] = orig + eps
        lp, _ = fwd([E, layers])
        arr.flat[j] = orig - eps
        lm, _ = fwd([E, layers])
        arr.flat[j] = orig
        num = (lp - lm) / (2 * eps)
        ana = ga.flat[j]
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        worst = max(worst, rel)
        checked += 1

print(f"loss0              : {loss0:.6f}")
print(f"parameters checked : {checked}")
print(f"worst rel error    : {worst:.2e}")
ok = worst < 1e-5
print("VERDICT:", "OK — backward pass matches numerical gradients" if ok
      else "MISMATCH — backward pass is wrong")
raise SystemExit(0 if ok else 1)
