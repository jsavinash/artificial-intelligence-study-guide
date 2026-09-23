"""torch_backend — PyTorch-first implementations of every concept in this tutorial.

Policy (module 27): use PyTorch wherever it is the idiomatic tool, and fall back
to the pure-NumPy reference when torch is not installed. Every function here
therefore has the same contract in both modes, and every example that imports
this module keeps passing either way:

    HAS_TORCH=True  -> the torch path runs (autograd, nn.Module, real optimizers)
    HAS_TORCH=False -> the NumPy path runs (same math, hand-written gradients)

What lives here and why torch is the right tool:

    MLP / generic training     autograd instead of hand-derived backward()
    CNN                        F.conv2d / F.max_pool2d (fused kernels)
    RNN / LSTM                 nn.RNN / nn.LSTM + gradients through time
    attention / tiny LM        scaled_dot_product_attention, nn.Embedding
    VAE / diffusion / GAN      reparameterisation, ELBO, denoising nets
    DQN                        function approximation for Q-values
    LoRA                       trainable low-rank adapters on frozen nn.Linear
    k-means / PCA / ridge      tensor ops, and autograd where gradient-based
    matrix factorisation       nn.Embedding learned by SGD
    GNN                        message passing as nn.Module
    CLIP two-tower             contrastive dual encoder

Everything returns plain Python/NumPy values (losses, accuracies, seconds) so
callers never need torch to consume the results.
"""
from __future__ import annotations

import math
import time
import zlib

import numpy as np

try:                                        # torch is an optional accelerator
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
    HAS_MPS = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())
    HAS_CUDA = torch.cuda.is_available()
except Exception:                           # pragma: no cover - fallback path
    torch = None                            # type: ignore[assignment]
    nn = None                               # type: ignore[assignment]
    F = None                                # type: ignore[assignment]
    HAS_TORCH = False
    TORCH_VERSION = None
    HAS_MPS = HAS_CUDA = False


def get_device(prefer: str | None = None) -> str:
    """Best available device: cuda > mps > cpu (or an explicit choice)."""
    if not HAS_TORCH:
        return "cpu"
    if prefer in ("cpu", "cuda", "mps"):
        if prefer == "cuda" and not HAS_CUDA:
            return "cpu"
        if prefer == "mps" and not HAS_MPS:
            return "cpu"
        return prefer
    if HAS_CUDA:
        return "cuda"
    if HAS_MPS:
        return "mps"
    return "cpu"


def set_seed(seed: int = 0) -> None:
    """Reproducibility for every backend (torch seeds are a no-op without torch)."""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if HAS_CUDA:
            torch.cuda.manual_seed_all(seed)


def sync(device: str) -> None:
    """GPU work is async — timing it without this measures launch overhead only."""
    if not HAS_TORCH:
        return
    if device == "mps" and HAS_MPS:
        torch.mps.synchronize()
    elif device == "cuda" and HAS_CUDA:
        torch.cuda.synchronize()


def require_torch(feature: str = "this feature") -> None:
    """Fail loudly when a genuine torch implementation is unavailable.

    Several topics in this tutorial (autograd training loops, convolution,
    recurrence, VAEs, GANs, diffusion, DQN, LoRA) have no honest pure-NumPy
    equivalent worth shipping — a hand-rolled stand-in would teach the wrong
    thing and silently return misleading numbers. For those we raise with
    install instructions instead of degrading quietly.
    """
    if not HAS_TORCH:
        raise RuntimeError(
            f"{feature} requires PyTorch, which is not installed.\n"
            f"  Install it:  python3 -m pip install torch\n"
            f"  Or run a topic that has a genuine NumPy path (e.g. m00-m06, "
            f"m16-m22, m24-m26)."
        )


def backend_label() -> str:
    if not HAS_TORCH:
        return "numpy"
    return f"torch-{TORCH_VERSION}-{get_device()}"


def count_params(module) -> int:
    """Trainable parameter count — the number that matters for LLM scaling."""
    if not HAS_TORCH:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Generic supervised training (replaces every hand-written backward() loop)
# ---------------------------------------------------------------------------
def _make_optimizer(name: str, module, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(module.parameters(), lr=lr,
                               weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(module.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(module.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    return torch.optim.Adam(module.parameters(), lr=lr,
                            weight_decay=weight_decay)


def train_module(module, X, y, *, epochs=200, lr=0.05, batch_size=None,
                 optimizer="adam", weight_decay=0.0, device=None,
                 seed=0, log_every=None, eval_every=None) -> dict:
    """Train an nn.Module and return metrics as plain Python values.

    optimizer: "adam" | "sgd" | "adamw" | "rmsprop"
    X and y may be NumPy arrays; conversion happens here.
    """
    if not HAS_TORCH:
        raise RuntimeError("train_module requires torch (HAS_TORCH is False)")
    device = get_device(device)
    set_seed(seed)

    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    y_np = np.asarray(y)
    is_cls = y_np.dtype.kind in "iu"
    yt = torch.as_tensor(y_np, dtype=torch.long if is_cls else torch.float32,
                         device=device)

    module = module.to(device)
    opt = _make_optimizer(optimizer, module, lr, weight_decay)
    lossf = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    n = len(Xt)
    losses, evals, t0 = [], [], time.perf_counter()

    for epoch in range(epochs):
        if batch_size and batch_size < n:
            idx = torch.randperm(n, device=device)[:batch_size]
            xb, yb = Xt[idx], yt[idx]
        else:
            xb, yb = Xt, yt
        opt.zero_grad(set_to_none=True)
        loss = lossf(module(xb), yb)
        loss.backward()          # autograd: no backward() written by hand
        opt.step()
        losses.append(float(loss.item()))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"      epoch {epoch + 1:>4}  loss {losses[-1]:.4f}")
        if eval_every and (epoch + 1) % eval_every == 0:
            evals.append((epoch + 1, _evaluate(module, Xt, yt, is_cls)))

    sync(device)
    secs = time.perf_counter() - t0
    score = _evaluate(module, Xt, yt, is_cls)
    return {"losses": losses, "evals": evals, "seconds": secs, "acc": score,
            "score": score, "device": device, "backend": f"torch-{TORCH_VERSION}",
            "n_params": count_params(module), "model": module,
            "optimizer": optimizer}


def _evaluate(module, Xt, yt, is_cls) -> float:
    """Accuracy for classification, Pearson r for regression."""
    with torch.no_grad():
        was_training = module.training
        module.eval()
        out = module(Xt)
        if is_cls:
            score = float((out.argmax(1) == yt).float().mean().item())
        else:
            a, b = out.flatten(), yt.flatten()
            va, vb = a - a.mean(), b - b.mean()
            denom = (va.norm() * vb.norm()).item()
            score = float((va @ vb).item() / denom) if denom > 1e-12 else 0.0
        if was_training:
            module.train()
    return score


def build_mlp(in_dim: int, out_dim: int, hidden=(32,), dropout=0.0,
              in_shape=None, activation="relu") -> "nn.Module":
    """MLP with the standard shape (optionally flattening image input first)."""
    if not HAS_TORCH:
        raise RuntimeError("build_mlp requires torch")
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
           "sigmoid": nn.Sigmoid}[activation]
    layers: list = []
    if in_shape:
        layers.append(nn.Flatten())
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_classifier(X, y, *, hidden=(32,), epochs=200, lr=0.05,
                     optimizer="adam", dropout=0.0, weight_decay=0.0,
                     batch_size=None, device=None, seed=0) -> dict:
    """Build + train an MLP classifier. NumPy fallback included."""
    n_classes = int(np.max(y)) + 1
    if HAS_TORCH:
        model = build_mlp(X.shape[1], n_classes, hidden, dropout)
        return train_module(model, X, y, epochs=epochs, lr=lr,
                            optimizer=optimizer, weight_decay=weight_decay,
                            batch_size=batch_size, device=device, seed=seed)
    return _numpy_mlp_fallback(X, y, hidden, epochs, lr, dropout,
                               weight_decay, seed)


def _numpy_mlp_fallback(X, y, hidden, epochs, lr, dropout, weight_decay,
                        seed) -> dict:
    """Same model by hand — used when torch is not installed.

    `hidden=()` means "no hidden layer", i.e. multinomial logistic regression;
    that case is handled exactly rather than approximated by a 1-layer MLP.
    """
    rng = np.random.default_rng(seed)
    d, k, n = X.shape[1], int(y.max()) + 1, len(X)

    if not hidden:                                    # pure logistic regression
        W = np.zeros((d, k))
        b = np.zeros(k)
        losses, t0 = [], time.perf_counter()
        for _ in range(epochs):
            z = X @ W + b
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            losses.append(float(-np.log(p[np.arange(n), y] + 1e-12).mean()))
            g = p.copy()
            g[np.arange(n), y] -= 1
            g /= n
            W -= lr * (X.T @ g + weight_decay * W)
            b -= lr * g.sum(0)
        acc = float(((X @ W + b).argmax(1) == y).mean())

        def predict_lin(X_new):
            Xn = np.asarray(X_new, dtype=float)
            return (Xn @ W + b).argmax(1)

        return {"losses": losses, "evals": [], "acc": acc, "score": acc,
                "seconds": time.perf_counter() - t0, "device": "cpu",
                "backend": "numpy", "n_params": d * k,
                "model": (W, b), "predict": predict_lin}

    h = hidden[0]
    W1 = rng.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (d, h))
    b1 = np.zeros(h)
    W2 = rng.uniform(-1 / np.sqrt(h), 1 / np.sqrt(h), (h, k))
    b2 = np.zeros(k)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)
        if dropout > 0:
            a1 = a1 * (rng.random(a1.shape) > dropout) / (1 - dropout)
        logits = a1 @ W2 + b2
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(n), y] + 1e-12).mean()))
        dlog = p.copy()
        dlog[np.arange(n), y] -= 1
        dlog /= n
        a1c = np.maximum(z1, 0)
        W2 -= lr * (a1c.T @ dlog + weight_decay * W2)
        b2 -= lr * dlog.sum(0)
        da = dlog @ W2.T * (z1 > 0)
        W1 -= lr * (X.T @ da + weight_decay * W1)
        b1 -= lr * da.sum(0)
    a1 = np.maximum(X @ W1 + b1, 0)
    acc = float(((a1 @ W2 + b2).argmax(1) == y).mean())

    def predict(X_new):
        """Same contract as the torch path — callers never branch on backend."""
        Xn = np.asarray(X_new, dtype=float)
        return (np.maximum(Xn @ W1 + b1, 0) @ W2 + b2).argmax(1)

    return {"losses": losses, "evals": [], "seconds": time.perf_counter() - t0,
            "acc": acc, "score": acc, "device": "cpu", "backend": "numpy",
            "n_params": d * h + h * k, "model": (W1, b1, W2, b2),
            "predict": predict}


# ---------------------------------------------------------------------------
# Convolutional networks (module 09) — fused kernels instead of im2col loops
# ---------------------------------------------------------------------------
def build_cnn(in_channels: int, n_classes: int, *, width=8, hidden=64,
              in_size=8) -> "nn.Module":
    """Tiny VGG-style CNN: conv→relu→maxpool ×2, then a classifier head."""
    if not HAS_TORCH:
        raise RuntimeError("build_cnn requires torch")
    flat = width * 2 * (in_size // 4) * (in_size // 4)
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(flat, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


def train_cnn(X, y, *, in_size=8, epochs=40, lr=0.01, width=8, batch_size=64,
              device=None, seed=0) -> dict:
    """Train a CNN on (N, C, H, W) float arrays. Returns loss/acc history."""
    if HAS_TORCH:
        set_seed(seed)
        ch = X.shape[1] if X.ndim == 4 else 1
        model = build_cnn(ch, int(np.max(y)) + 1, width=width, in_size=in_size)
        return train_module(model, X, y, epochs=epochs, lr=lr,
                            batch_size=batch_size, device=device, seed=seed)
    return _numpy_cnn_fallback(X, y, epochs, lr)


def _numpy_cnn_fallback(X, y, epochs, lr) -> dict:
    """Single conv layer + ReLU + global pooling, trained by hand (no torch).

    Both the classifier head *and* the convolution kernels receive gradients —
    a frozen random filter bank cannot learn anything, so training only the head
    would report chance accuracy while claiming to be a CNN.

    Uses the same Adam update as the torch path (and the same head bias), so
    the two backends converge on the same schedule — vanilla GD at lr=0.01 for
    25 steps barely moves, which is why an earlier version sat at chance.
    """
    rng = np.random.default_rng(0)
    n, ch, h, w = X.shape
    k = int(y.max()) + 1
    nf = 4
    oh, ow = h - 2, w - 2
    ker = rng.normal(0, 0.3, (nf, ch, 3, 3))
    Wc = rng.normal(0, 0.3, (nf, k))
    bc = np.zeros(k)
    params = [ker, Wc, bc]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    losses, t0 = [], time.perf_counter()
    for t in range(1, epochs + 1):
        conv = np.stack([_np_conv3x3(X[i], ker) for i in range(n)])
        pooled = conv.mean((2, 3))
        feat = np.maximum(pooled, 0)
        logits = feat @ Wc + bc
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(n), y] + 1e-12).mean()))

        dlog = p.copy()
        dlog[np.arange(n), y] -= 1
        dlog /= n
        gWc = feat.T @ dlog                              # head gradient
        gbc = dlog.sum(0)                                # head bias
        dfeat = dlog @ Wc.T
        dpooled = dfeat * (pooled > 0)                   # through ReLU
        dconv = dpooled[:, :, None, None] / (oh * ow)    # through mean-pool
        # dL/dker[f,c,a,b] = sum_{i,j} dconv[f,i,j] * img[c, i+a, j+b]
        dker = np.zeros_like(ker)
        for f in range(nf):
            for c in range(ch):
                for a in range(3):
                    for b in range(3):
                        dker[f, c, a, b] = float(
                            np.sum(dconv[:, f] * X[:, c, a:a + oh, b:b + ow]))
        grads = [dker, gWc, gbc]
        for p_, mt, vt, gr in zip(params, mts, vts, grads):
            mt *= b1
            mt += (1 - b1) * gr
            vt *= b2
            vt += (1 - b2) * gr * gr
            p_ -= lr * (mt / (1 - b1 ** t)) / (
                np.sqrt(vt / (1 - b2 ** t)) + eps)       # bias-corrected Adam
    conv = np.stack([_np_conv3x3(X[i], ker) for i in range(n)])
    feat = np.maximum(conv.mean((2, 3)), 0)
    acc = float(((feat @ Wc + bc).argmax(1) == y).mean())
    return {"losses": losses, "evals": [], "seconds": time.perf_counter() - t0,
            "acc": acc, "score": acc, "device": "cpu", "backend": "numpy",
            "n_params": int(ker.size + Wc.size + bc.size),
            "model": (ker, Wc, bc)}



def _np_conv3x3(img, ker) -> np.ndarray:
    """valid 3x3 conv, no stride: (C,H,W) -> (nf, H-2, W-2)."""
    ch, h, w = img.shape
    nf = ker.shape[0]
    out = np.zeros((nf, h - 2, w - 2))
    for f in range(nf):
        for i in range(h - 2):
            for j in range(w - 2):
                out[f, i, j] = float(np.sum(img[:, i:i + 3, j:j + 3]
                                            * ker[f]))
    return out


# ---------------------------------------------------------------------------
# Sequences (module 10) — RNN / LSTM with real gradients through time
# ---------------------------------------------------------------------------
def build_rnn(in_dim: int, hidden=32, *, cell="lstm", n_layers=1,
              out_dim=1, bidirectional=False) -> "nn.Module":
    """Recurrent sequence model. cell: "rnn" | "lstm" | "gru"."""
    if not HAS_TORCH:
        raise RuntimeError("build_rnn requires torch")
    cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
    rnn = cls(in_dim, hidden, num_layers=n_layers,
              batch_first=True, bidirectional=bidirectional)
    head = nn.Linear(hidden * (2 if bidirectional else 1), out_dim)
    return nn.Sequential(rnn, _TakeLast(), head)


class _TakeLast(nn.Module if HAS_TORCH else object):
    """Last timestep of an (B, T, H) sequence — the standard seq→vec reduction."""

    def forward(self, x):
        out = x[0] if isinstance(x, tuple) else x
        return out[:, -1, :]


def train_rnn(X, y, *, hidden=32, cell="lstm", epochs=120, lr=0.01,
              device=None, seed=0) -> dict:
    """Train a recurrent net on (N, T, F) sequences. NumPy fallback included."""
    if HAS_TORCH:
        set_seed(seed)
        n_out = 1 if np.asarray(y).dtype.kind == "f" else int(np.max(y)) + 1
        model = build_rnn(X.shape[2], hidden, cell=cell, out_dim=n_out)
        return train_module(model, X, y, epochs=epochs, lr=lr,
                            device=device, seed=seed)
    return _numpy_rnn_fallback(X, y, hidden, epochs, lr)


def _numpy_rnn_fallback(X, y, hidden, epochs, lr) -> dict:
    """Elman RNN trained by hand-written BPTT (used when torch is absent)."""
    rng = np.random.default_rng(0)
    n, T, f = X.shape
    y_col = np.asarray(y, dtype=float).reshape(-1, 1)
    scale = 1.0 / np.sqrt(f)
    Wxh = rng.normal(0, scale, (f, hidden))
    Whh = rng.normal(0, scale, (hidden, hidden))
    Why = rng.normal(0, scale, (hidden, 1))
    bh, by = np.zeros(hidden), np.zeros(1)

    def forward():
        hs, h = [], np.zeros((n, hidden))
        for t in range(T):
            h = np.tanh(X[:, t] @ Wxh + h @ Whh + bh)
            hs.append(h)
        return hs, hs[-1] @ Why + by

    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        hs, pred = forward()
        losses.append(float(((pred - y_col) ** 2).mean()))
        d = 2.0 * (pred - y_col) / n                     # dL/dpred
        grad_Why, grad_by = hs[-1].T @ d, d.sum(0)
        dh = d @ Why.T                                   # back through time
        grad_Wxh = np.zeros_like(Wxh)
        grad_Whh = np.zeros_like(Whh)
        grad_bh = np.zeros_like(bh)
        for t in reversed(range(T)):
            dz = dh * (1 - hs[t] ** 2)
            grad_Wxh += X[:, t].T @ dz
            grad_bh += dz.sum(0)
            if t > 0:
                grad_Whh += hs[t - 1].T @ dz
                dh = dz @ Whh.T
        for p, g in ((Wxh, grad_Wxh), (Whh, grad_Whh), (Why, grad_Why),
                     (bh, grad_bh), (by, grad_by)):
            p -= lr * g

    _, pred = forward()
    score = float(np.corrcoef(pred.flatten(), y_col.flatten())[0, 1])
    return {"losses": losses, "evals": [], "seconds": time.perf_counter() - t0,
            "acc": score, "score": score, "device": "cpu", "backend": "numpy",
            "n_params": int(Wxh.size + Whh.size + Why.size), "model": (Wxh, Why)}


# ---------------------------------------------------------------------------
# Attention & transformers (module 11) — scaled_dot_product_attention
# ---------------------------------------------------------------------------
def build_attention_head(d_model=32, d_head=16, *, causal=True):
    """A single QKV attention head: the unit every transformer is built from."""
    if not HAS_TORCH:
        raise RuntimeError("build_attention_head requires torch")
    return _AttentionHead(d_model, d_head, causal)


class _AttentionHead(nn.Module if HAS_TORCH else object):
    """Multi-head attention is a stack of these — here, one head, spelled out."""

    def __init__(self, d_model, d_head, causal):
        if HAS_TORCH:
            super().__init__()
            self.Wq = nn.Linear(d_model, d_head, bias=False)
            self.Wk = nn.Linear(d_model, d_head, bias=False)
            self.Wv = nn.Linear(d_model, d_head, bias=False)
            self.causal = causal
            self.last_attn = None

    def forward(self, x):
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        # one fused call replaces the explicit QK^T/scale/mask/softmax chain
        out = F.scaled_dot_product_attention(q, k, v,
                                             is_causal=self.causal)
        self.last_attn = out.detach()
        return out


def attention_output(x, *, d_head=16, causal=True, seed=0) -> np.ndarray:
    """Run one attention head over (T, D) or (B, T, D); returns NumPy output.

    Falls back to the explicit NumPy formula from ai_core.accelerators so the
    same call works with or without torch.
    """
    x = np.asarray(x, dtype=np.float32)
    single = x.ndim == 2
    if single:
        x = x[None]
    if HAS_TORCH:
        set_seed(seed)
        head = build_attention_head(x.shape[2], d_head, causal=causal)
        with torch.no_grad():
            out = head(torch.as_tensor(x)).numpy()
    else:
        from .accelerators import numpy_attention
        rng = np.random.default_rng(seed)
        Wq, Wk, Wv = (rng.normal(0, 0.3, (x.shape[2], d_head))
                      for _ in range(3))
        out = np.stack([numpy_attention(X @ Wq, X @ Wk, X @ Wv,
                                        causal=causal)[0] for X in x])
    return out[0] if single else out


def build_tiny_lm(vocab, d_model=48, n_heads=4, n_layers=2, block_size=16):
    """Decoder-only transformer: the GPT architecture at tutorial scale."""
    if not HAS_TORCH:
        raise RuntimeError("build_tiny_lm requires torch")
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
        dropout=0.0, batch_first=True, activation="gelu")
    return nn.Sequential(
        _TokenEmbed(vocab, d_model, block_size),
        nn.TransformerEncoder(layer, num_layers=n_layers),
        nn.Linear(d_model, vocab),
    )


class _TokenEmbed(nn.Module if HAS_TORCH else object):
    """Token embedding + learned positional embedding (gpt-2 style, no sinusoids)."""

    def __init__(self, vocab, d_model, block_size):
        if HAS_TORCH:
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(block_size, d_model)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.tok(idx) + self.pos(pos)


def train_tiny_lm(tokens, vocab=None, *, block_size=16, epochs=150, lr=3e-3,
                  d_model=48, n_heads=4, n_layers=2, device=None,
                  seed=0) -> dict:
    """Train a tiny next-token predictor. Returns loss history + samples."""
    tokens = np.asarray(tokens, dtype=np.int64)
    vocab = vocab or int(tokens.max()) + 1
    if not HAS_TORCH:
        return _numpy_lm_fallback(tokens, vocab, block_size, epochs, lr,
                                  d_model=d_model, n_layers=n_layers)
    set_seed(seed)
    device = get_device(device)
    model = build_tiny_lm(vocab, d_model, n_heads, n_layers,
                          block_size).to(device)
    X, Y = _lm_windows(tokens, block_size)
    Xt = torch.as_tensor(X, device=device)
    Yt = torch.as_tensor(Y, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(Xt)
        loss = lossf(logits.reshape(-1, vocab), Yt.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        pred = model(Xt).argmax(-1)
        acc = float((pred == Yt).float().mean().item())
    return {"losses": losses, "acc": acc, "seconds": time.perf_counter() - t0,
            "backend": f"torch-{TORCH_VERSION}", "device": device,
            "n_params": count_params(model), "model": model, "vocab": vocab}


def _lm_windows(tokens, block_size):
    """(X, Y) next-token pairs: Y is X shifted one position left."""
    X = np.stack([tokens[i:i + block_size]
                  for i in range(len(tokens) - block_size)])
    Y = np.stack([tokens[i + 1:i + 1 + block_size]
                  for i in range(len(tokens) - block_size)])
    return X, Y


def _numpy_lm_fallback(tokens, vocab, block_size, epochs, lr, d_model=32,
                       n_layers=1, ctx=4) -> dict:
    """Embedding + ReLU MLP next-token model trained by Adam (torch-free stand-in).

    Not a transformer (no attention, so token order is a fixed window rather
    than a learned relation), but a genuinely *trained* LM whose **parameter
    count scales with `d_model` and `n_layers`**. That size knob is what makes
    the m14 scaling sweep meaningful when torch is absent: more parameters must
    still buy lower loss, or the lesson would be untestable.

    The output layer is zero-initialised, so the first loss is exactly log(V) —
    the same chance baseline the transformer path compares itself against.
    """
    X, Y = _lm_windows(tokens, block_size)
    xs, ys = X[:, -ctx:], Y[:, -1]          # last ctx tokens -> next token
    n = len(xs)
    rng = np.random.default_rng(0)

    # --- parameters: E, then n_layers hidden layers, then the output layer ---
    E = rng.normal(0, 0.1, (vocab, d_model))
    dims = [ctx * d_model] + [d_model] * n_layers + [vocab]
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        if i == len(dims) - 2:               # zero-init output => loss starts at log V
            W, b = np.zeros((fan_in, fan_out)), np.zeros(fan_out)
        else:
            W = rng.normal(0, 1 / np.sqrt(fan_in), (fan_in, fan_out))
            b = np.zeros(fan_out)
        layers.append([W, b])
    n_params = E.size + sum(W.size + b.size for W, b in layers)

    # --- Adam state (deterministic: fixed seed, full batch) ---
    params = [E] + [p for W, b in layers for p in (W, b)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    losses, t0 = [], time.perf_counter()
    for step in range(1, epochs + 1):
        emb = E[xs]                                   # (n, ctx, d_model)
        h = emb.reshape(n, -1)                        # fixed-window concat
        acts = [h]
        for i, (W, b) in enumerate(layers[:-1]):      # hidden layers + ReLU
            h = np.maximum(0.0, h @ W + b)
            acts.append(h)
        Wo, bo = layers[-1]
        logits = h @ Wo + bo

        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(n), ys] + 1e-12).mean()))

        g = p.copy()
        g[np.arange(n), ys] -= 1.0
        g /= n
        grads = [None] * len(params)
        gWo, gbo = h.T @ g, g.sum(0)
        grads[-2], grads[-1] = gWo, gbo
        gh = g @ Wo.T
        for i in range(len(layers) - 2, -1, -1):      # back through hidden layers
            dz = gh * (acts[i + 1] > 0)
            W, _ = layers[i]
            # params layout is [E, W0, b0, W1, b1, ...] -> layer i sits at 1+2i
            grads[1 + 2 * i] = acts[i].T @ dz
            grads[2 + 2 * i] = dz.sum(0)
            gh = dz @ W.T
        gE = np.zeros_like(E)                          # scatter to embeddings
        np.add.at(gE, xs.reshape(-1), gh.reshape(-1, d_model))
        grads[0] = gE

        for j, (prm, gr) in enumerate(zip(params, grads)):   # Adam update
            m[j] = b1 * m[j] + (1 - b1) * gr
            v[j] = b2 * v[j] + (1 - b2) * gr ** 2
            mh = m[j] / (1 - b1 ** step)
            vh = v[j] / (1 - b2 ** step)
            prm -= lr * mh / (np.sqrt(vh) + eps)

    emb = E[xs].reshape(n, -1)
    h = emb
    for W, b in layers[:-1]:
        h = np.maximum(0.0, h @ W + b)
    acc = float(((h @ layers[-1][0] + layers[-1][1]).argmax(1) == ys).mean())
    return {"losses": losses, "acc": acc, "seconds": time.perf_counter() - t0,
            "backend": "numpy", "device": "cpu", "n_params": int(n_params),
            "model": layers, "vocab": vocab, "d_model": d_model,
            "n_layers": n_layers}



# ---------------------------------------------------------------------------
# Generative models (module 12) — VAE / GAN / diffusion
# ---------------------------------------------------------------------------
def build_vae(in_dim, hidden=64, latent=8):
    """Variational autoencoder with the reparameterisation trick built in."""
    if not HAS_TORCH:
        raise RuntimeError("build_vae requires torch")
    return _VAE(in_dim, hidden, latent)


class _VAE(nn.Module if HAS_TORCH else object):
    def __init__(self, in_dim, hidden, latent):
        if HAS_TORCH:
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
            self.mu = nn.Linear(hidden, latent)
            self.logvar = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, in_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)      # reparameterisation
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    """ELBO = reconstruction (BCE) + KL divergence to the standard normal."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kl) / x.shape[0], bce.item() / len(x), kl.item() / len(x)


def train_vae(X, *, hidden=64, latent=8, epochs=150, lr=1e-3, device=None,
              seed=0) -> dict:
    """Train a VAE on (N, D) data in [0,1]. NumPy fallback included."""
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X, 0.0, 1.0)
    if not HAS_TORCH:
        require_torch("train_vae (a real variational autoencoder)")
    set_seed(seed)
    device = get_device(device)
    model = _VAE(X.shape[1], hidden, latent).to(device)
    Xt = torch.as_tensor(X, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        recon, mu, logvar = model(Xt)
        loss, _, _ = vae_loss(recon, Xt, mu, logvar)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        recon, mu, _ = model(Xt)
        mse = float(F.mse_loss(recon, Xt).item())
    return {"losses": losses, "recon_mse": mse,
            "seconds": time.perf_counter() - t0, "model": model,
            "backend": f"torch-{TORCH_VERSION}", "latent": latent,
            "n_params": count_params(model)}


def build_gan(latent=8, out_dim=64, hidden=64):
    """Generator + discriminator pair with the standard BCE objectives."""
    if not HAS_TORCH:
        raise RuntimeError("build_gan requires torch")
    G = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                      nn.Linear(hidden, hidden), nn.ReLU(),
                      nn.Linear(hidden, out_dim), nn.Sigmoid())
    D = nn.Sequential(nn.Linear(out_dim, hidden), nn.LeakyReLU(0.2),
                      nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                      nn.Linear(hidden, 1))
    return G, D


def train_gan(X, *, latent=8, epochs=200, lr=2e-3, device=None, seed=0) -> dict:
    """Train a GAN — alternating D/G updates, the classic adversarial loop."""
    X = np.asarray(X, dtype=np.float32)
    if not HAS_TORCH:
        require_torch("train_gan (adversarial training)")
    set_seed(seed)
    device = get_device(device)
    G, D = build_gan(latent, X.shape[1])
    G, D = G.to(device), D.to(device)
    Xt = torch.as_tensor(X, device=device)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    n = len(Xt)
    d_losses, g_losses, t0 = [], [], time.perf_counter()
    d_loss = g_loss = None
    for _ in range(epochs):
        # --- discriminator: separate real from fake ---
        z = torch.randn(n, latent, device=device)
        fake = G(z).detach()
        optD.zero_grad(set_to_none=True)
        d_loss = 0.5 * (bce(D(Xt), torch.ones(n, 1, device=device))
                        + bce(D(fake), torch.zeros(n, 1, device=device)))
        d_loss.backward()
        optD.step()
        # --- generator: make the discriminator say "real" ---
        z = torch.randn(n, latent, device=device)
        optG.zero_grad(set_to_none=True)
        g_loss = bce(D(G(z)), torch.ones(n, 1, device=device))
        g_loss.backward()
        optG.step()
        d_losses.append(float(d_loss.item()))
        g_losses.append(float(g_loss.item()))
    sync(device)
    return {"d_losses": d_losses, "g_losses": g_losses,
            "seconds": time.perf_counter() - t0, "model": (G, D),
            "backend": f"torch-{TORCH_VERSION}",
            "n_params": count_params(G) + count_params(D),
            "final_d": d_losses[-1], "final_g": g_losses[-1]}


def train_diffusion(X, *, timesteps=40, epochs=120, lr=1e-3, hidden=64,
                    device=None, seed=0) -> dict:
    """Denoising diffusion: learn eps(x_t, t), then sample by reverse chain."""
    X = np.asarray(X, dtype=np.float32)
    if not HAS_TORCH:
        require_torch("train_diffusion (a denoising diffusion model)")
    set_seed(seed)
    device = get_device(device)
    d = X.shape[1]
    betas = torch.linspace(1e-4, 0.2, timesteps, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, 0)
    net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, d)).to(device)
    Xt = torch.as_tensor(X, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, losses, t0 = len(Xt), [], time.perf_counter()
    for _ in range(epochs):
        t = torch.randint(0, timesteps, (n,), device=device)
        eps = torch.randn_like(Xt)
        x_t = (abar[t].sqrt()[:, None] * Xt
               + (1 - abar[t]).sqrt()[:, None] * eps)
        t_norm = (t.float() / timesteps)[:, None]
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(net(torch.cat([x_t, t_norm], dim=1)), eps)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    # reverse chain: pure noise -> sample (the generative process)
    with torch.no_grad():
        x = torch.randn(16, d, device=device)
        for step in reversed(range(timesteps)):
            tt = torch.full((16, 1), step / timesteps, device=device)
            eps_hat = net(torch.cat([x, tt], dim=1))
            a, a_b = alphas[step], abar[step]
            x = (x - (1 - a) / (1 - a_b).sqrt() * eps_hat) / a.sqrt()
            if step > 0:
                x = x + betas[step].sqrt() * torch.randn_like(x)
    return {"losses": losses, "samples": x.cpu().numpy(),
            "seconds": time.perf_counter() - t0, "model": net,
            "backend": f"torch-{TORCH_VERSION}", "timesteps": timesteps,
            "n_params": count_params(net)}


# ---------------------------------------------------------------------------
# Reinforcement learning (module 13) — DQN with a learned Q-network
# ---------------------------------------------------------------------------
class GridWorld:
    """Deterministic grid world — the standard DQN tutorial environment.

    One-hot state (row, col), 4 actions (up/down/left/right), reward -1 per step,
    +10 at the goal (bottom-right), +5 at a trap. Learning to avoid the trap and
    reach the goal is the classic test of credit assignment over many steps.
    """

    def __init__(self, size: int = 4, max_steps: int = 40, trap=(1, 1)):
        self.size = size
        self.n_actions = 4
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)
        self.trap = trap
        self.state_dim = size * size

    def _vec(self, pos):
        v = np.zeros(self.state_dim, dtype=np.float32)
        v[pos[0] * self.size + pos[1]] = 1.0
        return v

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._vec(self.pos)

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        else:
            c = min(self.size - 1, c + 1)
        self.pos = (r, c)
        self.steps += 1
        reward, done = -1.0, False
        if self.pos == self.goal:
            reward, done = 10.0, True
        elif self.pos == self.trap:
            reward, done = -5.0, True
        elif self.steps >= self.max_steps:
            done = True
        return self._vec(self.pos), reward, done, {}


def build_qnet(state_dim, n_actions, hidden=64):
    """Tiny Q-network: state -> Q-value per action. The heart of DQN."""
    if not HAS_TORCH:
        raise RuntimeError("build_qnet requires torch")
    return nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_actions))


def train_dqn(env_factory=None, *, episodes=300, gamma=0.95, lr=1e-2, hidden=64,
              epsilon_start=1.0, epsilon_end=0.05, batch_size=64,
              buffer_size=5000, target_sync=25, device=None,
              seed=0) -> dict:
    """DQN: replay buffer + target network, the two stabilising tricks.

    env_factory() must return an object with reset()->state and
    step(action)->(state, reward, done, info). Defaults to GridWorld.
    """
    if not HAS_TORCH:
        require_torch("train_dqn (a neural Q-network)")
    env_factory = env_factory or GridWorld
    set_seed(seed)
    device = get_device(device)
    env = env_factory()
    s = env.reset()
    n_actions = getattr(env, "n_actions", 4)
    q = build_qnet(len(s), n_actions, hidden).to(device)
    q_target = build_qnet(len(s), n_actions, hidden).to(device)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf: list = []
    returns, t0 = [], time.perf_counter()

    for ep in range(episodes):
        s, done, total = env.reset(), False, 0.0
        eps = max(epsilon_end,
                  epsilon_start - (epsilon_start - epsilon_end)
                  * ep / max(1, episodes * 0.6))
        while not done:
            if np.random.rand() < eps:
                a = int(np.random.randint(n_actions))
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(s, dtype=torch.float32,
                                              device=device)).argmax().item())
            s2, r, done, _ = env.step(a)
            buf.append((s, a, r, s2, done))
            if len(buf) > buffer_size:
                buf.pop(0)
            s, total = s2, total + r
            if len(buf) >= batch_size:
                idx = np.random.choice(len(buf), batch_size, replace=False)
                batch = [buf[i] for i in idx]
                S = torch.as_tensor(np.array([b[0] for b in batch]),
                                    dtype=torch.float32, device=device)
                A = torch.as_tensor([b[1] for b in batch],
                                    dtype=torch.long, device=device)
                R = torch.as_tensor([b[2] for b in batch],
                                    dtype=torch.float32, device=device)
                S2 = torch.as_tensor(np.array([b[3] for b in batch]),
                                     dtype=torch.float32, device=device)
                D = torch.as_tensor([b[4] for b in batch],
                                    dtype=torch.float32, device=device)
                with torch.no_grad():
                    target = R + gamma * (1 - D) * q_target(S2).max(1).values
                pred = q(S).gather(1, A[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(pred, target)   # Huber: robust TD error
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
        returns.append(total)
        if (ep + 1) % target_sync == 0:
            q_target.load_state_dict(q.state_dict())    # stabilise the target
    sync(device)
    n = max(1, episodes // 10)
    return {"returns": returns, "avg_last": float(np.mean(returns[-n:])),
            "avg_first": float(np.mean(returns[:n])),
            "seconds": time.perf_counter() - t0, "model": q, "device": device,
            "backend": f"torch-{TORCH_VERSION}", "n_params": count_params(q),
            "losses": []}


# ---------------------------------------------------------------------------
# PEFT / LoRA (module 15) — trainable low-rank adapters on frozen weights
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module if HAS_TORCH else object):
    """y = Wx + (alpha/r)·B(Ax) with W frozen — the LoRA update, in 6 lines.

    Only A and B are trainable. Parameter count drops from d·k to r·(d+k).
    """

    def __init__(self, base: "nn.Linear", r: int = 4, alpha: float = 8.0):
        if HAS_TORCH:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False          # freeze the pretrained weight
            # Create the adapter on the SAME device/dtype as the frozen weight.
            # Wrapping an already-moved model (e.g. after .to("mps")) otherwise
            # fails with "Tensor for argument #2 'mat2' is on CPU..."
            ref = base.weight
            self.A = nn.Parameter(
                torch.randn(r, base.in_features, device=ref.device,
                            dtype=ref.dtype) * 0.01)
            self.B = nn.Parameter(torch.zeros(base.out_features, r,
                                              device=ref.device,
                                              dtype=ref.dtype))
            self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self) -> "nn.Linear":
        """Fold the adapter back into W: W' = W + scale·B·A — zero inference cost."""
        with torch.no_grad():
            self.base.weight += self.scale * (self.B @ self.A)
        return self.base


def apply_lora(module, r=4, alpha=8.0, target="Linear") -> dict:
    """Freeze the model and wrap every matching layer with a LoRA adapter.

    Walks the module tree and swaps each matching child *in place on its real
    parent* via setattr — the standard PyTorch module-replacement pattern. This
    works at any nesting depth (Sequential, nested blocks, Hugging Face models).
    """
    if not HAS_TORCH:
        total = sum(p.numel() for p in module.parameters()) if module else 0
        return {"model": module, "trainable": 0, "total": total, "percent": 0.0,
                "adapters": 0, "lora_params": 0, "pairs": []}

    for p in module.parameters():
        p.requires_grad = False            # freeze the pretrained weights

    adapters: list = []

    def _wrap(parent):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == target:
                lora = LoRALinear(child, r, alpha)
                setattr(parent, name, lora)          # in-place swap
                adapters.append((name, lora))
            else:
                _wrap(child)                          # recurse into containers

    _wrap(module)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    lora_params = sum(a.A.numel() + a.B.numel() for _, a in adapters)
    return {"model": module, "trainable": trainable, "total": total,
            "lora_params": lora_params, "adapters": len(adapters),
            "percent": 100.0 * trainable / max(1, total),
            "pairs": [n for n, _ in adapters]}


def merge_lora(module) -> int:
    """Fold every adapter back into its base weight; returns layers merged.

    After merging, inference cost is identical to the original model — this is
    why LoRA is the default for serving many fine-tunes from one base.
    """
    if not HAS_TORCH:
        return 0
    merged = 0
    for sub in module.modules():
        for name, child in list(sub.named_children()):
            if isinstance(child, LoRALinear):
                setattr(sub, name, child.merge())
                merged += 1
    return merged



def lora_param_math(d_in: int, d_out: int, r: int) -> dict:
    """The arithmetic behind LoRA's efficiency claim (works without torch)."""
    full = d_in * d_out
    adapter = r * (d_in + d_out)
    return {"full": full, "adapter": adapter, "ratio": adapter / full,
            "percent": 100.0 * adapter / full}


def train_peft(X, y, *, hidden=(64,), r=4, alpha=8.0, epochs=150, lr=1e-2,
               device=None, seed=0) -> dict:
    """Pretrain a backbone, freeze it, then fine-tune with LoRA adapters only."""
    if not HAS_TORCH:
        base = train_classifier(X, y, hidden=hidden, epochs=epochs, lr=lr)
        base["lora"] = {"percent": 0.0, "adapters": 0, "note": "torch absent"}
        return base
    set_seed(seed)
    device = get_device(device)
    n_classes = int(np.max(y)) + 1
    # 1) "pretrained" backbone on the general objective
    backbone = build_mlp(X.shape[1], n_classes, hidden)
    pretrain = train_module(backbone, X, y, epochs=epochs // 3, lr=lr,
                            device=device, seed=seed)
    # 2) freeze + wrap with LoRA, then fine-tune the adapters
    info = apply_lora(backbone, r=r, alpha=alpha)
    frozen = sum(p.numel() for p in backbone.parameters()
                 if not p.requires_grad)
    adapters = [m for m in backbone.modules() if isinstance(m, LoRALinear)]
    opt = torch.optim.Adam([p for m in adapters for p in m.parameters()
                            if p.requires_grad], lr=lr)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(backbone(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    with torch.no_grad():
        acc = float((backbone(Xt).argmax(1) == yt).float().mean().item())
    return {"losses": losses, "acc": acc, "seconds": time.perf_counter() - t0,
            "model": backbone, "device": device, "backend": f"torch-{TORCH_VERSION}",
            "n_params": count_params(backbone),
            "lora": {"r": r, "alpha": alpha, "adapters": len(adapters),
                     "trainable": trainable, "total": total,
                     "percent": 100.0 * trainable / max(1, total),
                     "frozen": frozen, "pretrain_acc": pretrain["acc"]},
            "pretrain_losses": pretrain["losses"]}










# ---------------------------------------------------------------------------
# Classical estimators as trainable modules (modules 03, 04, 26)
# ---------------------------------------------------------------------------
def fit_linear_regression(X, y, *, epochs=400, lr=0.05, ridge=0.0,
                          device=None, seed=0) -> dict:
    """Linear regression by gradient descent on nn.Linear, plus the closed form.

    Teaching point: for a convex problem the iterative and closed-form answers
    must coincide — which is exactly how you verify an autograd implementation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    Xc = np.hstack([X, np.ones((len(X), 1))])                # add bias column
    A = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    closed = np.linalg.solve(A, Xc.T @ y)

    if not HAS_TORCH:
        # the NumPy path *is* the closed form, so agreement is exact by definition
        pred = Xc @ closed
        mse = float(((pred - y) ** 2).mean())
        return {"losses": [mse], "weights": closed.ravel()[:-1],
                "bias": float(closed[-1, 0]), "mse": mse, "seconds": 0.0,
                "backend": "numpy-closed-form", "closed_form": closed.ravel(),
                "agrees_with_closed_form": True}

    device = get_device(device)
    set_seed(seed)
    model = nn.Linear(X.shape[1], 1).to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=ridge)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        mse = float(nn.functional.mse_loss(model(Xt), yt).item())
        w = model.weight.detach().cpu().numpy().ravel()
        b = float(model.bias.detach().cpu().item())
    return {"losses": losses, "weights": w, "bias": b, "mse": mse,
            "seconds": time.perf_counter() - t0,
            "backend": f"torch-{TORCH_VERSION}", "closed_form": closed.ravel(),
            "agrees_with_closed_form": bool(np.allclose(w, closed.ravel()[:-1],
                                                        atol=1e-3)),
            "model": model}


def train_logistic_regression(X, y, *, epochs=300, lr=0.1, l2=0.0,
                              device=None, seed=0) -> dict:
    """Binary logistic regression = nn.Linear + BCEWithLogitsLoss."""
    if not HAS_TORCH:
        return train_classifier(X, y, hidden=(), epochs=epochs, lr=lr)
    device = get_device(device)
    set_seed(seed)
    model = nn.Linear(X.shape[1], 1).to(device)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32,
                         device=device).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    lossf = nn.BCEWithLogitsLoss()
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(Xt))
        acc = float(((prob > 0.5).float() == yt).float().mean().item())

    def predict(X_new):
        """Class predictions for unseen data — the whole point of a model."""
        with torch.no_grad():
            z = model(torch.as_tensor(np.asarray(X_new), dtype=torch.float32,
                                      device=device))
            return (torch.sigmoid(z) > 0.5).long().cpu().numpy().ravel()

    return {"losses": losses, "acc": acc,
            "seconds": time.perf_counter() - t0, "model": model,
            "device": device, "backend": f"torch-{TORCH_VERSION}",
            "n_params": count_params(model), "predict": predict}




# ---------------------------------------------------------------------------
# Unsupervised learning (module 04) — k-means, PCA, Gaussian mixture
# ---------------------------------------------------------------------------
def _kmeans_plusplus_init(X, k, rng):
    """k-means++ seeding: pick centres with probability proportional to D².

    Plain random seeding lands in bad local optima often (on a 3-blob test,
    seed 0 gave inertia 2998 vs the optimal 90). k-means++ is O(log k)-
    competitive and makes Lloyd's algorithm reliable. Cost: one extra pass.
    """
    n = len(X)
    idx = [int(rng.integers(n))]
    closest = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0:                       # fewer distinct points than k
            idx.append(int(rng.integers(n)))
        else:
            probs = closest / total
            idx.append(int(rng.choice(n, p=probs)))
        closest = np.minimum(closest, ((X - X[idx[-1]]) ** 2).sum(1))
    return X[idx].copy()


def kmeans(X, k=3, *, epochs=60, device=None, seed=0, tol=1e-6) -> dict:
    """Lloyd's algorithm on tensors: assign (broadcast distances) / update (mean)."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if not HAS_TORCH:
        return _numpy_kmeans(X, k, epochs, tol, rng)

    device = get_device(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    centroids = torch.as_tensor(_kmeans_plusplus_init(X, k, rng),
                                dtype=torch.float32, device=device)
    inertia_hist, t0 = [], time.perf_counter()
    for _ in range(epochs):
        # ||x - c||^2 expanded to avoid a (n, k, d) intermediate
        d2 = (Xt.pow(2).sum(1, keepdim=True)
              - 2 * Xt @ centroids.T
              + centroids.pow(2).sum(1))
        labels = d2.argmin(1)
        new = torch.stack([Xt[labels == j].mean(0) if (labels == j).any()
                           else centroids[j] for j in range(k)])
        shift = float((new - centroids).norm().item())
        centroids = new
        inertia_hist.append(float(d2[torch.arange(len(X)), labels].sum().item()))
        if shift < tol:
            break
    sync(device)
    d_final = (Xt.pow(2).sum(1, keepdim=True) - 2 * Xt @ centroids.T
               + centroids.pow(2).sum(1))
    labels = d_final.argmin(1)
    return {"centroids": centroids.detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
            "inertia": inertia_hist[-1], "inertia_history": inertia_hist,
            "iterations": len(inertia_hist), "k": k,
            "seconds": time.perf_counter() - t0,
            "backend": f"torch-{TORCH_VERSION}", "device": device}


def _numpy_kmeans(X, k, epochs, tol, rng) -> dict:
    c = _kmeans_plusplus_init(X, k, rng)      # same seeding as the torch path
    hist, t0 = [], time.perf_counter()
    for _ in range(epochs):
        d2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j]
                        for j in range(k)])
        shift = float(np.linalg.norm(new - c))
        c = new
        hist.append(float(d2[np.arange(len(X)), lab].sum()))
        if shift < tol:
            break
    d2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return {"centroids": c, "labels": d2.argmin(1), "inertia": hist[-1],
            "inertia_history": hist, "iterations": len(hist), "k": k,
            "seconds": time.perf_counter() - t0, "backend": "numpy",
            "device": "cpu"}



def _numpy_gmm_em(X, k, seed, epochs=250, tol=1e-6) -> dict:
    """Gaussian mixture by the EM algorithm — the exact twin of the SGD path.

    E-step computes responsibilities r[i,j] = P(component j | x_i);
    M-step re-estimates means/sigmas/weights from those soft counts.
    Both paths expose the same keys (incl. `responsibilities`) so callers do
    not care which backend produced the result.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # init: pick k random points as means, unit variance, uniform weights
    means = X[rng.choice(n, k, replace=False)].copy()
    sigmas = np.ones((k, d))
    weights = np.full(k, 1.0 / k)
    ll_hist, t0 = [], time.perf_counter()
    prev_ll = -np.inf
    for _ in range(epochs):
        # --- E-step ------------------------------------------------------
        logp = np.zeros((n, k))
        for j in range(k):
            diff = (X - means[j]) / sigmas[j]
            logp[:, j] = (-0.5 * (diff ** 2).sum(1)
                          - np.log(sigmas[j]).sum()
                          - 0.5 * d * np.log(2 * math.pi)
                          + np.log(weights[j] + 1e-300))
        m = logp.max(1, keepdims=True)
        ll = (m + np.log(np.exp(logp - m).sum(1, keepdims=True)))
        ll_hist.append(float(ll.sum() / n))
        resp = np.exp(logp - ll)                     # (n, k), rows sum to 1
        # --- M-step ------------------------------------------------------
        Nk = resp.sum(0) + 1e-10
        means = (resp.T @ X) / Nk[:, None]
        var = (resp.T @ (X ** 2)) / Nk[:, None] - means ** 2
        sigmas = np.sqrt(np.maximum(var, 1e-6))
        weights = Nk / n
        if abs(ll_hist[-1] - prev_ll) < tol:
            break
        prev_ll = ll_hist[-1]
    return {"means": means, "sigmas": sigmas, "weights": weights,
            "responsibilities": resp, "labels": resp.argmax(1),
            "losses": [-v for v in ll_hist], "k": k,
            "log_likelihood": ll_hist, "final_nll": -ll_hist[-1],
            "seconds": time.perf_counter() - t0, "backend": "numpy",
            "device": "cpu"}


def pca(X, n_components=2, *, device=None) -> dict:
    """PCA via torch.linalg.svd — same LAPACK path, one API for CPU/GPU."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0, keepdims=True)
    if not HAS_TORCH:
        Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return _pca_pack(Xc, U, S, Vt, mu, n_components, 0.0, "numpy", "cpu")

    device = get_device(device)
    t0 = time.perf_counter()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Xc = Xt - Xt.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    sync(device)
    return _pca_pack(Xc.detach().cpu().numpy(), U.detach().cpu().numpy(),
                     S.detach().cpu().numpy(), Vt.detach().cpu().numpy(),
                     mu, n_components, time.perf_counter() - t0,
                     f"torch-{TORCH_VERSION}", device)


def _pca_pack(Xc, U, S, Vt, mu, n_components, secs, backend, device) -> dict:
    k = min(n_components, len(S))
    var = (S ** 2) / max(1, (len(Xc) - 1))
    ratio = var / var.sum()
    proj = Xc @ Vt[:k].T
    recon = proj @ Vt[:k] + mu
    return {"components": Vt[:k], "explained_variance_ratio": ratio[:k],
            "cumulative": float(ratio[:k].sum()), "projections": proj,
            "recon_mse": float(((recon - (Xc + mu)) ** 2).mean()),
            "singular_values": S, "n_components": k, "seconds": secs,
            "backend": backend, "device": device}


def train_gmm(X, k=3, *, epochs=250, lr=0.02, device=None, seed=0) -> dict:
    """Gaussian mixture by gradient ascent on the log-likelihood (EM's SGD twin)."""
    X = np.asarray(X, dtype=np.float64)
    if not HAS_TORCH:
        return _numpy_gmm_em(X, k, seed)
    device = get_device(device)
    set_seed(seed)
    d = X.shape[1]
    means = nn.Parameter(torch.randn(k, d, device=device) * 0.5)
    log_sigma = nn.Parameter(torch.zeros(k, d, device=device))
    logits = nn.Parameter(torch.zeros(k, device=device))
    opt = torch.optim.Adam([means, log_sigma, logits], lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)

    def log_prob():
        sigma = torch.exp(log_sigma) + 1e-6
        return (-0.5 * (((Xt[:, None, :] - means[None]) / sigma[None]) ** 2)
                - log_sigma[None] - 0.5 * math.log(2 * math.pi)).sum(-1)

    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        # logsumexp over components = numerically stable mixture log-likelihood
        ll = torch.logsumexp(log_prob() + torch.log_softmax(logits, 0)[None],
                             dim=1)
        loss = -ll.mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        resp = torch.softmax(log_prob() + torch.log_softmax(logits, 0)[None],
                             dim=1)
        hard = resp.argmax(1).cpu().numpy()
    return {"means": means.detach().cpu().numpy(),
            "sigmas": (torch.exp(log_sigma) + 1e-6).detach().cpu().numpy(),
            "weights": torch.softmax(logits, 0).detach().cpu().numpy(),
            # soft assignment: a k x n simplex — point i belongs to each
            # component with probability r[i, j], and the rows sum to 1.
            "responsibilities": resp.detach().cpu().numpy(),
            "labels": hard, "losses": losses, "k": k,
            # log-likelihood is just -NLL, so it must increase over training
            "log_likelihood": [-v for v in losses],
            "final_nll": losses[-1], "seconds": time.perf_counter() - t0,
            "backend": f"torch-{TORCH_VERSION}", "device": device}


# ---------------------------------------------------------------------------
# Advanced & specialised (modules 20, 23) — MF, GNN, CLIP
# ---------------------------------------------------------------------------
def train_matrix_factorization(ratings, n_factors=8, *, epochs=400, lr=0.02,
                               weight_decay=1e-3, mask=None, device=None,
                               seed=0) -> dict:
    """Recommendation via learned embeddings — nn.Embedding, not SVD.

    ratings: (n_users, n_items) with NaN for missing entries.
    mask:    optional boolean array marking observed entries. If omitted, the
             observed set is `~isnan(ratings)`. Pass one explicitly when your
             ratings matrix is dense but only some entries are real feedback.

    Returns rmse (on observed entries) and baseline_rmse (the global-mean
    predictor) so the learned model is always compared against doing nothing.
    """
    R = np.asarray(ratings, dtype=np.float64)
    if mask is None:
        mask_np = ~np.isnan(R)
    else:
        mask_np = np.asarray(mask, bool) & ~np.isnan(R)
    if mask_np.sum() == 0:
        raise ValueError("no observed ratings — check `mask`/NaN placement")
    R_filled = np.nan_to_num(R)
    baseline_rmse = float(np.sqrt(((R_filled - R_filled[mask_np].mean())
                                   [mask_np] ** 2).mean()))
    if not HAS_TORCH:
        out = _numpy_mf(R, mask_np, n_factors, epochs, lr, weight_decay, seed)
        out["baseline_rmse"] = baseline_rmse
        return out

    device = get_device(device)
    set_seed(seed)
    n_users, n_items = R.shape
    E = nn.Embedding(n_users, n_factors).to(device)
    Fm = nn.Embedding(n_items, n_factors).to(device)
    bu = nn.Embedding(n_users, 1).to(device)
    bi = nn.Embedding(n_items, 1).to(device)
    nn.init.normal_(E.weight, std=0.1)
    nn.init.normal_(Fm.weight, std=0.1)
    nn.init.zeros_(bu.weight)
    nn.init.zeros_(bi.weight)

    uidx = torch.arange(n_users, device=device).unsqueeze(1).expand(-1, n_items)
    iidx = torch.arange(n_items, device=device).unsqueeze(0).expand(n_users, -1)
    Rt = torch.as_tensor(R_filled, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_np, device=device)

    opt = torch.optim.Adam(list(E.parameters()) + list(Fm.parameters())
                           + list(bu.parameters()) + list(bi.parameters()),
                           lr=lr, weight_decay=weight_decay)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = (E(uidx) * Fm(iidx)).sum(-1) + bu(uidx).squeeze(-1) \
            + bi(iidx).squeeze(-1)
        # masked MSE: only observed ratings contribute to the loss
        loss = (((pred - Rt) ** 2) * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        pred = ((E(uidx) * Fm(iidx)).sum(-1) + bu(uidx).squeeze(-1)
                + bi(iidx).squeeze(-1)).cpu().numpy()
    err = float(np.sqrt(((pred - R_filled)[mask_np] ** 2).mean()))
    return {"rmse": err, "baseline_rmse": baseline_rmse,
            "predictions": pred, "losses": losses,
            "user_factors": E.weight.detach().cpu().numpy(),
            "item_factors": Fm.weight.detach().cpu().numpy(),
            "n_params": count_params(E) + count_params(Fm)
            + count_params(bu) + count_params(bi),
            "seconds": time.perf_counter() - t0,
            "backend": f"torch-{TORCH_VERSION}", "device": device}


def _numpy_mf(R, mask, n_factors, epochs, lr, wd, seed) -> dict:
    """Exact NumPy replica of the torch path: U·Vᵀ + user/item biases, trained
    with the same Adam update (coupled weight decay) so both backends solve the
    identical optimisation problem and reach the same RMSE.
    """
    rng = np.random.default_rng(seed)
    R_filled = np.nan_to_num(R)
    n_users, n_items = R.shape
    U = rng.normal(0, 0.1, (n_users, n_factors))
    V = rng.normal(0, 0.1, (n_items, n_factors))
    bu = np.zeros((n_users, 1))
    bi = np.zeros((n_items, 1))
    m_den = float(mask.sum())

    # Adam moments for every trainable tensor (same order as torch path)
    params = [U, V, bu, bi]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    losses, t0 = [], time.perf_counter()
    for t in range(1, epochs + 1):
        pred = U @ V.T + bu + bi.T                       # (n_users, n_items)
        err = np.where(mask, pred - R_filled, 0.0)       # only observed entries
        losses.append(float((err ** 2).sum() / m_den))
        g = 2.0 * err / m_den                            # dL/dpred (masked MSE)
        grads = [
            g @ V + wd * U,                              # dL/dU (coupled L2)
            g.T @ U + wd * V,                            # dL/dV
            g.sum(1, keepdims=True) + wd * bu,           # dL/dbu
            g.sum(0, keepdims=True).T + wd * bi,         # dL/dbi
        ]
        for p, mt, vt, gr in zip(params, mts, vts, grads):
            mt *= b1
            mt += (1 - b1) * gr
            vt *= b2
            vt += (1 - b2) * gr * gr
            mhat = mt / (1 - b1 ** t)                    # bias-corrected steps
            vhat = vt / (1 - b2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    pred = U @ V.T + bu + bi.T
    rmse = float(np.sqrt(((pred - R_filled)[mask] ** 2).mean()))
    return {"rmse": rmse, "predictions": pred, "losses": losses,
            "user_factors": U, "item_factors": V,
            "n_params": int(U.size + V.size + bu.size + bi.size),
            "seconds": time.perf_counter() - t0,
            "backend": "numpy", "device": "cpu"}


def train_gnn(adj, features, labels, *, train_mask=None, hidden=16, epochs=150,
              lr=0.05, device=None, seed=0) -> dict:
    """Graph neural network: GCN layer = A_norm @ X @ W, then classify nodes.

    adj: (n, n) adjacency (unweighted), features: (n, d), labels: (n,) ints.
    Message passing in torch is just two matmuls — that is the whole idea.
    """
    adj = np.asarray(adj, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(adj)
    train_mask = (np.ones(n, bool) if train_mask is None
                  else np.asarray(train_mask, bool))
    n_classes = int(labels.max()) + 1
    # symmetric normalisation: D^-1/2 (A + I) D^-1/2 — self-loops keep your
    # own features, normalisation stops degree from dominating the scale.
    A = adj + np.eye(n)
    Dinv = np.diag(1.0 / np.sqrt(np.maximum(A.sum(1), 1e-9)))
    A_norm = Dinv @ A @ Dinv

    if not HAS_TORCH:
        return _numpy_gnn(A_norm, features, labels, train_mask, n_classes, seed)

    device = get_device(device)
    set_seed(seed)
    d = features.shape[1]
    # NOTE: the division must happen *before* requires_grad is set. A tensor
    # that is the result of an op on a requires_grad tensor is a *non-leaf*,
    # and optimizers with weight_decay refuse it ("can't optimize a non-leaf
    # Tensor"). .requires_grad_(True) on a plain result gives a leaf tensor.
    W1 = (torch.randn(d, hidden, device=device) / math.sqrt(d)
          ).requires_grad_(True)
    W2 = (torch.randn(hidden, n_classes, device=device) / math.sqrt(hidden)
          ).requires_grad_(True)
    At = torch.as_tensor(A_norm, dtype=torch.float32, device=device)
    Xt = torch.as_tensor(features, dtype=torch.float32, device=device)
    yt = torch.as_tensor(labels, dtype=torch.long, device=device)
    tm = torch.as_tensor(train_mask, device=device)
    opt = torch.optim.Adam([W1, W2], lr=lr, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        h = torch.relu(At @ Xt @ W1)          # message pass + transform
        out = At @ h @ W2                     # second hop
        loss = lossf(out[tm], yt[tm])         # semi-supervised: labels on subset
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        h = torch.relu(At @ Xt @ W1)
        pred = (At @ h @ W2).argmax(1).cpu().numpy()
    return {"labels": pred, "acc": float((pred == labels).mean()),
            "train_acc": float((pred[train_mask] == labels[train_mask]).mean()),
            "losses": losses, "seconds": time.perf_counter() - t0,
            "n_params": int(W1.numel() + W2.numel()), "n_nodes": n,
            "backend": f"torch-{TORCH_VERSION}", "device": device}


def _numpy_gnn(A_norm, features, labels, train_mask, n_classes, seed,
               epochs=400, lr=0.5) -> dict:
    """GCN trained by hand — same two matmuls as the torch path, plus the
    backward pass. Without this the function would return *random* weights,
    which is not a graph neural network by any definition.
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    h_dim = 16
    W1 = rng.normal(size=(d, h_dim)) * np.sqrt(2.0 / d)
    W2 = rng.normal(size=(h_dim, n_classes)) * np.sqrt(2.0 / h_dim)
    eye = np.eye(n_classes)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        # forward: message passing = A_norm @ (features @ W)
        Z1 = A_norm @ (features @ W1)
        H = np.maximum(Z1, 0)
        logits = A_norm @ (H @ W2)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        # loss on the TRAIN nodes only (transductive setting)
        loss = float(-np.log(p[train_mask, labels[train_mask]] + 1e-12).mean())
        losses.append(loss)
        # backward — the softmax + cross-entropy gradient collapses to
        # (p - onehot) / n_train, masked to the labelled nodes
        n_train = max(int(train_mask.sum()), 1)
        g_logits = (p - eye[labels]) * train_mask[:, None] / n_train
        gH = A_norm.T @ g_logits @ W2.T
        gZ = (Z1 > 0) * gH
        gW2 = (A_norm @ H).T @ g_logits
        gW1 = features.T @ (A_norm.T @ gZ)
        W1 -= lr * gW1
        W2 -= lr * gW2
    Z1 = A_norm @ (features @ W1)
    logits = A_norm @ (np.maximum(Z1, 0) @ W2)
    pred = logits.argmax(1)
    return {"labels": pred, "acc": float((pred == labels).mean()),
            "train_acc": float((pred[train_mask] == labels[train_mask]).mean()),
            "losses": losses, "seconds": time.perf_counter() - t0,
            "n_params": int(W1.size + W2.size), "n_nodes": n,
            "backend": "numpy", "device": "cpu"}


def train_clip(image_emb, text_emb, *, temperature=0.07, epochs=300, lr=0.01,
               embed_dim=16, device=None, seed=0) -> dict:
    """Two-tower contrastive model (CLIP): match paired image/text embeddings.

    Trains projection heads so paired samples land close in a shared space;
    retrieval accuracy is then nearest-neighbour in that space.
    """
    image_emb = np.asarray(image_emb, dtype=np.float64)
    text_emb = np.asarray(text_emb, dtype=np.float64)
    n = len(image_emb)

    if not HAS_TORCH:
        return _numpy_clip(image_emb, n, seed)

    device = get_device(device)
    set_seed(seed)
    img_head = nn.Sequential(nn.Linear(image_emb.shape[1], embed_dim),
                             nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    txt_head = nn.Sequential(nn.Linear(text_emb.shape[1], embed_dim),
                             nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    heads = nn.ModuleList([img_head, txt_head]).to(device)
    I = torch.as_tensor(image_emb, dtype=torch.float32, device=device)
    T = torch.as_tensor(text_emb, dtype=torch.float32, device=device)
    target = torch.arange(n, device=device)          # diagonal = the true pairs
    opt = torch.optim.Adam(heads.parameters(), lr=lr)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        zi = nn.functional.normalize(img_head(I), dim=-1)
        zt = nn.functional.normalize(txt_head(T), dim=-1)
        logits = zi @ zt.T / temperature
        # symmetric InfoNCE: image->text and text->image both must be right
        loss = 0.5 * (nn.functional.cross_entropy(logits, target)
                      + nn.functional.cross_entropy(logits.T, target))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    with torch.no_grad():
        zi = nn.functional.normalize(img_head(I), dim=-1)
        zt = nn.functional.normalize(txt_head(T), dim=-1)
        sim = (zi @ zt.T).cpu().numpy()
    # i2t / t2i retrieval: is the correct partner rank 1?
    return {"losses": losses, "i2t_top1": float((sim.argmax(1) == np.arange(n)).mean()),
            "t2i_top1": float((sim.argmax(0) == np.arange(n)).mean()),
            "similarity": sim, "seconds": time.perf_counter() - t0,
            "n_params": count_params(img_head) + count_params(txt_head),
            "backend": f"torch-{TORCH_VERSION}", "device": device}


def _numpy_clip(image_emb, n, seed) -> dict:
    z = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-9)
    sim = z @ z.T
    return {"losses": [], "i2t_top1": float((sim.argmax(1) == np.arange(n)).mean()),
            "t2i_top1": float((sim.argmax(0) == np.arange(n)).mean()),
            "similarity": sim, "seconds": 0.0, "n_params": 0,
            "backend": "numpy", "device": "cpu"}


# ---------------------------------------------------------------------------
# LLM scaling laws — measured, not quoted (module 14)
# ---------------------------------------------------------------------------
def fit_power_law(sizes, losses) -> dict:
    """Least-squares fit of L = a * N^(-alpha) + ... in log-log space.

    Returns the exponent alpha: |alpha| is the empirical "how fast loss falls
    with parameters" number. Chinchilla reports ~0.34 for model size; anything
    in 0.05-0.6 is expected for a tiny model on a toy corpus.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    x, y = np.log(sizes), np.log(losses)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"alpha": float(-slope), "log_a": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
            "predicted": np.exp(pred).tolist()}


def scaling_law_sweep(tokens, *, sizes=None, block_size=12, epochs=120,
                      lr=4e-3, seed=0, verbose=True) -> dict:
    """Train tiny transformers of increasing size and measure the loss curve.

    This turns the Chinchilla rule from a quoted constant into something the
    learner observes: more parameters -> lower loss, and the trend is a power
    law, i.e. straight in log-log space. Each point is a genuinely trained
    model, so the numbers cannot be faked by the analytical formula.

    sizes: list of (d_model, n_heads, n_layers) tuples.
    """
    if sizes is None:
        sizes = [(16, 2, 1), (24, 2, 1), (32, 4, 1), (48, 4, 2), (64, 4, 2)]

    points = []
    for d_model, n_heads, n_layers in sizes:
        if HAS_TORCH:
            res = train_tiny_lm(tokens, block_size=block_size, epochs=epochs,
                                lr=lr, d_model=d_model, n_heads=n_heads,
                                n_layers=n_layers, seed=seed)
        else:
            # The NumPy path has no attention, but it *does* have a real size
            # knob (embedding width x depth), so the sweep still measures a
            # genuine parameters->loss relationship instead of repeating one
            # identical model five times.
            res = train_tiny_lm(tokens, block_size=block_size, epochs=epochs,
                                lr=lr, d_model=d_model, n_layers=n_layers)
        n_params = res.get("n_params") or (d_model * d_model * 4 * n_layers)
        points.append({"d_model": d_model, "n_heads": n_heads,
                       "n_layers": n_layers, "n_params": int(n_params),
                       "final_loss": float(res["losses"][-1]),
                       "seconds": res.get("seconds", 0.0)})
        if verbose:
            print(f"    d_model={d_model:>3} layers={n_layers} "
                  f"params={int(n_params):>7}  loss={res['losses'][-1]:.4f}  "
                  f"({res.get('seconds', 0.0):.2f}s)")

    fit = fit_power_law([p["n_params"] for p in points],
                        [p["final_loss"] for p in points])
    return {"points": points, "fit": fit, "backend": backend_label()}


def chinchilla_tokens(n_params: float, ratio: float = 20.0) -> dict:
    """Compute-optimal tokens and training FLOPs for a model of N parameters.

    Chinchilla: D* ~= ratio * N, and forward+backward cost ~= 6 * N * D FLOPs.
    """
    D = ratio * n_params
    return {"n_params": n_params, "tokens": D, "ratio": ratio,
            "flops": 6.0 * n_params * D,
            "gpu_hours_at_1e15_fps": 6.0 * n_params * D / (1e15 * 3600)}


# ---------------------------------------------------------------------------
# Autograd / calculus (module 00) — the chain rule, computed not derived
# ---------------------------------------------------------------------------
def autograd_grad(fn, x, *, device=None) -> dict:
    """d/dx of a scalar function via reverse-mode autodiff.

    Replaces hand-derived gradients: fn must be a differentiable torch
    expression. Returns value, gradient, and the analytic companion if given.
    """
    if not HAS_TORCH:
        return {"value": float(fn(x)), "grad": None, "backend": "numpy"}
    device = device or get_device()
    t = torch.tensor(float(x), dtype=torch.float32, device=device,
                     requires_grad=True)
    y = fn(t)
    y.backward()
    return {"value": float(y.item()), "grad": float(t.grad.item()),
            "backend": f"torch-{TORCH_VERSION}", "device": device}


def jacobian(fn, x, *, device=None) -> np.ndarray:
    """Full Jacobian J[i,j] = d f_i / d x_j — one backward pass per output row.

    This is what autograd buys you over hand-differentiation: an arbitrary
    vector-valued function differentiated without writing a single derivative.
    """
    if not HAS_TORCH:
        raise RuntimeError("jacobian requires torch")
    device = device or get_device()
    xt = torch.tensor(np.asarray(x, dtype=np.float32), device=device,
                      requires_grad=True)
    y = fn(xt)
    rows = []
    for i in range(y.numel()):
        (g,) = torch.autograd.grad(y.reshape(-1)[i], xt, retain_graph=True)
        rows.append(g.detach().cpu().numpy().ravel())
    return np.vstack(rows)


def gradient_descent_autograd(fn, x0=(-4.0,), *, lr=0.1, iters=100,
                              optimizer="sgd", device=None) -> dict:
    """Minimise a differentiable fn from x0 using a real torch optimizer.

    The same algorithm as m00's hand-written `x -= lr * grad(x)`, except the
    gradient comes from autograd — so it works for any function, any dimension.
    """
    if not HAS_TORCH:
        raise RuntimeError("gradient_descent_autograd requires torch")
    device = device or get_device()
    x = torch.tensor(list(x0), dtype=torch.float32, device=device,
                     requires_grad=True)
    opt = (torch.optim.Adam([x], lr=lr) if optimizer == "adam"
           else torch.optim.SGD([x], lr=lr))
    hist = []
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = fn(x)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    return {"x": x.detach().cpu().numpy().tolist(), "loss": hist[-1],
            "losses": hist, "optimizer": optimizer, "device": device,
            "backend": f"torch-{TORCH_VERSION}"}


def check_gradient(fn, x, *, eps=1e-4, points=None) -> dict:
    """Compare autograd against central finite differences.

    The universal correctness test for any hand-written backward pass: if the
    two disagree, the analytic gradient has a bug (this caught real bugs in
    this repo's NumPy implementations).
    """
    if not HAS_TORCH:
        raise RuntimeError("check_gradient requires torch")
    x = np.asarray(x, dtype=np.float64)
    points = points if points is not None else \
        [[1.3, -0.7, 2.1], [0.5, 0.5, 0.5], [-1.2, 0.9, -2.4]]
    max_err = 0.0
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        xt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        fn(xt).backward()
        auto = xt.grad.detach().numpy().astype(np.float64)
        num = np.zeros_like(auto)
        for i in range(len(p)):
            hi, lo = p.copy(), p.copy()
            hi[i] += eps
            lo[i] -= eps
            with torch.no_grad():
                num[i] = ((float(fn(torch.tensor(hi, dtype=torch.float32)))
                           - float(fn(torch.tensor(lo, dtype=torch.float32))))
                          / (2 * eps))
        max_err = max(max_err, float(np.max(np.abs(auto - num))))
    return {"max_abs_err": max_err, "ok": max_err < 1e-2,
            "backend": f"torch-{TORCH_VERSION}", "n_points": len(points)}


def softmax_cross_entropy(logits, targets) -> dict:
    """Cross-entropy the framework way, with the gradient available for free.

    Also returns the hand-computed (softmax - onehot) gradient, so the identity
    behind every classification training loop is verifiable rather than asserted.
    """
    if not HAS_TORCH:
        raise RuntimeError("softmax_cross_entropy requires torch")
    lt = torch.tensor(np.asarray(logits), dtype=torch.float32,
                      requires_grad=True)
    tt = torch.tensor(np.asarray(targets), dtype=torch.long)
    loss = F.cross_entropy(lt, tt)
    loss.backward()
    p = F.softmax(lt.detach(), dim=-1).numpy()
    onehot = np.zeros_like(p)
    onehot[np.arange(len(tt)), tt.numpy()] = 1.0
    return {"loss": float(loss.item()),
            "autograd_grad": lt.grad.numpy(),
            "analytic_grad": (p - onehot) / len(tt),
            "backend": f"torch-{TORCH_VERSION}"}


# ---------------------------------------------------------------------------
# Calibration & decision thresholds (module 05)
# ---------------------------------------------------------------------------
def temperature_scale(logits, y, *, epochs=300, lr=0.05, device=None) -> dict:
    """Learn a single temperature T so that softmax(logits/T) is calibrated.

    A network's confidence is usually wrong even when its accuracy is right
    (Guo et al. 2017). Fitting one scalar with autograd fixes most of it —
    a perfect example of using torch because it is the right tool.
    """
    if not HAS_TORCH:
        raise RuntimeError("temperature_scale requires torch")
    device = device or get_device()
    lt = torch.tensor(np.asarray(logits), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(y), dtype=torch.long, device=device)
    log_t = torch.zeros(1, device=device, requires_grad=True)   # T = exp(log_t)
    opt = torch.optim.Adam([log_t], lr=lr)
    before = float(F.cross_entropy(lt, yt).item())
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(lt / torch.exp(log_t), yt)
        loss.backward()
        opt.step()
    T = float(torch.exp(log_t).item())
    after = float(F.cross_entropy(lt / torch.exp(log_t.detach()), yt).item())
    return {"temperature": T, "nll_before": before, "nll_after": after,
            "improved": after <= before + 1e-6, "device": device,
            "backend": f"torch-{TORCH_VERSION}"}


def sweep_thresholds(y, scores, n=50) -> dict:
    """Search decision thresholds for the best F1, on-device and vectorised.

    Shows the accuracy paradox: the threshold maximising *accuracy* is rarely
    the one maximising F1 or recall on imbalanced data.
    """
    if not HAS_TORCH:
        raise RuntimeError("sweep_thresholds requires torch")
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    st = torch.tensor(np.asarray(scores), dtype=torch.float32)
    grid = torch.linspace(0.05, 0.95, n)
    rows = []
    for t in grid:
        pred = (st >= t).float()
        tp = float(((pred == 1) & (yt == 1)).sum())
        fp = float(((pred == 1) & (yt == 0)).sum())
        fn = float(((pred == 0) & (yt == 1)).sum())
        tn = float(((pred == 0) & (yt == 0)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1v = 2 * prec * rec / (prec + rec + 1e-12)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        rows.append({"threshold": float(t), "precision": prec, "recall": rec,
                     "f1": f1v, "accuracy": acc})
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])
    return {"rows": rows, "best_f1": best_f1, "best_acc": best_acc,
            "thresholds_differ": abs(best_f1["threshold"]
                                     - best_acc["threshold"]) > 1e-9}


def expected_calibration_error(probs, y, n_bins=10) -> dict:
    """ECE = sum_b (n_b/N) * |acc(b) - conf(b)| — the calibration metric."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).astype(int)
    conf = probs if probs.ndim == 1 else probs.max(1)
    pred = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc_b = float((pred[m] == y[m]).mean())
        conf_b = float(conf[m].mean())
        ece += float(m.mean()) * abs(acc_b - conf_b)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": acc_b, "conf": conf_b})
    return {"ece": float(ece), "bins": bins, "n_bins": n_bins}


# ---------------------------------------------------------------------------
# Feature engineering (module 06) — learned embeddings for high-cardinality
# categoricals instead of one-hot columns
# ---------------------------------------------------------------------------
def _vocab_size(codes, holdout=None, *, reserve_unk: bool = True) -> int:
    """Vocabulary must be sized over train AND holdout, or transform-time codes
    index past the table. The extra leading slot is the unknown/OOV bucket —
    the same convention as `handle_unknown='ignore'` in sklearn's OneHotEncoder,
    and what keeps a production encoder from crashing on a new category.
    """
    top = int(np.asarray(codes).max())
    if holdout is not None:
        top = max(top, int(np.asarray(holdout[0]).max()))
    return top + 1 + (1 if reserve_unk else 0)


def learn_embeddings(codes, y, *, dim=4, epochs=300, lr=0.05, device=None,
                     seed=0, holdout=None) -> dict:
    """Learn a dense embedding table for a categorical feature by SGD.

    This is the torch-native answer to feature engineering's hardest case:
    a categorical with thousands of levels. One-hot needs one weight per level
    per output; an embedding needs `dim` per level, so the composed
    `head @ embedding` is a **rank-`dim` factorisation** of the one-hot weight
    matrix. That low-rank constraint is what lets rare categories borrow
    statistical strength from frequent ones — the reason embeddings beat
    one-hot on high-cardinality features.

    codes:   integer category ids, shape (n,).
    y:       integer labels, shape (n,).
    holdout: optional (codes_te, y_te) scored with the *learned* table, so
             memorisation cannot inflate the number.
    Returns embeddings, loss curve, accuracy, parameter counts, and
    holdout_acc / holdout_acc_rare when a holdout is supplied.
    """
    codes = np.asarray(codes).astype(np.int64).ravel()
    y = np.asarray(y).astype(np.int64).ravel()
    n_cat = _vocab_size(codes, holdout)
    n_out = int(y.max()) + 1

    if HAS_TORCH:
        device = device or get_device()
        set_seed(seed)
        model = _EmbModel(n_cat, dim, n_out).to(device)
        ct = torch.as_tensor(codes, dtype=torch.long, device=device)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        losses, t0 = [], time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(ct), yt)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sync(device)
        secs = time.perf_counter() - t0
        with torch.no_grad():
            acc = float((model(ct).argmax(1) == yt).float().mean().item())
            emb = model.emb.weight.detach().cpu().numpy()
            out = {"embeddings": emb, "losses": losses, "seconds": secs,
                   "acc": acc, "dim": dim, "n_categories": n_cat,
                   "emb_params": n_cat * dim,
                   "one_hot_params": n_cat * n_out,
                   "backend": f"torch-{TORCH_VERSION}", "device": device}
            out.update(_holdout_torch(model, holdout, codes))
        return out

    return _numpy_embeddings(codes, y, n_cat, n_out, dim, epochs, lr, seed,
                             holdout)


def _holdout_clamped(codes_te, y_te, n_cat: int):
    """Map any unseen-at-fit time code onto the unknown bucket instead of
    letting it index past the table."""
    codes_te = np.asarray(codes_te).astype(np.int64).ravel()
    y_te = np.asarray(y_te).astype(np.int64).ravel()
    oov = codes_te >= n_cat
    safe = np.where(oov, n_cat - 1, codes_te)      # last slot = unknown
    return safe, y_te, oov


def _holdout_torch(model, holdout, train_codes) -> dict:
    c_te, y_te = holdout
    n_cat = model.emb.num_embeddings
    safe, y_te, oov = _holdout_clamped(c_te, y_te, n_cat)
    dev = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.as_tensor(safe, dtype=torch.long,
                                     device=dev)).argmax(1).cpu().numpy()
    correct = pred == y_te
    counts = np.bincount(np.asarray(train_codes).ravel(), minlength=n_cat)
    rare = counts[safe] <= 2
    return {"holdout_acc": float(correct.mean()),
            "holdout_acc_rare": float(correct[rare].mean()) if rare.any() else None,
            "holdout_n": int(len(y_te)), "holdout_rare_n": int(rare.sum()),
            "holdout_oov_n": int(oov.sum())}



def _numpy_embeddings(codes, y, n_cat, n_out, dim, epochs, lr, seed,
                      holdout=None) -> dict:
    """Same embedding + linear head by hand (lookup-table SGD).

    The backward pass is a *scatter* (`np.add.at`) rather than a matmul — that
    is the whole trick behind embeddings: only the rows that were looked up
    receive gradient, so cost is O(batch) not O(vocab).
    """
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, (n_cat, dim))
    W = rng.normal(0, 0.1, (dim, n_out))
    b = np.zeros(n_out)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        h = E[codes]                       # forward: row lookup = the embedding
        logits = h @ W + b
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()))
        dlog = p.copy()
        dlog[np.arange(len(y)), y] -= 1
        dlog /= len(y)
        gE = dlog @ W.T                    # scatter back into the table
        np.add.at(E, codes, -lr * gE)
        W -= lr * (h.T @ dlog)
        b -= lr * dlog.sum(0)
    h = E[codes]
    out = {"embeddings": E, "losses": losses,
           "seconds": time.perf_counter() - t0,
           "acc": float(((h @ W + b).argmax(1) == y).mean()), "dim": dim,
           "n_categories": n_cat, "emb_params": n_cat * dim,
           "one_hot_params": n_cat * n_out, "backend": "numpy", "device": "cpu"}
    if holdout is not None:
        c_te, y_te = holdout
        # same unknown-category policy as the torch path — never index past
        # the table, bucket unseen codes into the reserved unknown slot
        safe, y_te, oov = _holdout_clamped(c_te, y_te, n_cat)
        pred = (E[safe] @ W + b).argmax(1)
        correct = pred == y_te
        counts = np.bincount(codes, minlength=n_cat)
        rare = counts[safe] <= 2
        out.update({
            "holdout_acc": float(correct.mean()),
            "holdout_acc_rare": (float(correct[rare].mean())
                                 if rare.any() else None),
            "holdout_n": int(len(y_te)), "holdout_rare_n": int(rare.sum()),
            "holdout_oov_n": int(oov.sum()),
        })
    return out


def one_hot_baseline(codes, y, *, n_out=None, epochs=300, lr=0.05,
                     device=None, seed=0, holdout=None) -> dict:
    """The alternative feature-engineering approach: one-hot + linear layer.

    Same head, same optimiser, same epochs — the only difference is how the
    category is encoded, which isolates the encoding's effect. Parameters scale
    as n_categories × n_out, and unseen categories carry no information at all.
    """
    codes = np.asarray(codes).astype(np.int64).ravel()
    y = np.asarray(y).astype(np.int64).ravel()
    # size the vocabulary over train AND holdout, with the reserved unknown slot
    # (for one-hot this becomes an all-zeros row => the model falls back to its
    # bias for an unseen category, which is the correct default)
    n_cat = _vocab_size(codes, holdout)
    n_out = n_out or int(y.max()) + 1
    set_seed(seed)

    if HAS_TORCH:
        device = device or get_device()
        model = nn.Linear(n_cat, n_out).to(device)
        ct = torch.as_tensor(codes, dtype=torch.long, device=device)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        oh = F.one_hot(ct, num_classes=n_cat).float()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        losses, t0 = [], time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(oh), yt)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sync(device)
        with torch.no_grad():
            acc = float((model(oh).argmax(1) == yt).float().mean().item())
        out = {"losses": losses, "acc": acc, "n_params": n_cat * n_out,
               "seconds": time.perf_counter() - t0, "encoding": "one_hot",
               "backend": f"torch-{TORCH_VERSION}", "device": device}
        if holdout is not None:
            safe, y_np, oov = _holdout_clamped(holdout[0], holdout[1], n_cat)
            c_te = torch.as_tensor(safe, dtype=torch.long, device=device)
            y_te = torch.as_tensor(y_np, dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model(F.one_hot(c_te, num_classes=n_cat).float()).argmax(1)
            correct = (pred == y_te).cpu().numpy()
            counts = np.bincount(codes, minlength=n_cat)
            rare = counts[safe] <= 2
            out.update({"holdout_acc": float(correct.mean()),
                        "holdout_acc_rare": (float(correct[rare].mean())
                                             if rare.any() else None),
                        "holdout_n": int(len(y_np)),
                        "holdout_rare_n": int(rare.sum()),
                        "holdout_oov_n": int(oov.sum())})
        return out

    # NumPy fallback: one-hot + linear, solved by gradient descent
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (n_cat, n_out))
    b = np.zeros(n_out)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        logits = W[codes] + b                    # one-hot matmul == row gather
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()))
        dlog = p.copy()
        dlog[np.arange(len(y)), y] -= 1
        dlog /= len(y)
        np.add.at(W, codes, -lr * dlog)
        b -= lr * dlog.sum(0)
    out = {"losses": losses,
           "acc": float((W[codes] + b).argmax(1).__eq__(y).mean()),
           "n_params": n_cat * n_out, "seconds": time.perf_counter() - t0,
           "encoding": "one_hot", "backend": "numpy", "device": "cpu"}
    if holdout is not None:
        # same unknown-category policy as the torch path
        safe, y_te, oov = _holdout_clamped(holdout[0], holdout[1], n_cat)
        correct = (W[safe] + b).argmax(1) == y_te
        counts = np.bincount(codes, minlength=n_cat)
        rare = counts[safe] <= 2
        out.update({"holdout_acc": float(correct.mean()),
                    "holdout_acc_rare": (float(correct[rare].mean())
                                         if rare.any() else None),
                    "holdout_n": int(len(y_te)),
                    "holdout_rare_n": int(rare.sum()),
                    "holdout_oov_n": int(oov.sum())})
    return out


class _EmbModel(nn.Module if HAS_TORCH else object):
    """Embedding lookup + linear head — the standard categorical encoder."""

    def __init__(self, n_categories: int, dim: int, n_out: int):
        super().__init__()
        self.emb = nn.Embedding(n_categories, dim)
        self.head = nn.Linear(dim, n_out)

    def forward(self, codes):
        return self.head(self.emb(codes))


# ---------------------------------------------------------------------------
# Learned dense retrieval (module 17) — a real trained encoder, not hashing
# ---------------------------------------------------------------------------
def bow_features(texts, *, dim=256) -> np.ndarray:
    """Deterministic hashed bag-of-words — the *input features* a retriever
    learns from. Hashing keeps this dependency-free (no tokenizer download).

    Uses crc32, not builtin hash(): Python randomises str hashing per process
    (PYTHONHASHSEED), which would make embeddings differ between runs.
    """
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            w = "".join(c for c in w if c.isalnum())
            if w:
                X[i, zlib.crc32(w.encode()) % dim] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-9)


def _hits_at_1(q_emb, p_emb, positives) -> int:
    """Cosine retrieval accuracy: does each query's top-1 passage match gold?"""
    qn = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-9)
    pn = p_emb / np.maximum(np.linalg.norm(p_emb, axis=1, keepdims=True), 1e-9)
    sims = qn @ pn.T
    return int((sims.argmax(1) == np.asarray(positives)).sum())


def train_retriever(queries, passages, positives, *, dim=64, epochs=300,
                    lr=0.05, temperature=0.1, device=None, seed=0) -> dict:
    """Train a two-tower dense retriever with InfoNCE (the standard IR recipe).

    Demonstrates the payoff of *learning* embeddings: an untrained projection is
    near-random, a contrastively trained one pulls each query to its passage.
    Returns hits@1 before/after training (identical evaluation, only weights differ).
    """
    set_seed(seed)
    Q = bow_features(queries)
    P = bow_features(passages)
    pos = np.asarray(positives)

    if not HAS_TORCH:
        return _numpy_retriever_fallback(Q, P, pos, dim, epochs, lr, seed)

    device = device or get_device()
    Qt = torch.as_tensor(Q, device=device)
    Pt = torch.as_tensor(P, device=device)
    pt = torch.as_tensor(pos, device=device)

    q_enc = nn.Linear(Q.shape[1], dim, bias=False).to(device)
    p_enc = nn.Linear(P.shape[1], dim, bias=False).to(device)

    def encode(q_enc, p_enc):
        with torch.no_grad():
            return (q_enc(Qt).cpu().numpy(), p_enc(Pt).cpu().numpy())

    q0, p0 = encode(q_enc, p_enc)                 # random init
    before = _hits_at_1(q0, p0, pos)

    opt = torch.optim.Adam(list(q_enc.parameters()) + list(p_enc.parameters()),
                           lr=lr)
    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        qz = F.normalize(q_enc(Qt), dim=-1)
        pz = F.normalize(p_enc(Pt), dim=-1)
        logits = qz @ pz.T / temperature
        loss = F.cross_entropy(logits, pt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    sync(device)
    secs = time.perf_counter() - t0

    q1, p1 = encode(q_enc, p_enc)
    after = _hits_at_1(q1, p1, pos)
    return {"hits_before": before, "hits_after": after, "n_queries": len(pos),
            "losses": losses, "seconds": secs, "device": device, "dim": dim,
            "backend": f"torch-{TORCH_VERSION}",
            "encoder": (q_enc, p_enc),
            "query_emb": q1, "passage_emb": p1}


def _numpy_retriever_fallback(Q, P, pos, dim, epochs, lr, seed) -> dict:
    """Same two-tower model + InfoNCE by hand, with the *correct* gradients.

    dL/dlogits = (prob - onehot)/n ; logits = q̂ p̂ᵀ/T ; and because the
    encoders are L2-normalised, the gradient flows back through that
    normalisation via the projection (I - ẑẑᵀ)/‖z‖.
    """
    rng = np.random.default_rng(seed)
    Wq = rng.normal(0, 0.1, (Q.shape[1], dim))
    Wp = rng.normal(0, 0.1, (P.shape[1], dim))
    T = 0.1

    def unit(Z):
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

    def unit_backward(Z, Zh, G):
        """dL/dZ given dL/dẐ through L2 normalisation."""
        nrm = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return (G - Zh * np.sum(G * Zh, axis=1, keepdims=True)) / nrm

    before = _hits_at_1(unit(Q @ Wq), unit(P @ Wp), pos)
    losses, t0 = [], time.perf_counter()
    n = len(pos)

    for _ in range(epochs):
        Zq, Zp = Q @ Wq, P @ Wp
        qh, ph = unit(Zq), unit(Zp)
        logits = qh @ ph.T / T
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        losses.append(float(-np.log(prob[np.arange(n), pos] + 1e-12).mean()))
        if losses[-1] <= 1e-4:
            break

        gl = prob.copy()                       # dL/dlogits
        gl[np.arange(n), pos] -= 1.0
        gl /= n
        gl /= T                                # chain through the /T inside logits
        g_qh = gl @ ph                         # (n × dim)
        g_ph = gl.T @ qh                       # (n × dim)
        Wq -= lr * (Q.T @ unit_backward(Zq, qh, g_qh))
        Wp -= lr * (P.T @ unit_backward(Zp, ph, g_ph))

    q1, p1 = unit(Q @ Wq), unit(P @ Wp)
    after = _hits_at_1(q1, p1, pos)
    return {"hits_before": before, "hits_after": after, "n_queries": len(pos),
            "losses": losses, "seconds": time.perf_counter() - t0,
            "device": "cpu", "dim": dim, "backend": "numpy",
            "encoder": (Wq, Wp), "query_emb": q1, "passage_emb": p1}


# ---------------------------------------------------------------------------
# Safety: learned text classification (module 22)
# ---------------------------------------------------------------------------
def char_ngram_features(texts, *, n=3, dim=512) -> np.ndarray:
    """L2-normalised hashed character n-grams (n=3 by default).

    Character rather than word n-grams is the whole point: they survive the
    character-level obfuscation that defeats literal regex rules, because
    "1gnore prev10us" still shares most of its 3-grams with
    "ignore previous". This is why production guardrails pair rules with a
    learned classifier instead of relying on patterns alone.
    """
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, raw in enumerate(texts):
        t = " ".join(str(raw).lower().split())      # collapse whitespace
        t = f"^{t}$"                                # boundary markers
        for j in range(len(t) - n + 1):
            X[i, zlib.crc32(t[j:j + n].encode()) % dim] += 1.0
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _text_predict_factory(res, n, dim):
    """Uniform predict_proba/predict closures for both backends."""
    if HAS_TORCH:
        model = res["model"]

        def proba(texts):
            X = char_ngram_features(texts, n=n, dim=dim)
            model.eval()
            with torch.no_grad():
                p = torch.softmax(
                    model(torch.as_tensor(X, dtype=torch.float32,
                                          device=res["device"])), dim=1)
            return p[:, 1].cpu().numpy()
    else:
        W1, b1, W2, b2 = res["model"]

        def proba(texts):
            X = char_ngram_features(texts, n=n, dim=dim)
            h = np.maximum(X @ W1 + b1, 0.0)
            logits = h @ W2 + b2
            e = np.exp(logits - logits.max(1, keepdims=True))
            return (e / e.sum(1, keepdims=True))[:, 1]

    def predict(texts, threshold=0.5):
        return (proba(texts) >= threshold).astype(int).tolist()

    def features_of(texts):
        return char_ngram_features(texts, n=n, dim=dim)

    return proba, predict, features_of


def train_text_classifier(texts, y, *, n=3, dim=512, hidden=(48,), epochs=400,
                          lr=0.5, device=None, seed=0) -> dict:
    """Binary text classifier over hashed char n-grams; `1` = positive class.

    Autograd + Adam under torch, hand-written backprop otherwise — the same
    architecture on both paths, so the comparison is fair.
    """
    X = char_ngram_features(texts, n=n, dim=dim)
    y = np.asarray(y, dtype=np.int64)
    feats = X.shape[1]

    if HAS_TORCH:
        model = build_mlp(feats, 2, hidden)
        # Use the SAME optimizer/step rule as the NumPy fallback (plain SGD at
        # lr) — Adam at lr=0.5 is 500x its default and trains unstably, which
        # showed up as run-to-run variance. Matching algorithms keeps both
        # backends equivalent and reproducible.
        res = train_module(model, X, y, epochs=epochs, lr=lr, optimizer="sgd",
                           device=device, seed=seed)
    else:
        res = _numpy_mlp_fallback(X, y, hidden, epochs, lr, 0.0, 0.0, seed)
        res["model"] = (res["model"][0], res["model"][1],
                        res["model"][2], res["model"][3])

    proba, predict, features_of = _text_predict_factory(res, n, dim)
    train_acc = float((np.asarray(predict(texts)) == y).mean())
    return {"proba": proba, "predict": predict, "features": features_of,
            "train_acc": train_acc, "losses": res["losses"],
            "backend": res["backend"], "device": res["device"],
            "n_params": res["n_params"], "dim": dim, "n": n,
            "model": res["model"]}


def group_thresholds(y, scores, group, *, target_rate=None) -> dict:
    """Post-processing fairness mitigation: pick one threshold per group.

    Demographic parity requires P(ŷ=1 | group) to match. A single global
    threshold lets a score-shifted group through at a different rate, so we
    solve for a per-group threshold instead. Equalized odds is stricter — it
    constrains TPR *and* FPR per group — and is generally unattainable by
    thresholds alone when base rates differ (see doc 22).
    """
    y, scores = np.asarray(y), np.asarray(scores)
    group = np.asarray(group)
    if target_rate is None:                     # match the overall selection rate
        target_rate = float((scores >= np.median(scores)).mean())
    out, rates = {}, {}
    for g in np.unique(group):
        s = scores[group == g]
        thr = float(np.quantile(s, 1.0 - target_rate))
        out[int(g)] = round(thr, 4)
        rates[int(g)] = round(float((s >= thr).mean()), 3)
    dp = abs(rates[max(rates)] - rates[min(rates)])
    return {"thresholds": out, "selection_rates": rates,
            "demographic_parity_diff": round(dp, 3)}









