"""Optional PyTorch accelerator layer (module 27).

Why this file exists
--------------------
The tutorial's deep-learning examples (m07-m15, m23) are deliberately written in
pure NumPy so they run with **zero extra dependencies** and expose the math.
That is the right default for teaching. But it leaves one real question open:
*when is it worth reaching for a framework?*

This module answers that with measurement instead of opinion. PyTorch is treated
as an **optional accelerator**, never a requirement:

* if torch is installed  -> accelerated kernels are used where they pay off
* if torch is absent     -> the pure-NumPy path runs and nothing breaks

The amortization rule (the actual lesson)
-----------------------------------------
Importing torch costs ~1.2 s on this Apple M1 / Python 3.14 setup, while a
typical tutorial example finishes in ~0.15 s. So for the *existing* examples,
adding torch makes them ~8x SLOWER, even though each forward/backward pass is
faster in isolation. Torch only wins once the workload is large enough to
amortize the import:

    numpy_time T, speedup s  ->  torch_time = T/s + IMPORT_COST
    break-even:  T > IMPORT_COST * s / (s - 1)

For the kernels here (s = 3-70x) that puts the crossover at roughly
**1.4-1.6 s of NumPy work**. Below it: stay in NumPy. Above it: use torch.

Measured on this machine (Apple M1, 8 cores, CPU unless noted):
    conv2d     48x48 k=5   x20 : numpy 0.117s -> torch 0.002s   (12-70x)
    attention  S=512 d=64  x200: numpy 0.415s -> torch 0.084s   (  5x)
    MLP train  20000x64    x100: numpy 2.671s -> torch 0.684s   (3.9x)
    matmul     100000x512  x50 : numpy 1.801s -> torch 1.561s   (  ~1x, BLAS)
"""
from __future__ import annotations

import time

import numpy as np

try:  # optional dependency — absence is a supported configuration
    import torch
    import torch.nn as nn

    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
    HAS_MPS = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False
    TORCH_VERSION = None
    HAS_MPS = False

# Measured cost of `import torch` on this machine (seconds). Used by
# should_accelerate() so the decision is explicit rather than hand-wavy.
IMPORT_COST = 1.24

__all__ = [
    "HAS_TORCH", "TORCH_VERSION", "HAS_MPS", "IMPORT_COST",
    "backend_info", "should_accelerate", "numpy_conv2d", "torch_conv2d",
    "numpy_attention", "torch_attention", "train_mlp", "benchmark_backends",
]


def backend_info(device: str = "cpu") -> dict:
    """Describe the compute backend actually available at runtime."""
    return {
        "torch_available": HAS_TORCH,
        "torch_version": TORCH_VERSION,
        "mps_available": HAS_MPS,
        "device": device if HAS_TORCH else "numpy",
        "import_cost_s": IMPORT_COST,
    }


def should_accelerate(numpy_seconds: float, speedup: float,
                      import_cost: float = IMPORT_COST) -> bool:
    """True when switching to torch would beat staying in NumPy.

    Compares `numpy_seconds` against the amortization break-even
    T > import_cost * s / (s - 1). Values are estimates — callers pass the
    measured speedup for their kernel class.
    """
    if not HAS_TORCH or speedup <= 1.0:
        return False
    break_even = import_cost * speedup / (speedup - 1.0)
    return numpy_seconds > break_even


# --------------------------------------------------------------------------
# 2D convolution — the clearest framework win (im2col/GEMM vs Python loops)
# --------------------------------------------------------------------------
def numpy_conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Reference implementation: explicit nested loops (what m09 teaches)."""
    n = img.shape[0]
    k = kernel.shape[0]
    o = (n - k) // stride + 1
    out = np.zeros((o, o))
    for i in range(o):
        for j in range(o):
            out[i, j] = (img[i * stride:i * stride + k,
                            j * stride:j * stride + k] * kernel).sum()
    return out


def torch_conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Same math, executed by torch's optimized kernel (im2col + BLAS GEMM).

    Falls back to numpy_conv2d when torch is unavailable, so callers never
    need to branch.
    """
    if not HAS_TORCH:
        return numpy_conv2d(img, kernel, stride)
    x = torch.tensor(img, dtype=torch.float32).reshape(1, 1, *img.shape)
    w = torch.tensor(kernel, dtype=torch.float32).reshape(1, 1, *kernel.shape)
    with torch.no_grad():
        out = torch.nn.functional.conv2d(x, w, stride=stride)
    return out.reshape(out.shape[-2], out.shape[-1]).numpy()


# --------------------------------------------------------------------------
# Scaled dot-product attention
# --------------------------------------------------------------------------
def numpy_attention(Q, K, V, causal: bool = False):
    """softmax(QK^T / sqrt(d_k)) V — the NumPy reference from m11."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        T = scores.shape[0]
        scores = np.where(np.tril(np.ones((T, T))) == 1, scores, -1e9)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ V, p


def torch_attention(Q, K, V, causal: bool = False):
    """Fused scaled-dot-product-attention kernel (flash attention on GPU)."""
    if not HAS_TORCH:
        return numpy_attention(Q, K, V, causal)
    q, k, v = (torch.tensor(a, dtype=torch.float32) for a in (Q, K, V))
    with torch.no_grad():
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal)
        p = torch.softmax(q @ k.T / (q.shape[-1] ** 0.5), dim=-1)
    return out.numpy(), p.numpy()


# --------------------------------------------------------------------------
# MLP training — where autograd replaces hand-written backward()
# --------------------------------------------------------------------------
def train_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 100, hidden: int = 128,
              lr: float = 0.5, backend: str = "auto", device: str = "cpu",
              seed: int = 0) -> dict:
    """Train a 2-layer MLP and report loss trajectory + wall time.

    backend: "numpy" | "torch" | "auto"  (auto = torch when installed)
    Returns dict(losses, seconds, backend, device, final_acc).
    """
    use_torch = HAS_TORCH and backend in ("auto", "torch")
    if use_torch:
        return _train_mlp_torch(X, y, epochs, hidden, lr, device, seed)
    return _train_mlp_numpy(X, y, epochs, hidden, lr, seed)


def _train_mlp_numpy(X, y, epochs, hidden, lr, seed) -> dict:
    """Hand-written forward + backward (no autograd) — the m07/m08 approach.

    Init reproduces torch's nn.Linear default exactly: U(-1/sqrt(fan_in),
    1/sqrt(fan_in)) for weights, zeros for bias. Comparing backends is only
    meaningful if they start from the same distribution.
    """
    rng = np.random.default_rng(seed)
    d, n_out = X.shape[1], int(y.max()) + 1

    def torch_like(rows, cols, rng):
        bound = 1.0 / np.sqrt(rows)
        return rng.uniform(-bound, bound, size=(rows, cols))

    W1 = torch_like(d, hidden, rng)
    b1 = np.zeros(hidden)
    W2 = torch_like(hidden, n_out, rng)
    b2 = np.zeros(n_out)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        z1 = X @ W1 + b1
        h = np.maximum(0.0, z1)                   # forward
        logits = h @ W2 + b2
        p = softmax(logits)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        g = p.copy()                              # backward (analytical)
        g[np.arange(len(y)), y] -= 1.0
        g /= len(y)
        gW2, gb2 = h.T @ g, g.sum(0)
        gh = (g @ W2.T) * (z1 > 0)
        gW1, gb1 = X.T @ gh, gh.sum(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
        losses.append(float(loss))
    secs = time.perf_counter() - t0
    # NOTE: ReLU must be applied here too — scoring the pre-activation would
    # under-report accuracy while the loss keeps falling (a silent bug).
    acc = float((np.maximum(0.0, X @ W1 + b1) @ W2 + b2).argmax(1)
                .__eq__(y).mean())
    return {"losses": losses, "seconds": secs, "backend": "numpy",
            "device": "cpu", "final_acc": acc}


def _train_mlp_torch(X, y, epochs, hidden, lr, device, seed) -> dict:
    """Same model via autograd: no backward() is written by hand."""
    torch.manual_seed(seed)
    dev = device if (device == "cpu" or HAS_MPS or _cuda_ok(device)) else "cpu"
    n_out = int(y.max()) + 1
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, n_out)).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)

    losses, t0 = [], time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()                           # autograd does the calculus
        opt.step()
        losses.append(float(loss.item()))
    _synchronize(dev)
    secs = time.perf_counter() - t0
    with torch.no_grad():
        acc = float((net(Xt).argmax(1) == yt).float().mean().item())
    return {"losses": losses, "seconds": secs, "backend": "torch",
            "device": dev, "final_acc": acc}


def _cuda_ok(device: str) -> bool:
    return bool(HAS_TORCH and device == "cuda" and torch.cuda.is_available())


def _synchronize(device: str) -> None:
    """GPU work is async — without this the timing would be a lie."""
    if not HAS_TORCH:
        return
    if device == "mps" and HAS_MPS:
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Head-to-head benchmark
# --------------------------------------------------------------------------
def benchmark_backends(device: str = "cpu", quiet: bool = False) -> dict:
    """Time NumPy vs torch on the three kernels that matter, then apply the
    amortization rule to decide which backend each workload *should* use.
    """
    from sklearn.datasets import make_classification

    report: dict = {"device": device, "torch": HAS_TORCH,
                    "torch_version": TORCH_VERSION, "results": []}

    def record(name, np_s, t_s, note=""):
        entry = {"workload": name, "numpy_s": round(np_s, 4),
                 "torch_s": round(t_s, 4) if t_s else None,
                 "speedup": round(np_s / t_s, 2) if t_s else None,
                 "worth_it": bool(t_s and should_accelerate(np_s, np_s / t_s)),
                 "note": note}
        report["results"].append(entry)
        if not quiet:
            sp = f"{entry['speedup']}x" if entry["speedup"] else "n/a"
            verdict = "USE TORCH" if entry["worth_it"] else "stay in NumPy"
            ts = f"{entry['torch_s']:6.3f}" if entry["torch_s"] else "  --- "
            print(f"  {name:<32} numpy {np_s:6.3f}s  torch {ts}s  "
                  f"{sp:>7}  -> {verdict}")
        return entry

    # 1) convolution
    img, ker = np.random.rand(48, 48), np.random.rand(5, 5)
    t = time.perf_counter()
    for _ in range(20):
        numpy_conv2d(img, ker)
    np_s = time.perf_counter() - t
    t_s = None
    if HAS_TORCH:
        torch_conv2d(img, ker)                       # warm up
        t = time.perf_counter()
        for _ in range(20):
            torch_conv2d(img, ker)
        t_s = time.perf_counter() - t
    record("conv2d 48x48 k=5 (x20)", np_s, t_s, "im2col+GEMM vs Python loops")

    # 2) attention
    S, d = 512, 64
    X = np.random.rand(S, d)
    Wq, Wk, Wv = [np.random.rand(d, d) * 0.1 for _ in range(3)]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    t = time.perf_counter()
    for _ in range(200):
        numpy_attention(Q, K, V)
    np_s = time.perf_counter() - t
    t_s = None
    if HAS_TORCH:
        torch_attention(Q, K, V)
        t = time.perf_counter()
        for _ in range(200):
            torch_attention(Q, K, V)
        t_s = time.perf_counter() - t
    record("attention S=512 d=64 (x200)", np_s, t_s, "fused SDPA kernel")

    # 3) MLP training
    Xc, yc = make_classification(n_samples=20000, n_features=64, n_classes=10,
                                n_informative=16, random_state=0)
    np_r = train_mlp(Xc, yc, epochs=100, backend="numpy")
    t_r = (train_mlp(Xc, yc, epochs=100, backend="torch", device=device)
           if HAS_TORCH else {"seconds": None, "final_acc": None})
    record("MLP train 20000x64 (100 ep)", np_r["seconds"], t_r.get("seconds"),
           "autograd vs hand-written backward")
    report["mlp_numpy_acc"] = round(np_r["final_acc"], 3)
    report["mlp_torch_acc"] = round(t_r.get("final_acc") or 0.0, 3)
    return report



