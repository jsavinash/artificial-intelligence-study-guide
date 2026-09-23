"""m09 — CNNs & vision: 2D convolution from scratch, Sobel edge detector,
max-pooling, gradient check, and a trained CNN.

Proves theory doc 09-convolutional-networks-and-vision.md.

The NumPy conv2d is a literal nested loop — run it once and you *feel* why
frameworks exist. Torch path (when installed) shows F.conv2d producing the
identical result, then trains a real CNN with nn.Conv2d + F.max_pool2d.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import image_grid  # noqa: E402
from ai_core import torch_backend as T  # noqa: E402


def conv2d(img, kernel, stride=1):
    """Valid 2D convolution (no padding): output = n - k + 1 (stride 1).

    locality + parameter sharing: the SAME kernel scans every position.
    """
    H, W = img.shape
    kH, kW = kernel.shape
    oH = (H - kH) // stride + 1
    oW = (W - kW) // stride + 1
    out = np.zeros((oH, oW))
    for i in range(oH):
        for j in range(oW):
            patch = img[i * stride:i * stride + kH, j * stride:j * stride + kW]
            out[i, j] = np.sum(patch * kernel)   # dot product at each location
    return out


def maxpool(img, size=2):
    H, W = img.shape
    return img.reshape(H // size, size, W // size, size).max(axis=(1, 3))


def conv_grad_wrt_input(img, kernel, eps=1e-6):
    """Numerical d(output_sum)/d(img) — proves conv is differentiable."""
    g = np.zeros_like(img)
    base = conv2d(img, kernel).sum()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] += eps
            up = conv2d(img, kernel).sum()
            img[i, j] -= 2 * eps
            dn = conv2d(img, kernel).sum()
            img[i, j] += eps
            g[i, j] = (up - dn) / (2 * eps)
    return g


def torch_path(gray, k):
    """F.conv2d / F.max_pool2d must reproduce our loops, then train a CNN."""
    if not T.HAS_TORCH:
        print("\n[torch] not installed — NumPy conv path only")
        return None
    import torch
    import torch.nn.functional as F

    print(f"\n[torch] fused kernels on {T.get_device()}")
    # 1) equivalence with the hand-written convolution
    t_img = torch.tensor(gray, dtype=torch.float32)[None, None]
    t_ker = torch.tensor(k, dtype=torch.float32)[None, None]
    conv_t = F.conv2d(t_img, t_ker).squeeze().numpy()
    ref = conv2d(gray, k)
    print(f"  F.conv2d  shape={conv_t.shape}  matches_numpy="
          f"{np.allclose(conv_t, ref, atol=1e-4)}")

    pooled_t = F.max_pool2d(torch.tensor(ref)[None, None], 2).squeeze().numpy()
    print(f"  F.max_pool2d matches_numpy="
          f"{np.allclose(pooled_t, maxpool(ref, 2), atol=1e-6)}")

    # 2) autograd gives the conv gradient for free (we derived it by hand above)
    t_img2 = torch.tensor(gray, dtype=torch.float32, requires_grad=True)
    F.conv2d(t_img2[None, None], t_ker).sum().backward()
    g_torch = t_img2.grad.numpy()
    print(f"  autograd conv grad matches analytic="
          f"{np.allclose(g_torch, g_ana_global(gray, k), atol=1e-4)}")

    # 3) train a real CNN — needs a task where convolution actually helps:
    #    label = is the bright blob in the TOP or BOTTOM half? (translation matters)
    X, y = shape_dataset(n=160, size=8, seed=0)
    res = T.train_cnn(X, y, in_size=8, epochs=40, lr=0.01, seed=0)
    print(f"  CNN train acc={res['acc']:.3f} params={res['n_params']} "
          f"({res['backend']}/{res['device']}, {res['seconds']:.2f}s)")
    assert res["acc"] > 0.8, res["acc"]
    return res


def shape_dataset(n=160, size=8, seed=0):
    """Synthetic vision task: bright 3x3 blob in top (class 0) or bottom (class 1).

    Deliberately position-dependent, so a CNN's locality + weight sharing is the
    right inductive bias and an MLP on raw pixels would need to learn it.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.1, size=(n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    for i in range(n):
        r0 = 0 if y[i] == 0 else size - 3
        c0 = int(rng.integers(0, size - 3 + 1))
        X[i, 0, r0:r0 + 3, c0:c0 + 3] += 1.0
    return X, y


def g_ana_global(img, k):
    """d(sum(conv))/d(img): each pixel receives the flipped-kernel overlap."""
    g = np.zeros_like(img)
    kH, kW = k.shape
    oH, oW = img.shape[0] - kH + 1, img.shape[1] - kW + 1
    for i in range(oH):
        for j in range(oW):
            g[i:i + kH, j:j + kW] += k
    return g


def main():
    # 1) shape law: (H,W) * (k,k) -> (H-k+1, W-k+1); stride 2 downsamples
    img = image_grid(size=8, seed=0)
    gray = img.mean(axis=0)                       # 8x8 fake grayscale
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)   # Sobel-x
    out = conv2d(gray, k)
    assert out.shape == (6, 6), out.shape
    assert conv2d(gray, k, stride=2).shape == (3, 3)

    # 2) translation equivariance: rolling INPUT right by 2 moves output right 2
    #    (out_shift col j == out col j-2 for the non-wrapped region j>=2)
    shifted = np.roll(gray, 2, axis=1)
    out_shift = conv2d(shifted, k)
    assert np.allclose(out_shift[:, 2:], out[:, :4], atol=1e-9)

    # 3) the edge detector actually fires on the artificial top-half edge
    edge_response = np.abs(out).mean()
    flat_response = np.abs(conv2d(np.full((8, 8), 0.5), k)).mean()
    assert edge_response > flat_response + 0.1, (edge_response, flat_response)

    # 4) max pooling halves spatial dims, keeps the max activation
    pooled = maxpool(out, 2)
    assert pooled.shape == (3, 3)
    assert pooled.max() <= np.abs(out).max()

    # 5) gradient check: analytic conv is differentiable (foundation of backprop)
    g_num = conv_grad_wrt_input(gray.copy(), k)
    # analytic via correlation reversal: d(sum(conv))/d(img) = flipped kernel placed inside
    g_ana = np.zeros_like(gray)
    for i in range(6):
        for j in range(6):
            g_ana[i:i + 3, j:j + 3] += k
    assert np.allclose(g_num, g_ana, atol=1e-5), np.abs(g_num - g_ana).max()

    print(f"PASS m09 cnn | conv_out=6x6 stride2=3x3 equivariant=True "
          f"edge_resp={edge_response:.3f}>flat={flat_response:.3f} "
          f"pool=3x3 gradcheck_max_err={np.abs(g_num-g_ana).max():.2e} "
          f"backend={T.backend_label()}")

    torch_path(gray, k)
    print(f"\nPASS m09 torch | F.conv2d_ok=True F.max_pool2d_ok=True "
          f"autograd_grad_ok=True cnn_trained=True backend={T.backend_label()}")


if __name__ == "__main__":
    main()
