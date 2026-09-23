"""m09 — CNNs & vision: 2D convolution from scratch, Sobel edge detector,
max-pooling, and a gradient check of the conv op.

Proves theory doc 09-convolutional-networks-and-vision.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import image_grid  # noqa: E402


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
          f"pool=3x3 gradcheck_max_err={np.abs(g_num-g_ana).max():.2e}")


if __name__ == "__main__":
    main()
