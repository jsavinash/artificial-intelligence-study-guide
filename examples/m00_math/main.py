"""m00 — Mathematical foundations: gradient descent, Bayes rule, cross-entropy, KL.

Proves theory doc 00-mathematical-foundations.md with executable math.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))


def gradient_descent(f, grad, x0=-4.0, lr=0.1, iters=100):
    """Minimize f by descent — the engine behind ALL training in this repo."""
    x = float(x0)
    for _ in range(iters):
        x -= lr * grad(x)
    return x


def backprop_by_hand():
    """2-layer net by hand: z = w*x; a = relu(z); L = (a - y)^2 -> dL/dw."""
    x, w, y = 2.0, 0.5, 1.0
    z = w * x
    a = max(z, 0.0)
    L = (a - y) ** 2
    dL_da = 2 * (a - y)
    da_dz = 1.0 if z > 0 else 0.0
    dz_dw = x
    dL_dw = dL_da * da_dz * dz_dw
    return L, dL_dw


def bayes_medical_test():
    """P(disease|positive) with 1% prevalence, 90% sens, 95% spec (LR=19)."""
    p_d, sens, spec = 0.01, 0.90, 0.95
    p_pos = sens * p_d + (1 - spec) * (1 - p_d)
    post = sens * p_d / p_pos
    return post


def main():
    # 1. gradient descent finds the minimum of f(x)=x^2
    xmin = gradient_descent(lambda x: x ** 2, lambda x: 2 * x)
    assert abs(xmin) < 1e-6, xmin

    # 2. chain-rule backprop returns finite loss + gradient
    L, g = backprop_by_hand()
    assert np.isfinite(L) and np.isfinite(g)

    # 3. Bayes: positive test still only ~16% posterior (base-rate intuition)
    post = bayes_medical_test()
    assert 0.10 < post < 0.30, post

    # 4. cross-entropy punishes confident-wrong predictions
    y = np.array([0])
    confident_right = np.array([[0.999, 0.001]])
    confident_wrong = np.array([[0.001, 0.999]])
    ce = lambda p: -np.log(np.clip(p[0, y[0]], 1e-12, 1))
    assert ce(confident_wrong) > ce(confident_right) * 10

    # 5. KL divergence: identical distributions -> 0; shifted -> positive
    p = np.array([0.7, 0.3])
    kl_same = float(np.sum(p * np.log(p / p)))
    q = np.array([0.2, 0.8])
    kl_diff = float(np.sum(p * np.log(p / q)))
    assert kl_same == 0.0 and kl_diff > 0.5

    # 6. matrix shapes for a linear layer: (n,d) @ (d,h) -> (n,h)
    X, W = np.ones((5, 4)), np.ones((4, 3))
    assert (X @ W).shape == (5, 3)

    print(f"PASS m00 math | gd_min={xmin:.6f} bayes_post={post:.3f} "
          f"backprop_L={L:.3f} kl_diff={kl_diff:.3f}")


if __name__ == "__main__":
    main()
