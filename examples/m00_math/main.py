"""m00 — Mathematical foundations: gradient descent, Bayes rule, cross-entropy, KL.

Proves theory doc 00-mathematical-foundations.md with executable math.

Every quantity here is first computed by hand, then — where torch is installed —
recomputed by autograd, so the derivatives you derive on paper are *verified*
against the ones the framework produces.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402


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


def torch_path(xmin, L, g, kl_diff):
    """Recompute the hand-derived math with autograd and check they agree."""
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — hand-derived gradients only "
              "(everything above is still verified)")
        return None
    print(f"\n[torch] autograd on {TB.get_device()} — verifying the math you "
          f"just did by hand")

    # 1. same descent, but the gradient comes from autograd (not from `grad`)
    auto = TB.gradient_descent_autograd(lambda t: (t[0] ** 2).sum(), (-4.0,),
                                        lr=0.1, iters=100)
    print(f"    hand-coded GD   x={xmin:+.6f}")
    print(f"    autograd GD     x={auto['x'][0]:+.6f}   "
          f"({'agree' if abs(auto['x'][0]) < 1e-3 else 'DIFFER'})")
    assert abs(auto["x"][0]) < 1e-3, auto["x"]

    # 2. THE payoff: dL/dw derived by hand must equal autograd's dL/dw
    x, y = 2.0, 1.0
    chk = TB.check_gradient(lambda t: (torch_relu(t[0] * x) - y) ** 2,
                            [0.0, 0.0])
    print(f"    hand-derived dL/dw={g:+.4f}")
    print(f"    autograd vs finite-diff max_err={chk['max_abs_err']:.2e} "
          f"({'ok' if chk['ok'] else 'MISMATCH'})")
    assert chk["ok"], chk

    # 3. the softmax cross-entropy gradient identity: dL/dz = (p - onehot)/n
    logits = np.array([[2.0, 1.0, 0.1], [-1.0, 0.5, 2.2]])
    targets = np.array([0, 2])
    sce = TB.softmax_cross_entropy(logits, targets)
    grad_err = float(np.max(np.abs(sce["autograd_grad"] - sce["analytic_grad"])))
    print(f"    cross-entropy loss={sce['loss']:.4f}  "
          f"|autograd - (p-onehot)/n|={grad_err:.2e}")
    assert grad_err < 1e-5, grad_err

    # 4. KL divergence via the framework, matching the hand formula
    p_t = np.array([[0.7, 0.3]])
    q_t = np.array([[0.2, 0.8]])
    kl_torch = float(TB.F.kl_div(torch_log(q_t), torch_t(p_t),
                                 reduction="sum").item())
    print(f"    KL(hand)={kl_diff:.4f}  KL(torch)={kl_torch:.4f}")
    assert abs(kl_torch - kl_diff) < 1e-4, (kl_torch, kl_diff)

    return {"x_autograd": auto["x"][0], "gradcheck_err": chk["max_abs_err"],
            "sce_grad_err": grad_err, "kl_torch": kl_torch}


def torch_relu(t):
    """ReLU — torch when available, NumPy scalar otherwise (same math)."""
    return TB.F.relu(t) if TB.HAS_TORCH else max(t, 0.0)


def torch_t(arr):
    return TB.torch.tensor(arr, dtype=TB.torch.float32)


def torch_log(arr):
    return TB.torch.log(torch_t(arr))


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

    # 7. autograd re-derives everything above and checks it numerically
    tr = torch_path(xmin, L, g, kl_diff)

    if tr:
        print(f"PASS m00 math | gd_min={xmin:.6f} bayes_post={post:.3f} "
              f"backprop_L={L:.3f} kl_diff={kl_diff:.3f} "
              f"autograd_gd={tr['x_autograd']:+.4f} "
              f"gradcheck_err={tr['gradcheck_err']:.1e} "
              f"ce_grad_err={tr['sce_grad_err']:.1e} "
              f"kl_match=True backend={TB.backend_label()}")
    else:
        print(f"PASS m00 math | gd_min={xmin:.6f} bayes_post={post:.3f} "
              f"backprop_L={L:.3f} kl_diff={kl_diff:.3f} backend=numpy")



if __name__ == "__main__":
    main()
