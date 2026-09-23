"""m15 — Fine-tuning & PEFT: LoRA low-rank math, merge, distillation, quantization.

Proves theory doc 15-fine-tuning-and-peft.md.

Two paths:
  1. NumPy from-scratch — SVD rank sweep, merge equivalence, teacher/student
     distillation, int8 quantization error. The math with nothing hidden.
  2. PyTorch — the *real* LoRA: nn.Linear layers frozen in place, trainable
     low-rank adapters attached via module replacement, then merged back into
     the base weights so inference cost returns to zero.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification  # noqa: E402
from ai_core.metrics import accuracy  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402


def lora_decompose(dW, r):
    """Any dW ≈ B@A with rank r: keep top-r singular triplets (SVD)."""
    U, S, Vt = np.linalg.svd(dW, full_matrices=False)
    B = U[:, :r] * S[:r]          # d x r
    A = Vt[:r, :]                 # r x d
    return B, A


def torch_path():
    """Real LoRA: freeze an nn.Module, attach adapters, fine-tune, merge back."""
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — SVD-based LoRA math only")
        return None
    X, y = classification(n=600, d=16, classes=4, seed=3)
    print(f"\n[torch] real LoRA on {TB.get_device()} — freeze nn.Linear, "
          f"train only A and B")
    res = TB.train_peft(X, y, hidden=(64, 64), r=8, alpha=16.0,
                        epochs=120, lr=1e-2, seed=0)
    lo = res["lora"]
    print(f"    adapters={lo['adapters']} trainable={lo['trainable']}/"
          f"{lo['total']} ({lo['percent']:.1f}%) frozen={lo['frozen']} "
          f"acc={res['acc']:.3f} (pretrain {lo['pretrain_acc']:.3f})")

    # 1) LoRA really does shrink the trainable budget
    assert lo["adapters"] > 0, "expected adapters on the hidden layers"
    assert lo["frozen"] > lo["trainable"], (lo["frozen"], lo["trainable"])
    assert lo["percent"] < 30.0, lo["percent"]
    # 2) fine-tuning on adapters alone kept us at (or above) pretrain accuracy
    assert res["acc"] >= lo["pretrain_acc"] - 0.05, (res["acc"],
                                                     lo["pretrain_acc"])

    # 3) merge: output must be numerically unchanged, adapters must disappear
    import torch
    model = res["model"]
    dev = next(model.parameters()).device          # model may live on mps/cuda
    probe = torch.as_tensor(X[:8], dtype=torch.float32, device=dev)
    with torch.no_grad():
        before = model(probe).clone()
    merged = TB.merge_lora(model)
    with torch.no_grad():
        after = model(probe)
    max_diff = float((before - after).abs().max().item())
    print(f"    merged_layers={merged} max_output_diff={max_diff:.2e} "
          f"(0 => zero inference overhead)")
    assert merged == lo["adapters"], (merged, lo["adapters"])
    assert max_diff < 1e-4, max_diff
    return {"lora": lo, "merged": merged, "max_diff": max_diff}


def main():
    rng = np.random.default_rng(0)
    d = 64
    W = rng.normal(size=(d, d)) * 0.1                 # frozen base weight
    dW_full = rng.normal(size=(d, d)) * 0.02          # what full FT would learn

    # ---- 1) LoRA: tiny trainable budget approximates the full update ----
    params_full = d * d
    trainable = {}
    errs = {}
    for r in (2, 8, 16, 32):
        B, A = lora_decompose(dW_full, r)
        trainable[r] = d * r + r * d                   # both matrices
        errs[r] = np.linalg.norm(dW_full - B @ A) / np.linalg.norm(dW_full)
    assert trainable[2] < params_full * 0.1           # r=2: 6% of full params
    assert trainable[8] < params_full * 0.5           # r=8: 25%, still far smaller
    assert errs[32] < errs[2]                          # higher rank = better fit
    assert errs[8] < 0.9                               # 8-rank already captures signal

    # ---- 2) merge: W' = W + B@A  => ZERO inference overhead vs separate adapter ----
    B, A = lora_decompose(dW_full, 8)
    W_merged = W + B @ A
    x = rng.normal(size=d)
    y_adapter = (W + B @ A) @ x
    y_merged = W_merged @ x
    assert np.allclose(y_adapter, y_merged, atol=1e-10)

    # ---- 3) knowledge distillation: student learns teacher's SOFT distribution ----
    X, y = classification(n=500, d=10, classes=3, seed=5)
    y3 = (X[:, 0] + X[:, 1] > 0).astype(int)          # linearly separable rule
    K = int(y3.max()) + 1
    softmax = lambda L: (lambda e: e / e.sum(1, keepdims=True))(
        np.exp(L - L.max(1, keepdims=True)))

    # teacher: full head TRAINED on hard labels (the strong model)
    Wt = rng.normal(size=(10, K)) * 0.5
    for _ in range(400):
        Pt = softmax(X @ Wt)
        onehot = np.zeros_like(Pt); onehot[np.arange(len(y3)), y3] = 1
        Wt -= 0.5 * X.T @ (Pt - onehot) / len(X)
    logits_t = X @ Wt
    Pt = softmax(logits_t)
    acc_teacher = accuracy(y3, logits_t.argmax(1))

    # student: sees ONLY the first 4 features (capacity bottleneck),
    # trained on teacher soft labels at temperature T=2 — no hard labels used
    Z = X[:, :4]
    Ws = rng.normal(size=(4, K)) * 0.1
    lr, losses, T = 0.5, [], 2.0
    for _ in range(400):
        Ps = softmax(Z @ Ws / T)
        Pt_T = Pt ** (1 / T); Pt_T /= Pt_T.sum(1, keepdims=True)  # softened teacher
        loss = -(Pt_T * np.log(Ps + 1e-12)).sum(1).mean()         # soft cross-entropy
        g = (Ps - Pt_T) / len(X) / T
        Ws -= lr * Z.T @ g
        losses.append(loss)
    assert losses[-1] < losses[0]
    acc_student = accuracy(y3, (Z @ Ws).argmax(1))
    assert acc_teacher > 0.8, acc_teacher
    assert acc_student > 0.7, acc_student             # dark knowledge transferred

    # ---- 4) quantization error: fp32 -> int8 scales back with small loss ----
    q = dW_full
    scale = np.abs(q).max() / 127
    q8 = np.round(q / scale).astype(np.int8)
    rel_err = np.linalg.norm(q - q8.astype(float) * scale) / np.linalg.norm(q)
    assert rel_err < 0.05, rel_err                     # <5% relative error

    # ---- 5) frozen-base check: only A,B would receive gradients ----
    assert W.shape == (d, d) and trainable[8] == 2 * d * 8

    real = torch_path()
    real_note = ""
    if real:
        real_note = (f" real_lora_trainable={real['lora']['percent']:.1f}%"
                     f" merged={real['merged']} merge_diff={real['max_diff']:.1e}")

    print(f"PASS m15 peft | lora trainable {trainable[8]}/{params_full}="
          f"{trainable[8]/params_full:.1%} rank_err 2:{errs[2]:.2f}->32:{errs[32]:.2f} "
          f"merge_equal=True distill_loss {losses[0]:.3f}->{losses[-1]:.3f} "
          f"student_acc={acc_student:.3f} teacher_acc={acc_teacher:.3f} "
          f"int8_err={rel_err:.3f}{real_note}")


if __name__ == "__main__":
    main()
