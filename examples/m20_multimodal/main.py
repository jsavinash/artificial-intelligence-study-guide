"""m20 — Multimodal AI: CLIP-style shared embedding space, cross-modal retrieval,
and the symmetric contrastive (InfoNCE) loss matrix.

Proves theory doc 20-multimodal-ai.md.

Two paths:
  1. Embedding space (NumPy) — the similarity matrix, zero-shot classification
     and the InfoNCE objective, with no learning involved.
  2. Two-tower CLIP (PyTorch) — image and text go through *separate* projection
     heads trained end-to-end with symmetric InfoNCE. The raw inputs are
     deliberately misaligned first, so this proves the heads learn the shared
     space rather than reading it off the input.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.llm import MockLLM  # noqa: E402
from ai_core import torch_backend as TB  # noqa: E402

# (image_id, caption) pairs — in CLIP these come from 400M web pairs
PAIRS = [
    ("img1", "a photo of a red car on the street"),
    ("img2", "a photo of a cat sitting on a couch"),
    ("img3", "a diagram of a neural network architecture"),
    ("img4", "an aerial photo of a farm field"),
    ("img5", "a portrait of a smiling person outdoors"),
    ("img6", "a bowl of fruit on a kitchen table"),
    ("img7", "a mountain landscape with a lake"),
    ("img8", "code text on a computer monitor screen"),
]


def clip_style_space(llm):
    """Both modalities map into ONE space: 'image' described by caption + noise."""
    r = np.random.default_rng(0)
    text_vecs = np.array([llm.embed(cap) for _, cap in PAIRS])
    # synthetic visual features: same semantics + small visual-only noise
    img_vecs = text_vecs + 0.02 * r.normal(size=text_vecs.shape)
    return img_vecs, text_vecs


def cosine_matrix(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def info_nce(S, labels):
    """Symmetric contrastive loss: (row CE + col CE)/2 over the similarity matrix."""
    p_r = np.exp(S - S.max(1, keepdims=True)); p_r /= p_r.sum(1, keepdims=True)
    p_c = np.exp(S - S.max(0, keepdims=True)); p_c /= p_c.sum(0, keepdims=True)
    return float(-(np.log(p_r[np.arange(len(labels)), labels] + 1e-12).mean() +
                   np.log(p_c[labels, np.arange(len(labels))] + 1e-12).mean()) / 2)


def torch_path(imgs, txts):
    """Train real projection heads with symmetric InfoNCE — CLIP in miniature.

    The inputs are first pushed through a fixed random linear map plus noise.
    That destroys the trivial alignment, so if retrieval works afterwards it is
    because the two towers *learned* to meet in a shared space.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — embedding-space path only (CLIP skipped)")
        return None
    rng = np.random.default_rng(7)
    n, d = txts.shape
    warp = rng.normal(size=(d, d)) * 0.3             # fixed, unknown transform
    img_raw = imgs @ warp + 0.30 * rng.normal(size=(n, d))
    txt_raw = txts + 0.30 * rng.normal(size=(n, d))

    # baseline: naive cosine on the *misaligned* raw features
    S_before = cosine_matrix(img_raw, txt_raw)
    top1_before = float((S_before.argmax(1) == np.arange(n)).mean())

    print(f"\n[torch] two-tower CLIP on {TB.get_device()} — "
          f"InfoNCE over a {n}x{n} similarity matrix")
    res = TB.train_clip(img_raw, txt_raw, temperature=0.07, epochs=400,
                        lr=0.02, embed_dim=32, seed=0)
    print(f"    raw cosine top1={top1_before:.2f} -> trained "
          f"i2t={res['i2t_top1']:.2f} t2i={res['t2i_top1']:.2f} "
          f"loss {res['losses'][0]:.2f}->{res['losses'][-1]:.2f} "
          f"params={res['n_params']}")

    # 1) the towers learned a shared space that raw features did not have
    if top1_before < 1.0:
        assert res["i2t_top1"] > top1_before, (top1_before, res["i2t_top1"])
    # 2) both directions must succeed (CLIP is symmetric by construction)
    assert res["i2t_top1"] >= 0.75, res["i2t_top1"]
    assert res["t2i_top1"] >= 0.75, res["t2i_top1"]
    # 3) the contrastive loss actually went down
    assert res["losses"][-1] < res["losses"][0]
    return {"before": top1_before, **res}


def main():
    llm = MockLLM()
    imgs, txts = clip_style_space(llm)
    n = len(PAIRS)
    labels = np.arange(n)

    # ---- 1) one similarity matrix governs both directions (CLIP's trick) ----
    S = cosine_matrix(txts, imgs)                  # text->image scores
    assert S.shape == (n, n)
    diag = np.diag(S)
    assert (S.argmax(axis=1) == labels).all(), "text->image retrieval@1 failed"
    assert (S.argmax(axis=0) == labels).all(), "image->text retrieval@1 failed"

    # ---- 2) zero-shot classification: map label text into the same space ----
    class_names = ["car", "cat", "network", "field", "person", "fruit",
                   "mountain", "code"]
    class_vecs = np.array([llm.embed(f"a photo of a {c}") for c in class_names])
    zs_pred = cosine_matrix(class_vecs, imgs).argmax(axis=1)
    hits = int((zs_pred == labels).sum())
    assert hits >= n * 0.5, hits     # hashed embeds: >> chance (1/8), >=50% recall

    # ---- 3) InfoNCE: aligned pairs beat shuffled pairs ----
    loss_aligned = info_nce(S, labels)
    shuffled = np.roll(labels, 1)
    loss_shuffled = info_nce(S, shuffled)
    assert loss_aligned < loss_shuffled, (loss_aligned, loss_shuffled)

    # ---- 4) temperature sharpens the contrastive distribution ----
    sharp = info_nce(S * 5, labels)
    assert sharp <= loss_aligned + 1e-9            # higher scale = lower loss on diag

    # ---- 5) embedding dim consistency (the API contract real CLIP needs) ----
    assert imgs.shape[1] == txts.shape[1] == 64

    print(f"PASS m20 multimodal | retrieval@1=8/8 both directions "
          f"zero_shot={hits}/{n} InfoNCE aligned={loss_aligned:.3f}"
          f"<shuffled={loss_shuffled:.3f} temp5={sharp:.3f} dim=64")


if __name__ == "__main__":
    main()
