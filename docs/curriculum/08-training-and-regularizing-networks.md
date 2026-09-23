# 08 — Training & Regularizing Deep Networks

> Back to [index](../../README.md) · Prev: [07 DL Fundamentals](07-deep-learning-fundamentals.md) · Next: [09 CNNs & Vision](09-convolutional-networks-and-vision.md)

Getting a net to *train well* is a craft. This is the toolbox.

## 1. Optimizers (the update rules)

| Optimizer | Idea | Notes |
|---|---|---|
| **SGD + momentum** | accumulate velocity | classic default for vision; often best *final* accuracy with tuning |
| **RMSProp** | per-param adaptive LR from RMS of grads | good for RNNs |
| **Adam** | momentum + RMSProp, bias-corrected | **default for almost everything**; lr 1e-3 typical |
| **AdamW** | Adam + **decoupled weight decay** | **the transformer/LLM default** (see below) |
| Adagrad | per-param LR, decays | sparse features |
| LAMB/LARS | layer-wise scaled batches | very large-batch training |

- **Weight decay vs L2:** in Adam they differ — AdamW applies decay directly to weights (true regularization); classic Adam L2 is entangled with adaptive moments. Use AdamW.
- Typical starting LRs: SGD 1e-1 (with momentum), Adam/AdamW 1e-3, transformers 1e-4–5e-4 (or 1e-3 for LoRA).

## 2. Learning-rate schedules

- **Warmup** (linear LR rise over first N steps) — critical for transformers; avoids early unstable gradients.
- **Step/cosine decay** — cosine annealing to ~0 is the modern default for fine-tuning and pretraining.
- **Cyclic/1cycle** — aggressive LR exploration then consolidation; also a great LR-finder.
- **Reduce-on-plateau** — reactive fallback.
- Rule: schedule and batch size are coupled (linear scaling rule for SGD).

## 3. Initialization

- **Xavier/Glorot** (tanh/sigmoid): `Var(w) = 2/(fan_in+fan_out)`.
- **He/Kaiming** (ReLU): accounts for ReLU zeroing half the units.
- Random labels on an uninitialized net should give ~chance accuracy — if not, init is broken.
- Bad init ⇒ activations/gradients explode or vanish with depth.

## 4. Normalization techniques

- **BatchNorm:** normalize per-feature across the batch; adds learnable scale/shift. Stabilizes, allows higher LR; caveats: batch-size sensitive (small batches hurt), train/inference running stats differ (keep `model.eval()` right), doesn't apply cleanly to autoregressive decoding.
- **LayerNorm:** normalize per-sample — **no batch dependence** → transformers use this.
- **Group Norm / Instance Norm:** batch-independent alternatives for small-batch vision.
- Modern trend: **Pre-LN transformers** (LayerNorm before sublayer) — much more stable than Post-LN.
- **RMSNorm:** simplified LayerNorm without mean-centering — used in LLaMA/modern LLMs ([14](14-large-language-models.md)).

## 5. Regularization (fighting overfitting)

- **Dropout** (Srivastava 2014): randomly zero units with prob p during *training only*, scale at inference. Ensemble-of-subnetworks intuition. p=0.1–0.5. **Not** used inside most LLM pretraining (LN + massive data instead) — do use for small tabular nets.
- **Weight decay** — shrink weights toward zero.
- **Early stopping** — monitor val loss, checkpoint best, patience.
- **Data augmentation** — the most honest regularizer: label-preserving transforms.
  - Images: random crop+flip+rotate+color jitter, **RandAugment/AutoAugment**, cutout, Mixup/CutMix (blend two images+labels).
  - Text: back-translation, synonym swap, token dropout (or just more data).
  - Audio: time stretch, pitch shift, SpecAugment (mask time/freq bands).
- **Label smoothing** — soften targets (0.9/0.1 instead of 1/0); reduces overconfidence, improves calibration. Standard in transformers.
- **Stochastic depth / weight tying** — for very deep nets / shared embeddings.

## 6. Data & batching decisions

- **Batch size:** bigger = stabler grads, faster epochs on GPU, needs LR scaling; smaller = regularization noise, fits memory. 32–256 typical; LLMs use micro-batching + gradient accumulation to emulate huge batches.
- **Shuffle every epoch** (except true sequential/streaming data).
- **Gradient accumulation:** sum N micro-batch losses / step less often → effective large batch on small GPUs.
- **Mixed precision (fp16/bf16):** `torch.autocast` + GradScaler; ~2× speed, half memory. **bf16 preferred** (same exponent range as fp32, no scaler needed) — standard in LLM training.

## 7. Debugging a training run (in order)

1. **Overfit one tiny batch** (20 samples → ~0 loss). If you can't, there's a bug — wrong labels, loss, or lr.
2. Plot train/val loss: loss ≈ log(#classes) at init? NaN → lr too high / bad data / need grad clipping.
3. Check input shapes/dtypes/scaling and label ranges.
4. Baseline comparison: is your net beating the classical baseline ([03](03-supervised-learning-algorithms.md))?
5. Sane ranges: lr ∈ [1e-5, 1e-2], watch grad-norm histograms.
6. Gradient clipping (`clip_grad_norm_(..., 1.0)`) for RNN/transformer instability.

## 8. Distributed training (concepts)

- **Data parallelism:** each GPU holds a full model on different samples; all-reduce gradients. Default (DDP).
- **Gradient/ tensor parallelism & pipeline parallelism:** model too big for one GPU — split by layers/parameters (how LLM clusters train; covered again in [14](14-large-language-models.md), [21](21-mlops-and-production-ml.md)).

## Mastery Checklist

- [ ] Explains Adam vs AdamW and why transformers use AdamW + warmup + cosine decay
- [ ] Chooses initialization, normalization, and dropout for a given architecture
- [ ] Designs an augmentation pipeline for images or text with label-preservation in mind
- [ ] Uses label smoothing, early stopping, and weight decay coherently
- [ ] Debugs a non-learning model using the 5-step procedure
- [ ] Uses mixed precision and gradient accumulation to train bigger/faster
