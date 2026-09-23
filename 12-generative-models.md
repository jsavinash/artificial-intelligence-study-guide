# 12 — Generative Models

> Back to [index](README.md) · Prev: [11 Transformers](11-transformers-and-foundation-models.md) · Next: [13 Reinforcement Learning](13-reinforcement-learning.md)

**Discriminative** models learn P(y|x) (decide). **Generative** models learn P(x) or P(x|y) (create). Language models are generative — they're in this family too.

## 1. The taxonomy

```
Likelihood-based     → autoregressive, flows, VAEs
Adversarial          → GANs
Diffusion/energy     → DDPM, score matching
Neural autoregressive → PixelCNN, WaveNet, GPT (discrete/sequence generation)
```

## 2. Autoregressive generation

Factorize joint as chain rule: `P(x) = ∏ P(x_t | x_<t)` — models a *tractable* likelihood; train = cross-entropy; generate = sequential sampling. This is GPT ([11](11-transformers-and-foundation-models.md)) and WaveNet audio.

**Sampling strategies at inference** (critical vocabulary for LLMs too):
- **Greedy / temperature τ:** divide logits by τ before softmax — low τ = conservative, high τ = creative; τ→0 greedy, τ→∞ uniform.
- **Top-k:** sample among k most likely tokens.
- **Top-p (nucleus):** smallest set with cumulative prob ≥ p.
- Beam search for likelihood-maximizing tasks (translation); free-form sampling for chat ([14](14-large-language-models.md)).

## 3. VAE (Variational Autoencoder)

- **Encoder** q(z|x) → mean & variance of a latent distribution; **sample z** (reparameterization trick: `z = μ + σ·ε` makes it differentiable); **decoder** p(x|z) reconstructs.
- **Loss = reconstruction + KL(q(z|x)‖N(0,I))** — compression vs structure tradeoff (β-VAE balances).
- **Posterior collapse** risk with powerful decoders; blurry samples (Gaussian likelihoods average modes).
- **Uses:** disentangled latents, anomaly detection, molecular design, Stable Diffusion's latent space cousin.

## 4. GAN (Generative Adversarial Network)

- **Generator** fakes data; **discriminator** classifies real vs fake — minimax game: `min_G max_D E[log D(x)] + E[log(1−D(G(z)))]`.
- **Training is an art:** mode collapse (generator finds one fooling image), no likelihood to monitor, delicate equilibrium — Wasserstein GAN, spectral norm, progressive growing, StyleGAN's style-based latents helped.
- **Wins:** photorealistic faces (StyleGAN), super-resolution (ESRGAN), image-to-image (Pix2Pix), translation (CycleGAN unpaired).
- **Loss of:** diffusion models on quality/stability — but GAN losses still appear as perceptual/adversarial terms inside other pipelines.

## 5. Diffusion models — how images got good

**Forward process:** slowly add Gaussian noise over T steps until data → pure noise (fixed schedule).
**Reverse process:** a network learns to *denoise* step by step — i.e., predicts the noise ε (or the score ∇ log p).
**Sampling:** start from noise, apply learned reverse steps (DDIM makes it fewer/faster; DPM-solver ~10–25 steps).

Key ideas & variants:
- **Score matching / denoising score matching** — the underlying math (learn the gradient of the data density).
- **Latent diffusion (Stable Diffusion):** run diffusion in a **VAE-compressed latent space** → cheap, enables consumer GPUs; the架构 behind SD, DALL·E 2/3 pieces, and video models.
- **Conditioning:** text via cross-attention (CLIP/T5 embeddings) — classifier-free guidance (CFG): trade fidelity vs diversity by extrapolating conditional/unconditional scores.
- **Flow matching / rectified flows:** straighter probability paths → fewer sampling steps; current research frontier (used in some SD3/Flux-style models).
- **Video & 3D:** temporal attention layers, latent video diffusion; consistency models for 1-step generation.

## 6. Evaluation of generators (hard on purpose)

- **Likelihood/ELBO** where defined (AR, VAE); not for GANs/diffusion directly.
- **FID** (Fréchet Inception Distance): feature-space distance — lower = closer *and* diverse (biased for small sample counts).
- **IS** (Inception Score) — older, gameable.
- **Precision/Recall for distributions** (fidelity vs coverage), **CLIP score** for text alignment.
- **Human eval / A-B** remains the real arbiter for images and text.

## 7. How this connects forward

- LLMs = autoregressive generative models over text ([14](14-large-language-models.md)).
- Diffusion + transformers = modern image/video generation ([20](20-multimodal-ai.md)).
- RL fine-tunes generators against reward ([13](13-reinforcement-learning.md), [14](14-large-language-models.md)).

## Mastery Checklist

- [ ] Decomposes P(x) autoregressively and connects it to GPT training
- [ ] Explains temperature/top-k/top-p effects on a generated distribution
- [ ] States the VAE loss terms and the reparameterization trick's purpose
- [ ] Describes the GAN minimax objective and why mode collapse happens
- [ ] Walks through forward/reverse diffusion and why latent diffusion mattered
- [ ] Compares FID, CLIP score, and human eval — and when each misleads
