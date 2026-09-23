# 20 — Multimodal AI

> Back to [index](README.md) · Prev: [19 LLM App Engineering](19-llm-application-engineering.md) · Next: [21 MLOps](21-mlops-and-production-ml.md)

Models that perceive and/or generate **across modalities** — text, image, audio, video. The dominant direction of frontier research.

## 1. The unifying idea: shared embedding spaces

Project every modality into one vector space where **aligned items sit together** ([11](11-transformers-and-foundation-models.md)):

- **CLIP (OpenAI 2021):** contrastive training on 400M image–text pairs → image encoder + text encoder, cosine similarity aligns them. Enables **zero-shot classification** (compare image to class-name prompts), semantic image search, and **guidance** for generators (CFG, [12](12-generative-models.md)).
- **ALIGN/Florence, SigLIP:** same recipe, bigger/noisier data or loss tweaks.
- **Embedding-space trick:** once aligned, *any* task becomes similarity search or generation in that space.

## 2. Vision–language models (image understanding)

- **Architecture:** vision encoder (ViT [09](09-convolutional-networks-and-vision.md) or CLIP ViT) → **projector** (linear/MLP/Q-Former) → **LLM** that reads visual tokens like text tokens.
- **GPT-4V/4o, Gemini, Claude vision, LLaVA, Qwen-VL, InternVL:** read screenshots, charts, documents, photos; answer, extract, reason.
- **OCR & document AI** (layout-aware parsing powers RAG over PDFs — [17](17-retrieval-augmented-generation.md)).
- **Visual grounding:** pointing/bounding boxes referring to image regions (helps UI agents — [18](18-ai-agents-and-tool-use.md)).
- Typical training: alignment pretraining (freeze encoders, train projector) → instruction fine-tuning on image–task data.

## 3. Image generation

- **Autoregressive** (DALL·E 1 style, MAR), **GAN** lineage (fast, older), **diffusion** (dominant — mechanics in [12](12-generative-models.md)).
- **Text→image systems:** prompt encoder (T5/CLIP text) + latent diffusion U-Net or **DiT (diffusion transformer)** + VAE codec (SD 3/Flux-class).
- **Control:** ControlNet (depth/edges/pose conditioning), inpainting/outpainting, img2img, LoRA style/subject fine-tunes ([15](15-fine-tuning-and-peft.md)), aesthetic preference tuning.
- **Evaluation:** FID, CLIP score, human preference ([12](12-generative-models.md)); production concerns: NSFW filtering, provenance/C2PA metadata, copyright/licensing ([22](22-ai-safety-security-ethics.md)).

## 4. Speech & audio

- **ASR (speech→text):** CTC losses, RNN-Transducer; end-to-end transformers — **Whisper** (weakly supervised, multilingual, robust); streaming variants; diarization separates speakers.
- **TTS (text→speech):** pipeline (text frontend → acoustic model → vocoder like HiFi-GAN) vs **end-to-end** (VALL-E-style: *voice cloning from seconds of reference audio* — deepfake risk, [22](22-ai-safety-security-ethics.md)); neural codecs (EnCodec/SoundStorm) tokenize audio like text.
- **Audio understanding:** spectrograms → transformers; music generation (MusicGen, Suno-class), sound effects, audio events detection.
- **Latency reality:** real-time voice agents need VAD, streaming ASR, fast LLM (or small model), streaming TTS — target sub-second round trips ([19](19-llm-application-engineering.md)).

## 5. Video

- **Understanding:** sample frames + temporal modeling (TimeSformer/VideoMAE), long-video LLMs with frame sequences; action recognition benchmarks (Kinetics).
- **Generation:** latent video diffusion — image diffusion + temporal layers + 3D/时空 VAE; heavy compute; consistency across frames is the core challenge. Text→video (Sora-class, Runway, Veo) and video editing/extend.

## 6. Omni models

- **Any-to-any:** models handling text+image+audio **in and out** (GPT-4o-class, Gemini): interleaved multimodal conversation, native speech latency, interruptible voice.
- Trend: modality tokenizers feeding *one* autoregressive transformer over a joint stream — the convergence point of [12](12-generative-models.md)/[14](14-large-language-models.md)/this module.

## 7. Building multimodal apps (practical)

- Send base64/images with size limits; downsample first — image tokens eat context and money ([19](19-llm-application-engineering.md)).
- **Grounding-sensitive evals:** OCR-heavy tasks, chart QA, UI screenshots — verify with golden sets; VLMs hallucinate layout details.
- Guardrails for user-uploaded images (NSFW, prompt injection hidden in screenshots — [22](22-ai-safety-security-ethics.md)).
- Search stacks: CLIP embeddings unify image+text queries in one vector index ([17](17-retrieval-augmented-generation.md)).

## Mastery Checklist

- [ ] Explains CLIP's contrastive objective and three ways it's reused
- [ ] Sketches a VLM: vision encoder → projector → LLM, with training stages
- [ ] Compares diffusion vs GAN vs AR for image generation on speed/quality/control
- [ ] Names the components of a real-time voice pipeline and its latency budget
- [ ] Describes how video generation extends image diffusion (temporal layers/VAE)
- [ ] Lists multimodal-specific risks (deepfakes, image prompt-injection) and mitigations
