# 15 — Fine-Tuning & Parameter-Efficient Methods

> Back to [index](../../README.md) · Prev: [14 Large Language Models](14-large-language-models.md) · Next: [16 Prompt Engineering](16-prompt-engineering.md)

**When prompting + RAG aren't enough** — to change *style, format, domain knowledge, or behavior* baked into the weights — you fine-tune.

## 1. Decide: prompt vs RAG vs fine-tune

| Need | Solution |
|---|---|
| New facts/knowledge, cite-able | **RAG** ([17](17-retrieval-augmented-generation.md)) |
| Formatting, tone, consistent structure, domain jargon, classifying at scale | **Fine-tuning** |
| Few examples, quick iteration | **Few-shot prompting** ([16](16-prompt-engineering.md)) |
| Private data + no extra retrieval infra | Fine-tune (weights absorb it — irreversible; privacy tradeoff) |

**Sequence:** prompt-engineer first → add RAG for knowledge → fine-tune for behavior/format; combine freely. Fine-tuning is a **fixed cost** (dev + training + eval), prompting/RAG is an **operational cost**.

## 2. Transfer-learning foundations

Fine-tuning = supervised training of a pretrained model on task data with small LRs (1e-5–5e-5 for full FT). Why it works: general features from pretraining specialize cheaply ([09](09-convolutional-networks-and-vision.md) showed this for vision first).

**Classic strategies:** feature extraction (freeze backbone) vs full fine-tune; gradual unfreezing; discriminative LRs (lower for early layers).

## 3. Full fine-tuning (the baseline)

- Update **all** parameters with (instruction, response) data: SFT ([14](14-large-language-models.md)).
- **Cost reality:** 70B full FT needs multi-GPU memory ≈ params × (4 bytes + optimizer states ~8–12 bytes) — 8-bit optimizers (AdamW8bit) and FSDP/ZeRO help but it's expensive.
- **Risks:** **catastrophic forgetting** (loses general abilities — mix in replay/general data, keep LR small), overfitting small datasets (early stopping, few epochs: 1–3), alignment tax.

## 4. Parameter-efficient fine-tuning (PEFT)

Train **<1–10% of parameters**, keep the base frozen.

### LoRA (Low-Rank Adaptation) — the default
- Frozen weight W; learn small **ΔW = B·A** where A: d→r, B: r→d, **rank r = 4–64** (≪ d).
- Inject into attention projections (Q,K,V,O) — often also FFN.
- Orders of magnitude fewer trainable params; adapters can be **swapped/merged** (`W' = W + BA` — zero inference overhead after merge); supports multi-task via multiple adapters.
- **QLoRA:** base model quantized to **4-bit (NF4)** while training LoRA in bf16 + paged optimizers — **fine-tune a 65B on one 48GB GPU**; the popular open recipe (HF PEFT library).

### Other PEFT families
- **Adapters:** small bottleneck modules inserted between layers (Houlsby); sequential, slight latency.
- **Prefix/P-tuning / prompt tuning:** learn continuous prompt embeddings (scale to many tasks; weak for small models).
- **BitFit:** bias terms only.
- **DoRA:** weight-decomposed LoRA (magnitude/direction split) — stronger at small ranks.

**Practical defaults:** LoRA r=16, α=32 (scaling α/r), dropout 0.05, target q,k,v,o; lr 1e-4–2e-4; 1–3 epochs; packing short examples; verify chat template correctness — the #1 silent bug.

## 5. Data preparation (the real work)

- **Format:** dataset → chat template tokens; include system prompt if used at inference.
- Quality bar: curated demonstrations > scraped volume; **dedupe, decontaminate against evals**, filter wrong/unsafe answers.
- Sources: manual (hundreds of strong examples), distillation (strong teacher → filter → student), synthetic (generate + human-verify), domain logs with consent/privacy review ([22](22-ai-safety-security-ethics.md)).
- **Mix in general SFT data** to prevent catastrophic forgetting.

## 6. Evaluation before & after

- **Hold-out eval** never seen in training; **benchmark suite** for regressions (general knowledge, instruction following, safety — HF Open LLM Leaderboard style).
- **LLM-as-judge + human review** on style/faithfulness ([19](19-llm-application-engineering.md)); A/B against the base.
- Ship only if: target metric ↑ **and** general metrics didn't fall **and** safety evals hold.

## 7. Distillation & compression (siblings of fine-tuning)

- **Knowledge distillation:** train a small **student** on the big **teacher's soft logits** (dark knowledge in the probabilities) — how many 7–8B models match yesterday's 70B.
- **Quantization:** fp16/bf16 → int8 → **int4/NF4/GPTQ/AWQ** — smaller memory, faster inference, minor quality loss ([21](21-mlops-and-production-ml.md)); applies to base or fine-tuned weights.
- **Pruning/MoE conversion:** sparsity research; speculative decoding (small draft model verifies with big model) for latency.

## Mastery Checklist

- [ ] Chooses prompt vs RAG vs fine-tune for a given product requirement
- [ ] Explains LoRA math (low-rank ΔW), rank's role, and merge semantics
- [ ] Sets up QLoRA training (4-bit base + LoRA) with sane hyperparameters
- [ ] Detects chat-template and train/inference mismatch bugs
- [ ] Prevents catastrophic forgetting with data mixing and small LR
- [ ] Designs a before/after eval suite including regression and safety checks
- [ ] Explains distillation and int4 quantization tradeoffs
