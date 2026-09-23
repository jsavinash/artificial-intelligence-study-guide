# 14 — Large Language Models

> Back to [index](README.md) · Prev: [13 Reinforcement Learning](13-reinforcement-learning.md) · Next: [15 Fine-Tuning & PEFT](15-fine-tuning-and-peft.md)

Everything in [11](11-transformers-and-foundation-models.md), scaled up and post-trained into a usable assistant. This module is the center of gravity of modern AI.

## 1. Anatomy of an LLM

- **Decoder-only transformer** (GPT/LLaMA/Mistral/Qwen/DeepSeek families), trained for **next-token prediction**.
- **Scale vocabulary:** parameters ≈ 12 · n_layer · d_model² (plus embeddings). Family examples: 7B → 70B → 405B parameters.
- **Key architecture choices:** rotary positional embeddings (**RoPE**), **RMSNorm** + **Pre-LN**, SwiGLU activations in the FFN, **GQA** (grouped-query attention) to shrink the KV cache, and **MoE** (Mixture-of-Experts: route tokens to a subset of FFN experts → more capacity, similar FLOPs per token).
- **Context window:** 4K → 128K–1M+; long context costs attention compute + KV-cache memory ([11](11-transformers-and-foundation-models.md)).

## 2. Data & pretraining pipeline

1. **Corpus curation:** web crawls (CommonCrawl → filtered), books, code (GitHub), papers, multilingual; quality classifiers, dedup (MinHash), PII removal, toxicity filtering, license checks. **Data quality > quantity** at the frontier.
2. **Tokenizer:** BPE/SentencePiece (32K–256K vocab).
3. **Pretraining objective:** cross-entropy on next tokens, trillions of tokens.
4. **Scaling laws (Kaplan/Chinchilla):** loss ≈ power law in **N** (params), **D** (tokens), **C** (compute). *Chinchilla (2022):* for compute-optimality, ≈**20 tokens per parameter** — many "undertrained" models were later retrained (Llama-3 style: 15T+ tokens for 8B). Inference-optimal tradeoffs now favor *smaller, well-trained* models.
5. **Compute:** thousands of GPUs, fp8/bf16, sharded data/tensor/pipeline parallelism, ZeRO/FSDP, gradient checkpointing ([08](08-training-and-regularizing-networks.md), [21](21-mlops-and-production-ml.md)).

## 3. Post-training (what makes a model *usable*)

Stage pipeline: **pretrain → SFT → preference alignment (RLHF/DPO) → safety/robustness tuning.**

### Supervised fine-tuning (SFT)
- Train on high-quality **instruction–response pairs** (formatted conversations with system/roles).
- Chat templates matter — mismatches degrade behavior.
- Quality/diversity of demonstrations dominates quantity (distillation from a stronger teacher is common).

### Preference alignment
- **RLHF:** humans compare responses → train **reward model** (Bradley–Terry: `P(A≻B) = σ(r(A)−r(B))`) → **PPO** optimizes the policy for reward **subject to KL constraint** to the SFT policy; reward-model overfitting & KL tuning are the craft.
- **DPO:** direct preference optimization — a classification-style loss on (chosen, rejected) pairs, no separate RM, no RL loop. **RLAIF:** AI (constitutional/self-critique) labels instead of humans; **ORPO, KTO, SimPO, GRPO** (group-relative, used by DeepSeek): family of simpler alternatives.

## 4. Inference & decoding (talking to a model)

- **Logits → distribution:** temperature τ scaling; **top-k**; **top-p/nucleus**; repetition/frequency penalties; **min-p** (proportional to top token — used by newer stacks).
- **Stop sequences, max tokens, structured output constraints** ([16](16-prompt-engineering.md), [19](19-llm-application-engineering.md)).
- **Prompt caching:** prefix reuse for system prompts/documents (big cost/latency win — [19](19-llm-application-engineering.md)).
- **Reasoning models (2024+):** trained with RL (RLVR — verifiable rewards) to emit long **chain-of-thought before answering**; budget/adaptive thinking; dramatically better at math/code/agentic tasks; trade latency/tokens for accuracy. Distilled into smaller "think" variants.

## 5. Capabilities & failure modes

**Emergent-ish abilities:** in-context learning (few-shot examples shape behavior without weight updates), instruction following, reasoning, code, tool use ([18](18-ai-agents-and-tool-use.md)).

**Inherent limitations:**
- **Hallucination** — next-token objective rewards plausibility, not truth. Mitigations: RAG ([17](17-retrieval-augmented-generation.md)), citations, verification loops, abstention training.
- **No native calculation/memory:** arithmetic errors, cutoff knowledge, stateless across calls (you manage history).
- **Context limits** — everything outside the window is invisible.
- **Sycophancy & bias** — trained to please; reflect training data's biases ([22](22-ai-safety-security-ethics.md)).

## 6. The model landscape (conceptual map)

- **Proprietary frontier:** GPT / Claude / Gemini families — top capability, API only, strong tooling.
- **Open-weight:** LLaMA, Mistral, Qwen, DeepSeek, Gemma, OLMo — runnable locally, fine-tunable ([15](15-fine-tuning-and-peft.md)); open weights ≠ open data/training details.
- **Small models (1–8B):** laptops/phones via quantization ([15](15-fine-tuning-and-peft.md)); surprisingly capable after distillation/RL.
- **Multimodal:** vision + audio inputs, sometimes outputs ([20](20-multimodal-ai.md)).

## Mastery Checklist

- [ ] Diagrams the LLM lifecycle: data → pretrain → SFT → preference → deploy
- [ ] Explains scaling laws and the Chinchilla 20-tokens-per-param rule
- [ ] Describes RLHF's reward model + PPO + KL, and how DPO replaces it
- [ ] Chooses decoding settings (τ/top-p) for creative vs factual tasks
- [ ] Explains *why* hallucination is structural, not a bug — and 3 mitigations
- [ ] Compares open-weight vs proprietary tradeoffs for a deployment decision
- [ ] Explains what a reasoning model does differently and its cost
