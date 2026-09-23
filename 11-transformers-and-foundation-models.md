# 11 — Transformers & Foundation Models

> Back to [index](README.md) · Prev: [10 RNNs](10-rnn-and-sequence-modeling.md) · Next: [12 Generative Models](12-generative-models.md)

The transformer ("Attention Is All You Need," Vaswani et al. 2017) is the architecture behind **every LLM, CLIP, Whisper, AlphaFold-adjacent model, and modern recommendation system.** Master this module.

## 1. Tokenization first

Text must become integers before learning:

- **Word-level:** clean but vocab explosion + OOV problem.
- **Subword (the standard):** **Byte-Pair Encoding (BPE)** — iteratively merge most frequent symbol pairs (GPT, LLaMA); **WordPiece** (BERT, likelihood-based merges); **Unigram/SentencePiece** (probabilistic, language-agnostic); **byte-level** fallback (GPT-2/4 never sees OOV).
- Why it matters: vocab size (32K–256K) trades off sequence length vs embedding-table size; tokenizer behavior silently affects every downstream task (numbers, code, multilingual).

## 2. Self-attention, computed

For queries Q, keys K, values V (projections of the same sequence):

```
Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
```

- `Q·Kᵀ` = all-pairs compatibility scores; **√d_k scaling** keeps logits from saturating softmax as dims grow; softmax → attention *weights*; weighted sum of V.
- **Multi-Head Attention (MHead):** run h smaller attentions in parallel (d_model split into h heads), concatenate + project — different heads learn different relations (syntactic, positional, coreference).
- **Complexity:** O(L²·d) — quadratic in sequence length (the cost motivating FlashAttention, sliding windows, and linear-attention research).
- **Masking:** causal mask (row i attends only to ≤ i) for decoders; padding mask for batches; encoder attends everywhere.

**Intuition:** attention = soft, differentiable, content-based database lookup — every token gathers what it needs from every other token, in one parallel step.

## 3. The transformer block

```
x → [Multi-Head Self-Attention] → (+ residual, LayerNorm)   # "mix information between positions"
  → [Position-wise FFN: d→4d→d, GELU] → (+ residual, LayerNorm)   # "process each position"
  → × N layers
```

- **Residual connections** keep gradients alive (ResNet trick) — enables 10s–100s of layers.
- **LayerNorm** per-sample (batch-independent); modern LLMs use **Pre-LN** (norm before sublayer) + **RMSNorm**.
- **FFN** = the per-token "knowledge storage"; MoE models sparsify it ([14](14-large-language-models.md)).
- **Positional information:** attention is permutation-invariant → inject position: *sinusoidal* (original), **learned absolute**, **RoPE (rotary)** — rotates Q/K by position, the modern LLM default (relative, extrapolates better), **ALiBi** (distance-biased slopes).

### Three architectural flavors

| Type | Masking | Use | Examples |
|---|---|---|---|
| **Encoder-only** | bidirectional | classification, embeddings | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | causal | **generation — what LLMs use** | GPT family, LLaMA, Mistral |
| **Encoder-decoder** | both | translation, summarization | T5, BART, mBART |

## 4. Pretraining objectives (self-supervised)

- **MLM (masked LM):** predict randomly masked tokens → bidirectional understanding (BERT). Used for ~15% of tokens with 80/10/10 mask policy.
- **Next-token prediction (causal LM):** predict token t+1 from 1..t (**GPT** — the recipe that scaled to LLMs).
- **T5 span-corruption:** masked spans → decoder fills (text-to-text framing).
- **Denoising** (BART), **permutation LM** (XLNet), **replaced-token detection** (ELECTRA — efficient).
- **CLIP-style contrastive:** align image–text pairs in shared space ([20](20-multimodal-ai.md)).

## 5. The foundation-model era

**Idea:** one big self-supervised-pretrained model → **fine-tune or prompt** for many downstream tasks. Transfer at scale ([15](15-fine-tuning-and-peft.md)).

- **Fine-tuning era (2018–2021):** BERT for everything (GLUE/SQuAD records).
- **Prompting era (2020+):** GPT-3 "language models are few-shot learners" — task as text, no weight updates.
- **Scaling era (2022+):** capability emerges from scale ([14](14-large-language-models.md)).
- **Embedding models:** mean/CLS-pooled transformer outputs as dense vectors for **semantic search & RAG** ([17](17-retrieval-augmented-generation.md)) — Sentence-BERT, E5, bge, OpenAI text-embedding-3.

## 6. Efficiency techniques (know the acronyms)

- **KV cache:** during decoding, cache past K/V — avoids recomputing attention every token; grows linearly with context × layers (the memory bottleneck of LLM serving; [21](21-mlops-and-production-ml.md)).
- **FlashAttention:** IO-aware exact attention — big speed/memory wins, enables long context.
- **GQA/MQA:** share key/value heads across query heads → smaller cache (LLaMA-2/3 use GQA).
- Sliding-window/local attention (Mistral), **sparse/MoE** FFNs, context extension (RoPE scaling, YaRN), and prompt/context compression.

## Mastery Checklist

- [ ] Writes the attention formula with the √d_k scaling and explains each term
- [ ] Sketches a full transformer block with residuals and norms in order
- [ ] Compares encoder-only / decoder-only / encoder-decoder with use cases
- [ ] Explains BPE tokenization mechanics and vocab-size tradeoffs
- [ ] Contrasts MLM vs next-token pretraining and why GPT chose causal
- [ ] Explains what the KV cache stores and why serving is memory-bound by it
- [ ] Describes RoPE and why positional encodings are needed at all
