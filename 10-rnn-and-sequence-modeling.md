# 10 — RNNs & Sequence Modeling

> Back to [index](README.md) · Prev: [09 CNNs & Vision](09-convolutional-networks-and-vision.md) · Next: [11 Transformers](11-transformers-and-foundation-models.md)

Sequences — text, speech, time series, genomes. RNNs were *the* answer before transformers, and their ideas (state, gating, attention) live on inside every LLM.

## 1. Word representations (the precursor battle)

- **One-hot:** V-dim sparse — no similarity notion (all pairs orthogonal).
- **TF-IDF/BOW:** counts with rarity weighting — strong classical baselines ([06](06-feature-engineering.md)).
- **Word2Vec (Mikolov 2013):** predict context (Skip-gram) or context→word (CBOW); dense vectors where `king − man + woman ≈ queen`. Shallow, trained on huge corpora.
- **GloVe:** factorizes global co-occurrence matrix — combines count statistics with prediction.
- **FastText:** subword (character n-gram) embeddings → handles OOV/morphology.
- **Key insight:** distributional semantics — *you shall know a word by the company it keeps*. Modern contextual embeddings ([11](11-transformers-and-foundation-models.md)) fixed the fatal flaw: "bank" river vs bank money got *one* vector forever.

**Classical NLP toolkit (pre-embeddings) — know the pipeline:**
- **Preprocessing:** sentence splitting, tokenization, lowercasing, stemming/lemmatization, stopword handling.
- **Structural tagging:** part-of-speech tagging, chunking, **named-entity recognition** (rules → CRF → BiLSTM-CRF), constituency/dependency parsing — still powers information extraction.
- **Classic tasks:** sentiment (lexicon or BOW + Naive Bayes), topic classification, **statistical machine translation** (phrase tables — the motivation for attention), information retrieval (**BM25** — alive today inside hybrid search, [17](17-retrieval-augmented-generation.md)).
- **Classic evaluation:** BLEU/ROUGE for generation ([05](05-model-evaluation-and-tuning.md)), perplexity for language models.
- Transformers ([11](11-transformers-and-foundation-models.md)) collapsed *pipelines of separate models* into one contextual model — but the task taxonomy and evaluation habits remain.

## 2. Vanilla RNN

```
h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)    # hidden state = memory
y_t = W_hy·h_t
```

- Processes sequences step-by-step, shares parameters across time (like conv shares across space).
- **Problems:** (1) **vanishing/exploding gradients** through time (BPTT) → can't learn long-range dependencies; (2) sequential computation — no parallelism.

## 3. Gated architectures

**LSTM (Hochreiter & Schmidhuber 1997)** — cell state `c_t` as an information highway with three gates:
- **Forget gate** `f = σ(W_f·[h,x])` — what to erase from memory
- **Input/update gate** — what new info to store
- **Output gate** — what to expose as `h_t`
Sigmoid gates solve vanishing gradients by letting gradients flow multiplicatively along `c` (≈identity path).

**GRU:** simplified LSTM — reset & update gates, no separate cell; fewer params, often equal performance.

**Bidirectional RNNs:** run forward + backward, concatenate — sees both directions (offline tasks only; unusable for true streaming prediction).

**Stacking/depth:** multi-layer RNNs learn hierarchical features — but transformers ate this market for language.

## 4. Seq2Seq & attention (the bridge to transformers)

**Encoder–decoder (Sutskever 2014):** encoder compresses input to a *fixed vector*; decoder generates output — the architecture behind NMT.

**The bottleneck:** one vector can't hold a whole sentence → **Bahdanau attention (2014)**: at each decode step, the decoder attends over *all* encoder hidden states:

```
score(h_decoder, h_i) → softmax → weighted sum of encoder states
```

**This is attention** — and it's exactly what became the transformer's core mechanism ([11](11-transformers-and-foundation-models.md)).

**Beam search:** keep top-B partial hypotheses, score = Σ log P(token) (length-normalized) — better than greedy for translation; sampling variants (temperature, top-k, top-p) covered in [14](14-large-language-models.md).

**Teacher forcing:** feed the true previous token during training (fast, stable) vs scheduled sampling (exposure-bias fix).

## 5. Where RNNs still matter

- Time-series forecasting with strict causality (though TCN/transformer/linear models like N-BEATS, PatchTST compete).
- On-device/streaming inference where KV-cache transformers are heavier.
- Understanding history: every LLM still *looks* like an RNN conceptually — decoding maintains a growing state (the KV cache).

## 6. Transition to transformers (preview)

| RNN pain | Transformer fix |
|---|---|
| sequential (slow) | **parallel over positions** |
| long-range vanishing grads | **attention = direct all-to-all paths** |
| fixed bottleneck vector | **attend to every position** |
| order = computation order | **positional encodings restore order** |

## Mastery Checklist

- [ ] Explains distributional semantics and why word2vec worked
- [ ] States the two failure modes of vanilla RNNs and how gates fix each
- [ ] Describes LSTM's three gates and the cell-state highway
- [ ] Explains encoder–seq2seq bottleneck and how attention solved it
- [ ] Implements/train an LSTM sequence classifier from memory
- [ ] Maps each RNN limitation to its transformer solution
