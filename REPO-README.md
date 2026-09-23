# 🧰 AI Tutorial — Executable Monorepo

Working, real-life implementations for every topic in the companion theory docs
([README.md](../README.md) · [TOPICS-SUMMARY.md](../TOPICS-SUMMARY.md)).

## Capacity-aware design decisions

- **Pure NumPy/sklearn** for all ML/DL (this machine: Apple M1, 8 cores, Python 3.14, **no PyTorch**)
- **MockLLM fallback** — no API keys in env today; every LLM example runs offline and
  transparently upgrades to OpenAI/Anthropic when `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` are set
- **stdlib HTTP server** for the API (no FastAPI install needed) + vanilla-JS web UI (Node 24 available but unused)

## Quickstart

```bash
make setup                 # verify environment
make run M=11              # run ONE example (module 11 — transformers)
make run-all               # run ALL 27 examples (~2 min)
make test                  # pytest smoke suite
make serve                 # JSON API on http://127.0.0.1:8000
make ui                    # web UI on http://127.0.0.1:8080 (use API too)
```

## Layout

| Path | Contents |
|---|---|
| `packages/ai_core/` | Shared library: datasets, metrics, **MockLLM/real LLM client**, vector store (cosine + BM25), model registry |
| `examples/m00_math/ … m26_interviews/` | One runnable `main.py` per theory module — each prints `PASS mXX …` |
| `apps/api_server/` | Production-shape JSON API: `/health` `/predict` `/rag` `/agents/run` |
| `apps/web_ui/` | Full-stack browser client (predict + RAG chat) |
| `tests/` | pytest smoke suite — executes every example, asserts PASS |
| `scripts/run_all.sh` | Sequential runner with pass/fail tally |
| `artifacts/` | Model registry + run outputs written here |

## Example index ↔ theory module

| Ex | Theory topic it proves |
|---|---|
| m00 | Gradient descent on f(x)=x², Bayes rule, cross-entropy from scratch |
| m01 | NumPy/pandas/sklearn pipeline workflow |
| m02 | End-to-end workflow + **data-leakage demo** |
| m03 | Linear/logistic regression, KNN, trees, boosting from scratch + sklearn |
| m04 | K-Means, PCA, DBSCAN |
| m05 | Metrics, cross-validation, imbalance handling, random-search tuning |
| m06 | Feature pipelines, safe target encoding |
| m07 | **MLP + backprop from scratch (NumPy)** |
| m08 | Optimizers (SGD/Adam), dropout, regularization comparison |
| m09 | Conv layer from scratch + transfer-ready CNN pieces |
| m10 | RNN forward pass, word2vec-style skip-gram toy |
| m11 | **Attention + mini-GPT training on text (NumPy)** |
| m12 | VAE loss terms / diffusion forward–reverse toy |
| m13 | Q-learning gridworld → policy learned |
| m14 | BPE tokenizer from scratch + scaling-law calculator |
| m15 | **LoRA ΔW = B·A** rank math + distillation demo |
| m16 | Prompt library + golden-set eval harness (works offline) |
| m17 | **Full local RAG**: chunk → embed → hybrid search → citations |
| m18 | **Agent loop**: tools, ReAct trace, budgets (offline) |
| m19 | Eval harness, guardrails, cost/latency tracker |
| m20 | CLIP-style contrastive similarity (text↔"image" vectors) |
| m21 | Registry, drift monitor, **batch vs online serving** |
| m22 | Prompt-injection detector, fairness metrics, PII redaction |
| m23 | Recommender (matrix factorization), time-series CV, GNN message passing |
| m24 | Runs/validates the project-ladder capstone checklist |
| m25 | Executable ML design-doc generator (11 sections) |
| m26 | From-scratch implementations quiz (self-scored) |
