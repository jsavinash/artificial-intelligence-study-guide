# 24 — Study Plan & Projects

> Back to [index](README.md) · Prev: [23 Advanced Topics](23-advanced-and-specialized-topics.md)

Three executable paths + a project ladder + final mastery gate. Assumes ~10–12 hrs/week; compress or stretch accordingly.

## Path A — Complete beginner → practitioner (≈ 24 weeks)

| Weeks | Modules | Milestone |
|---|---|---|
| 1–3 | [00](00-mathematical-foundations.md) + [01](01-python-and-data-tooling.md) | NumPy fluency; Bayes & backprop by hand |
| 4–6 | [02](02-machine-learning-foundations.md) + [03](03-supervised-learning-algorithms.md) | Kaggle Titanic + House Prices with boosted trees |
| 7–8 | [04](04-unsupervised-learning.md) + [05](05-model-evaluation-and-tuning.md) + [06](06-feature-engineering.md) | Full CV + tuning report with error analysis |
| 9–12 | [07](07-deep-learning-fundamentals.md) + [08](08-training-and-regularizing-networks.md) | MNIST/CIFAR net from scratch in PyTorch ≥90% |
| 13–14 | [09](09-convolutional-networks-and-vision.md) + [10](10-rnn-and-sequence-modeling.md) | Transfer-learned classifier; sentiment LSTM |
| 15–16 | [11](11-transformers-and-foundation-models.md) | Attention from scratch; fine-tune a BERT |
| 17 | [12](12-generative-models.md) + [13](13-reinforcement-learning.md) | Train a small GAN; solve CartPole with DQN |
| 18–20 | [14](14-large-language-models.md) + [15](15-fine-tuning-and-peft.md) + [16](16-prompt-engineering.md) | QLoRA fine-tune + prompt golden set |
| 21–22 | [17](17-retrieval-augmented-generation.md) + [18](18-ai-agents-and-tool-use.md) | RAG chatbot with citations + tool agent |
| 23 | [19](19-llm-application-engineering.md) + [21](21-mlops-and-production-ml.md) | Deploy with evals, tracing, monitoring |
| 24 | [20](20-multimodal-ai.md) + [22](22-ai-safety-security-ethics.md) + [23](23-advanced-and-specialized-topics.md) | Capstone demo + model card + red-team notes |

## Path B — Developer → AI/LLM engineer (≈ 10 weeks)

- **W1:** 02 skim + 05 (eval rigor) + 07 PyTorch loop.
- **W2–3:** 11 transformers + 14 LLMs — build attention from scratch, then prompt APIs.
- **W4:** 16 prompt engineering + golden eval set in CI.
- **W5–6:** 17 RAG — full pipeline, hybrid search, reranker, RAGAS-style evals.
- **W7:** 15 fine-tuning — QLoRA on domain data; compare to RAG-only.
- **W8:** 18 agents — hand-built loop, then LangGraph; trajectory evals.
- **W9:** 19 + 21 — streaming, caching, routing, guardrails, deploy/monitor.
- **W10:** 22 safety + capstone ship.

## Path C — Research/theory depth (≈ 16 weeks)

00 (full rigor) → 03/05 proofs-adjacent → 07/08 → 11 (re-derive attention) → 12/13 → 14/15 → 22 (alignment/interpretability) → 23 (pick two frontiers) → read one paper/week in your lane (arXiv + *The Illustrated Transformer* + d2l.ai as companions).

## Project ladder (build in order — 24 ideas)

**Foundations:** 1) House-price regressor + full report. 2) Spam/credit classifier with pipelines & threshold tuning. 3) Customer-segmentation notebook (k-means/DBSCAN + business readout). 4) Time-series forecaster with walk-forward CV.

**Deep learning:** 5) MNIST/CIFAR CNN from scratch (no transfer). 6) Transfer-learned image classifier API. 7) Object detection fine-tune (YOLO). 8) Sentiment/NER with LSTM then BERT — compare.

**Transformers/generative:** 9) Attention from scratch (numpy→torch, verify vs reference). 10) Mini-GPT character-level text generator. 11) Text-to-image with SD + a ControlNet condition. 12) Voice assistant: Whisper ASR → LLM → TTS.

**LLM applied:** 13) Prompt library + CI eval harness with LLM-as-judge. 14) **Domain RAG** (your docs/PDFs) with hybrid search, reranker, citations, abstention. 15) Structured-extraction service (invoices → validated JSON). 16) QLoRA fine-tune on a niche task; beat prompting + document it. 17) Multi-tool agent (search + code exec + DB) with budgets & traces. 18) Multimodal: chart/screenshot QA or CLIP image search engine.

**Production:** 19) Deploy model (FastAPI + Docker) with shadow deploy & drift monitor. 20) LLM app with streaming, caching, model router, cost dashboard. 21) Voice workflow with latency budget. 22) Fine-tune + distill + int4-quantize a small model for local use. 23) Red-team your own app (injection, jailbreak, data leak) → fix report. 24) **Capstone:** end-to-end product — data → model → evals → deploy → monitoring → model card ([22](22-ai-safety-security-ethics.md)) → 5-min demo.

## Final mastery gate (answer without notes)

1. Derive backprop for a 2-layer net; write the attention formula and block diagram.
2. Diagnose a model from learning curves + confusion matrix; choose metrics for a fraud use case.
3. Explain pretrain→SFT→RLHF/DPO and why hallucination is structural — with three mitigations.
4. Design a RAG system end-to-end and its eval suite; state when you'd fine-tune instead.
5. Build an agent loop with tools, budgets, guardrails, and trajectory evals — and justify why not a workflow.
6. Sketch production architecture: streaming, caching, routing, monitoring, drift→retrain.
7. Red-team it: name 4 attacks and your layered defenses ([22](22-ai-safety-security-ethics.md)).

If you can do all seven **and** you've shipped projects 14, 17, and 24 — you have working command of the field this tutorial covers.

## Companion resources

- Math/viz intuition: 3Blue1Brown · Practice code: d2l.ai · Structured course: Ng ML Specialization · DL practice: fast.ai · LLM stack: Hugging Face LLM Course · Short applied courses: DeepLearning.AI · Keep current: paper summaries + provider changelogs (verify anything >6 months old).
