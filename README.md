# 📚 The Exhaustive AI Tutorial — Theory + Executable Monorepo

A complete curriculum covering **every major AI topic** — math foundations → classical ML → deep learning → transformers → LLMs/agents → MLOps/safety → ML system design — **paired with a runnable implementation for every module**, a JSON API, a web UI, and a test suite.

## ⚡ Quickstart

```bash
make setup                 # verify environment (all deps preinstalled)
make run M=11              # run ONE example (module 11 — transformers)
make run-all               # run ALL 28 examples (~2 min) → 28/28 PASS
make test                  # pytest suite (examples + live API + accelerators) → 12/12
make bench                 # NumPy vs PyTorch benchmark → when acceleration actually pays
make notebook              # open the module-00 MATH NOTEBOOK in JupyterLab (50 cells, all graphs)
make notebook-run          # rebuild → re-execute → verify the notebook headlessly
make serve                 # JSON API  → http://127.0.0.1:8000
make ui                    # web UI    → http://127.0.0.1:8080
export OPENAI_API_KEY=…    # optional: same code switches from MockLLM to a real model
```

## 📁 Repository layout

| Path | What lives there |
|---|---|
| [`docs/curriculum/`](docs/curriculum/) | **28 theory modules** (`00`–`27`), each with a Mastery Checklist |
| [`docs/figures/`](docs/figures/) | Generated didactic figures — 31 diagrams for module 00's math topics (`python3 tools/make_math_figures.py`); every worked number verified by `tools/calc_00_math.py` |
| [`notebooks/`](notebooks/) | **Jupyter notebook for module 00** — every formula/graph/table of the math syllabus, executable: [`00-mathematical-foundations.ipynb`](notebooks/00-mathematical-foundations.ipynb) (source: `00_math_foundations.py` → rebuild with `tools/build_math_notebook.py`) |
| [`docs/TOPICS-SUMMARY.md`](docs/TOPICS-SUMMARY.md) | One-page map of every topic covered |
| [`examples/`](examples/) | **28 runnable `main.py`** — one per theory module · **[catalog & sample outputs](docs/EXAMPLES.md)** |
| [`benchmarks/`](benchmarks/) | NumPy vs PyTorch benchmark (`make bench`) — decides when accelerating pays off |
| [`packages/ai_core/`](packages/ai_core/) | Shared lib: datasets, from-scratch metrics, MockLLM/real-LLM factory, vector store (cosine+BM25+RRF), model registry + PSI drift, **`torch_backend`** — PyTorch-first training for 19 examples with automatic NumPy fallback |
| [`apps/api_server/`](apps/api_server/) | stdlib JSON API: `/health` `/predict` `/rag` `/agents/run` `/generate` |
| [`apps/web_ui/`](apps/web_ui/) | Full-stack browser client (vanilla JS, CORS-enabled) |
| [`tests/`](tests/) | pytest: smoke-runs every example + boots & tests the API |
| [`scripts/run_all.sh`](scripts/run_all.sh) | Sequential runner with pass/fail tally |
| [`artifacts/`](artifacts/) | Generated outputs (registry models, prediction logs, design docs) — git-ignored |

## 🖥️ Capacity-aware design decisions

This machine (**Apple M1 · 8 cores · Python 3.14 · PyTorch 2.14 w/ MPS · no API keys**) drove the architecture:

- **PyTorch-first training with a genuine NumPy fallback** — MLP/CNN/RNN/transformer/VAE/GAN/diffusion/DQN/LoRA/CLIP/InfoNCE train via autograd in [`torch_backend`](packages/ai_core/torch_backend.py); when torch is absent every function runs the *same algorithm* by hand (verified parity — e.g. recommender RMSE torch 0.608 vs NumPy 0.596), so `make run-all` needs zero installs. `make setup` reports which backend is active
- **MockLLM fallback** — every LLM example (prompts/RAG/agents/evals) runs fully offline and transparently upgrades to OpenAI/Anthropic when a key exists
- **stdlib-only serving** — no FastAPI install; vanilla-JS UI — no npm build
- **Acceleration is measured, not assumed** — module [27](docs/curriculum/27-accelerated-computing-and-pytorch.md) derives the amortization rule (torch wins past ~1.5 s of NumPy work; its own sub-second kernels don't qualify)

## 🧠 Theory curriculum — [`docs/curriculum/`](docs/curriculum/)

| # | Module | # | Module |
|---|---|---|---|
| 00 | [Mathematical Foundations](docs/curriculum/00-mathematical-foundations.md) | 14 | [Large Language Models](docs/curriculum/14-large-language-models.md) |
| 01 | [Python & Data Tooling](docs/curriculum/01-python-and-data-tooling.md) | 15 | [Fine-Tuning & PEFT](docs/curriculum/15-fine-tuning-and-peft.md) |
| 02 | [ML Foundations](docs/curriculum/02-machine-learning-foundations.md) | 16 | [Prompt Engineering](docs/curriculum/16-prompt-engineering.md) |
| 03 | [Supervised Algorithms](docs/curriculum/03-supervised-learning-algorithms.md) | 17 | [Retrieval-Augmented Generation](docs/curriculum/17-retrieval-augmented-generation.md) |
| 04 | [Unsupervised Learning](docs/curriculum/04-unsupervised-learning.md) | 18 | [AI Agents & Tool Use](docs/curriculum/18-ai-agents-and-tool-use.md) |
| 05 | [Evaluation & Tuning](docs/curriculum/05-model-evaluation-and-tuning.md) | 19 | [LLM Application Engineering](docs/curriculum/19-llm-application-engineering.md) |
| 06 | [Feature Engineering](docs/curriculum/06-feature-engineering.md) | 20 | [Multimodal AI](docs/curriculum/20-multimodal-ai.md) |
| 07 | [DL Fundamentals](docs/curriculum/07-deep-learning-fundamentals.md) | 21 | [MLOps & Production ML](docs/curriculum/21-mlops-and-production-ml.md) |
| 08 | [Training & Regularization](docs/curriculum/08-training-and-regularizing-networks.md) | 22 | [AI Safety, Security & Ethics](docs/curriculum/22-ai-safety-security-ethics.md) |
| 09 | [CNNs & Computer Vision](docs/curriculum/09-convolutional-networks-and-vision.md) | 23 | [Advanced & Specialized Topics](docs/curriculum/23-advanced-and-specialized-topics.md) |
| 10 | [RNNs & Sequences](docs/curriculum/10-rnn-and-sequence-modeling.md) | 24 | [Study Plan & Projects](docs/curriculum/24-study-plan-and-projects.md) |
| 11 | [Transformers & Foundation Models](docs/curriculum/11-transformers-and-foundation-models.md) | 25 | [ML System Design](docs/curriculum/25-ml-system-design.md) |
| 12 | [Generative Models](docs/curriculum/12-generative-models.md) | 26 | [Interviews & From-Scratch Coding](docs/curriculum/26-ml-interview-prep-and-coding.md) |
| 13 | [Reinforcement Learning](docs/curriculum/13-reinforcement-learning.md) | 27 | [Accelerated Computing & PyTorch](docs/curriculum/27-accelerated-computing-and-pytorch.md) |

📖 Full topic index: **[docs/TOPICS-SUMMARY.md](docs/TOPICS-SUMMARY.md)** ·
💻 Every example with concepts + real output: **[docs/EXAMPLES.md](docs/EXAMPLES.md)**

## 🗺️ Suggested paths

- **Beginner → practitioner (24 wks):** curriculum 00→13, then 14–19, 21, 24 (full schedule in [24-study-plan](docs/curriculum/24-study-plan-and-projects.md))
- **Developer → AI engineer (10 wks):** 02, 05, 07, 11, 14, 16, 17, 18, 19, 21 + matching examples
- **Interview sprint:** 25 (design docs) + 26 (from-scratch drills) with examples m25/m26

## 🔗 Companion external resources

3Blue1Brown (math intuition) · Andrew Ng *ML Specialization* · *fast.ai* · Hugging Face *LLM Course* · DeepLearning.AI short courses · d2l.ai — see [study plan](docs/curriculum/24-study-plan-and-projects.md) for the full pairing guide.
