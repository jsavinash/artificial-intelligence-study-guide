# 23 — Advanced & Specialized Topics

> Back to [index](../../README.md) · Prev: [22 Safety & Ethics](22-ai-safety-security-ethics.md) · Next: [24 Study Plan & Projects](24-study-plan-and-projects.md)

The topics beyond the core track — each a full subfield; this is your map and where to go deeper.

## 1. Recommender systems

- **Collaborative filtering:** user–item matrix factorization (SVD, ALS for implicit feedback), neighborhood methods; **deep CF** (two-tower).
- **Content-based:** item features → user profile; **hybrid** = production standard.
- **Sequential/session-based:** Transformers/RNNs over click history (SASRec).
- **Learning-to-rank:** LambdaMART → neural rankers; two-stage **retrieve → rank → re-rank** (diversity, business rules) — same shape as RAG ([17](17-retrieval-augmented-generation.md)).
- **Evaluation:** NDCG/Recall@k with time-based splits; **online A/B is truth**; popularity bias & cold start (content features, exploration from [13](13-reinforcement-learning.md)).
- **LLM era:** LLM re-rankers, conversational recommenders, LLMs as explainers.

## 2. Time series & forecasting

- **Classical:** ARIMA/ETS, exponential smoothing — still competitive on many business series.
- **Features:** lags, rolling stats, calendar effects ([06](06-feature-engineering.md)); stationarity; **walk-forward CV** ([05](05-model-evaluation-and-tuning.md)).
- **Deep:** RNNs, TCN, transformers (Informer, Autoformer), **N-BEATS, PatchTST, TimesFM-style foundation models**, and strong **linear models (DLinear)**.
- **Anomaly detection:** forecast-error thresholds, Prophet ([04](04-unsupervised-learning.md)).
- Exogenous regressors, hierarchical reconciliation (SKU→category), probabilistic forecasts (quantiles).

## 3. Graph neural networks (GNNs)

- **Data:** social, citation, molecules, transactions, knowledge graphs.
- **Mechanism:** **message passing** — aggregate neighbor embeddings, update node state, repeat; readout for graph-level tasks (molecular property prediction).
- **Architectures:** GCN, GraphSAGE (inductive), GAT (attention over edges), GIN; **→** node classification, link prediction, **recommendation**.
- **Beyond:** graph transformers; knowledge-graph embeddings (TransE) powering fact-aware RAG ([17](17-retrieval-augmented-generation.md)).

## 4. Self-supervised & contrastive learning

- **Contrastive (InfoNCE):** pull pairs together, push negatives apart — **SimCLR, MoCo** (images), **CLIP** (cross-modal, [20](20-multimodal-ai.md)), SimCSE (text).
- **Non-contrastive:** BYOL, **DINO/iBOT** (teacher-student, no negatives) → **DINOv2** backbone-grade features.
- **Masked modeling:** MAE reconstructs patches — BERT's logic for pixels ([11](11-transformers-and-foundation-models.md)).
- **Why:** unlabeled data → representations → small-label fine-tune ([15](15-fine-tuning-and-peft.md)).

## 5. Transfer, multi-task & continual learning

- **Domain adaptation:** source→target shift; domain randomization (Sim2Real robotics).
- **Multi-task:** shared trunk + heads; instruction-mix training is the LLM version ([14](14-large-language-models.md)).
- **Continual learning:** catastrophic forgetting; EWC, replay, per-task adapters ([15](15-fine-tuning-and-peft.md)).
- **Zero/few-shot & open-vocabulary:** prompt-based ([16](16-prompt-engineering.md)); CLIP-powered detection/segmentation.

## 6. AutoML & system-building

- **Auto-sklearn, AutoGluon, FLAML, H2O:** stacked ensembles + Bayesian search, zero code — **baselines to beat** before hand-tuning ([05](05-model-evaluation-and-tuning.md)).
- **NAS:** weight-sharing methods — largely historical vs hand-designed + scaling laws ([14](14-large-language-models.md)).
- **Meta-learning:** MAML (learn to init), prototypical networks for few-shot.

## 7. Privacy-preserving & causal ML

- **Federated learning:** train across devices without raw data leaving (FedAvg + secure aggregation + DP); communication cost is the battle.
- **Secure computation:** homomorphic encryption (slow), enclaves, MPC; private LLM inference is active research.
- **Causal inference:** uplift modeling, double ML, do-calculus — *what happens if we act*, not *what correlates* ([00](00-mathematical-foundations.md)).

## 8. Frontiers worth knowing by name

- **World models & embodiment:** learn dynamics to plan (MuZero, V-JEPA-style video prediction, robot foundation models) — RL meets multimodal ([13](13-reinforcement-learning.md)).
- **Mechanistic interpretability:** circuits, sparse autoencoders ([22](22-ai-safety-security-ethics.md)).
- **Scientific AI:** AlphaFold-class prediction, materials/drug discovery, LLM theorem proving.
- **Efficiency:** MoE sparsity, quantization/distillation ([15](15-fine-tuning-and-peft.md)), low-bit training research.
- **AGI discourse:** scaling optimists vs reasoning/data walls — treat claims as hypotheses; evaluate with tasks ([19](19-llm-application-engineering.md)).

## 9. Uncertainty quantification & beyond-attention architectures

- **Why:** confident wrong answers are costly (medical, finance). **Deep ensembles** (train N, average — strongest practical baseline) · **MC dropout** (dropout at inference → predictive variance) · Bayesian layers (variational posteriors) · **conformal prediction** (distribution-free coverage guarantees) · calibration ([05](05-model-evaluation-and-tuning.md)).
- **Beyond softmax attention:** state-space models (**Mamba/RWKV** — linear-time sequence modeling for very long streams; your `machine-learning/annotated_deep_learning_paper_implementations` has RWKV annotated) · ConvMixer (minimal patch mixer) · **capsule networks** (pose-aware part routing — research) · hypernetworks (nets that generate weights) · adaptive computation (think longer dynamically — ancestor of reasoning models, [14](14-large-language-models.md)).
- Treat as **awareness topics:** know each name, its one-line intuition, and why it exists; go deep only when benchmarks or papers demand it.

## Mastery Checklist

- [ ] Designs a two-stage retrieve→rank→re-rank recommender with offline+online eval
- [ ] Chooses walk-forward validation and lag features for a forecasting problem
- [ ] Explains message passing in a GNN with an application example
- [ ] Contrasts contrastive vs masked self-supervised objectives (SimCLR/CLIP/MAE)
- [ ] Names where AutoML fits in the workflow and its honest value
- [ ] Maps each frontier topic to its prerequisite modules in this tutorial
