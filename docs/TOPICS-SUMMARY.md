# AI Tutorial — Summary of Topics

One-page map of everything covered. Each line = a topic; **bold** = the key idea. Full depth in the linked module.

## Foundations
**[00](curriculum/00-mathematical-foundations.md) Mathematical Foundations**
Linear algebra (matrices, eigenvectors, SVD) · calculus & chain rule → **backprop** · probability (Bayes, distributions, MLE) · statistics (estimators, tests, correlation≠causation) · optimization (gradient descent, convexity, Bayesian opt) · information theory (entropy, cross-entropy, KL).

**[01](curriculum/01-python-and-data-tooling.md) Python & Data Tooling**
Python essentials · NumPy vectorization · pandas (groupby, joins, missing data) · visualization · Jupyter & reproducibility · sklearn pipelines · toolchain map (PyTorch, vector DBs, MLflow).

## Classical Machine Learning
**[02](curriculum/02-machine-learning-foundations.md) ML Foundations**
What learning is · supervised/unsupervised/RL/self-supervised · the ML workflow · data splits (group/time) · **leakage** · baselines · bias–variance intuition · No Free Lunch.

**[03](curriculum/03-supervised-learning-algorithms.md) Supervised Algorithms**
Linear & logistic regression (L1/L2) · KNN · Naive Bayes · decision trees · random forest (bagging) · **gradient boosting** (XGBoost/LightGBM/CatBoost) · SVM & kernels · calibration · stacking · algorithm-selection guide.

**[04](curriculum/04-unsupervised-learning.md) Unsupervised Learning**
K-Means (elbow/silhouette) · hierarchical (dendrogram) · DBSCAN · GMM/EM · spectral · PCA · t-SNE/UMAP (viz only) · autoencoders · anomaly detection · association rules · topic modeling (LDA, BERTopic).

**[05](curriculum/05-model-evaluation-and-tuning.md) Evaluation & Tuning**
**Bias²+variance+noise** · regression & classification metrics (F1, PR-AUC, thresholds) · cross-validation (stratified/group/time, nested) · regularization · grid/random/**Bayesian** tuning (Optuna) · class imbalance (weights/SMOTE) · error analysis.

**[06](curriculum/06-feature-engineering.md) Feature Engineering**
Scaling, skew, binning, interactions · missing-value strategy · categorical encodings (**target encoding safely in-fold**) · cyclical dates & lag features · TF-IDF · feature selection (filter/wrapper/embedded).

## Deep Learning
**[07](curriculum/07-deep-learning-fundamentals.md) DL Fundamentals**
Neuron/MLP · why nonlinearity matters · losses per task · **backprop & vanishing/exploding gradients** · SGD/mini-batch · activations (ReLU/GELU/sigmoid/softmax) · PyTorch loop · embeddings.

**[08](curriculum/08-training-and-regularizing-networks.md) Training & Regularization**
Adam/**AdamW** · warmup + cosine schedules · Xavier/He init · BatchNorm/LayerNorm/RMSNorm · dropout · weight decay · early stopping · **augmentation** (Mixup, RandAugment) · label smoothing · mixed precision · gradient accumulation · debugging recipe.

**[09](curriculum/09-convolutional-networks-and-vision.md) CNNs & Computer Vision**
Convolution math (stride/padding/dilation) · locality & parameter sharing · AlexNet→VGG→**ResNet** lineage · detection (YOLO, Faster R-CNN, mAP) · segmentation (U-Net, Mask R-CNN) · **transfer learning** · ViT.

**[10](curriculum/10-rnn-and-sequence-modeling.md) RNNs & Sequences**
Word2Vec/GloVe/fastText · vanilla RNN failure modes · **LSTM/GRU gates** · bidirectionals · seq2seq bottleneck · **attention's origin** (Bahdanau) · beam search · teacher forcing · classical NLP toolkit (POS/NER/MT/BLEU).

**[11](curriculum/11-transformers-and-foundation-models.md) Transformers & Foundation Models**
BPE tokenization · **self-attention formula (QKᵀ/√d)** · multi-head · transformer block (residuals, LayerNorm, FFN) · RoPE · encoder/decoder-only/enc-dec · MLM vs next-token pretraining · foundation-model era · embeddings · **KV cache, FlashAttention, GQA**.

**[12](curriculum/12-generative-models.md) Generative Models**
Autoregressive likelihoods · sampling (temperature/top-k/top-p) · **VAE** (ELBO, reparameterization) · **GAN** (minimax, mode collapse) · **diffusion** (forward/reverse, latent diffusion, CFG, flow matching) · FID/CLIP-score eval.

**[13](curriculum/13-reinforcement-learning.md) Reinforcement Learning**
MDPs, returns, values · TD/Q-learning · DQN (+replay, target net) · policy gradients · actor-critic · **PPO** · model-based (MuZero) · exploration · multi-agent · **RLHF connection** · reward hacking.

## Modern LLM Stack
**[14](curriculum/14-large-language-models.md) Large Language Models**
LLM anatomy (RoPE, RMSNorm, MoE) · data pipeline · **scaling laws & Chinchilla** · post-training: **SFT → RLHF/DPO/GRPO** · decoding · reasoning models · hallucination causes · open-weight vs proprietary landscape.

**[15](curriculum/15-fine-tuning-and-peft.md) Fine-Tuning & PEFT**
Prompt vs RAG vs fine-tune decision · full fine-tuning & forgetting · **LoRA/QLoRA math** · adapters/prompt-tuning · data prep & chat templates · eval-before/after · **distillation** · quantization (int4/GPTQ).

**[16](curriculum/16-prompt-engineering.md) Prompt Engineering**
System/user/few-shot anatomy · 7 core patterns · **chain-of-thought**, self-consistency, decomposition · structured JSON output · system-prompt design · failure-mode table · golden-set evals.

**[17](curriculum/17-retrieval-augmented-generation.md) RAG**
Ingest→chunk→embed→index pipeline · embeddings & **hybrid search + RRF** · vector DB choice · query rewriting · **rerankers** · Graph RAG · agentic retrieval · grounded generation with citations · **faithfulness evals** (RAGAS).

**[18](curriculum/18-ai-agents-and-tool-use.md) Agents & Tool Use**
Agent loop (observe/think/act/memory) · **function calling** · ReAct vs plan-execute vs reflexion · tool-design rules · workflow-vs-agent decision · multi-agent (orchestrator) · frameworks (LangGraph, LlamaIndex) · **trajectory evals** & budgets · **MCP/A2A protocols**, AgentOps & Ng's 4 patterns.

**[19](curriculum/19-llm-application-engineering.md) LLM Application Engineering**
API patterns & streaming · structured outputs · **eval harnesses & LLM-as-judge** · guardrails (defense in depth) · cost/latency (caching, routing, TTFT) · context engineering · production architecture & feedback loop · GenAI patterns (map-reduce summarization, many-label classification, hallucination-minimization).

## Systems, Ethics, Frontiers
**[20](curriculum/20-multimodal-ai.md) Multimodal AI**
CLIP & shared embedding spaces · VLM architecture (encoder→projector→LLM) · image generation (ControlNet) · speech (Whisper, TTS, voice agents) · video diffusion · omni models.

**[21](curriculum/21-mlops-and-production-ml.md) MLOps & Production**
Lifecycle · data/versioning & **train-serve skew** · experiment tracking/registry · deployment patterns (batch/online/edge) · shadow/canary · **drift monitoring** · CI/CD/CT · LLMOps deltas.

**[22](curriculum/22-ai-safety-security-ethics.md) Safety, Security & Ethics**
Alignment & reward hacking · **fairness-metric tradeoffs** · privacy (DP, federated) · **prompt injection/jailbreaks** · poisoning & provenance (safetensors, C2PA) · XAI (SHAP) · model cards · EU AI Act/NIST.

**[23](curriculum/23-advanced-and-specialized-topics.md) Advanced Topics**
Recommenders (two-tower, LTR) · time series (walk-forward, PatchTST) · **GNNs/message passing** · contrastive & self-supervised (SimCLR/DINO/MAE) · continual learning · AutoML · causal ML · uncertainty quantification (ensembles/MC dropout/conformal) · SSM/Mamba & beyond-attention architectures · world models & frontiers.

**[24](curriculum/24-study-plan-and-projects.md) Study Plan & Projects**
3 paths (beginner 24 wks / developer 10 wks / research 16 wks) · 24-project ladder · final mastery gate · companion resources · **read [25](curriculum/25-ml-system-design.md)+[26](curriculum/26-ml-interview-prep-and-coding.md) for system design & interview prep.**

## Practice & Career
**[25](curriculum/25-ml-system-design.md) ML System Design & Case Studies**
Problem space vs solution space · 11-section design doc · Mercari pattern catalog (serving/QA/training/operation/lifecycle + antipatterns) · shadow & A/B testing · 8-step case-study method (search, recsys, ads, fraud) · **ML technical debt** (Sculley) & data cascades.

**[26](curriculum/26-ml-interview-prep-and-coding.md) Interviews & From-Scratch Coding**
Six interview modules · NumPy implementations (linear reg, CART, MLP backprop, attention head, autograd) · fundamentals drill · 45-min system-design framework · agentic/GenAI decision questions · STAR behavioral stories · 4-week prep sprint.

---

## Systems & Performance
**[27](curriculum/27-accelerated-computing-and-pytorch.md) Accelerated Computing & PyTorch**
Why frameworks exist · **the amortization rule** (`break-even = import_cost × s/(s−1)`) · what torch gives you over NumPy (autograd, fused kernels, devices, compile) · the async **synchronize()** timing trap · **GPU slower than CPU** · BLAS already makes large matmul optimal · bandwidth vs compute bound · MFU · scaling levers ranked by cost (batch → compile → bf16 → grad-accum → checkpointing → DDP → FSDP) · measurement discipline · decision playbook.

---

**Totals:** 31 docs · 28 theory modules · ~2,400 lines · every module ends with a Mastery Checklist · all cross-links verified.
**Executable monorepo:** 28 runnable examples (m00–m27) + JSON API + web UI + pytest suite — see [monorepo guide](../README.md) · `make run-all` (28/28 green) · `make test` (12/12 green) · `make bench` (NumPy vs PyTorch).

