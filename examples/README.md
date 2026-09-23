# 💻 Examples Catalog — 27 runnable implementations

One `main.py` per theory module in [`docs/curriculum/`](../docs/curriculum/).
Every example is **self-contained, offline-capable, and assertion-checked** — it prints `PASS …` and exits 0 only if every teaching claim actually holds.

```bash
make run M=11        # one example          # or: python3 examples/m11_transformers/main.py
make run-all         # all 27 (~2 min) → tally
```

**Reading an entry:** `→ doc` = theory it proves · `Shows` = key implementations · `Output` = the real PASS line this example prints (captured from a live run).

---

## Foundations (00–06) — classical ML

### m00 · Mathematical Foundations
→ [00](../docs/curriculum/00-mathematical-foundations.md) · `make run M=00`
Gradient descent on f(x)=x², by-hand 2-layer chain rule, Bayes medical-test posterior (base-rate trap), CE punishing confident-wrong, KL divergence, layer shape law.
```text
PASS m00 math | gd_min=-0.000000 bayes_post=0.154 backprop_L=0.000 kl_diff=0.583
```

### m01 · Python & Data Tooling
→ [01](../docs/curriculum/01-python-and-data-tooling.md) · `make run M=01`
NumPy broadcasting, pandas filter→groupby→merge→impute, **sklearn Pipeline with preprocessing inside CV** (the golden anti-leakage pattern), 5-fold scores.
```text
PASS m01 tooling | cv_acc=0.957±0.017 test_acc=0.930 top_city=Tokyo
```

### m02 · ML Foundations + **data-leakage demo**
→ [02](../docs/curriculum/02-machine-learning-foundations.md) · `make run M=02`
Baseline first; leaky model (label leaked as feature) shows a fantasy 1.000 score vs honest 0.950; generalization gap measured.
```text
PASS m02 workflow | baseline=0.518 clean_test=0.950 leaky_fantasy=1.000 (illusory) gap=0.022
```

### m03 · Supervised Algorithms (from scratch + sklearn)
→ [03](../docs/curriculum/03-supervised-learning-algorithms.md) · `make run M=03`
`linreg_gd`, `logistic_gd`, `knn_predict`, `best_gini_split` by hand; logistic/tree/forest/**GBM** compared — boosting beats the single tree.
```text
PASS m03 supervised | scratch: linreg=0.949 logreg=0.943 knn=0.880 tree_split(j=4,g=0.424) | logistic=0.943 tree=0.800 forest=0.846 gbm=0.851
```

### m04 · Unsupervised Learning
→ [04](../docs/curriculum/04-unsupervised-learning.md) · `make run M=04`
Lloyd's k-means (ARI 0.99 on blobs), elbow curve, PCA variance ratio, DBSCAN on non-convex moons — the k-means-vs-DBSCAN contrast.
```text
PASS m04 unsupervised | kmeans_ari=0.990 silhouette=0.743 pca_var=0.929 dbscan_k=2 noise=0.01 elbow=2012->287->238->155
```

### m05 · Evaluation & Tuning
→ [05](../docs/curriculum/05-model-evaluation-and-tuning.md) · `make run M=05`
P/R/F1 identities, rank-based ROC-AUC, stratified 5-fold, **class weights rescue minority recall 0.33→0.44**, RandomizedSearchCV.
```text
PASS m05 eval | acc=0.887 P=0.934 R=0.814 F1=0.870 ROC-AUC=0.969 cv=0.878 imbalance_rate=0.030 recall_plain=0.33->weighted=0.44 tuned=0.887
```

### m06 · Feature Engineering
→ [06](../docs/curriculum/06-feature-engineering.md) · `make run M=06`
**Safe target encoding** (in-fold + smoothing) vs leaky version, log de-skew, cyclical hour sin/cos, TF-IDF, permutation-importance selection.
```text
PASS m06 features | tfidf_nnz=17 safe_vs_leaky_differs=True acc=0.928 top_feature=f5
```

---

## Deep Learning from scratch (07–13) — pure NumPy

### m07 · DL Fundamentals — **MLP + backprop + gradient check**
→ [07](../docs/curriculum/07-deep-learning-fundamentals.md) · `make run M=07`
Forward/ReLU/softmax + manual backward; 400 descent steps; analytic gradient vs central finite differences — identical to 5 decimals.
```text
PASS m07 dl_fundamentals | loss 1.101->0.064 acc=0.991 gradcheck_analytic=0.00177 numeric=0.00177
```

### m08 · Training & Regularization
→ [08](../docs/curriculum/08-training-and-regularizing-networks.md) · `make run M=08`
Adam (momentum + bias correction) vs SGD; **LR=30 diverges** (2.672 vs 0.072); dropout masks with inference scaling; weight decay shrinks ‖W‖.
```text
PASS m08 training | sgd_final=0.072 adam_final=0.131 sgd_lr30_final=2.672(worse) dropout_final=0.079 ||W||_plain=3.618>reg=1.780 test_acc=0.920
```

### m09 · CNNs & Vision
→ [09](../docs/curriculum/09-convolutional-networks-and-vision.md) · `make run M=09`
`conv2d` sliding-window from scratch, shape law (8→6; stride 2→3), **translation equivariance**, Sobel edge detection, max-pool, conv gradient check.
```text
PASS m09 cnn | conv_out=6x6 stride2=3x3 equivariant=True edge_resp=1.298>flat=0.000 pool=3x3 gradcheck_max_err=1.22e-09
```

### m10 · RNNs & Sequences
→ [10](../docs/curriculum/10-rnn-and-sequence-modeling.md) · `make run M=10`
Vanilla RNN recurrence, **vanishing-gradient demo** (|g| 1e-2 → 3e-25 over 60 steps), word2vec-style embeddings (cat–dog 0.99 vs cat–the −0.72), Naive-Bayes sentiment.
```text
PASS m10 sequences | rnn_T=10 vanishing: |g|_5=1.13e-02>|g|_60=3.04e-25 word2vec_sim(cat,dog)=0.990>sim(cat,the)=-0.722 nb_sentiment_acc=1.00
```

### m11 · Transformers — **attention + next-token LM**
→ [11](../docs/curriculum/11-transformers-and-foundation-models.md) · `make run M=11`
Scaled dot-product + causal mask (rows sum 1, future masked), √d_k saturation demo, multi-head split, **KV-cache equivalence proof**, char-LM below uniform entropy.
```text
PASS m11 transformers | causal_mask=True rows_sum=1 kv_cache_equiv=True mh=(8, 16) nxttok_loss 2.891->1.420 < uniform=2.890
```

### m12 · Generative Models
→ [12](../docs/curriculum/12-generative-models.md) · `make run M=12`
VAE reparameterization + KL≥0 (0.632), diffusion forward schedule (ᾱ 1.0→4e-5 destroys signal), **oracle reverse recovers x₀ exactly**, temperature → entropy.
```text
PASS m12 generative | vae_kl=0.632 recon=2.153 diffusion_sig 1.00->0.00 a_bar_T=4.0e-05 oracle_recover=True tau_entropy low=-0.00<high=1.20
```

### m13 · Reinforcement Learning
→ [13](../docs/curriculum/13-reinforcement-learning.md) · `make run M=13`
Q-learning (TD + ε-decay) on a 4×4 gridworld: return −0.48→+0.67, greedy policy finds the 6-step path, avoids the hole; final policy arrows printed.
```text
PASS m13 rl | return early=-0.48 -> late=0.67 path_len=6 goal_reached=True V_start=0.729 V_goal=0.000 policy=↓←↓↓↓·↓↓↓→→↓→→→·
```

---

## LLM stack (14–19) — offline via MockLLM, live with an API key

### m14 · LLMs — **BPE tokenizer + scaling laws**
→ [14](../docs/curriculum/14-large-language-models.md) · `make run M=14`
BPE merges trained on a corpus (25 merges, 8 tokens vs 21 chars), Chinchilla calculator: 7B @ 6e21 FLOPs → **ratio 20.4 tokens/param**, temperature flattening.
```text
PASS m14 llm | bpe_merges=25 vocab=44 encode('the cat sat')=8tok vs 21chars | 7B_opt_D=142.9B ratio=20.4 | 13B_chinchilla_D=260.0B | llm=mock-llm
```

### m15 · Fine-Tuning & PEFT — **LoRA + distillation + quantization**
→ [15](../docs/curriculum/15-fine-tuning-and-peft.md) · `make run M=15`
SVD low-rank ΔW = B·A (r=8 → 25% of params, rank-error sweep 0.94→0.32), **merge equivalence** (adapter ≡ merged), teacher→student soft-label distillation, int8 error 0.8%.
```text
PASS m15 peft | lora trainable 1024/4096=25.0% rank_err 2:0.94->32:0.32 merge_equal=True distill_loss 0.670->0.261 student_acc=0.992 teacher_acc=0.994 int8_err=0.008
```

### m16 · Prompt Engineering — **golden-set eval harness**
→ [16](../docs/curriculum/16-prompt-engineering.md) · `make run M=16`
5 prompt patterns (summarize / classify / JSON extract / few-shot / chain-of-thought) run against a golden set with checkers; determinism verified (temp=0).
```text
PASS m16 prompt-engineering | golden 5/5 (summarize=OK classify=OK extract_json=OK few_shot=OK chain_of_thought=OK) deterministic=True provider=mock-llm
```

### m17 · RAG — **chunk → embed → hybrid search → citations**
→ [17](../docs/curriculum/17-retrieval-augmented-generation.md) · `make run M=17`
Overlap chunking, hashed embeddings, **cosine + BM25 fused with RRF**, retrieval benchmark hit@1=3/3 (LoRA/BM25/Chinchilla queries), grounded answers cite `[dN]`.
```text
PASS m17 rag | chunks=8 hit@1=3/3 top_doc=d2 citations_in_answer=True hybrid=cosine+BM25+RRF provider=mock-llm
```

### m18 · Agents — **tool loop with ReAct trace + evals**
→ [18](../docs/curriculum/18-ai-agents-and-tool-use.md) · `make run M=18`
Observe→think→act→memory loop with `calculator` + `search_docs` tools, step/budget caps, and 3-level trajectory/step/outcome evaluation.
```text
PASS m18 agents | tools=['calculator', 'search_docs'] goal1_steps=2 tool_calls=1 answer='The result is 96.0.' evals[traj=True step=True outcome=True] trace=[s1:ACT calculator('12.0*8.0') | s2:FINAL]
```

### m19 · LLM App Engineering — **evals, guardrails, cost tracking**
→ [19](../docs/curriculum/19-llm-application-engineering.md) · `make run M=19`
Golden-set harness with latency + $ cost per case, injection/PII input guards, output leak checks, LLM-as-judge rubric score, semantic cache (1 hit saved).
```text
PASS m19 llm-app | eval_pass=100% p95=0.11ms total_cost=$0.000122 guards[injection_blocked=True pii_flagged=True] judge=1.0 cache_hits=1 provider=mock-llm
```

---

## Systems, safety & career (20–26)

### m20 · Multimodal — **CLIP-style shared space + InfoNCE**
→ [20](../docs/curriculum/20-multimodal-ai.md) · `make run M=20`
Cross-modal retrieval@1 = 8/8 both directions, zero-shot label classification (4/8 with hashed embeds — 6× chance), symmetric contrastive loss aligned < shuffled.
```text
PASS m20 multimodal | retrieval@1=8/8 both directions zero_shot=4/8 InfoNCE aligned=1.648<shuffled=2.127 temp5=0.493 dim=64
```

### m21 · MLOps — **registry, drift, batch vs online serving**
→ [21](../docs/curriculum/21-mlops-and-production-ml.md) · `make run M=21`
Save→load→**promote staging→production** in the filesystem registry, PSI drift verdicts (stable / retrain), batch vs per-row latency, prediction logging to CSV.
```text
PASS m21 mlops | acc=0.960 registry=production drift[stable=stable,shifted=retrain] batch=0.09ms vs online_avg=0.05ms pred_log=150 rows
```

### m22 · Safety — **injection detector, PII redaction, fairness audit**
→ [22](../docs/curriculum/22-ai-safety-security-ethics.md) · `make run M=22`
Regex injection patterns (P=R=1.0 on fixture), email/SSN/phone redaction, **demographic-parity bias measured 0.251 → 0.027 after debiasing**, equalized-odds gaps.
```text
PASS m22 safety | injection P=1.00 R=1.00 pii_redacted=['email', 'ssn', 'phone'] bias_DP=0.251>fair_DP=0.027 EO_tpr=0.265
```

### m23 · Advanced — **recommender + time series + GNN**
→ [23](../docs/curriculum/23-advanced-and-specialized-topics.md) · `make run M=23`
ALS matrix factorization (RMSE 0.015 ≪ mean baseline), **walk-forward CV with causality check** (future never in train), 2-layer message passing → 100% label propagation.
```text
PASS m23 advanced | mf_rmse=0.015<baseline=0.346 walkforward_splits=9 (causal=True) ts_err naive=1.30 mean=4.21 gnn_label_acc=1.000
```

### m24 · Capstone — **end-to-end pipeline integration**
→ [24](../docs/curriculum/24-study-plan-and-projects.md) · `make run M=24`
Runs the whole project ladder in one process — data→baseline→split→GBM→metrics→register→drift→RAG index→design doc→**verifies all other 26 examples still green**.
```text
PASS m24 capstone | checklist 10/10 acc=0.925>baseline=0.521 registry=capstone-gbm drift=investigate store=8 examples=26green | ✅…✅examples_green
```

### m25 · ML System Design — **11-section design-doc generator**
→ [25](../docs/curriculum/25-ml-system-design.md) · `make run M=25`
Renders a complete interview-ready design doc (news-feed ranking) to `artifacts/design_docs/`, fills the 8-step case-study scaffold, serving-pattern picker, "is ML even the answer?" gate.
```text
PASS m25 system-design | doc_sections=11/11 written_to=news_feed_ranking.md (1555B) case_steps=8/8 serving_picker=OK ml_gate=OK
```

### m26 · Interviews — **from-scratch verification + quiz**
→ [26](../docs/curriculum/26-ml-interview-prep-and-coding.md) · `make run M=26`
Reloads your earlier implementations via importlib and re-verifies them (linreg recovers w=[2,−1,0.5], k-means finds planted centers, causal attention row-0), plus a 6-question keyed quiz.
```text
PASS m26 interview | from_scratch[linreg✓ kmeans✓ attention✓ backprop✓ logreg acc=0.95] quiz=6/6 star_stories=4_framework_ready
```

---

## 🧪 Verifying everything

```bash
make run-all     # RESULT: 27 passed, 0 failed
make test        # 4 passed  (example smoke ×27 + live API endpoint suite)
```

- Each `main.py` asserts its own claims — a changed number **fails loudly**, so the outputs above are executable documentation.
- LLM examples run on the deterministic **MockLLM** offline; export `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` to run the *same* code against a real model.
- New example? Drop `examples/mXX_name/main.py` with a `PASS mXX` print — the runner and test suite pick it up automatically.

