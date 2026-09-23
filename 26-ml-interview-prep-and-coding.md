# 26 — ML/AI Interviews & From-Scratch Coding

> Back to [index](README.md) · Prev: [25 ML System Design](25-ml-system-design.md)

Turning knowledge into interview performance — the six-module structure of FAANG-style ML interviews (AIMLInterviews), mapped onto this tutorial.

## 1. The interview map

| Module | Content | Where here |
|---|---|---|
| 1 | General coding (DSA) | adjacent skill — see coding-interview-university/system-design-primer |
| 2 | **ML coding from scratch** | §2 below |
| 3 | ML fundamentals breadth | module Mastery Checklists |
| 4 | **ML/GenAI/LLM system design** | [25](25-ml-system-design.md) + §4 |
| 5 | Agentic AI systems | [17](17-retrieval-augmented-generation.md)/[18](18-ai-agents-and-tool-use.md) + §5 |
| 6 | Behavioral & leadership | §6 |

## 2. ML coding from scratch (NumPy)

Build each in ≤40 lines; test shapes and hand-check gradients:

- Linear regression: closed form + gradient descent ([00](00-mathematical-foundations.md), [03](03-supervised-learning-algorithms.md))
- Logistic regression with a BCE training loop ([03](03-supervised-learning-algorithms.md))
- k-Means ([04](04-unsupervised-learning.md)) · PCA via SVD ([04](04-unsupervised-learning.md))
- CART decision tree with Gini splits ([03](03-supervised-learning-algorithms.md))
- 2-layer MLP: forward → loss → **manual backward** → update ([07](07-deep-learning-fundamentals.md))
- Single attention head with causal masking ([11](11-transformers-and-foundation-models.md))
- micrograd-style scalar autograd engine (Karpathy's lecture in your `machine-learning/nn-zero-to-hero`)
- Beam-search decoder ([10](10-rnn-and-sequence-modeling.md))

**Tactics:** narrate while coding · state time/space complexity · validate on a 3-line toy case · rubric = correct math → clean code → vectorization → edge cases.

## 3. ML fundamentals breadth (drill as question banks)

Use each module's Mastery Checklist ([03](03-supervised-learning-algorithms.md)–[08](08-training-and-regularizing-networks.md)): bias–variance, metric choice, imbalance, leakage, ensembles, BN vs LN, Adam vs SGD, vanishing gradients, regularization menu.
Modern add-ons: attention math ([11](11-transformers-and-foundation-models.md)) · scaling laws & post-training ([14](14-large-language-models.md)) · RAG vs fine-tune ([15](15-fine-tuning-and-peft.md), [17](17-retrieval-augmented-generation.md)) · agent patterns ([18](18-ai-agents-and-tool-use.md)) · hallucination, evals, safety ([19](19-llm-application-engineering.md), [22](22-ai-safety-security-ethics.md)) · multimodal ([20](20-multimodal-ai.md)).

## 4. ML/GenAI system-design interview framework

**45 minutes, eight steps** — exactly the [25](25-ml-system-design.md) design doc:
1. Clarify use case, users, scale, constraints (~5 min)
2. Metrics: offline + online + guardrails
3. Data & labeling
4. Baseline & validation schema
5. First model + improvement roadmap (features → complex → deep/LLM)
6. Serving pattern + latency/throughput math ([21](21-mlops-and-production-ml.md))
7. Experimentation (A/B) & monitoring
8. Risks, failure modes, cost

Drill prompts: *news-feed ranking* · *fraud detection* · *LLM customer-support copilot* · *RAG over company docs* · *agentic research assistant*.

## 5. Agentic & GenAI design specifics

Interviewers probe decisions: workflow vs agent ([18](18-ai-agents-and-tool-use.md)) · retrieval strategy & evals ([17](17-retrieval-augmented-generation.md)) · prompt/version management ([19](19-llm-application-engineering.md)) · prompt-injection defenses ([22](22-ai-safety-security-ethics.md)) · cost-per-request math (tokens × price) · when to fine-tune ([15](15-fine-tuning-and-peft.md)).

## 6. Behavioral & leadership (STAR)

Prepare 6–8 stories: conflict · failure/debug · ambiguous requirements · cross-team influence · quality vs speed · incident/rollback.
Structure: **Situation → Task → Action → Result** (with numbers + learning). Anchor to ownership, iteration, responsibility ([22](22-ai-safety-security-ethics.md)).

## 7. A 4-week prep sprint

- **W1:** daily DSA + 2 from-scratch implementations
- **W2:** fundamentals checklists aloud, 1 design doc per day
- **W3:** 3 full mock system designs (classic ML + GenAI + agent)
- **W4:** polished STAR stories, company-specific gap review, rest

## Mastery Checklist

- [ ] Implements linear regression, a tree split, MLP backprop, and an attention head from memory
- [ ] Delivers an 8-step ML system design answer in 45 minutes
- [ ] States metrics, validation, and monitoring for any proposed ML feature
- [ ] Compares RAG vs fine-tune vs agent with cost reasoning
- [ ] Has 6 polished STAR stories with measurable results
- [ ] Scores every module's Mastery Checklist at 100%
