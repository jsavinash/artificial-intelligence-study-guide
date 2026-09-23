# 22 — AI Safety, Security & Ethics

> Back to [index](../../README.md) · Prev: [21 MLOps](21-mlops-and-production-ml.md) · Next: [23 Advanced Topics](23-advanced-and-specialized-topics.md)

Non-optional for practitioners: systems that are **secure, fair, lawful, and aligned with intent.**

## 1. Alignment — getting models to do what we actually want

- **The core problem:** specifying human intent precisely is hard; optimizers find **specification gaps** (reward hacking — [13](13-reinforcement-learning.md)).
- **Scalable oversight:** RLHF/RLAIF, constitutional principles, debate/critique, **process reward models** (grade reasoning steps, not just answers) ([14](14-large-language-models.md)).
- **Inner alignment (research):** does the learned objective match the specified one? Untestable at frontier scale but frames the risk.
- **Emergence & deception concerns:** monitoring for deceptive sycophancy, sandbagging; evaluation under distribution shift — active research frontier.
- **Capability vs control:** more capable models need *more* oversight — control methods (auditing, sandboxing, tripwires).

## 2. Bias & fairness

- **Sources:** historical bias in data, representation imbalance, measurement/label bias, aggregation (Simpson's paradox), deployment feedback loops.
- **Fairness metrics — provably incompatible in general** (Kleinberg et al.): choose per context.
  - **Demographic parity:** equal selection rates.
  - **Equalized odds / equal opportunity:** equal TPR/FPR across groups.
  - **Calibration:** equal accuracy of probabilities within groups.
- **Mitigation:** re-sampling/reweighting, re-labeling, adversarial debiasing, in-processing constraints, per-group thresholding (legally fraught — involve counsel), model cards & datasheets.
- **Practice:** disaggregated evaluation by subgroup — averages hide harm ([05](05-model-evaluation-and-tuning.md)).

## 3. Privacy

- **Training data:** PII scrubbing, opt-outs, machine-unlearning, membership-inference and **model-inversion** attacks.
- **Inference privacy:** redact sensitive data before third-party APIs; **differential privacy** (ε-budgets); federated learning ([23](23-advanced-and-specialized-topics.md)).
- **Regulation:** GDPR (deletion, purpose limits), CCPA — plus provider DPAs.

## 4. Security — adversarial ML & LLM threats

- **Prompt injection (the new XSS):** instructions hidden in *user input, web pages, documents, images* that override your system prompt. **Indirect injection** via RAG/browser tools is worse. Defenses: treat untrusted content as data, minimize tool authority, output filtering, isolated instruction channels — **assume breach** ([16](16-prompt-engineering.md), [19](19-llm-application-engineering.md)).
- **Jailbreaks:** role-play, obfuscation, encoding, multi-turn persona drift; layered classifiers + rate limits — no prompt-only fix.
- **Data poisoning & supply chain:** poisoned fine-tuning data, malicious model files (**use safetensors, never pickle**), typosquatted packages, poisoned vector indexes.
- **Model extraction & provenance:** theft via queries; watermarking (SynthID) and C2PA credentials for generated content.
- **Classic adversarial ML:** evasion/perturbation attacks on CV systems ([09](09-convolutional-networks-and-vision.md)).
- **Agentic risk multiplier:** language attacks become *actions* — least privilege, approvals, sandboxing ([18](18-ai-agents-and-tool-use.md)).

## 5. Interpretability & XAI

- **Global:** feature & permutation importance, SHAP (game-theoretic attributions), partial dependence, surrogate rules ([03](03-supervised-learning-algorithms.md)).
- **Local:** why *this* prediction — LIME, SHAP values; attention inspection (weaker evidence than hyped).
- **Mechanistic interpretability (research):** circuits, features, sparse autoencoders decomposing LLM activations — toward reading internal concepts ([11](11-transformers-and-foundation-models.md)).
- **Use cases:** debugging, regulatory explanation (loan denials), trust, scientific insight.

## 6. Responsible practice & governance

- **Documentation:** model cards, datasheets for datasets, eval reports, incident logs.
- **Pre-deployment:** misuse red-teaming, FMEA-style hazard analysis, staged rollout, kill switch.
- **Harms taxonomy:** allocation (who loses opportunity), representational (stereotype amplification), physical safety (autonomy, medical), misinformation at scale.
- **Human-in-the-loop** for high-stakes decisions — with a clear escalation path.
- **Copyright & consent:** training-data legality unsettled; respect licenses, opt-outs, and scraped-content terms.

## 7. Regulation landscape (moving target — verify current status)

- **EU AI Act:** risk tiers (unacceptable → prohibited; high → strict obligations; limited → transparency; minimal) + GPAI duties for general-purpose models.
- **Sector rules already binding:** medical, finance (fair-lending), hiring, credit — model-risk guidance expects validation & documentation.
- **Standards:** NIST AI RMF, ISO/IEC 42001, red-teaming & incident-reporting norms.
- **Provenance infra:** C2PA content credentials, synthetic-media labeling laws.

## Mastery Checklist

- [ ] Names 3 alignment techniques and explains reward hacking concretely
- [ ] States the fairness-metric impossibility and picks metrics for hiring vs medical contexts
- [ ] Mitigates direct and indirect prompt injection in a RAG+tool system (defense in depth)
- [ ] Designs a privacy posture: PII handling, DP, provider terms
- [ ] Runs a red-team exercise with misuse scenarios before launch
- [ ] Produces a model card covering data, evals, limits, and intended use
- [ ] Explains SHAP vs attention as explanation tools and their limits

