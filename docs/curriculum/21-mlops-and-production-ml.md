# 21 — MLOps & Production ML

> Back to [index](../../README.md) · Prev: [20 Multimodal](20-multimodal-ai.md) · Next: [22 Safety & Ethics](22-ai-safety-security-ethics.md)

Models die in notebooks. MLOps = **reliably shipping, monitoring, and improving** models in production.

## 1. The ML system lifecycle

```
problem → data pipeline → train → validate → package → deploy
        → monitor (data + model + business) → retrain → repeat
```

**Team artifacts that make this real:** versioned data, versioned code, versioned models, reproducible training runs, documented metrics/SLOs.

## 2. Data engineering foundations

- **Pipelines:** ingestion → validation (schema/ranges/drift checks, Great Expectations) → transforms → feature computation.
- **Training/serving skew** = #1 killer: same code path features for both (libraries: Feast/Tecton feature stores, or unified transforms like tf.transform/Hopsworks).
- **Point-in-time correctness** for temporal features (no future leakage — [06](06-feature-engineering.md)).
- **Data versioning:** DVC, LakeFS, delta/Iceberg tables.
- **Label pipelines:** labeling tools, inter-annotator agreement, active learning loops ([04](04-unsupervised-learning.md)).

## 3. Experiment tracking & reproducibility

- Log: hyperparameters, code commit, data snapshot, env, metrics, artifacts per run.
- Tools: **MLflow** (open, self-host), **W&B** (teams, rich media), TensorBoard (local), Neptune.
- Seed everything ([01](01-python-and-data-tooling.md)); model registry stores "the approved version" with lineage.

## 4. Deployment patterns (classic ML & deep)

| Pattern | When |
|---|---|
| **Batch** (nightly scoring to DB) | predictions consumed in bulk; latency irrelevant |
| **Online endpoint** (REST/gRPC) | real-time requests |
| **Streaming** (Kafka/Spark) | event-driven scoring |
| **Edge/on-device** | privacy, offline, latency (Core ML, TFLite, ONNX Runtime) |
| **Embedded in app** (library) | model = feature inside product code |

- **Formats:** pickle/sklearn, **ONNX** (interchange), SavedModel/TorchScript, safetensors for LLM weights.
- **Serving stacks:** FastAPI/BentoML/Ray Serve/KFServing-Seldon/KServe/Triton (GPU, dynamic batching); TorchServe.
- **Canning:** containerize; A/B, **canary** (5%→100%), **shadow mode** (score live traffic, compare, don't serve) — rollback ready.

## 5. Scaling & performance

- Autoscaling on QPS/GPU util; cold-start problem (pre-warm, model server keeps loaded).
- **Batching** requests on GPU; quantization/fp16 for throughput ([15](15-fine-tuning-and-peft.md)).
- **Distillation** to smaller models for hot paths; caching frequent inputs (semantic caching for LLMs, [19](19-llm-application-engineering.md)).
- Cost dashboards per endpoint — model size = bill.

## 6. Monitoring (what to watch)

- **Data drift:** input distribution shift (PSI, KS test, chi², embedding drift) — *the early warning*.
- **Concept drift:** P(y|x) changed — performance labels arrive late (if at all).
- **Model metrics:** proxy quality (score distributions), delayed ground truth metrics, calibration drift.
- **Ops metrics:** latency p50/p95, error rate, throughput, resource saturation, **cost/request**.
- **LLM-specific ([19](19-llm-application-engineering.md)):** groundedness, refusal rate, tool-failure rate, judge scores sampled from prod, token spend.
- **Alerting + runbook → auto-retrain trigger** on sustained drift with champion/challenger gate before swap.

## 7. CI/CD/CT for ML

- **CI:** unit tests, data validation, training smoke test, eval suite on golden data, **prompt/regression tests** for LLM apps.
- **CD:** model + infra packaging, registry promotion, canary.
- **CT (continuous training):** scheduled or drift-triggered retraining; retraining must re-validate — models rot silently.
- Feature pipelines and training jobs orchestrated by **Airflow/Prefect/Dagster/Kubeflow**; compute on **Kubernetes** (or serverless endpoints).

## 8. LLMOps deltas vs classic MLOps

- No "training" in prod — but **prompts, indexes, tools, model versions all change behavior** → they *are* the artifacts to version ([19](19-llm-application-engineering.md)).
- **RAG index freshness** is your new data pipeline SLA ([17](17-retrieval-augmented-generation.md)).
- Evals-as-tests in CI; tracing every call (LangSmith/Braintrust/OpenTelemetry GenAI conventions).
- Guardrail monitors and human-feedback capture as first-class monitors ([22](22-ai-safety-security-ethics.md)).

## Mastery Checklist

- [ ] Designs train/serve pipelines that structurally prevent skew
- [ ] Picks a deployment pattern (batch/online/stream/edge) for a requirement
- [ ] Ships with shadow → canary → full with rollback
- [ ] Specifies a monitoring stack: data drift, performance, ops, cost
- [ ] Sets up experiment tracking and a model registry with lineage
- [ ] Defines CI gates for a model — and for an LLM app (prompts/indexes/evals versioned)
- [ ] Explains how feedback becomes retraining data without leaking
