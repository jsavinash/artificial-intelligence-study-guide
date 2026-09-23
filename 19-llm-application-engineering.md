# 19 — LLM Application Engineering

> Back to [index](README.md) · Prev: [18 Agents](18-ai-agents-and-tool-use.md) · Next: [20 Multimodal AI](20-multimodal-ai.md)

The engineering discipline around LLM calls: APIs, evals, guardrails, cost, and observability. **Prompting ([16](16-prompt-engineering.md)) is input design; this is system design.**

## 1. API interaction patterns

```python
# Canonical call shape (provider-agnostic concepts)
resp = client.chat.completions.create(
    model="...", messages=[{"role":"system",...},{"role":"user",...}],
    temperature=0.2, max_tokens=512, stream=True,
    tools=[...], response_format={"type":"json_object"})
```

- **Roles:** system (behavior), user, assistant (history — you must send prior turns), tool (results).
- **Streaming:** SSE token streams — essential for perceived latency (first-token time vs total time).
- **Structured output:** JSON mode (advisory) vs schema-constrained decoding (guaranteed) vs tool calls — use the strongest enforcement available.
- **Batching APIs:** async batch endpoints at ~50% cost for offline jobs (eval, backfills).
- **Batch multiple requests yourself** for parallel document processing.
- **Idempotency keys, retries with jitter, timeouts** — treat LLM calls like any unreliable network dependency.

## 2. Evals — the product's test suite

Levels:
1. **Unit-style:** single call → check format, required fields, refusal behavior (regex/assertions).
2. **Golden dataset:** 50–500 real cases with expected properties → run on every prompt/model change in CI.
3. **LLM-as-judge:** strong model scores responses against a **rubric** (faithfulness 1–5, tone, completeness) — calibrate on a human-labeled subset first; judges inherit biases (position, verbosity, self-preference).
4. **Human review:** gold standard for high stakes; sample-based.
5. **Online:** A/B, thumbs-up/down, task completion, downstream ticket reduction.

**Metrics that matter:** task success rate, groundedness, format validity, refusal appropriateness, **p95 latency**, **cost per task**, safety violation rate.
**Process:** version prompts/models/datasets together; detect regressions before prod; never eval only on data you tuned against (contamination, [15](15-fine-tuning-and-peft.md)).

## 3. Guardrails (defense in depth)

- **Input side:** prompt-injection classifiers, PII detection/redaction, topic blocklists, jailbreak heuristics ([22](22-ai-safety-security-ethics.md)).
- **Output side:** schema validation, factuality/groundedness checks vs retrieved context, citation verification, toxicity/profanity filters, allow-listed claims for high stakes.
- **Structural:** system-prompt isolation of untrusted content (delimit + "treat as data"), tool allow-lists, human approval for actions, sandboxed execution.
- **Never rely on the prompt alone.** Prompt text is data the model can be talked out of ([16](16-prompt-engineering.md)).

## 4. Cost & latency engineering

**Cost** = tokens × price; system prompts & RAG context dominate. Levers:
- **Prompt caching** (provider-managed): stable prefixes cached → big discounts; put static instructions/documents first, dynamic content last.
- **Smaller/cheaper models per tier:** route easy calls (classification, extraction) to small models; escalate hard ones — **model routing** with a fast classifier.
- **Trim context:** rerank→top-k ([17](17-retrieval-augmented-generation.md)); compress history; summarize long threads.
- Batch offline work; max_tokens caps; stop sequences.

**Latency stack:** TTFT (streaming, caching, smaller models, shorter prompts) vs total time (output length, reasoning models, retries). Techniques: streaming UI, speculative/parallel calls, **semantic caching** (embed query; if similar past answer exists & fresh → return it), keep prompts short, avoid unnecessary tool round-trips. Reasoning models trade latency for quality — make it a product choice ([14](14-large-language-models.md)).

## 5. Context engineering

Ordering (critical rules first/last), keeping *only* relevant history, packing retrieved chunks without duplication, token budgets per section, explicit "what the model should ignore." The craft that separates demos from products — big context windows don't absolve you (lost-in-the-middle, distraction, cost).

## 6. Architecture of a production app

```
client → gateway (auth, rate limit, cost meter)
      → orchestrator: prompt/version mgmt → model router
      → services: retrieval (17) · tools/agents (18) · fine-tuned models (15)
      → guardrails in/out → response (+ streaming)
cross-cutting: tracing/observability · eval harness · feedback loop → dataset → re-tune
```

- **State:** conversation store (Redis/DB), session summaries for long threads.
- **Feedback loop:** user signals → labeled set → eval → prompt/RAG/fine-tune improvement. This loop *is* the moat.
- Safety, privacy, and audit logging per [22](22-ai-safety-security-ethics.md).

## 7. Recurring GenAI design patterns

- **Large-document summarization:** *map* (summarize sections in parallel) → *reduce* (merge) → **map-reduce-refine** (sequential pass where each summary builds on the previous) — for corpora beyond the context budget (ingest side shared with [17](17-retrieval-augmented-generation.md)).
- **Classification with a large number of labels** (thousands of tags): LLM zero/few-shot with label descriptions → **hierarchical classification** (coarse→fine) → embedding similarity against label prototypes → distill into a small classifier from LLM-labeled data ([15](15-fine-tuning-and-peft.md)). Cheaper and sharper than one giant softmax prompt.
- **Minimizing hallucination (checklist):** grounded context + citations · explicit abstention ("not in context") · low temperature for factual turns · claim-level verification pass · narrower question scoping.
- **Output structure enforcement:** schema-constrained decoding / tool calls over prose requests (§2, [16](16-prompt-engineering.md)) — the production version of "please return JSON."

## Mastery Checklist

- [ ] Implements streaming with structured outputs, retries, and timeouts
- [ ] Builds a golden-set eval in CI with LLM-as-judge calibrated to humans
- [ ] Designs defense-in-depth guardrails (input, output, structural)
- [ ] Calculates and reduces cost/task using caching, routing, and context trimming
- [ ] Distinguishes TTFT from total latency and picks the right lever for each
- [ ] Applies context-engineering ordering and budgeting rules
- [ ] Sketches the full production architecture including the feedback loop
