# 16 — Prompt Engineering

> Back to [index](README.md) · Prev: [15 Fine-Tuning](15-fine-tuning-and-peft.md) · Next: [17 RAG](17-retrieval-augmented-generation.md)

Prompting = programming a model in natural language. Structured, teachable, and the fastest lever in applied GenAI.

## 1. Anatomy of a prompt

```
[system prompt]   role, rules, constraints, tone, boundaries
[user prompt]     task + context + input data + output requirements
[few-shot examples]  optional: input → desired output pairs
```

**Principles:** be specific (vague in → vague out); give the model an *out* ("if unsure, say I don't know"); format instructions matter more than polite words; one clear task per prompt beats a bundle.

## 2. Core patterns (the DeepLearning.AI/OpenAI taxonomy)

1. **Writing instructions** — delimit context with `"""`/XML tags; state audience & purpose; "explain step-by-step" / "think before answering."
2. **Iterative development** — prompt → inspect output → fix the *specific* failure → repeat. The real workflow.
3. **Summarizing** — "summarize for a 2nd grader," length & focus constraints.
4. **Inferring** — classify/sentiment/extract from unstructured text.
5. **Transforming** — rewrite tone, translate, format conversions, edit-preservation tasks.
6. **Expanding** — take short input → longer text (careful: amplifies hallucination surface).
7. **Chatbots / multi-turn** — maintain history; system prompt holds standing instructions.

## 3. Reasoning techniques

- **Chain-of-thought (CoT):** "think step by step" or show reasoning in examples — decompose before answering. Huge on math/logic; near-free (few extra tokens).
- **Zero-shot CoT** ("Let's think step by step") vs **few-shot CoT** (examples *with* reasoning traces).
- **Structured scratchpad:** "list givens → plan → compute → check" — pseudo-reasoning models; modern **reasoning models** do this natively with trained hidden/thinking budgets ([14](14-large-language-models.md)).
- **Self-consistency:** sample N reasoning paths, majority-vote the answer — accuracy↑, cost×N.
- **Decomposition:** split complex tasks into sub-prompts chained by code ([18](18-ai-agents-and-tool-use.md)).
- **Checklist/verification:** "review your answer against these rules before final output."

## 4. Few-shot prompting

- **3–8 examples** typical; match output format *exactly* (the model mimics format as much as content).
- Cover edge cases in examples (empty input, ambiguous input, refusal cases).
- Order matters somewhat; keep examples consistent — contradictory examples = confused model.

## 5. Structured & constrained output

- Explicit schema: "Respond ONLY with JSON: {field: type}" + list fields + nullability.
- Show one complete example of the schema; demand no extra prose.
- **API-level enforcement:** JSON mode / tool-choice / grammar-constrained decoding (guaranteed-valid JSON by masking invalid tokens) — use these over prose-only requests in production ([19](19-llm-application-engineering.md)).
- For anything parseable, **prefer function/tool calling** over free-text JSON ([18](18-ai-agents-and-tool-use.md)).

## 6. System-prompt design patterns

- Persona ("You are a senior reviewer for X") + audience + scope + out-of-scope list.
- Behavioral rules: citation policy, refusal policy, tone, language, verbosity.
- Guardrail statements → but see [22](22-ai-safety-security-ethics.md): **prompt text alone is not security** (prompt injection can override it).
- Keep stable instructions in the **system/prefix** (cacheable) and dynamic content in the user turn (prompt caching economics, [19](19-llm-application-engineering.md)).

## 7. Failure modes & fixes

| Symptom | Fix |
|---|---|
| Confidently wrong (hallucination) | Ground in context (RAG), demand citations, allow "I don't know", lower temperature |
| Ignores instructions late in prompt | Put critical rules **first and last**; repeat; shorten context |
| Format drift | One perfect example; JSON mode; validators + retry loop |
| Verbosity/slop | "Answer in ≤3 sentences"; "no preamble" |
| Sycophancy | System rule: "Disagree when warranted; never flatter" |
| Prompt injection in inputs | Delimit untrusted content, strip instructions from data, output filters ([22](22-ai-safety-security-ethics.md)) |
| Non-determinism breaks tests | Fix seed/temp=0 where possible; test semantic equivalence, not exact strings |

## 8. Evaluation of prompts

- Build a **golden set** of 30–100 real inputs with expected properties before changing anything.
- Automated: exact/regex match, rubric checks, **LLM-as-judge with a fixed rubric** ([19](19-llm-application-engineering.md)).
- Version prompts like code; measure cost & latency per change; regressions are common when "improving" a prompt.

## Mastery Checklist

- [ ] Writes a complete system+user+few-shot prompt for an unfamiliar task
- [ ] Applies CoT, self-consistency, and decomposition — and states when each helps
- [ ] Produces reliably parseable JSON (and knows when to use grammar/tool enforcement instead)
- [ ] Diagnoses a failing prompt using the failure-mode table
- [ ] Builds a golden test set and judges prompt changes with it
- [ ] Explains why system prompts are not a security boundary
