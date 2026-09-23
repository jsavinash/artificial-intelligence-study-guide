# 18 — AI Agents & Tool Use

> Back to [index](README.md) · Prev: [17 RAG](17-retrieval-augmented-generation.md) · Next: [19 LLM App Engineering](19-llm-application-engineering.md)

**An agent = LLM + tools + loop + memory, pursuing a goal with intermediate steps.** Where prompting (single call) ends and autonomous systems begin.

## 1. Anatomy

```
goal
 └─ loop:
     1. observe (state, results, memory)
     2. think   (plan / choose next step)      ← LLM call
     3. act     (call a tool)                  ← structured output
     4. record  → memory / transcript
     until done | max_steps | budget | failure
```

- **Tools:** functions/APIs the model may call — search, code executor, SQL, browser, file I/O, ticketing, math.
- **Memory:** *context window* (working), *retrieval memory* (vector store of past episodes — [17](17-retrieval-augmented-generation.md)), *external state* (files, DBs).
- **Stop conditions:** explicit final answer, step limit, token/cost budget — **always cap steps** (runaway loops burn money).

## 2. Tool use / function calling (the mechanism)

1. Declare tools: name, description, JSON schema of params — **descriptions are the API docs the model reads**.
2. Model responds with a *tool call* (name + arguments) — enforced structured decoding.
3. Your code executes it (with auth/validation), returns a **tool result** message.
4. Model continues, incorporating the result.

**Design rules:** verbs not sentences (names); few, well-scoped tools beat 100 overlapping ones; validate & sanitize arguments; timeouts; never expose destructive actions without confirmation gates; return *concise, structured* results (huge tool outputs waste context).

## 3. Reasoning/control patterns

- **ReAct (Reason + Act):** interleave `Thought / Action / Observation` — the classic agent scaffold; grounding observations prevents hallucinated tool results.
- **Plan-and-execute:** first produce a step plan (or plan via a planner model), then execute steps — better for long-horizon; replan on failure.
- **Reflexion / self-critique:** after a step or task, critique output, add to memory, retry.
- **Code as action (CodeAct):** the model writes executable code instead of rigid tool calls — flexible for data/math work.
- **Workflow ≠ agent (important):** a *fixed DAG* of LLM calls (classify → retrieve → answer) is a **workflow** — deterministic, testable, cheap. Reserve **agents** for tasks where the *path isn't known upfront*. Most production wins in 2024–26 came from workflows + RAG, not open-ended autonomy.

## 4. Multi-agent systems

- **Orchestrator–worker:** planner delegates subtasks to specialized agents (researcher, coder, reviewer) and aggregates.
- **Debate/critic panels:** multiple agents propose/critique → better reasoning & evaluation.
- **Handoffs:** agent transfers the conversation to another specialist (support triage).
- **Shared blackboard/memory** vs message passing.
- **Costs:** coordination overhead, error compounding, harder evals — add structure before adding agents.

## 5. Frameworks (know the landscape, keep the core visible)

- **LangChain / LangGraph:** chains + graph-based stateful control flow (explicit state machine for agents); **LlamaIndex:** data/RAG-first, agent tools on top; **Haystack:** pipelines; **AutoGen / CrewAI / Semantic Kernel / OpenAI & Anthropic SDKs:** conversation/multi-agent scaffolds.
- **Advice:** build one agent loop *by hand first* (~100 lines) — framework magic otherwise hides failure modes; then adopt a framework for production features (streaming, retries, observability).

## 6. Agent reliability engineering (the hard part)

- **Structured outputs everywhere** (tool args, intermediate notes) — grammar-constrained decoding ([16](16-prompt-engineering.md)).
- **Environment feedback:** agents should *see errors* (stack traces, test failures) and self-correct — sandboxes for code execution.
- **Determinism aids:** temperature ~0 for planning steps; fixed seeds where possible; replayable transcripts.
- **Eval:** **trajectory evals** (was the step sequence valid?), **step-level** (correct tool? correct args?), **outcome-level** (task succeeded?). Compare against simpler baselines — agents must beat a workflow to justify complexity. Frameworks: promptfoo, LangSmith, Braintrust, W&B Weave.
- **Guardrails:** step/token/cost budgets, action allow-lists, human-in-the-loop approval for irreversible actions, rollback plans ([22](22-ai-safety-security-ethics.md)).
- **Observability:** full traces of every LLM call, tool call, and context — you cannot debug what you can't see ([19](19-llm-application-engineering.md)).

## 7. Where agents shine (and fail)

**Good fits:** research & synthesis over many sources; code fix→test→retry loops; ops triage with read-mostly tools; data wrangling pipelines; personal-assistant-style multi-step goals.
**Bad fits:** high-stakes irreversible actions without gates; tasks a 20-line script handles; domains without environment feedback (the agent can't tell it's wrong); extremely long horizons (errors compound).

## 8. Protocols & production agent patterns

- **Ng's four agentic design patterns** — *reflection* (critique & revise), *tool use*, *planning* (decompose → schedule → replan), *multi-agent* (collaborate) — the canonical vocabulary; §3's control patterns are their implementations.
- **MCP (Model Context Protocol):** open standard where servers expose tools/resources/prompts and clients discover them — "USB-C for AI tooling"; replaces N×N custom integrations. Core stack skill for 2026 agents.
- **A2A (Agent-to-Agent):** protocol for agents owned by *different* systems to discover each other and hand off tasks securely.
- **Long-running agents & AgentOps:** durable execution — checkpoint state, resume after failure, queues, human-approval steps, budget meters; DevOps thinking applied to agent runs (ties to [21](21-mlops-and-production-ml.md)).
- **Harness engineering:** the surrounding scaffolding (prompts, tool schemas, retries, context budgets) often decides success more than model choice — **context engineering** ([19](19-llm-application-engineering.md)) is its core skill.

## Mastery Checklist

- [ ] Defines the agent loop and contrasts it with a deterministic workflow
- [ ] Implements function calling end-to-end with schema, validation, and result messages
- [ ] Explains ReAct vs plan-and-execute vs reflexion
- [ ] Applies tool-design rules (naming, scoping, concise results)
- [ ] Adds budgets, action gates, and trajectory/step/outcome evals to an agent
- [ ] Knows when to choose a workflow over an agent — and defends the choice
- [ ] Sketches an orchestrator–worker multi-agent design with its failure modes
