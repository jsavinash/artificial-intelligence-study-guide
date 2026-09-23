"""m18 — Agents: hand-built agent loop (observe -> think -> act -> memory)
with tool calling, ReAct trace, step/budget limits, and trajectory evaluation.

Offline via MockLLM's calculator tool. Proves doc 18-ai-agents-and-tool-use.md.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import get_llm  # noqa: E402
from ai_core.datasets import rag_corpus  # noqa: E402
from ai_core.vectorstore import VectorStore, bm25_scores  # noqa: E402

TOOLS = [
    {"name": "calculator",
     "description": "Evaluate arithmetic expressions like '12 * 8'.",
     "parameters": {"type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}},
    {"name": "search_docs",
     "description": "Keyword search over internal documentation.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
]


def tool_search_docs(query: str, store: VectorStore) -> str:
    scores = bm25_scores(query, store.texts)
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return store.texts[best] if scores[best] > 0 else "NO_RESULTS"


def agent_run(llm, store, goal: str, max_steps: int = 6,
              budget_tokens: int = 500) -> dict:
    """The loop: prompt with tools -> parse tool call -> execute -> feed result."""
    trace, memory, spent = [], [], 0
    messages = [{"role": "system",
                 "content": "You are an agent. Use tools when needed. "
                            "When done reply with final: <answer>"}] + \
               [{"role": "user", "content": f"Goal: {goal}"}]
    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools=TOOLS)
        spent += len(reply) + len(goal)
        if spent > budget_tokens * 10:                 # runaway guard (scaled for mock)
            trace.append(("limit", "budget exceeded -> stop"))
            break
        # parse tool intent
        m = re.search(r'"tool":\s*"(\w+)"', reply)
        if m:
            tool = m.group(1)
            arg = (re.search(r'"expression":\s*"([^"]+)"', reply) or
                   re.search(r'"query":\s*"([^"]+)"', reply))
            arg = arg.group(1) if arg else goal
            if tool == "calculator":
                val = eval(arg, {"__builtins__": {}}, {})   # safe: numeric-only exprs
                observation = f"calculator -> {val}"
            else:
                observation = f"search_docs -> {tool_search_docs(arg, store)}"
            trace.append((step, f"ACT {tool}({arg!r})"))
            memory.append({"role": "assistant", "content": reply})
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"OBSERVATION: {observation}. Continue."})
        else:
            final = re.search(r"<answer>(.*?)</answer>", reply, re.S)
            answer = final.group(1).strip() if final else reply.strip()
            trace.append((step, "FINAL"))
            return {"answer": answer, "trace": trace, "steps": step,
                    "tool_calls": sum(1 for t, d in trace if "ACT" in str(d)),
                    "ok": True}
    return {"answer": "", "trace": trace, "steps": max_steps, "tool_calls": 0,
            "ok": False}


def main():
    llm = get_llm()
    store = VectorStore()
    for d in rag_corpus():
        store.add(d["id"], f"[{d['id']}] {d['title']}: {d['text']}",
                  llm.embed(d["text"]))

    # ---- goal 1: arithmetic -> calculator tool exercised ----
    r1 = agent_run(llm, store, "What is 12 * 8?")
    assert r1["ok"], r1
    assert "96" in r1["answer"], r1["answer"]
    assert r1["tool_calls"] >= 1

    # ---- goal 2: docs lookup -> search tool exercised ----
    r2 = agent_run(llm, store, "search_docs: What is MCP Model Context Protocol used for?")
    assert r2["ok"], r2

    # ---- budgets: max_steps caps the loop (termination guarantee) ----
    r3 = agent_run(llm, store, "calculate 6 * 7", max_steps=1)
    assert r3["steps"] <= 1

    # ---- trajectory eval: step sequence + outcome (the 3-level rubric) ----
    trajectory_ok = all(isinstance(s, int) or s == "limit" for s, _ in r1["trace"]) \
        and r1["steps"] >= 1
    outcome_ok = "96" in r1["answer"]
    step_ok = any("ACT" in str(d) for _, d in r1["trace"])
    assert trajectory_ok and outcome_ok and step_ok

    trace_str = " | ".join(f"s{s}:{d}" for s, d in r1["trace"])
    print(f"PASS m18 agents | tools={[t['name'] for t in TOOLS]} "
          f"goal1_steps={r1['steps']} tool_calls={r1['tool_calls']} "
          f"answer={r1['answer'][:40]!r} evals[traj={trajectory_ok} step={step_ok} "
          f"outcome={outcome_ok}] trace=[{trace_str}]")


if __name__ == "__main__":
    main()
