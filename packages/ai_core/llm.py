"""Unified LLM interface: real API when keys exist, deterministic MockLLM otherwise.

Capacity note: this machine has NO API keys in env, so MockLLM is the default
execution path — every example in the monorepo runs fully offline.
Drop OPENAI_API_KEY or ANTHROPIC_API_KEY into the environment and the same
examples transparently call a real model.
"""
from __future__ import annotations

import json
import os
import re
import zlib


# ----------------------------------------------------------------- mock
class MockLLM:
    """Deterministic, rule-based stand-in with the shape of a chat API.

    Supports: chat(messages, json_mode, tools), streaming, and embeddings.
    Used by RAG/agent/prompt examples when no API key is configured.
    """

    name = "mock-llm"

    def __init__(self, seed: int = 0):
        self.calls = 0

    def chat(self, messages: list[dict], json_mode: bool = False,
             tools: list[dict] | None = None, temperature: float = 0.0) -> str:
        self.calls += 1
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        lower = text.lower()

        # 0) agent loop: consume an OBSERVATION and produce a final <answer>
        if "observation:" in lower:
            obs = re.split(r"observation:", text, maxsplit=1, flags=re.I)[1].strip()
            if "search_docs ->" in obs:
                snippet = obs.split("->", 1)[1].strip()[:220]
                return f"<answer>{snippet}</answer>"
            m0 = re.search(r"(-?\d+(?:\.\d+)?)", obs)
            return f"<answer>The result is {m0.group(1) if m0 else '?'}.</answer>"

        # 1a) tool use: search_docs (agent needs a doc lookup)
        if tools and any(t.get("name") == "search_docs" for t in tools) and \
                ("search_docs" in lower or "documentation" in lower):
            q = text.split(":", 1)[1].strip() if ":" in text else text
            return json.dumps({"tool": "search_docs", "args": {"query": q[:80]}})

        # 1) tool use: calculator arithmetic (agent loop)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*])\s*(-?\d+(?:\.\d+)?)", text)
        if tools and any(t.get("name") == "calculator" for t in tools) and m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            val = {"+": a + b, "-": a - b, "*": a * b}[op]
            return json.dumps({"tool": "calculator",
                               "args": {"expression": f"{a}{op}{b}",
                                        "result": val}})

        # 2) grounded RAG answer built from [Sn]/[dN] citations in context
        ctx_ids = re.findall(r"\[(d\d+)\]", text)
        if "context" in lower and ctx_ids:
            unique = list(dict.fromkeys(ctx_ids))
            answer = "Based on the context, " + self._summarize(text, unique) + \
                     f" Sources: {', '.join(unique)}."
            if json_mode:
                return json.dumps({"answer": answer, "citations": unique, "confident": True})
            return answer

        # 3) structured extraction
        if json_mode:
            emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
            amount = re.search(r"\$?(\d+(?:,\d{3})*(?:\.\d+)?)", text)
            payload = {"summary": self._summarize(text, []),
                       "emails": emails,
                       "amount": float(amount.group(1).replace(",", "")) if amount else None,
                       "sentiment": "negative" if any(w in lower for w in ("fail", "error", "broken"))
                                    else "neutral"}
            return json.dumps(payload)

        # 4) summarization
        if "summar" in lower:
            return self._summarize(text, [])

        # 5) chain-of-thought / rate problems ("... X km in Y hours ... speed")
        m_sp = re.search(r"(\d+(?:\.\d+)?)\s*km\s+in\s+(\d+(?:\.\d+)?)\s*hours", lower)
        if "step by step" in lower or m_sp:
            if m_sp:
                speed = float(m_sp.group(1)) / float(m_sp.group(2))
                return (f"Givens: distance={m_sp.group(1)} km, time={m_sp.group(2)} h. "
                        f"Plan: speed = distance / time. "
                        f"Final answer: {speed:g} (km per hour).")
            return "Step 1: parse givens. Step 2: reason. Final answer: done."

        # 6) classification: label picked from a bracketed list
        m = re.search(r"classify.*?into\s*\[([^\]]+)\]", text, re.I | re.S)
        if m:
            labels = [l.strip() for l in m.group(1).split(",")]
            return next((l for l in labels if l.lower() in lower), labels[0])

        # 6) default conversational reply
        return (f"I am the offline MockLLM. You said: {text[:120]!r}. "
                "Set OPENAI_API_KEY/ANTHROPIC_API_KEY for live model answers.")

    def _summarize(self, text: str, cites: list[str]) -> str:
        words = re.sub(r"\[.*?\]", "", text).split()
        keep, seen = [], set()
        for w in words:
            lw = w.lower().strip(".,!?")
            if lw not in seen and len(lw) >= 3:
                seen.add(lw)
                keep.append(lw)
            if len(keep) >= 14:
                break
        s = " ".join(keep)
        return (s[0].upper() + s[1:] + ".") if s else "No content."

    def stream(self, messages: list[dict], **kw):
        for tok in self.chat(messages, **kw).split():
            yield tok + " "

    def embed(self, text: str, dim: int = 64) -> list[float]:
        """Deterministic hashed bag-of-words embedding (cosine-friendly)."""
        v = [0.0] * dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


# --------------------------------------------------------------- real APIs
class OpenAIClient:
    name = "openai"

    def __init__(self):
        from openai import OpenAI  # lazy import — optional dependency
        self.c = OpenAI()

    def chat(self, messages, json_mode=False, tools=None, temperature=0.0):
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        if tools:
            kw["tools"] = [{"type": "function", **t} for t in tools]
        r = self.c.chat.completions.create(model="gpt-4o-mini", messages=messages,
                                           temperature=temperature, **kw)
        return r.choices[0].message.content or ""

    def embed(self, text, dim=1536):
        return self.c.embeddings.create(model="text-embedding-3-small",
                                         input=text).data[0].embedding


class AnthropicClient:
    name = "anthropic"

    def __init__(self):
        import anthropic
        self.c = anthropic.Anthropic()

    def chat(self, messages, json_mode=False, tools=None, temperature=0.0):
        sys = "Respond with valid JSON only." if json_mode else None
        r = self.c.messages.create(model="claude-3-5-haiku-latest", max_tokens=512,
                                   system=sys, messages=messages, temperature=temperature)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    def embed(self, text, dim=1024):
        return MockLLM().embed(text, dim)  # Anthropic has no public embeddings endpoint


def get_llm(prefer: str | None = None):
    """Factory: real client if key present, else MockLLM (never raises).

    Order: explicit prefer > LLM_PROVIDER env > auto-detect keys > mock.
    """
    prefer = (prefer or os.getenv("LLM_PROVIDER", "")).lower()
    try:
        if prefer == "mock":
            return MockLLM()
        if prefer in ("", "auto"):
            if os.getenv("ANTHROPIC_API_KEY"):
                return AnthropicClient()
            if os.getenv("OPENAI_API_KEY"):
                return OpenAIClient()
            return MockLLM()
        if prefer == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIClient()
        if prefer == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicClient()
    except Exception:
        pass  # any import/config failure falls back — examples must always run
    return MockLLM()

