"""Production-shape JSON API (stdlib http.server — no FastAPI install needed,
per the capacity analysis). Endpoints:

  GET  /health            -> liveness + provider info
  POST /predict           -> tabular model inference (registry-backed)
  POST /rag               -> retrieval-augmented answer with citations
  POST /agents/run        -> agent loop with tools + trace
  POST /generate          -> raw LLM chat

Run: make serve   (http://127.0.0.1:8000)   UI: make ui
"""
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_core import VectorStore, get_llm  # noqa: E402
from ai_core.datasets import rag_corpus  # noqa: E402
from ai_core.registry import ModelRegistry  # noqa: E402

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))     # PORT override used by tests
LLM = get_llm()
REG = ModelRegistry()

# ---- warm RAG index at boot (cold-start latency would hit the first request) ----
STORE = VectorStore()
for _d in rag_corpus():
    STORE.add(_d["id"], f"[{_d['id']}] {_d['title']}: {_d['text']}",
              LLM.embed(f"{_d['title']} {_d['text']}"))


def load_latest_model():
    import numpy as np
    models = sorted(REG.list(), key=lambda m: m.get("created_at", 0), reverse=True)
    if not models:
        return None, None
    mid = models[0]["model_id"]
    payload = REG.load(mid)
    # ravel: sklearn stores binary coef_ as (1, n_features)
    return (np.array(payload["coef"], float).ravel(),
            float(np.ravel(payload["intercept"])[0]),
            int(payload["n_features"])), mid


MODEL_INFO = {"model": None, "id": None}


def refresh_model():
    MODEL_INFO["model"], MODEL_INFO["id"] = load_latest_model()


refresh_model()


def rag_answer(question: str, k: int = 3):
    hits = STORE.search(LLM.embed(question), k=k)
    context = "\n".join(t for _, _, t in hits)
    ids = sorted({m.group(1) for _, _, t in hits
                  if (m := re.match(r"\[(d\d+)\]", t))})
    reply = LLM.chat([
        {"role": "system",
         "content": "Answer ONLY from the context. Cite [dN]. If absent: I don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])
    return {"answer": reply, "citations": ids,
            "sources": [{"id": i, "score": round(s, 3)} for i, s, _ in hits],
            "provider": LLM.name}


def agent_run(goal: str, max_steps: int = 6):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m18", str(Path(__file__).resolve().parents[2] /
                   "examples/m18_agents/main.py"))
    m18 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m18)
    result = m18.agent_run(LLM, STORE, goal, max_steps=max_steps)
    return {"answer": result["answer"], "steps": result["steps"],
            "tool_calls": result["tool_calls"],
            "trace": [{"step": s, "detail": d} for s, d in result["trace"]],
            "ok": result["ok"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # CORS so the separate UI origin (port 8080) can call us
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):                                  # CORS preflight
        self._send(204, {})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "provider": LLM.name,
                             "model": MODEL_INFO["id"], "index_size": len(STORE),
                             "uptime_s": round(time.time() - START, 1)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        t0 = time.perf_counter()
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/predict":
                import numpy as np
                if MODEL_INFO["model"] is None:
                    refresh_model()
                if MODEL_INFO["model"] is None:
                    self._send(503, {"error": "no model registered — run `make run M=21` first"})
                    return
                coef, intercept, n_feat = MODEL_INFO["model"]
                x = np.array(data.get("features", []), float)
                if x.size != n_feat:
                    self._send(422, {"error": f"expected {n_feat} features, got {x.size}"})
                    return
                logit = float(x @ coef + intercept)
                prob = 1 / (1 + pow(2.718281828, -logit))
                self._send(200, {"model": MODEL_INFO["id"], "score": round(prob, 4),
                                 "label": int(prob >= 0.5),
                                 "ms": round((time.perf_counter() - t0) * 1000, 2)})
            elif self.path == "/rag":
                q = (data.get("question") or "").strip()
                if not q:
                    self._send(422, {"error": "question required"}); return
                out = rag_answer(q, k=int(data.get("k", 3)))
                out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
                self._send(200, out)
            elif self.path == "/agents/run":
                goal = (data.get("goal") or "").strip()
                if not goal:
                    self._send(422, {"error": "goal required"}); return
                out = agent_run(goal, max_steps=int(data.get("max_steps", 6)))
                out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
                self._send(200, out)
            elif self.path == "/generate":
                msgs = data.get("messages") or [{"role": "user",
                                                  "content": data.get("prompt", "")}]
                out = LLM.chat(msgs, json_mode=bool(data.get("json_mode")))
                self._send(200, {"output": out, "provider": LLM.name,
                                 "ms": round((time.perf_counter() - t0) * 1000, 2)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:                              # never leak a stacktrace
            self._send(500, {"error": type(e).__name__, "detail": str(e)[:300]})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


START = time.time()

if __name__ == "__main__":
    print(f"API listening on http://{HOST}:{PORT}  provider={LLM.name} "
          f"model={MODEL_INFO['id']} index={len(STORE)}")
    print("Endpoints: GET /health | POST /predict /rag /agents/run /generate")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

