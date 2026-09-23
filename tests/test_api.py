"""API smoke test: boots the server as a subprocess and exercises every endpoint."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PORT = _free_port()          # dynamic: never collides with a dev server on 8000/8765


def _req(path, payload=None, method=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:       # 4xx/5xx are data, not exceptions, here
        return e.code, json.loads(e.read() or b"{}")


def test_all_endpoints():
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/packages:{ROOT}", PORT=str(PORT))
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "apps" / "api_server" / "server.py")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    try:
        for _ in range(50):                                # wait for boot (warm index)
            time.sleep(0.2)
            try:
                status, health = _req("/health")
                break
            except Exception:
                continue
        else:
            raise AssertionError("server never came up: " + proc.stderr.read()[-500:])

        assert status == 200 and health["status"] == "ok"
        assert health["index_size"] == 8

        # model may not exist yet in a fresh checkout -> register one first
        if health["model"] is None:
            subprocess.run([sys.executable, str(ROOT / "examples/m21_mlops/main.py")],
                           env=env, capture_output=True, timeout=120)
            status, health = _req("/health")
        assert health["model"] is not None, "expected a registered model after m21"

        st, out = _req("/predict", {"features": [0.1, -0.4, 1.2, 0.8, -1.0, 0.3]})
        assert st == 200 and 0 <= out["score"] <= 1 and out["model"]

        st, out = _req("/predict", {"features": [1, 2]})
        assert st == 422                                   # schema validation

        st, out = _req("/rag", {"question": "How does LoRA fine-tune a frozen model?"})
        assert st == 200 and out["citations"] and "answer" in out

        st, out = _req("/agents/run", {"goal": "What is 12 * 8?"})
        assert st == 200 and "96" in out["answer"] and out["tool_calls"] >= 1

        st, out = _req("/generate", {"prompt": "summarize retrieval augmented generation"})
        assert st == 200 and out["output"]

        st, out = _req("/nope")
        assert st == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
