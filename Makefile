# AI Tutorial Monorepo — build/test/run automation
# Usage: make setup | make run M=11 | make run-all | make test | make serve | make ui
SHELL := /bin/bash
PY := python3
export PYTHONPATH := $(CURDIR)/packages:$(CURDIR)

.PHONY: setup run run-all test serve ui bench clean

setup: ## Verify environment (deps already present on this machine)
	$(PY) -c "import numpy,pandas,sklearn,scipy,matplotlib,pytest; print('env OK: all core deps present')"
	$(PY) -c "import importlib.util as u; print('optional torch:', 'present (accelerator examples enabled)' if u.find_spec('torch') else 'absent (NumPy-only path, m27 still passes)')"

run: ## Run one example: make run M=11
	$(PY) examples/m$(M)_*/main.py

run-all: ## Run every example
	./scripts/run_all.sh

test: ## Pytest smoke suite over all examples
	$(PY) -m pytest tests/ -q

bench: ## Benchmark NumPy vs PyTorch: make bench [DEVICE=mps]
	$(PY) benchmarks/bench_backends.py --device $(or $(DEVICE),cpu)

serve: ## Start the JSON API server (http://127.0.0.1:8000)
	$(PY) apps/api_server/server.py

ui: ## Serve the web UI (opens alongside make serve)
	$(PY) -m http.server 8080 --directory apps/web_ui

clean:
	rm -rf **/__pycache__ .pytest_cache artifacts/*.tmp
