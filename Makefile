SHELL := /bin/bash
PYTHON     ?= python
VENV       := $(HOME)/venv/bin/activate
LMCACHE    := $(HOME)/distributed-tiered-kv-cache/LMCache
CFG        := lmcache_config.yaml
MACHINE_B  ?= 172.31.0.80
LOG        := $(HOME)/vllm.log

.PHONY: help setup proto run run-stress test-grpc logs status stop ping-b chat

help:         ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"}{printf "  %-14s %s\n",$$1,$$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:        ## Install LMCache from source + pip dependencies
	source $(VENV) && pip install -e $(LMCACHE) && pip install -r requirements.txt
	@echo "[setup] done"

PROTO_DIR  := $(LMCACHE)/lmcache/v1/storage_backend

proto:        ## (Re)generate gRPC Python stubs from evicpress.proto
	source $(VENV) && cd $(PROTO_DIR) && python -m grpc_tools.protoc \
		-I . \
		--python_out=. \
		--grpc_python_out=. \
		evicpress.proto
	@echo "[proto] stubs regenerated in $(PROTO_DIR)/"

# ── Run ────────────────────────────────────────────────────────────────────────

run:          ## Run the vLLM + LMCache end-to-end test (foreground)
	source $(VENV) && LMCACHE_CONFIG_FILE=$(CFG) PYTHONHASHSEED=0 $(PYTHON) run_vllm_hopefully.py

run-stress:   ## Run the gRPC stress test against Machine B
	source $(VENV) && LMCACHE_CONFIG_FILE=$(CFG) $(PYTHON) stress_test_grpc_backend.py

# ── Test / Debug ───────────────────────────────────────────────────────────────

test-grpc:    ## Direct gRPC smoke test against Machine B (no vLLM)
	source $(VENV) && MACHINE_B=$(MACHINE_B) $(PYTHON) smoke_test_b.py

ping-b:       ## TCP connectivity check to Machine B gRPC port
	@source $(VENV) && $(PYTHON) -c "import socket,os; \
a=os.environ.get('MACHINE_B','$(MACHINE_B)'); \
s=socket.socket(); s.settimeout(3); r=s.connect_ex((a,50051)); s.close(); \
print(f'[ping-b] {a}:50051 ->', 'OPEN' if r==0 else f'CLOSED (err={r})')"

status:       ## Show running LMCache/vLLM processes and GPU usage
	@echo "=== Processes ==="
	@ps aux | grep -E "[p]ython.*(vllm|run_vllm|lmcache)" || echo "  none"
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

logs:         ## Tail the vLLM log (Ctrl-C to exit)
	tail -f $(LOG)

chat:         ## Start interactive chat (LMCache + EvicPress backend)
	source $(VENV) && LMCACHE_CONFIG_FILE=$(CFG) PYTHONHASHSEED=0 MACHINE_B=$(MACHINE_B) $(PYTHON) chat.py

stop:         ## Kill any running vLLM python processes
	@pkill -f "python.*run_vllm" 2>/dev/null && echo "[stop] killed" || echo "[stop] nothing running"
