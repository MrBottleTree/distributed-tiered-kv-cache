# Distributed Tiered KV Cache

This is the Machine A side of a distributed KV-cache prototype. vLLM uses LMCache to send and retrieve KV blocks over gRPC; the separate `evicpress-core` repository runs Machine B's tiered storage service. Remote reuse and tiered storage are implemented. Per-head placement and attention-mass-guided decisions are future work.

The default model is pinned [Mistral-7B-Instruct-v0.3 AWQ](https://huggingface.co/solidrust/Mistral-7B-Instruct-v0.3-AWQ/tree/95b1295ddd1a8673117cdc7bd2a4da2a457bb3f7): **4-bit model weights, FP16 activations and GPU KV cache**. The original tokenizer/revision and frozen prompts are unchanged. `run_tests.py` defaults remote runs to `remote_fp16`; selecting `remote_alpha_*` or `all` explicitly enables Machine B cache-compression experiments. Manual chats use B's existing policy, so disable B quantization separately for an FP16-only remote run. Quantized weights can change answers; use the same checkpoint across profiles and measure quality rather than assuming identical output. GPU fit/quality remain unverified. The previous FP16-weight snapshot is commit `7bf1a6f` (on a clean node checkout: `git switch --detach 7bf1a6f`).

```text
vLLM / Machine A -> LMCache gRPC backend -> EvicPress / Machine B -> RAM and disk tiers
```

## Project files

| Path | Purpose |
| --- | --- |
| `Makefile` | Setup, launch, and Machine B smoke/stress commands. |
| `run_tests.py` | New-machine setup/validation: unit tests, preflight diagnostics, and separate single/remote GPU benchmark runs with logs and resume. |
| `requirements.txt` | Machine A Python dependencies. |
| `model_settings.py` | Shared pinned model/tokenizer loading options; keeps weight and KV precision separate. |
| `lmcache_config.yaml` | LMCache configuration for the remote backend. |
| `chat.py`, `new_chat.py` | Interactive vLLM + LMCache demonstrations using different models. |
| `tests/` | Smoke checks, integration diagnostics, needle retrieval, and benchmark-runner unit tests. Details below. |
| `benchmarks/suite.py` | Manifest-driven benchmark preparation, preflight, single-profile runs, and five-profile matrix. |
| `benchmarks/manifest.json` | Pinned model/source revisions, test definitions, and profiles. |
| `benchmarks/machine_b_base.yaml` | Shared Machine B capacity/configuration baseline. |
| `benchmarks/requirements.txt`, `prepare-requirements.txt` | Runtime scoring and isolated data-preparation dependencies. |
| `benchmarks/data/` | Frozen RULER/LongBench inputs and `provenance.json` hashes. |
| `attention_probe/`, `pyproject.toml` | Optional one-request vLLM FlashAttention mass probe and its plugin registration. |
| `PROGRESS_REPORT.md` | Completed implementation work and verified status across the two-node project. |
| `Pipeline implementation 25th march/` | Archived earlier Python/gRPC KV-cache prototype; not used by the current runner. |
| `package.json`, `package-lock.json` | Ancillary JavaScript dependency metadata; not required for Python serving or benchmarks. |
| `Untitled.java` | Legacy scratch file; not used by the current runtime or test runner. |
| `.gitignore` | Excludes generated dependencies, caches, and benchmark results. |
| `.gitattributes` | Preserves frozen benchmark bytes/hashes across Windows and Linux clones. |
| `AGENTS.md` | Repository instructions for coding agents. `Plan.md` is the concise remaining-work roadmap, not runnable code. |

The `LMCache/` tree is an upstream/vendor checkout with its own code, tests, and documentation. It is intentionally unchanged and is not enumerated here.

## What current tests cover

| Test | Coverage |
| --- | --- |
| `tests/smoke_test_b.py` | Direct gRPC miss/store/hit/retrieve round trip with Machine B. |
| `tests/test_grpc_backend.py` | Manual LMCache backend tensor put/get diagnostic. |
| `tests/stress_test_grpc_backend.py` | Multi-chunk/prefix fetches, repeated lookup, batched contains, and timing. |
| `tests/run_vllm_hopefully.py` | Repeated-prompt vLLM/LMCache integration smoke check. |
| `tests/needle_haystack.py` | Retrieval quality/latency at different needle positions; expected HIT/MISS labels are not proof of remote fetch. |
| `tests/test_benchmark_suite.py` | Seven GPU-free tests of model/KV settings, tokenizer provenance, profiles, config rendering, metrics, counters, and result output. |
| `tests/test_attention_probe.py` | One synthetic CPU check of paged-key order, GQA, causal masking, and window mass. |
| `tests/test_new_machine_runner.py` | One GPU-free check of stage order, five-profile isolation, and passed/failed resume behavior. |

The first five are runnable diagnostics, not a comprehensive automated correctness suite. Remote checks need Machine B; the three unit-test files do not need a GPU.

## Benchmarks and profiles

`benchmarks/suite.py` includes:

- Official-source RULER NIAH: single-needle/multi-key at 4K/8K, 20 samples per task.
- LongBench Qasper and 2WikiMQA: 25 frozen samples each, a pilot subset (not leaderboard-comparable).
- LMCache `long_doc_qa`, plus the five project diagnostics above.

Profiles: `plain`, `remote_fp16`, and remote quantization with alpha 0, 1, or 5. Results include quality where supported, latency/throughput, configuration, and Machine B counters, saved under `benchmarks/results/`.

## First-time setup

The serving setup is for Linux; Machine A needs a compatible NVIDIA GPU/driver. Clone the two repositories separately. On A, select the branch containing this runner:

```sh
git clone --branch vishrut-addition https://github.com/MrBottleTree/distributed-tiered-kv-cache.git
cd distributed-tiered-kv-cache
git rev-parse HEAD       # retain this revision with the experiment results
```

**Known GPU startup blocker:** the vendored `LMCache/setup.py` currently comments out the `lmcache.c_ops` CUDA extension, although its GPU runtime imports that module. A fresh editable install therefore does not build this required extension. Resolve that build issue before remote GPU runs; local unit-test success does not establish GPU readiness. The vendored tree has not been changed in this update.

On Machine B:

```sh
git clone https://github.com/MrBottleTree/evicpress-core.git
cd evicpress-core/machine_b
make setup
# Set tier capacities/data directory in config/config.yaml; keep ports 50051 and 8080 private to the A/B network.
make run                 # foreground; use make start for background
make ping                # B-local gRPC check, in a second terminal when using make run
```

### Recommended new-machine entry point

Run `run_tests.py` **on Machine A**, from the repository root. It wraps the existing official-source benchmark runner; it does not create EC2 instances or install/start Machine B. Use Linux, Python 3.10–3.13, a working NVIDIA driver, Git, a C++ compiler, and sufficient model/cache disk space. On a minimal Ubuntu image, install prerequisites first: `sudo apt-get update && sudo apt-get install -y git build-essential python3-venv`. GPU serving still needs a compatible driver/runtime image.

```sh
python3 run_tests.py --mode both --config hardware.json --dry-run  # inspect stages without executing
# Choose one GPU workflow (both already includes the single-machine baseline):
python3 run_tests.py --setup --mode single                         # first GPU node: unit tests, then all seven workloads
python3 run_tests.py --setup --mode both --config hardware.json     # first two-node campaign: plain + remote FP16
python3 run_tests.py --mode both --config hardware.json --remote-profile all  # explicit five-profile cache experiment
python3 run_tests.py --mode unit                                  # nine local tests; no GPU/network/B required
```

`--setup` creates `.validation-venv`, installs pinned serving/scoring dependencies, then installs **this checkout's** LMCache and the optional probe package. Later invocations automatically use that environment; shell activation is unnecessary. Setup is explicit and does not install drivers or alter system packages. Without it, an existing environment must already contain the dependencies (CPU tests need PyYAML and PyTorch).

For manual Python commands below, activate `source .validation-venv/bin/activate` if using this setup; for Makefile commands also pass `VENV="$PWD/.validation-venv/bin/activate" LMCACHE="$PWD/LMCache"` to override their old defaults.

For the two-node run, create an ignored `hardware.json` on A, replacing the example host and absolute B paths. This example keeps remote KV in FP16; override `--remote-profile all` only for cache-compression comparisons:

```json
{
  "b_host": "MACHINE_B_PRIVATE_IP",
  "b_ssh": "ubuntu@MACHINE_B_PRIVATE_IP",
  "b_config_path": "/opt/evicpress-core/machine_b/config/benchmark.yaml",
  "b_repo_path": "/opt/evicpress-core",
  "b_restart_command": "cd /opt/evicpress-core/machine_b && make restart CONFIG=config/benchmark.yaml",
  "remote_profile": ["remote_fp16"]
}
```

B must already be running and reachable on private ports `50051` and `8080`; configure noninteractive SSH and its verified host key beforehand. These explicit SSH settings authorize configuration upload and B restart for each remote profile, with a fresh disk directory. Without SSH, select one profile and pass `--b-host HOST --b-data-dir /data/kv_cache/FRESH_RUN`; manually restart B with that matching profile/directory first. `--mode single` never contacts B; `--mode remote` omits the plain baseline. Remote mode defaults to `remote_fp16`; `--remote-profile all` explicitly selects all four remote profiles, including KV-compression experiments.

Stages are: setup if requested → nine unit tests → dependency/import, CUDA execution and free-port checks → pinned scorer sources/frozen-input hashes and scorer calls → B connectivity/real gRPC round trip when requested → pinned model download → benchmarks. Failures stop immediately; no automatic paid retries. Attention probing is disabled for these baseline runs. The default workloads are four RULER NIAH tasks, two LongBench pilot subsets, and LMCache long-document QA, using the existing manifest/model and two cold/warm prompt repeats. Old manual model/needle diagnostics are not all rerun automatically; they are available below and are not official benchmarks.

```sh
python3 run_tests.py --mode single --test ruler_niah_single_4k     # focused first GPU pass
python3 run_tests.py --mode both --config hardware.json --preflight-only  # no model download/benchmark
python3 run_tests.py --resume benchmarks/results/validation/CAMPAIGN_ID   # retry after environment/service fix
```

Results are separated into `single/PROFILE/` and `remote/PROFILE/` beneath the printed campaign directory; `comparison.csv` combines metrics from passed profiles. Keep `state.json`, `environment.json`, per-attempt `logs/`, failure details, and each benchmark's `run.json`, predictions, `summary.json`/`summary.csv`, and `vllm.log`. They contain configuration/provenance, quality scores, latency/throughput, and B fetch/tier counters. Remote workloads with no Tier 2/3 hits stop validation **even if generation succeeded**; investigate local residency/B capacity before claiming remote reuse. `--allow-no-remote-hits` retains them as explicitly warned diagnostic results, not verified remote-fetch measurements.

Resume reruns units/preflight but skips successful benchmark **profiles** after verifying their summaries; a failed profile restarts in full, never merging partial cold/warm samples. Code/config/input changes require a new campaign. Change `--timeout-minutes` (default 60 per stage) on resume if a legitimate stage was too slow. Ctrl-C/timeouts interrupt only this runner's process group, not unrelated jobs; B is left running and results are preserved. A power/process crash may leave `.running`: check its recorded PID before removing that lock. This orchestrator is locally tested; actual GPU/two-node runs remain unverified.

### Existing Makefile setup

The older `make setup` only installs dependencies; it is not the validated new-machine workflow. Prefer `run_tests.py --setup --mode unit`, which installs dependencies before the local checkout and avoids replacing it with the historical LMCache Git pin. If maintaining the old environment manually, install PyTorch/build prerequisites first:

```sh
python3 -m venv "$HOME/venv"
source "$HOME/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0
make setup VENV="$HOME/venv/bin/activate" LMCACHE="$PWD/LMCache"
python -m pip install -e ./LMCache --no-deps --no-build-isolation  # restore this checkout
python -m pip install -r benchmarks/requirements.txt
python -m pip install -e . --no-deps --no-build-isolation  # register the optional attention probe
```

The A Makefile defaults to `/home/ubuntu/venv` and `$HOME/distributed-tiered-kv-cache/LMCache`; override `VENV` and `LMCACHE` as above if your user or clone path differs. Root `requirements.txt` also pins an LMCache Git revision, so install `./LMCache` editable after `make setup` to ensure the local integration is used.

Edit `lmcache_config.yaml` so `extra_config.grpc_server` is B's private address on port `50051`. Then fetch/verify the pinned benchmark scorer sources and list tests:

```sh
python benchmarks/suite.py prepare --sources-only
python benchmarks/suite.py list
```

## Test and smoke-test commands

There is no single `make test` target; `python3 run_tests.py --mode unit` runs the local unit checks together. Individually:

```sh
python -m unittest discover -s tests -p test_benchmark_suite.py -v
python -m unittest discover -s tests -p test_attention_probe.py -v
python -m unittest discover -s tests -p test_new_machine_runner.py -v
```

The CPU test constructs fake KV pages in shuffled physical order. It checks that the probe rebuilds logical token order, maps grouped-query heads to KV heads, applies a causal mask, and assigns uniform attention over 300 tokens to 256-token windows as `256/300` and `44/300`. It does **not** run vLLM, measure live attention, contact Machine B, or measure speed.

The GPU smoke command loads the pinned Mistral model for one request. With `--probe`, normal FlashAttention runs first; an extra Q/K softmax then calculates window mass **during each decode step** for one selected layer (default 0). It uses live Q/K but reconstructs probabilities independently rather than reading FlashAttention's internal values. It writes per-step and cumulative per-query-head mass, auxiliary GPU time (`probe_gpu_ms`), and peak worker GPU allocation to `benchmarks/results/attention_probe_*.json`. Prefill scoring is off unless `--include-prefill` is passed. The scores do not affect eviction/tiering; the command does not assert numerical parity, compare generated tokens with probe-off, prove a remote cache hit, or report isolated end-to-end per-token latency (`wall_s` covers the whole generation). GPU behavior remains unverified until run on hardware.

On GPU Machine A, after the setup above, run the no-remote check first:

```sh
python -m attention_probe.run --enforce-eager          # probe-off comparison
python -m attention_probe.run --probe --enforce-eager  # expect four decode-step records
python -m attention_probe.run --probe                  # check normal CUDA-graph mode
```

Inspect the printed result path: each decode event should have nonzero mass and each query head's window values should sum to about 1. Use the same prompt/settings for probe-off timing; repeat both modes with `--prompt-tokens 4000 --max-model-len 8192 --decode-steps 32` for an initial overhead check. To check the connector later, start B, set its address in `lmcache_config.yaml`, then on A run:

```sh
export LMCACHE_CONFIG_FILE="$PWD/lmcache_config.yaml"
python -m attention_probe.run --remote --enforce-eager          # warm/store the prompt
python -m attention_probe.run --remote --probe --enforce-eager  # repeat; verify B hit counters/logs separately
```

On B, run `make stats` or inspect its logs to confirm a remote hit; `--remote` alone is not proof.

Use `--chunk-size N` if LMCache is not using its 256-token default. The probe requires a GPU/backend supporting `FLASH_ATTN`, one request, and unquantized GPU KV. These smoke runs are not a replacement for the benchmark suite.

For a complete current A/B diagnostic pass, run the relevant commands below after B is running. These scripts are manual integration checks, not a comprehensive automated correctness suite; the backend smoke and vLLM checks require the remote service, and the latter also needs a GPU/model download.

```sh
export MACHINE_B=10.0.0.5
make ping-b MACHINE_B="$MACHINE_B"             # TCP port check only
make test-grpc MACHINE_B="$MACHINE_B"           # gRPC miss/store/hit/retrieve round trip
MACHINE_B="$MACHINE_B" python tests/test_grpc_backend.py
MACHINE_B="$MACHINE_B" make run-stress          # multi-chunk/prefix-fetch diagnostic
make run                                         # repeated-prompt vLLM/LMCache smoke
MACHINE_B="$MACHINE_B" python tests/needle_haystack.py
```

`python -m unittest discover -s tests` is intentionally not the documented command: `tests/test_grpc_backend.py` is a manual script with top-level execution, not a unittest module.

## Chat and backend verification

- `make new_chat MACHINE_B=10.0.0.5` and `make chat MACHINE_B=10.0.0.5` use the same pinned 4-bit Mistral through vLLM and LMCache. Both need a GPU, a running B service, and the correct `grpc_server` in `lmcache_config.yaml`.
- For a vLLM-only one-shot check, with LMCache not enabled in the shell, run: `python -c 'from vllm import LLM, SamplingParams; from model_settings import model_options; o=model_options(); o["max_model_len"]=2048; m=LLM(**o); print(m.generate(["Say hello."], SamplingParams(max_tokens=16))[0].outputs[0].text)'`.
- `make ping-b MACHINE_B=10.0.0.5` only proves TCP port `50051` is reachable. `make test-grpc MACHINE_B=10.0.0.5` verifies actual gRPC storage/retrieval. `python benchmarks/suite.py check --profile remote_alpha_1 --b-host 10.0.0.5` runs benchmark dependency/B preflight.
- Other A targets: `make help`, `make setup`, `make proto`, `make status`, `make logs`, `make stop`. `make logs` tails `$HOME/vllm.log` only if you have redirected a manual run there; benchmark logs are in their result directories. `make proto` regenerates stubs inside `LMCache/`; use only when intentionally changing its protocol. On B: `make help`, `make setup`, `make run`, `make start`, `make restart CONFIG=config/config.yaml`, `make stop`, `make status`, `make stats`, `make ping`, and `make logs`.

## Benchmark commands

Run one test or all official workloads for a profile. A remote single-profile run assumes B is already configured for that profile and uses a fresh disk directory:

```sh
python benchmarks/suite.py run --profile plain --test ruler_niah_single_4k
python benchmarks/suite.py check --profile remote_alpha_1 --b-host 10.0.0.5
python benchmarks/suite.py run --profile remote_alpha_1 --test ruler_niah_single_4k --b-host 10.0.0.5 --b-data-dir /data/kv_cache/alpha1_run1
```

Omit `--test` to run the official workload set for a profile. Use `--test NAME` more than once to select several workloads. For a complete five-profile matrix, the runner manages B by SSH, restarts it between profiles, and chooses fresh B data directories; configure the SSH target and absolute B config/repo paths for your machines:

```sh
python benchmarks/suite.py matrix --b-host 10.0.0.5 --b-ssh ubuntu@10.0.0.5 --b-config-path /opt/evicpress-core/machine_b/config/benchmark.yaml --b-repo-path /opt/evicpress-core --b-restart-command 'cd /opt/evicpress-core/machine_b && make restart CONFIG=config/benchmark.yaml'
```

For one manually configured remote profile, generate and install its B configuration, then run the preflight:

```sh
python benchmarks/suite.py render-b --profile remote_alpha_1 --data-dir /data/kv_cache/alpha1_run1 --output /tmp/evicpress-alpha1.yaml
scp /tmp/evicpress-alpha1.yaml ubuntu@10.0.0.5:/tmp/evicpress-alpha1.yaml
ssh ubuntu@10.0.0.5 'cd evicpress-core/machine_b && make restart CONFIG=/tmp/evicpress-alpha1.yaml'
python benchmarks/suite.py check --profile remote_alpha_1 --b-host 10.0.0.5
```

The matrix is the full GPU benchmark campaign and can be costly. Results are written to `benchmarks/results/`; retain those folders and `matrix_*.csv` for comparisons. `python benchmarks/suite.py --help` lists all options. Project documentation is limited to this README, `Plan.md` (remaining work), and `PROGRESS_REPORT.md` (completed work and verification); no live two-machine/GPU benchmark results are included yet.
