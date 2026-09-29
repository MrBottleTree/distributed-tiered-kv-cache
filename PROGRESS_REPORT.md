# Progress Report

Status as of 2026-09-29. Machine A and Machine B remain separate repositories.

## Completed

- Added a manifest-driven Machine A benchmark runner with five profiles: plain, remote FP16, and remote quantization at alpha 0, 1, and 5.
- Added pinned RULER and LongBench adapters, a long-document QA workload, six frozen benchmark input files with provenance, and per-run quality/latency/resource/fetch reporting.
- Grouped the project's smoke and integration diagnostics under `tests/`, updated the Makefile and manifest paths, and repaired the needle-test script so it is valid Python.
- In the Machine B companion repository, fixed oversized Tier 3 store rejection reporting and updated the service response, setup, configuration-aware restart, and dependency pins.
- Added an opt-in Machine A FlashAttention probe as a project-local vLLM plugin. It gathers logical keys from paged KV, computes post-softmax mass per query head and LMCache-sized token window during decode (optional chunked prefill), accumulates on GPU, and writes a small result after the requested steps. It leaves the normal attention output and eviction policy unchanged; this pilot supports one request/layer and unquantized GPU KV.
- Added `run_tests.py` as the new-machine entry point: optional isolated dependency setup, unit-first preflight, pinned model/scorer preparation, separate plain/remote benchmark profiles, strict remote-fetch evidence checks, combined metrics, per-stage logs/timeouts, and profile-level resume. It reuses the existing benchmark suite; it does not provision nodes. Captured subprocess errors now reach stage logs.

## Verified

- Machine A benchmark-harness tests: 6 passed locally.
- Machine B oversized-store regression test: 1 passed locally.
- Syntax, manifest, and whitespace checks passed after the test-file move.
- Attention-probe CPU correctness test: 1 passed (paged-key order, GQA, causal masking, and 256-token window sums).
- The probe package built as a wheel; syntax and command-line checks passed. The six benchmark-harness tests still pass.
- New-machine runner: eight local tests passed (six harness, one attention, one orchestration check); unit-mode execution/resume and five-profile dry-run checked. Linux dependency installation and live GPU/two-node orchestration remain unverified.
- Pre-commit verification reran all eight local tests successfully and checked documented CLI/Makefile entry points. README now specifies the working branch and distinguishes legacy setup from the new-machine runner.

These checks are local; the attention plugin has **not** been run inside a GPU vLLM worker, so live capture and overhead are still unverified. Remote diagnostics require Machine B. **No two-machine/GPU benchmark campaign has been run, so there are no measured baseline scores yet.** The RULER/LongBench sets are configured, but LongBench uses a small pilot subset and is not leaderboard-comparable.

Code inspection found a fresh-install GPU blocker: vendored LMCache disables building `lmcache.c_ops` while its GPU path imports it. The vendor tree is unchanged; this must be resolved before claiming remote GPU readiness.

## Not implemented yet

Per-head KV segmentation, attention-mass-guided placement/precision decisions, and QEvict/RedKnot comparative experiments remain future work in `Plan.md`.
