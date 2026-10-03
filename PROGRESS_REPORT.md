# Progress Report

Status as of 2026-10-03. Machine A and Machine B remain separate repositories.

## Direct-local adaptation — Milestones 1 and 2 completed

- Both checkouts use additive `feature/direct-local-baseline` branches from A `bf60c43` and B `3668180`. Original branches and distributed configurations are preserved. Record each current SHA with `git rev-parse HEAD`; commits are local, not published.
- B is installable as `evicpress-core` with a typed `LocalEvicPressClient` wrapping the existing manager, head groups and codecs. Added exclusive local writer locking, model/layout identity, fork rejection, thread-safe bounded prefetch, failure reporting and clean shutdown. Core imports neither gRPC nor service/dashboard modules.
- A shares tensor/mirror/head logic across `GRPCBackend` and `LocalEvicPressBackend`. Local mode creates one worker-owned manager, respects explicit plugin selection, rejects unsupported modes, fails startup on missing dependencies/configuration and has no remote fallback. Added standalone local configurations and tag-aware stable keys without changing untagged legacy keys.
- Fixed chunk invalidation leaving a stale CPU mirror and corrupt tensor handling. Shutdown releases backend-owned head buffers before closing their allocator. Local head counters now distinguish logical calls/payload from real RPC/network traffic; zero local RPCs are checked.
- Verification: **10 existing A unit tests, 10 new A adapter tests, and 11 B tests passed**. This includes real loopback chunk/head gRPC, FP16/INT8/INT4, warm/cold head restoration, read leases, directory ownership, key isolation, allocation exhaustion, corruption, prefetch failure and cleanup. Library installation was checked in an isolated temporary environment using Python 3.10.11 / CPU PyTorch 2.7.1; target CUDA/PyTorch versions are not verified.
- Adapter tests substitute CPU allocation/interface objects for GPU-dependent LMCache imports. No model loading, CUDA build, node/GPU tests or benchmark campaign was run. The existing benchmark suite/manifest/frozen prompts and pilot model remain unchanged, including its first-pass-only quality scoring. Official full evaluation and local runner profiles are still next work.

Head-mode companion B commit: `3668180` on `main`; pre-change B baseline: `207a894`. A's earlier runtime baseline is `37d143c`; use `git rev-parse HEAD` on each checkout to record the paired feature revisions. Commits are local; publication is not part of this change.

## Completed

- Implemented vLLM/LMCache → gRPC → EvicPress inclusive RAM/disk tiering, prefetch and FP16/INT8/INT4 backing. Fixed oversized canonical-store rejection and service/setup/restart issues on B.
- Added the manifest-driven suite: five profiles (plain, remote FP16, quantized alpha 0/1/5), pinned official RULER/LongBench adapters, six frozen inputs/provenance, long-document QA, per-run scores/latency/tier counters and isolated results. LongBench is a pilot subset, not leaderboard-comparable.
- Added the unit-first new-machine runner with optional setup, preflight/debug logs, separate single/remote runs, remote-fetch evidence checks and profile-level resume. Active loaders use pinned Mistral v0.3 AWQ INT4 **weights**, with FP16 activations/GPU KV and unchanged tokenizer/prompts. Earlier FP16-weight snapshot: `7bf1a6f`.
- Added the opt-in attention probe: logical paged keys, causal/GQA handling and post-softmax query-head/window mass for decode. It does not change native attention output or storage policy; GPU capture/overhead remain unverified.
- Added opt-in **head mode** across A/B: stable model/revision/layout namespaces, full-position head tables, independent `[2,T,D]` blobs for one layer/KV head, portable persistent descriptors, complete-group publication, read leases and bounded batch RPCs. Existing utility/bands operate per head, with parent-demand normalization and per-head K/V quantization scales. Prefetch preserves quantized backing; incomplete groups fail closed.
- A owns byte-bounded, pinned, reference-counted head mirrors in the existing mixed CPU pool; full-parent reconstruction shares that pool and bypasses automatic parent write-back. Actual T1 inventory reconciles B's ledger. Fixed retrieval-miss prefix-boundary/reference cleanup. Added `--granularity chunk|head`, capability checks, isolated run IDs, CSV head/parent/payload/RPC metrics and per-client allocation snapshots. Cold official runs require empty B tiers/zero demand counters; manual preflight uses a read-only ping. Default chunk mode remains available.

## Verified locally

- A: **10 tests passed** (7 harness, 1 attention scoring, 1 orchestration, 1 head split/gather/GQA/short-chunk/cache-budget/pinning check).
- B: **3 tests passed** (oversized store, head-group admission/lease/pressure/restart/INT4/missing-head checks, real loopback gRPC using A's actual head client). Warm and cold FP16 reconstruction matched exactly, and cache ownership returned to zero after cleanup.
- Syntax/protocol/CLI/dry-run checks and frozen-input hashes were checked; documented setup/benchmark entry points remain available. These are CPU checks, not live model or GPU performance results.

## Still unverified / remaining

- No live GPU/two-node benchmark campaign or measured baseline/granularity improvement exists. GPU AWQ loading/fit, end-to-end head recovery/recomputation, allocator pressure and network performance require hardware.
- Fresh GPU install is still blocked by vendored `LMCache/setup.py` disabling the required `lmcache.c_ops` build. Head support does not fix this unrelated build issue.
- Head mode retains all positions and reconstructs complete vLLM chunks; it does **not** provide sparse/per-head GPU attention, reduced HBM allocation, RedKnot attention-output reuse or attention-guided decisions. One unavailable head invalidates a chunk. CPU allocation snapshots are not whole-process RSS/GPU peaks. INT8/INT4 backing cannot recover original precision.
- Remaining hardware gates/comparisons and later attention-guided work are in `Plan.md`.
