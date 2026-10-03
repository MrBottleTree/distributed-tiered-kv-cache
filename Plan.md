# Remaining Implementation Plan

Updated 2026-10-03. This is the direct-local baseline adaptation agreed for the current phase. The existing head-storage implementation remains available; attention-guided policy and matched QEvict/RedKnot implementations are later research work.

## Milestone 1 — Local EvicPress library (completed, CPU-verified)

- Installable core package and typed local client around the existing manager/codecs.
- One directory owner, model/layout identity, fork rejection, thread-safe bounded prefetch and clean lifecycle.
- Existing gRPC service retained; core library does not import or start the service.

## Milestone 2 — Direct LMCache adapter (completed, CPU-verified)

- Shared chunk/head logic with local and existing remote transports.
- One manager in the GPU worker; scheduler lookup uses existing worker routing.
- Explicit backend selection, required initialization, no remote fallback, tag-aware keys and standalone local configurations.
- CPU tests exercise real storage/codecs/adapters with CPU allocation stand-ins; full GPU integration is not verified.

## Milestone 3 — Target-node configuration and hardware readiness (pending)

- Pin Mistral-7B-Instruct-v0.2 model/tokenizer at `63a8b081895390a26e140280378bc85ec8bce07a`, FP16 weights/activations/KV. Preserve the current AWQ pilot separately.
- Set the common 32,768-token total limit, batch/concurrency one, chunked prefill and identical serving settings for all profiles.
- Increase research staging capacity: the synchronous retrieval path holds the full approximately 4 GiB 32K KV before GPU copying; the current 2 GiB pool is insufficient. Measure physical mirrors, process RAM, live/reserved KV and peak GPU memory separately.
- Resolve the vendored CUDA-extension build blocker, then verify Linux installation, model fit and short/8K/32K complete restoration on the 24 GB GPU / 64 GB RAM node. Do not silently shorten context or quantize weights to make the final baseline fit.

## Milestone 4 — Official benchmark suite and results (next code phase)

Implement the runner changes before connecting to the node; execute only after the hardware gates above pass.

- Separate research manifest and frozen inputs, leaving pilot inputs/provenance unchanged. Add plain-vLLM, local-FP16 and local-current-compression profiles; a small matched loopback transport control is supplementary.
- Full 13-task RULER at 32K with 500 examples/task and the 12-task LongBench evaluation used in QEvict's table, every test example. Use pinned official preparation/scoring, task prompts/generation/post-processing and tokenizer-aware limits; no fitting-only subsets.
- Isolate examples with run/sample tags, wait for writes and independently score first and replay answers against references. Report real restored/recomputed tokens, source tiers and backing precision; repeat agreement is not accuracy.
- Freeze policy/settings before evaluation. Save predictions, complete counts, both repository SHAs, input/config hashes and environment/quality/latency/resource metrics. Report every omission, error, OOM and experimental deviation.
- Explicitly disclose Mistral-versus-Llama RULER model differences, unspecified paper settings, hardware/engine/precision/batching, replay as an additional protocol and CPU/disk resources. This is a full-cache baseline, not a paper-number reproduction or QEvict-style 5%/10%/20% active-KV experiment.

## Later research

Attention-mass-driven per-head residency/compression, reduced active GPU KV and matched competing-method runs are separate changes. Head segmentation alone preserves all positions and reconstructs complete native chunks; it does not implement sparse attention, QEvict recovery/routing or RedKnot attention-output reuse. Current utility is still the compressibility/frequency/bandwidth proxy, not attention history.
