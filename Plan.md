# Remaining Implementation Plan

This replaces the former week-by-week plan. It is the agreed four-milestone roadmap; detailed task breakdowns will be added separately only when requested.

## Milestone 1 — Verify and baseline

- Confirm the current implementation with existing smoke/correctness checks; resolve repository or environment issues and select the two-machine hardware setup.
- Resolve the vendored LMCache CUDA-extension build blocker (`lmcache.c_ops` is imported on GPU but its build is commented out), then verify a fresh Linux/GPU install.
- Verify the INT4-weight snapshot on the available GPU: confirm AWQ/Marlin loading and FP16 KV in the worker logs, then run the same plain/remote-FP16 workloads. Measure quality and memory/latency; 16 GB fit and numerical agreement with the earlier FP16-weight model have not been tested.
- Choose the benchmark set and freeze model, prompts, configuration, and measurement procedure. Record current quality and efficiency as the baseline.
- Finish the opt-in attention probe's **hardware verification** (implementation and CPU scoring check are in `PROGRESS_REPORT.md`): install this package on Machine A, run `python -m attention_probe.run --probe --enforce-eager` with one GPU and no remote backend, and confirm four nonzero per-step records, each query head's window mass sums to about 1, and generated output matches a probe-off run. Then remove `--enforce-eager` to check normal CUDA-graph operation. This has **not** been run locally: no CUDA GPU/vLLM runtime is available here.
- After Machine B is available, repeat with `--remote`; compare probe on/off at 1K/4K/8K contexts, reporting added latency and peak GPU memory. Check that retrieved resident chunks are scored correctly; remote-only/unloaded chunks have no observable attention mass. If the chosen GPU cannot run `FLASH_ATTN`, adapt the wrapper to its selected backend; if overhead is unacceptable, investigate fused window aggregation before Milestone 3. These GPU/two-machine checks and baseline benchmarks remain **unrun**.

**Exit:** the current setup is reproducible, baseline results exist, and the attention-instrumentation path and risks are understood.

## Milestone 2 — Per-head granularity

- Test configurations with remote storage disabled/enabled, FP16 storage, and selected quantization/alpha settings.
- Try QEvict and RedKnot on the same workloads if their code and hardware permit; record limitations and controlled cases where they underperform.
- Extend block identity, storage, and reconstruction to per-head segments while retaining the current utility policy. Add the required schema and correctness checks.
- Run the same benchmark suite and configurations as the baseline to measure the effect of granularity alone.

### Remaining validation — Implemented head mode

The code now supports `extra_config.grpc_granularity: chunk|head` (default `chunk`) and benchmark `--granularity`. One unit is K/V for one layer's shared KV head and a 256-token chunk (short final chunks supported). All positions remain visible, and the existing utility/bands apply independently per segment. This follows RedKnot's logical-segment/physical-storage separation, not its attention-output plugin or sparse attention.

- **Hardware gate:** resolve Milestone 1's `lmcache.c_ops` build issue; verify the pinned AWQ model with FP16 KV and this checkout's LMCache. Head mode currently rejects multi-GPU, MLA, layerwise/multi-group, P2P/NIXL/PD and async-loading configurations. CPU tests and real loopback gRPC passed; GPU integration has **not** run.
- **Correctness on GPU:** run cold/warm generation in chunk and head FP16 modes, including short chunks. Force a missing/corrupt head and canonical tier pressure; verify the lookup/retrieval lease, whole-parent miss/recomputation, prefix-boundary buffer cleanup, and unchanged output. Repeat with INT8/INT4 and measure quality; dequantization cannot restore lost FP16 information.
- **Memory/performance:** confirm actual CPU pool admission/eviction and staging limits under sustained load. A's existing MixedMemoryAllocator serves both head mirrors and parent assembly; no assembled-parent CPU write-back is allowed. Logs record per-client physical head-cache and returned-assembly peaks, not process RSS or combined live peaks. Measure process RAM and peak GPU allocation separately, plus serialization/metadata overhead and p50/p95 latency.
- **Comparisons:** retain A baseline `37d143c` (AWQ) / `7bf1a6f` (FP16 weights) and B baseline `207a894`; use paired feature commits for head mode. Run identical frozen workloads and effective byte budgets, FP16 first, then the same alpha/compression profiles; use fresh B directories for each run. Record both SHAs, topology and parent/head counters, payload bytes, RPCs and scores. For a local demo, run B on `127.0.0.1` using the same host's RAM/disk in both modes; this does not measure network performance.
- **Limits:** one unavailable head still invalidates a native vLLM chunk. This change does not reduce vLLM HBM allocation or implement per-head token masks, local/global classification, attention-output reuse, or attention-guided decisions. Access frequencies can remain similar because whole chunks are retrieved together. Run QEvict/RedKnot comparisons when their runtime/hardware permits; do not assume per-head storage improves performance.

**Exit:** per-head segmentation is correct, and its quality, latency, and resource effects are measured against the current-granularity baseline.

## Milestone 3 — Attention-guided decisions

- Measure attention distribution again after per-head segmentation using the same inputs and collection method.
- Integrate an online attention-mass signal, adapting QEvict-style decision logic to the project's multiple storage tiers and quantization levels.
- Run the same benchmarks and configurations again, comparing against Milestones 1 and 2.

**Exit:** attention measurements drive tier/precision decisions and their incremental quality/efficiency effect is recorded.

## Milestone 4 — Final verification and comparison

- Verify correctness and end-to-end integration; fix issues found.
- Repeat the selected benchmark suite across the relevant configurations. Compare task quality, latency/throughput, GPU memory, tier use, and transfer behavior.
- Select the best-supported configuration and explain its trade-offs and limitations.

**Exit:** the implementation and results are reproducible, comparisons are complete, and conclusions match the evidence.
