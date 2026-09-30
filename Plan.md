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

### Next task — Per-head implementation plan

**Scope:** independently place, quantize, prefetch, and evict **one layer's one KV head for a token chunk**, with K/V together. Keep 256-token chunks and shorter final chunks. For GQA, segment shared KV heads, not individual query heads. vLLM's GPU layout/attention kernel and the decision policy remain unchanged; attention-guided decisions stay in Milestone 3.

1. **Baseline/configuration.** Preserve A's non-per-head `37d143c` on `vishrut-addition` (`7bf1a6f` uses FP16 weights). B's `main` is at `7d52118` with uncommitted service/setup fixes: snapshot those separately before editing overlapping files. Make separate feature commits in both repositories and record paired SHAs. Add `extra_config.grpc_granularity: chunk|head` to `lmcache_config.yaml`, default `chunk`; preserve legacy RPCs/behavior. Keep model, FP16 GPU KV, prompts, alpha, and placement bands fixed. Use separate cache namespaces/directories, not migration of old files.

2. **A: split/reassemble.** Add `LMCache/lmcache/v1/storage_backend/head_segments.py`: `make_segment_id()`, `split_chunk()`, `assemble_chunk()`. Actual `KV_2LTD` shape is `[2,L,T,H_kv*D]`; derive dimensions from runtime metadata, reshape to `[2,L,T,H_kv,D]`, and serialize contiguous `[2,T,D]` segments. A versioned descriptor records parent key, model/revision namespace, shape/dtype/format, token count, and expected `(layer,KV-head,ID)` entries. IDs include namespace, worker/shard, parent, layer, head, and token count. Update `grpc_backend.py`'s put/contains/get/remove and batched lookup/get, reassembling in descriptor order. Initially support single-GPU, non-layerwise, single-group FP16 only; reject unsupported layouts. Leave token hashing/GPU connectors unchanged.

3. **Protocol/completeness.** Extend B's `machine_b/proto/evicpress.proto`; synchronize A's copy and regenerate both stubs through the Makefiles, checking relative imports. Add batch store/lookup/retrieve/delete with descriptors and per-head status/tier; reuse repeated-ID prefetch. Bound batches by bytes, not one RPC/head. Add B `server/grpc_server.py` handlers and manager batch methods; persist descriptors in new `evicpress/head_groups.py` (B paths relative to `machine_b/`). Publish a parent only after every canonical write succeeds. Require all heads locally/remotely and validate shape/dtype/coordinates; missing/corrupt heads mean a whole-chunk miss, never zero-fill. Replace no-op pinning with bounded group leases protecting lookup-to-retrieve; verify miss/recomputation behavior, including races.

4. **A: genuine per-head Tier 1.** Add byte-bounded `HeadSegmentCache` retaining only Tier-1-admitted segments. In `storage_manager.py`, suppress assembled-parent write-back in `get()`/`batched_get()` for head mode; keep `LocalCPUBackend` for staging. Add a head-mode `MixedMemoryAllocator` path in `local_cpu_backend.py`: its current whole-chunk pages must not be allocated once per head. Share A's RAM budget between segments/staging and handle references, eviction, pinning, and error cleanup. Reconcile B's ledger with actual A admissions/evictions without deleting backing; ledger entries alone are not proof of availability. Account for actual FP16 T1 bytes, not compressed B bytes.

5. **B: same policy per segment.** Update `evicpress/block.py`, `manager.py`, `tier_ram.py`, `tier_disk.py` to preserve head/parent metadata through copies, eviction, and restart. Retain `U = (alpha * quality - size_bytes / tier_bandwidth) * (access_count + 1) / (total_accesses + 1)` and placement bands. Compute `_compute_quality()` independently per serialized head: zlib ratio over at most 64 KiB, clamped `[0.1,1]`; this is compressibility, not attention/fidelity. Separate head access counters from logical parent requests; use parent requests for the denominator, avoiding batch-size inflation, and expose both statistics. Reuse `quantize.py`'s codecs with per-head K/V scales. Preserve inclusive T3 backing, quantized-copy prefetch, and dequantization on retrieval; recheck parent completeness after head eviction. Lossy backing cannot restore original FP16 values.

6. **Minimal tests/hardware gate.** Add one focused A `tests/` file for split/reassembly, GQA, short chunks, reordered replies, and missing-head rejection with a fake service; include it in `run_tests.py`. Extend B's existing tests for batch rejection/completeness, restart, independent eviction, and one lossy round trip with tolerance; run them in B's checkout. On hardware, run gRPC/tensor smoke checks and cold/warm vLLM generation, forcing a head miss and tier pressure. FP16 round trips must be exact; compressed results need measured quality. Verify recomputation, leases, and byte limits. GPU/service checks remain pending; resolve Milestone 1's CUDA-extension blocker first.

7. **Reproducible comparisons.** Add `--granularity chunk|head` to `benchmarks/suite.py`/`run_tests.py`; update `make_a_config()`, preflight, metadata, and manifest profiles. Check B's capability; record both SHAs, topology, head/parent counters, bytes/RPCs, and actual tier/staging memory alongside scores/latency. Preserve frozen inputs. Compare FP16 first, then identical alpha/compression profiles and effective byte budgets; per-head scales also change quantization error. For local demonstration, manually start existing B on `127.0.0.1` with host RAM/disk in both modes: no new backend needed. T1/T2 share physical RAM; local results do not establish network performance. Keep local restart automation separate. After implementation, briefly update README/PROGRESS_REPORT and replace this section with remaining checks.

**Limit:** vLLM still needs all heads/layers for reuse; one unavailable head invalidates the chunk unless fully recoverable from another tier. This does not implement sparse attention or head-only recomputation. Jointly fetched heads may have similar access histories; improvements are not guaranteed. Measure serialization/metadata/RPC overhead too.

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
