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
