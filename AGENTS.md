# Repository Agent Instructions

## Project scope

- This checkout contains the Machine A integration, benchmark runner, and evaluation scripts. Machine B's EvicPress service is maintained in the separate `evicpress-core` repository; do not edit that repository unless the task asks for it.
- Treat `LMCache/` as a vendored/upstream tree. Do not modify its source, tests, or documentation unless a task explicitly requires a change there. Its own `AGENTS.md` contains the local instructions for that subtree.
- The current implementation uses vLLM/LMCache with a gRPC remote backend and tiered storage. Per-head placement and attention-mass-guided eviction are planned work, not current features.

## Source of truth

- Use the root `README.md` for the current architecture, file map, test coverage, and benchmark inventory.
- Use `Plan.md` only for the concise remaining-work roadmap and `PROGRESS_REPORT.md` for completed work and verification; do not present planned features as implemented.
- `benchmarks/manifest.json` defines benchmark inputs, profiles, and source revisions. Keep frozen input files and their `benchmarks/data/provenance.json` hashes consistent; if inputs change, update and explain provenance.

## Documentation

- Maintain only `README.md`, `Plan.md`, and `PROGRESS_REPORT.md` as project documentation, plus this `AGENTS.md` instruction file.
- Do not add another README, implementation report, plan, or progress note unless the user explicitly asks. Update the appropriate existing document instead.
- Keep `README.md` concise and implementation-focused, `Plan.md` limited to the agreed milestones, and `PROGRESS_REPORT.md` limited to completed work and verified status.

## Testing and benchmarks

- Run the local harness tests with `python -m unittest discover -s tests -p test_benchmark_suite.py -v` after changing the benchmark runner or its manifest.
- `make test-grpc MACHINE_B=HOST_OR_IP` runs the direct Machine B smoke check; `make run-stress` runs the LMCache/gRPC stress diagnostic. These require a reachable Machine B.
- `make run` launches the vLLM integration check and may require a GPU, model download, and Machine B. Do not launch it unless the task calls for runtime testing.
- The scripts in `tests/` are mostly manual diagnostics, not a comprehensive assertion-based correctness suite. Distinguish their output from the seven GPU-free benchmark-harness unit tests.
- All model-loading entry points use `model_settings.py` and the pinned manifest. Weight quantization is separate from GPU KV dtype and Machine B compression; keep GPU KV FP16 by default and preserve frozen tokenizer provenance when changing weights.
- Do not run paid GPU, AWS, or full remote benchmark matrices unless explicitly requested. The LongBench inputs are a 25-example-per-task pilot subset and are not paper-leaderboard results.
- When moving or renaming a test, update every relevant reference in `Makefile`, `benchmarks/manifest.json`, and the root `README.md`.

## Change discipline

- Keep changes scoped to the requested Machine A work; preserve unrelated and uncommitted user changes.
- Prefer the smallest targeted test set, and state clearly when remote or GPU validation was not run.
- Do not add credentials, private endpoints, generated model files, or benchmark results to source control.
- Before deleting or moving files, verify the exact target and preserve anything that may be user-authored or needed by another workflow.
