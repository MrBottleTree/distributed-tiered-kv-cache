# Developer Guide: Distributed Tiered KV Cache (vLLM + LMCache + EvicPress)

## System Overview
This project implements a disaggregated KV cache for LLMs, decoupling **Compute (Machine A)** from **Storage (Machine B)**.

### Key Components
- **vLLM:** The primary inference engine on Machine A.
- **LMCache:** Modified to handle deterministic routing and gRPC communication.
- **EvicPress:** The control logic on Machine B managing 3-tier storage and semantic eviction.

## Technical Specs
- **Communication:** gRPC bidirectional streaming.
- **Routing:** $O(1)$ deterministic hash-based block mapping.
- **Tiers:** 
  - Tier 1: Machine A RAM (Statically Partitioned)
  - Tier 2: Machine B RAM
  - Tier 3: Machine B Disk (NVMe)
- **Concurrency:** State-lock mechanism on Machine A for `PENDING_DEMAND_FETCH` and `RECEIVING_PREFETCH`.

## Development Workflow
- **Machine A:** Focus on vLLM integration and LMCache routing logic.
- **Machine B:** Focus on EvicPress metadata management and compression/eviction algorithms.
- **Testing:** Use the `Makefile` for building and running integration tests.

## Code Conventions
- Follow standard Python (PEP 8) and C++ (Google style) where applicable.
- Ensure all gRPC operations are asynchronous and handle timeouts/retries.
- Document all new LMCache hooks in `GEMINI.md`.
