"""Worker-local, one-request recorder; no per-token CPU copy or file write."""

import atexit
import json
import os
from pathlib import Path
import re

import torch


class ProbeRecorder:
    def __init__(self) -> None:
        self.window_size = int(os.getenv("DSTN_ATTENTION_PROBE_CHUNK_SIZE", "256"))
        self.target_layer = os.getenv("DSTN_ATTENTION_PROBE_LAYER", "0")
        self.include_prefill = os.getenv("DSTN_ATTENTION_PROBE_PREFILL") == "1"
        self.max_decode_steps = int(os.getenv("DSTN_ATTENTION_PROBE_STEPS", "4"))
        self.output = Path(
            os.getenv(
                "DSTN_ATTENTION_PROBE_OUTPUT",
                "benchmarks/results/attention_probe.json",
            )
        )
        if self.window_size <= 0 or self.max_decode_steps <= 0 or not self.target_layer.isdigit():
            raise ValueError("probe chunk size/steps must be positive and layer numeric")
        self.events: list[tuple[str, int, torch.Tensor]] = []
        self.timers: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.running: torch.Tensor | None = None
        self.decode_steps = 0
        self.finished = False
        atexit.register(self.flush)

    def wants_layer(self, layer_name: str) -> bool:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name)
        if match is None:
            raise RuntimeError(f"cannot identify layer number in {layer_name!r}")
        return match.group(1) == self.target_layer

    def add(
        self,
        phase: str,
        seq_len: int,
        mass: torch.Tensor,
        timer: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    ) -> None:
        if self.finished:
            return
        # Prefill can contain several query rows; QEvict sums their mass.
        step_mass = mass.detach().sum(dim=0)
        if self.running is None:
            self.running = torch.zeros_like(step_mass)
        if step_mass.shape[-1] > self.running.shape[-1]:
            self.running = torch.nn.functional.pad(
                self.running, (0, step_mass.shape[-1] - self.running.shape[-1])
            )
        self.running[..., : step_mass.shape[-1]] += step_mass
        self.events.append((phase, seq_len, step_mass))
        if timer is not None:
            self.timers.append(timer)
        if phase == "decode":
            self.decode_steps += 1
            if self.decode_steps >= self.max_decode_steps:
                self.flush()

    def flush(self) -> None:
        if self.finished or not self.events:
            return
        self.finished = True
        elapsed_ms = []
        for start, end in self.timers:
            end.synchronize()  # One final synchronization, not one per token.
            elapsed_ms.append(start.elapsed_time(end))
        result = {
            "schema_version": 1,
            "backend": "FLASH_ATTN",
            "batch_row": 0,
            "window_size": self.window_size,
            "target_layer": self.target_layer,
            "events": [
                {
                    "phase": phase,
                    "seq_len": seq_len,
                    "mass_by_query_head_and_window": mass.cpu().tolist(),
                }
                for phase, seq_len, mass in self.events
            ],
            "cumulative_mass": self.running.cpu().tolist(),
            "probe_gpu_ms": elapsed_ms,
            "peak_gpu_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


_recorder: ProbeRecorder | None = None


def get_recorder() -> ProbeRecorder:
    global _recorder
    if _recorder is None:
        _recorder = ProbeRecorder()
    return _recorder
