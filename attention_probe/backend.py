"""Opt-in FlashAttention wrapper for one-request attention-mass experiments."""

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)

from .recorder import get_recorder
from .scorer import compute_window_mass, gather_visible_keys


class ProbeFlashAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Normal attention and its output remain entirely in vLLM's implementation.
        result = super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        if attn_metadata is None:  # vLLM's profiling pass
            return result

        recorder = get_recorder()
        if recorder.finished or not recorder.wants_layer(layer.layer_name):
            return result
        q_len = attn_metadata.num_actual_tokens
        phase = "decode" if q_len == 1 else "prefill"
        if phase == "prefill" and not recorder.include_prefill:
            return result

        # The first probe is intentionally restricted; unsupported modes fail
        # visibly instead of producing numbers with the wrong normalization.
        if attn_metadata.block_table.shape[0] != 1 or attn_metadata.use_cascade:
            raise RuntimeError("attention probe currently supports one request, no cascade")
        if self.dcp_world_size != 1 or self.alibi_slopes is not None or self.sinks is not None:
            raise RuntimeError("attention probe does not support DCP, ALiBi, or sinks yet")
        if kv_cache.dtype != query.dtype:
            raise RuntimeError("attention probe currently requires unquantized GPU KV")

        start = end = None
        if query.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
        seq_len = attn_metadata.max_seq_len  # Host int; avoids a per-step GPU sync.
        visible_keys = gather_visible_keys(kv_cache, attn_metadata.block_table[0], seq_len)
        mass = compute_window_mass(
            query[:q_len],
            visible_keys,
            window_size=recorder.window_size,
            scale=self.scale,
            causal=attn_metadata.causal,
            sliding_window=self.sliding_window,
            softcap=self.logits_soft_cap,
        )
        if end is not None:
            end.record()
        recorder.add(phase, seq_len, mass, (start, end) if start is not None else None)
        return result


class ProbeFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[ProbeFlashAttentionImpl]:
        return ProbeFlashAttentionImpl
