"""Register the probe only when explicitly enabled before vLLM starts."""

import os


def register() -> None:
    if os.environ.get("DSTN_ATTENTION_PROBE") != "1":
        return

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "attention_probe.backend.ProbeFlashAttentionBackend",
    )
