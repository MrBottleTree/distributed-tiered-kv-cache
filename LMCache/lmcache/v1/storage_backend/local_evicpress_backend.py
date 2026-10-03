# SPDX-License-Identifier: Apache-2.0
"""Worker-owned direct EvicPress plugin, with no backend server or fallback."""

# Standard
from typing import Any

# Third Party
import torch

# Local
from .evicpress_backend import EvicPressBackend
from .evicpress_transport import LocalTransport


class LocalEvicPressBackend(EvicPressBackend):
    """Use Machine A RAM/disk while leaving native GPU attention unchanged."""

    def create_transport(self, config: Any, metadata: Any) -> LocalTransport:
        """Open a model-specific runtime; reject scheduler/unsupported modes.

        Missing package/config and identity conflicts propagate at startup.
        """
        extra = config.extra_config
        if (
            metadata is None
            or metadata.role == "scheduler"
            or metadata.world_size != 1
            or metadata.use_mla
            or metadata.get_num_groups() != 1
            or metadata.kv_dtype != torch.float16
            or config.enable_scheduler_bypass_lookup
            or config.enable_async_loading
            or config.enable_pd
            or config.enable_p2p
            or config.use_layerwise
            or extra.get("enable_nixl_storage")
        ):
            raise ValueError(
                "local EvicPress requires a single worker, worker-routed lookup, synchronous non-layerwise loading; PD/P2P/NIXL unsupported"
            )
        config_path = extra.get("evicpress_config_path")
        revision = extra.get(
            "evicpress_model_revision", extra.get("grpc_model_revision")
        )
        if not config_path or not revision:
            raise ValueError(
                "local EvicPress requires evicpress_config_path and a pinned model revision"
            )
        identity = {
            "schema_version": 1,
            "model": metadata.model_name,
            "model_revision": revision,
            "world_size": metadata.world_size,
            "kv_shape": list(metadata.kv_shape),
            "kv_dtype": str(metadata.kv_dtype),
            "granularity": "head" if self.manages_head_cache else "chunk",
        }
        return LocalTransport(config_path, identity)
