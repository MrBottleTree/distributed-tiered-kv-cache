# SPDX-License-Identifier: Apache-2.0
"""Backend selection without importing GPU runtime or manager dependencies."""

# Standard
from typing import Any


def configure_evicpress_plugin(config: Any) -> None:
    """Respect explicit plugins; keep gRPC only as the legacy unspecified default.

    ``extra_config.evicpress_backend`` optionally selects ``local`` or ``grpc``.
    An explicit empty plugin list disables injection. Local selection forbids
    other storage transports and requires initialization to succeed.
    """
    if config.extra_config is None:
        config.extra_config = {}
    extra = config.extra_config
    mode = extra.get("evicpress_backend")
    if mode not in (None, "local", "grpc"):
        raise ValueError("evicpress_backend must be local or grpc")
    selected = config.storage_plugins
    if mode is not None:
        expected = "evicpress_local" if mode == "local" else "grpc"
        if selected is not None and list(selected) != [expected]:
            raise ValueError(
                f"evicpress_backend={mode} requires storage_plugins: [{expected}]"
            )
        selected = [expected]
    elif selected is None:
        selected = ["grpc"]
    config.storage_plugins = list(selected)
    if "evicpress_local" in selected:
        if list(selected) != ["evicpress_local"]:
            raise ValueError(
                "local EvicPress cannot be combined with another storage plugin"
            )
        if (
            config.enable_pd
            or config.enable_p2p
            or config.enable_async_loading
            or config.use_layerwise
            or config.enable_scheduler_bypass_lookup
            or config.local_disk
            or config.remote_url
            or config.gds_path
            or extra.get("enable_nixl_storage")
            or extra.get("audit_backend_enabled")
        ):
            raise ValueError(
                "local EvicPress requires worker-routed synchronous lookup and exclusive RAM/disk ownership; PD/P2P/NIXL/layerwise/async/other storage/audit unsupported"
            )
        extra.setdefault(
            "storage_plugin.evicpress_local.module_path",
            "lmcache.v1.storage_backend.local_evicpress_backend",
        )
        extra.setdefault(
            "storage_plugin.evicpress_local.class_name", "LocalEvicPressBackend"
        )
        extra["storage_plugin.evicpress_local.required"] = True
    if "grpc" in selected:
        # Preserve the former project's remote default, not unrelated plugins.
        config.enable_pd = False
        extra.setdefault(
            "storage_plugin.grpc.module_path", "lmcache.v1.storage_backend.grpc_backend"
        )
        extra.setdefault("storage_plugin.grpc.class_name", "GRPCBackend")
