# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible remote EvicPress plugin; shared logic lives in the base."""

# Standard
from typing import Any

# Local
from .evicpress_backend import EvicPressBackend
from .evicpress_transport import GRPCTransport


class GRPCBackend(EvicPressBackend):
    """Keep the existing distributed path and its configuration keys available."""

    def create_transport(self, config: Any, metadata: Any) -> GRPCTransport:
        """Open the configured remote endpoint without changing manager policy."""
        self.server_addr = config.extra_config.get("grpc_server", "localhost:50051")
        max_message = int(
            config.extra_config.get(
                "evicpress_max_message_bytes",
                config.extra_config.get("grpc_max_message_bytes", 512 * 1024 * 1024),
            )
        )
        return GRPCTransport(self.server_addr, max_message)
