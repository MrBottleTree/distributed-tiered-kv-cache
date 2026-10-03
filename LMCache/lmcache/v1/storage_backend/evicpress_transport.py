# SPDX-License-Identifier: Apache-2.0
"""Thin local/remote facades; protobuf objects are values, not local RPCs."""

# Standard
from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable

# Third Party
import grpc

# Local
from . import evicpress_pb2 as pb


class TransportError(RuntimeError):
    """Storage/transport failed rather than reporting a genuine missing block."""


class GRPCTransport:
    """Forward existing stub methods and nonblocking prefetch to actual gRPC."""

    mode = "grpc"
    is_remote = True

    def __init__(self, server: str, max_message_bytes: int) -> None:
        # Local: generated bindings are imported only for remote mode.
        from . import evicpress_pb2_grpc as rpc

        self.channel = grpc.insecure_channel(
            server,
            options=[
                ("grpc.max_receive_message_length", max_message_bytes),
                ("grpc.max_send_message_length", max_message_bytes),
            ],
        )
        self.stub = rpc.EvicPressServiceStub(self.channel)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stub, name)

    def submit_prefetch(self, request: Any, timeout: float = 15) -> Any:
        """Send a nonblocking remote hint; demand retrieval stays synchronous."""
        return self.stub.Prefetch.future(request, timeout=timeout)

    def get_state(self) -> dict:
        """Identify remote mode; detailed remote state remains on the service."""
        return {"transport": "grpc"}

    def close(self) -> None:
        """Close the existing remote channel."""
        self.channel.close()


class LocalTransport:
    """Translate request values into installed core-library function calls.

    The core package never imports protobuf or gRPC. This facade keeps head
    response value shapes compatible, without invoking an RPC serializer.
    """

    mode = "local"
    is_remote = False

    def __init__(self, config_path: str, cache_identity: dict) -> None:
        # Third Party: explicit package install, not a sibling sys.path hack.
        from evicpress.local import LocalEvicPressClient

        self.client = LocalEvicPressClient(config_path, cache_identity=cache_identity)
        self.lock = RLock()
        self.metrics = {
            "calls": 0,
            "payload_bytes": 0,
            "rpc_calls": 0,
            "wire_bytes": 0,
            "errors": 0,
            "prefetch_hints": 0,
        }

    def Lookup(self, request: Any, timeout: float | None = None) -> Any:
        """Direct lookup; timeout is accepted for facade compatibility only."""
        result = self._call(self.client.lookup, request.block_id)
        return pb.LookupResponse(hit=result.hit, tier=result.tier)

    def Store(self, request: Any, timeout: float | None = None) -> Any:
        """Store canonical bytes and expose actual backing precision."""
        self._payload(len(request.data))
        result = self._call(
            self.client.store,
            request.block_id,
            request.data,
            request.quality_score or 1.0,
        )
        return SimpleNamespace(
            success=result.success,
            tier=result.tier,
            quant_level=result.quant_level,
            message="stored" if result.success else "canonical store rejected",
        )

    def Retrieve(self, request: Any, timeout: float | None = None) -> Any:
        """Restore demand bytes; missing backing is an explicit miss."""
        result = self._call(self.client.retrieve, request.block_id)
        if result is None:
            return SimpleNamespace(found=False, data=b"", tier=0, quant_level="")
        self._payload(len(result.data))
        return SimpleNamespace(
            found=True,
            data=result.data,
            tier=result.tier,
            quant_level=result.quant_level,
        )

    def Delete(self, request: Any, timeout: float | None = None) -> Any:
        """Delete unleased canonical backing."""
        return pb.DeleteResponse(
            success=self._call(self.client.delete, request.block_id)
        )

    def GetStats(self, request: Any, timeout: float | None = None) -> Any:
        """Expose schema capability and manager counters without RPC."""
        state = self._call(self.client.get_state)
        stats = state["stats"]
        tier1, tier2, tier3 = state["tier1"], state["tier2"], state["tier3"]
        return pb.StatsResponse(
            head_schema_version=state["config"]["head_schema_version"],
            tier1_blocks=tier1["block_count"],
            tier1_bytes=tier1["used_bytes"],
            tier2_blocks=tier2["block_count"],
            tier2_bytes=tier2["used_bytes"],
            tier3_blocks=tier3["block_count"],
            tier3_bytes=tier3["used_bytes"],
            tier1_promotions=stats["tier1_promotions"],
            evictions=stats["evictions"],
            hit_rate=stats["hit_rate"],
            total_hits=stats["total_hits"],
            total_misses=stats["total_misses"],
            tier2_hits=stats["tier2_hits"],
            tier3_hits=stats["tier3_hits"],
            parent_hits=stats["parent_hits"],
            parent_misses=stats["parent_misses"],
            head_rpc_calls=0,
            head_wire_bytes=0,
        )

    def StoreHeads(self, request: Any, timeout: float | None = None) -> Any:
        """Publish native groups and adapt their result values."""
        payloads = [(h.block_id, h.data) for h in request.heads]
        self._payload(sum(len(data) for _, data in payloads))
        result = self._call(
            self.client.store_heads,
            request.descriptor_json,
            payloads,
            request.client_id,
        )
        return pb.HeadGroupResponse(**result)

    def LookupHeads(self, request: Any, timeout: float | None = None) -> Any:
        """Lookup complete groups and preserve native read leases."""
        return self._read_heads(request, retrieve=False)

    def RetrieveHeads(self, request: Any, timeout: float | None = None) -> Any:
        """Retrieve native groups, retaining independent per-head backing."""
        return self._read_heads(request, retrieve=True)

    def DeleteHeads(self, request: Any, timeout: float | None = None) -> Any:
        """Delete complete parents with existing lease/force semantics."""
        return pb.DeleteResponse(
            success=self._call(
                self.client.delete_heads, list(request.parent_ids), request.force
            )
        )

    def ReleaseHeads(self, request: Any, timeout: float | None = None) -> Any:
        """Release this client's canonical read reservations."""
        self._call(
            self.client.release_heads, list(request.lease_ids), request.client_id
        )
        return pb.DeleteResponse(success=True)

    def SyncHeadTier1(self, request: Any, timeout: float | None = None) -> Any:
        """Reconcile physically resident head mirrors."""
        revoked = self._call(
            self.client.sync_head_tier1,
            request.client_id,
            {e.block_id: e.nbytes for e in request.entries},
        )
        return pb.HeadTier1Response(revoked_ids=revoked)

    def submit_prefetch(self, request: Any, timeout: float = 15) -> Future[int]:
        """Submit a local queue hint; no channel or RPC is created."""
        with self.lock:
            self.metrics["prefetch_hints"] += 1
        return self._call(self.client.prefetch, list(request.block_ids))

    def get_state(self) -> dict:
        """Return policy state and counters separating payload from traffic."""
        state = self.client.get_state()
        with self.lock:
            state["transport_metrics"] = dict(self.metrics)
        return state

    def close(self) -> None:
        """Stop the owned runtime without deleting persistent backing."""
        self.client.close()

    def _call(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            self.metrics["calls"] += 1
        try:
            return method(*args, **kwargs)
        except (OSError, RuntimeError, ValueError) as exc:
            with self.lock:
                self.metrics["errors"] += 1
            raise TransportError(str(exc)) from exc

    def _payload(self, size: int) -> None:
        with self.lock:
            self.metrics["payload_bytes"] += size

    def _read_heads(self, request: Any, *, retrieve: bool) -> Any:
        items = [(p.parent_id, set(p.local_ids), p.lease_id) for p in request.parents]
        result = self._call(
            self.client.read_heads,
            items,
            request.client_id,
            retrieve=retrieve,
            pin=request.pin,
        )
        self._payload(
            sum(len(h.get("data", b"")) for p in result for h in p.get("heads", []))
        )
        return pb.HeadBatchResponse(parents=result)
