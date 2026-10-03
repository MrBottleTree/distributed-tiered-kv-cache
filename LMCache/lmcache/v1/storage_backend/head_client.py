# SPDX-License-Identifier: Apache-2.0
"""Batched head storage adapter; native vLLM still receives complete chunks."""
# Standard
from collections import OrderedDict
from contextlib import ExitStack
import io
import json
import pickle
from threading import RLock
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

# Third Party
import grpc
import torch

# Local
from . import evicpress_pb2 as pb
from .head_segments import HeadSegmentCache, assemble_chunk, split_chunk, validate_descriptor
from .evicpress_transport import TransportError


def tensor_bytes(value: torch.Tensor) -> bytes:
    """Serialize independent CPU storage, never a view of the entire parent."""
    stream = io.BytesIO()
    torch.save(value.detach().cpu().contiguous(), stream)
    return stream.getvalue()


class HeadClient:
    """Manage immutable head identities, shared-pool mirrors and read leases.

    allocate(shape, dtype) returns a caller-owned MemoryObj or None. The cache
    owns one reference per retained head; returned parent objects belong to
    LMCache. Temporary serialization buffers are outside the allocator pool.
    """

    def __init__(self, stub: Any, namespace: str, layers: int, kv_heads: int,
                 head_dim: int, chunk_size: int, allocate: Callable[..., Any],
                 cache_bytes: int, max_message_bytes: int) -> None:
        self.stub = stub
        self.namespace = namespace
        self.layers, self.kv_heads, self.head_dim = layers, kv_heads, head_dim
        self.chunk_size = chunk_size
        self.allocate = allocate
        self.cache = HeadSegmentCache(cache_bytes, allocate)
        self.max_message_bytes = max_message_bytes
        self.batch_bytes = min(64 * 1024 * 1024, max_message_bytes * 3 // 4)
        self.client_id = uuid4().hex
        self.descriptors: OrderedDict[str, dict] = OrderedDict()
        self.pending: dict[str, tuple[str, list[str], float]] = {}
        self.lock = RLock()
        self.metrics = {"parent_hits": 0, "parent_misses": 0,
                        "rpc_calls": 0, "wire_bytes": 0, "cache_peak_bytes": 0,
                        "calls": 0, "payload_bytes": 0,
                        "assembly_returned_bytes_peak": 0, "cache_bytes": 0}
        capability = self.stub.GetStats(pb.StatsRequest(), timeout=15)
        if capability.head_schema_version != 1:
            raise ValueError("Machine B lacks head schema v1; update both repositories")

    def parent_id(self, key: str) -> str:
        """Preserve the context-sensitive key within the model/layout namespace."""
        return f"head-v1:{self.namespace}:{key}"

    def store_head_segment(self, key: str, tensor: torch.Tensor) -> bool:
        """Store a complete group's independent heads in one bounded RPC."""
        with self.lock:
            self._expire()
            parent = self.parent_id(key)
            descriptor, values = split_chunk(tensor, parent, self.namespace,
                                             self.kv_heads, self.head_dim)
            self._validate(descriptor, parent)
            request = pb.HeadStoreRequest(
                descriptor_json=json.dumps(descriptor, separators=(",", ":")),
                heads=[pb.HeadPayload(block_id=k, data=tensor_bytes(v)) for k, v in values.items()],
                client_id=self.client_id)
            if request.ByteSize() > self.max_message_bytes:
                raise ValueError("complete head store exceeds gRPC message limit")
            response = self._rpc(self.stub.StoreHeads, request)
            if not response.found:
                print(f"[HeadCache] store rejected: {response.message}")
                return False
            self._remember(self._descriptor(response, parent))
            for head in response.heads:
                if head.cache_in_tier1:
                    self.cache.retain(head.block_id, values[head.block_id])
            self._sync()
            return True

    def contains_many(self, keys: Sequence[str], pin: bool = False) -> int:
        """Return the complete-hit prefix; reserve local/canonical heads if pin."""
        with self.lock:
            self._expire()
            hit_count = 0
            parents = [self.parent_id(key) for key in keys]
            for batch in self._batches(parents):
                # Protect local inventory while B evaluates group completeness.
                local = {p: self._local_ids(p) for p in batch}
                for p in batch:
                    self._release(p)
                    self.cache.pin(local[p])
                response = None
                accepted: set[str] = set()
                try:
                    request = pb.HeadReadRequest(client_id=self.client_id, pin=pin,
                        parents=[pb.HeadReadItem(parent_id=p, local_ids=local[p]) for p in batch])
                    response = self._rpc(self.stub.LookupHeads, request)
                    groups = self._groups(response, batch)
                    prefix_open = True
                    prefetch = []
                    for parent in batch:
                        group = groups[parent]
                        if prefix_open and group.found:
                            self._remember(self._descriptor(group, parent))
                            hit_count += 1
                            if pin:
                                self.pending[parent] = (group.lease_id, local[parent], time.monotonic() + 55)
                                accepted.add(parent)
                            prefetch.extend(h.block_id for h in group.heads if h.tier == 3)
                        else:
                            prefix_open = False
                    if prefetch:
                        # One hint per batch, not one RPC per KV head.
                        hint = pb.PrefetchRequest(block_ids=prefetch)
                        self.metrics["calls"] += 1
                        self.metrics["payload_bytes"] += hint.ByteSize()
                        if getattr(self.stub, "is_remote", True):
                            self.metrics["rpc_calls"] += 1
                            self.metrics["wire_bytes"] += hint.ByteSize()
                        if hasattr(self.stub, "submit_prefetch"):
                            self.stub.submit_prefetch(hint, timeout=15)
                        else:
                            # Preserve use with raw generated stubs.
                            self.stub.Prefetch.future(hint, timeout=15)
                    if not prefix_open:
                        return hit_count
                finally:
                    for parent in batch:
                        if parent not in accepted:
                            self.cache.unpin(local[parent])
                    if response is not None:
                        unused = [g.lease_id for g in response.parents
                                  if g.lease_id and g.parent_id not in accepted]
                        self._release_remote(unused)
            return hit_count

    def gather_segment(self, keys: Sequence[str]) -> list[Any]:
        """Fetch bounded groups and scatter full-position heads into shared buffers.

        A partial, corrupt or incompatible parent returns None, never zero KV.
        Failure paths release lookup leases and any allocated parent buffer.
        """
        with self.lock:
            self._expire()
            parents = [self.parent_id(key) for key in keys]
            output = []
            for batch in self._batches(parents):
                with ExitStack() as stack:
                    local = {p: stack.enter_context(self.cache.lease(self._local_ids(p))) for p in batch}
                    request = pb.HeadReadRequest(client_id=self.client_id, parents=[
                        pb.HeadReadItem(parent_id=p, local_ids=list(local[p]),
                                       lease_id=self.pending.get(p, ("", [], 0))[0]) for p in batch])
                    try:
                        response = self._rpc(self.stub.RetrieveHeads, request)
                        groups = self._groups(response, batch)
                        for parent in batch:
                            obj = None
                            try:
                                group = groups[parent]
                                if not group.found:
                                    raise ValueError(group.message or "incomplete parent")
                                d = self._descriptor(group, parent)
                                self._remember(d)
                                tensors = dict(local[parent])
                                for h in group.heads:
                                    if h.block_id not in tensors:
                                        tensors[h.block_id] = torch.load(io.BytesIO(h.data), weights_only=True)
                                obj = self.allocate(d["shape"], torch.float16)
                                while obj is None and self.cache.evict_one():
                                    obj = self.allocate(d["shape"], torch.float16)
                                if obj is None:
                                    raise ValueError("shared CPU pool has no assembly space")
                                assemble_chunk(d, tensors, output=obj.get_tensor(0))
                                for h in group.heads:
                                    if h.cache_in_tier1:
                                        self.cache.retain(h.block_id, tensors[h.block_id])
                                output.append(obj)
                                obj = None  # Ownership passes to LMCache.
                                self.metrics["parent_hits"] += 1
                            except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError, TypeError, AttributeError) as exc:
                                if obj is not None:
                                    obj.ref_count_down()
                                print(f"[HeadCache] parent miss: {exc}")
                                self.metrics["parent_misses"] += 1
                                output.append(None)
                    except (grpc.RpcError, TransportError, ValueError) as exc:
                        print(f"[HeadCache] retrieve failed: {exc}")
                        output.extend([None] * len(batch))
                        self.metrics["parent_misses"] += len(batch)
                    finally:
                        for parent in batch:
                            self._release(parent)
                returned_bytes = sum(obj.metadata.phy_size for obj in output if obj is not None)
                self.metrics["assembly_returned_bytes_peak"] = max(
                    self.metrics["assembly_returned_bytes_peak"], returned_bytes)
                self._sync()
            return output

    def remove(self, key: str, force: bool) -> bool:
        """Remove a logical group and its local mirrors, respecting live leases."""
        with self.lock:
            parent = self.parent_id(key)
            response = self._rpc(self.stub.DeleteHeads,
                pb.HeadDeleteRequest(parent_ids=[parent], force=force))
            if response.success:
                self._release(parent)
                for h in self.descriptors.get(parent, {}).get("heads", []):
                    self.cache.remove(h["block_id"])
                self.descriptors.pop(parent, None)
                self._sync()
            return response.success

    def unpin(self, key: str) -> bool:
        """Release lookup protection; canonical backing is unaffected."""
        with self.lock:
            self._release(self.parent_id(key))
            return True

    def close(self) -> None:
        """Release reservations and shared-pool mirrors before closing transport."""
        with self.lock:
            for parent in list(self.pending):
                self._release(parent)
            self.cache.clear()
            self._sync()

    def _rpc(self, method: Any, request: Any) -> Any:
        self.metrics["calls"] += 1
        self.metrics["payload_bytes"] += request.ByteSize()
        remote = getattr(self.stub, "is_remote", True)
        if remote:
            self.metrics["rpc_calls"] += 1
            self.metrics["wire_bytes"] += request.ByteSize()
        response = method(request, timeout=30)
        self.metrics["payload_bytes"] += response.ByteSize()
        if remote:
            self.metrics["wire_bytes"] += response.ByteSize()
        return response

    def _validate(self, d: dict, parent: str) -> None:
        validate_descriptor(d)
        if (d["parent_id"] != parent or d["namespace"] != self.namespace
                or d["shape"][1] != self.layers or d["num_kv_heads"] != self.kv_heads
                or d["head_dim"] != self.head_dim or d["shape"][2] > self.chunk_size):
            raise ValueError("head descriptor disagrees with runtime model/layout")

    def _descriptor(self, group: Any, parent: str) -> dict:
        d = json.loads(group.descriptor_json)
        self._validate(d, parent)
        expected = {h["block_id"] for h in d["heads"]}
        if (len(group.heads) != len(expected) or {h.block_id for h in group.heads} != expected
                or not all(h.found for h in group.heads)):
            raise ValueError("incomplete or duplicated response heads")
        return d

    def _remember(self, d: dict) -> None:
        parent = d["parent_id"]
        self.descriptors[parent] = d
        self.descriptors.move_to_end(parent)
        # Metadata is bounded independently of the tensor byte pool.
        while len(self.descriptors) > 512:
            old, descriptor = self.descriptors.popitem(last=False)
            self._release(old)
            for h in descriptor["heads"]:
                self.cache.remove(h["block_id"])

    def _local_ids(self, parent: str) -> list[str]:
        inventory = self.cache.inventory()
        return [h["block_id"] for h in self.descriptors.get(parent, {}).get("heads", [])
                if h["block_id"] in inventory]

    def _batches(self, parents: Sequence[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        batch: list[str] = []
        used = 0
        for parent in parents:
            d = self.descriptors.get(parent)
            tokens = d["shape"][2] if d else self.chunk_size
            # Include per-head torch serialization and descriptor overhead.
            total_heads = self.layers * self.kv_heads
            remote_heads = total_heads - len(self._local_ids(parent))
            estimate = 4 * tokens * remote_heads * self.head_dim + 4096 * total_heads
            if estimate > self.batch_bytes:
                raise ValueError("one parent exceeds safe batch limit; raise grpc_max_message_bytes or reduce chunk size")
            if batch and (used + estimate > self.batch_bytes or len(batch) >= 64):
                batches.append(batch)
                batch, used = [], 0
            batch.append(parent)
            used += estimate
        if batch:
            batches.append(batch)
        return batches

    @staticmethod
    def _groups(response: Any, parents: Sequence[str]) -> dict[str, Any]:
        groups = {g.parent_id: g for g in response.parents}
        if len(response.parents) != len(parents) or set(groups) != set(parents):
            raise ValueError("response parent identities do not match request")
        return groups

    def _sync(self) -> None:
        inventory = self.cache.inventory()
        self.metrics["cache_bytes"] = sum(inventory.values())
        self.metrics["cache_peak_bytes"] = max(self.metrics["cache_peak_bytes"], sum(inventory.values()))
        try:
            response = self._rpc(self.stub.SyncHeadTier1, pb.HeadTier1Request(
                client_id=self.client_id, entries=[pb.HeadTier1Entry(block_id=k, nbytes=v) for k, v in inventory.items()]))
            for key in response.revoked_ids:
                self.cache.remove(key)
        except (grpc.RpcError, TransportError) as exc:
            print(f"[HeadCache] inventory sync deferred: {exc}")
        # Preserve snapshots even if the worker is terminated before close().
        snapshot = {"client_id": self.client_id, **self.metrics}
        print(f"[HeadCache] metrics={json.dumps(snapshot, sort_keys=True)}", flush=True)

    def _release_remote(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._rpc(self.stub.ReleaseHeads, pb.HeadLeaseRequest(lease_ids=ids, client_id=self.client_id))
        except (grpc.RpcError, TransportError):
            pass  # B expires leases if the connection is lost.

    def _release(self, parent: str) -> None:
        pending = self.pending.pop(parent, None)
        if pending:
            lid, keys, _ = pending
            self.cache.unpin(keys)
            self._release_remote([lid] if lid else [])

    def _expire(self) -> None:
        for parent, (_, _, expiry) in list(self.pending.items()):
            if expiry <= time.monotonic():
                self._release(parent)
