# SPDX-License-Identifier: Apache-2.0
"""Logical KV-head segments, independent of their physical storage.

Design reference: RedKnot 4eb99dbc, core/segpaged_v2/{visible_plan,page_table}.
This implementation retains every token; it does not copy sparse-attention
policies, zero-padding or RedKnot's attention-output reuse implementation.
"""
# Standard
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Any, Callable, Iterator, Mapping

# Third Party
import torch


def make_segment_id(parent: str, layer: int, head: int, tokens: int) -> str:
    """Return a stable storage ID for one layer/KV-head/token segment."""
    raw = json.dumps([parent, layer, head, tokens], separators=(",", ":"))
    return "head-v1:" + hashlib.sha256(raw.encode()).hexdigest()


def query_to_kv_heads(query_heads: int, kv_heads: int, ids: list[int]) -> tuple[int, ...]:
    """Map query heads to their shared KV heads without duplicating GQA data."""
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("invalid GQA geometry")
    if any(type(h) is not int or not 0 <= h < query_heads for h in ids):
        raise ValueError("invalid query head")
    return tuple(h // (query_heads // kv_heads) for h in ids)


@dataclass(frozen=True)
class HeadVisiblePlan:
    """Compact ordered positions; only full retention is currently supported."""

    token_count: int

    def as_dict(self) -> dict[str, int]:
        """Return portable original positions, not physical page offsets."""
        return {"start": 0, "count": self.token_count}


@dataclass(frozen=True)
class HeadSegment:
    """Logical identity and a storage handle for one layer/KV head."""

    parent_id: str
    layer: int
    kv_head: int
    block_id: str
    positions: HeadVisiblePlan


class HeadSegmentTable:
    """Coordinate lookup independent of RPC ordering or the CPU allocator."""

    def __init__(self, descriptor: dict) -> None:
        self.descriptor = validate_descriptor(descriptor)
        plan = HeadVisiblePlan(descriptor["shape"][2])
        self.segments = {
            (h["layer"], h["kv_head"]): HeadSegment(
                descriptor["parent_id"], h["layer"], h["kv_head"], h["block_id"], plan)
            for h in descriptor["heads"]
        }

    def get(self, layer: int, kv_head: int) -> HeadSegment:
        """Return a segment in original model coordinates."""
        return self.segments[(layer, kv_head)]


def validate_descriptor(d: dict) -> dict:
    """Reject incomplete/sparse layouts before exposing KV to native attention."""
    if not isinstance(d, dict) or type(d.get("schema_version")) is not int or d["schema_version"] != 1:
        raise ValueError("unsupported head descriptor")
    shape = d.get("shape")
    if (not isinstance(shape, list) or len(shape) != 4 or shape[0] != 2
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("expected [2,layers,tokens,KV_heads*head_dim]")
    heads, dim = d.get("num_kv_heads"), d.get("head_dim")
    if (type(heads) is not int or type(dim) is not int or heads <= 0 or dim <= 0
            or shape[3] != heads * dim or shape[1] * heads > 4096):
        raise ValueError("invalid head geometry")
    if d.get("dtype") != "torch.float16" or d.get("format") != "KV_2LTD":
        raise ValueError("head mode currently requires FP16 KV_2LTD")
    if d.get("positions") != HeadVisiblePlan(shape[2]).as_dict():
        raise ValueError("sparse retention requires a matching attention kernel")
    for field in ("parent_id", "namespace"):
        if not isinstance(d.get(field), str) or not 0 < len(d[field]) <= 4096:
            raise ValueError(f"invalid {field}")
    expected = [{"block_id": make_segment_id(d["parent_id"], l, h, shape[2]),
                 "layer": l, "kv_head": h}
                for l in range(shape[1]) for h in range(heads)]
    if d.get("heads") != expected:
        raise ValueError("missing, duplicated or inconsistent head coordinates")
    return d


def split_chunk(tensor: torch.Tensor, parent: str, namespace: str,
                kv_heads: int, head_dim: int) -> tuple[dict, dict[str, torch.Tensor]]:
    """Split FP16 KV_2LTD into independent contiguous [2,tokens,head_dim] blobs."""
    if tensor.ndim != 4 or tensor.dtype != torch.float16:
        raise ValueError("head mode requires an FP16 [2,L,T,H*D] tensor")
    d = {"schema_version": 1, "parent_id": parent, "namespace": namespace,
         "shape": list(tensor.shape), "dtype": str(tensor.dtype), "format": "KV_2LTD",
         "num_kv_heads": kv_heads, "head_dim": head_dim,
         "positions": HeadVisiblePlan(tensor.shape[2]).as_dict(),
         "heads": [{"block_id": make_segment_id(parent, l, h, tensor.shape[2]),
                    "layer": l, "kv_head": h}
                   for l in range(tensor.shape[1]) for h in range(kv_heads)]}
    validate_descriptor(d)
    view = tensor.reshape(2, tensor.shape[1], tensor.shape[2], kv_heads, head_dim)
    blobs = {h["block_id"]: view[:, h["layer"], :, h["kv_head"], :].detach().clone().contiguous()
             for h in d["heads"]}
    return d, blobs


def assemble_chunk(d: dict, tensors: Mapping[str, torch.Tensor],
                   output: torch.Tensor | None = None) -> torch.Tensor:
    """Scatter a complete set of heads into native order; never zero-fill misses."""
    validate_descriptor(d)
    if set(tensors) != {h["block_id"] for h in d["heads"]}:
        raise ValueError("a complete chunk requires every KV head")
    if output is None:
        output = torch.empty(d["shape"], dtype=torch.float16)
    if list(output.shape) != d["shape"] or output.dtype != torch.float16:
        raise ValueError("invalid assembly buffer")
    view = output.reshape(2, d["shape"][1], d["shape"][2], d["num_kv_heads"], d["head_dim"])
    for h in d["heads"]:
        value = tensors[h["block_id"]]
        if tuple(value.shape) != (2, d["shape"][2], d["head_dim"]) or value.dtype != output.dtype:
            raise ValueError("head tensor shape/dtype mismatch")
        view[:, h["layer"], :, h["kv_head"], :].copy_(value)
    return output


class HeadSegmentCache:
    """Byte-bounded read-only ownership of independently allocated head tensors.

    All heads and parent assembly buffers use the caller's same allocator.
    Leases protect cache references; eviction releases allocator ownership.
    """

    def __init__(self, max_bytes: int, allocate: Callable[..., Any]) -> None:
        self.max_bytes = max_bytes
        self.allocate = allocate
        self.entries: OrderedDict[str, Any] = OrderedDict()
        self.pins: dict[str, int] = {}
        self.used_bytes = 0
        self.lock = RLock()

    def inventory(self) -> dict[str, int]:
        """Return actual physical bytes for B's Tier 1 reconciliation."""
        with self.lock:
            return {key: obj.metadata.phy_size for key, obj in self.entries.items()}

    def remove(self, key: str) -> bool:
        """Drop an unleased mirror without deleting its canonical backing."""
        with self.lock:
            if self.pins.get(key, 0) or key not in self.entries:
                return False
            obj = self.entries.pop(key)
            self.used_bytes -= obj.metadata.phy_size
            obj.ref_count_down()
            return True

    def evict_one(self) -> bool:
        """Release the oldest unleased allocation; return False if all are pinned."""
        with self.lock:
            return any(self.remove(key) for key in list(self.entries))

    def retain(self, key: str, value: torch.Tensor) -> bool:
        """Clone an admitted head into the shared pool; decline oversized entries."""
        required = value.numel() * value.element_size()
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return True
            if required > self.max_bytes:
                return False
            while self.used_bytes + required > self.max_bytes:
                if not self.evict_one():
                    return False
            # Add a singleton layer axis so LMCache's token dimension remains 2.
            shape = (2, 1, value.shape[1], value.shape[2])
            obj = self.allocate(shape, value.dtype)
            while obj is None and self.evict_one():
                obj = self.allocate(shape, value.dtype)
            if obj is None:
                return False
            while self.used_bytes + obj.metadata.phy_size > self.max_bytes:
                if not self.evict_one():
                    obj.ref_count_down()
                    return False
            try:
                obj.get_tensor(0).copy_(value[:, None])
            except Exception:
                obj.ref_count_down()
                raise
            self.entries[key] = obj
            self.used_bytes += obj.metadata.phy_size
            return True

    def pin(self, keys: list[str]) -> None:
        """Protect currently resident entries across lookup and retrieval."""
        with self.lock:
            for key in set(keys) & self.entries.keys():
                self.pins[key] = self.pins.get(key, 0) + 1

    def unpin(self, keys: list[str]) -> None:
        """Release one lookup reservation per distinct key."""
        with self.lock:
            for key in set(keys):
                count = self.pins.get(key, 0)
                if count <= 1:
                    self.pins.pop(key, None)
                else:
                    self.pins[key] = count - 1

    @contextmanager
    def lease(self, keys: list[str]) -> Iterator[dict[str, torch.Tensor]]:
        """Keep resident allocations alive while the parent is reconstructed."""
        with self.lock:
            objects = {key: self.entries[key] for key in keys if key in self.entries}
            self.pin(list(objects))
            for key, obj in objects.items():
                obj.ref_count_up()
                self.entries.move_to_end(key)
        try:
            yield {key: obj.get_tensor(0)[:, 0] for key, obj in objects.items()}
        finally:
            with self.lock:
                self.unpin(list(objects))
                for obj in objects.values():
                    obj.ref_count_down()

    def clear(self) -> None:
        """Release all mirrors after the client has closed its outstanding leases."""
        with self.lock:
            self.pins.clear()
            for key in list(self.entries):
                self.remove(key)
