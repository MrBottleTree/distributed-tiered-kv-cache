"""
GRPCBackend — LMCache storage plugin that talks to Machine B (EvicPress).

Implements the StoragePluginInterface so LMCache's StorageManager can route
KV cache blocks to the remote EvicPress tiered storage node over gRPC.

Tiering is INCLUSIVE: Machine B always holds a Tier 3 canonical copy of every
block. When Machine B signals tier=1 in StoreResponse, Machine A mirrors the
block into its LocalCPUBackend as a pure cache copy. If Machine A evicts that
copy under its own LRU the block is simply dropped — Machine B still has it.
There is NO T1 return protocol.

Serialization:
  KV tensors are serialized with torch.save() into raw bytes. Machine B stores
  them opaquely — it has no knowledge of tensor shapes or dtypes. On retrieval,
  torch.load() reconstructs the original tensor exactly.

Key encoding:
  CacheEngineKey (model_name, world_size, worker_id, chunk_hash, dtype) is
  encoded as a canonical pipe-separated string, which becomes Machine B's
  block_id.

Prefetch:
  When contains() detects a block in Tier 3 (disk), it fires a non-blocking
  Prefetch RPC to hint Machine B to promote it to Tier 2 (RAM) before the
  imminent get_blocking() call.
"""

import io
from typing import Callable, List, Optional, Sequence, Union
from concurrent.futures import Future

import torch
import grpc

from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.v1.storage_backend import evicpress_pb2
from lmcache.v1.storage_backend import evicpress_pb2_grpc


class GRPCBackend(StoragePluginInterface):

    def __init__(
        self,
        dst_device: str = "cuda",
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop=None,
    ):
        super().__init__(dst_device, config, metadata, local_cpu_backend, loop)

        self.server_addr = config.extra_config.get("grpc_server", "localhost:50051")

        # quality_score sent to Machine B for every block.
        # 1.0 = highly sensitive to compression (preserve as-is).
        # Tune this once the compression profiler is wired up.
        self._quality_score: float = float(
            config.extra_config.get("grpc_quality_score", 1.0)
        )

        max_msg = int(
            config.extra_config.get("grpc_max_message_bytes", 512 * 1024 * 1024)
        )
        options = [
            ("grpc.max_receive_message_length", max_msg),
            ("grpc.max_send_message_length",    max_msg),
        ]
        self.channel = grpc.insecure_channel(self.server_addr, options=options)
        self.stub    = evicpress_pb2_grpc.EvicPressServiceStub(self.channel)
        print(f"[GRPCBackend] connected to Machine B at {self.server_addr}")

    # ------------------------------------------------------------------ #
    #  Key encoding                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_key(key: CacheEngineKey) -> str:
        """
        Canonical string block_id for Machine B.
        Pipe-separated so no field value can collide with the separator.
        """
        return f"{key.model_name}|{key.world_size}|{key.worker_id}|{key.chunk_hash}|{key.dtype}"

    # ------------------------------------------------------------------ #
    #  Tensor serialization                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        """Serialize a tensor to bytes, preserving shape and dtype."""
        buf = io.BytesIO()
        torch.save(tensor.clone().cpu(), buf)
        return buf.getvalue()

    @staticmethod
    def _bytes_to_tensor(data: bytes) -> torch.Tensor:
        """Deserialize bytes back to a tensor."""
        buf = io.BytesIO(data)
        return torch.load(buf, weights_only=True)

    # ------------------------------------------------------------------ #
    #  StorageBackendInterface implementation                              #
    # ------------------------------------------------------------------ #

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Local-first existence check: if Machine A's LocalCPUBackend (Tier 1)
        already holds the block, skip the network round-trip. Otherwise fall
        back to Machine B's Lookup RPC. If the block is in Tier 3 (disk),
        fire a non-blocking Prefetch hint so Machine B can promote it to RAM
        before the upcoming get_blocking().
        """
        if self.local_cpu_backend is not None and self.local_cpu_backend.contains(key, pin=pin):
            return True

        block_id = self._encode_key(key)
        try:
            resp = self.stub.Lookup(
                evicpress_pb2.LookupRequest(block_id=block_id)
            )
        except grpc.RpcError as e:
            print(f"[GRPCBackend] Lookup failed for {block_id[:32]}…: {e.details()}")
            return False

        if resp.hit and resp.tier == 3:
            # Fire-and-forget prefetch: promotes block from disk → RAM
            # before get_blocking() is called. Does not block here.
            self.stub.Prefetch.future(
                evicpress_pb2.PrefetchRequest(block_ids=[block_id])
            )

        return resp.hit

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False  # puts are synchronous; nothing is pending

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec=None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Store KV blocks in Machine B. Synchronous (simple, correct)."""
        for key, obj in zip(keys, objs):
            self._store_one(key, obj)
            if on_complete_callback:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    print(f"[GRPCBackend] on_complete_callback error: {e}")
        return None

    def _store_one(self, key: CacheEngineKey, obj: MemoryObj) -> None:
        """
        Serialize one MemoryObj and send it to Machine B via Store RPC.

        The full KV tensor (shape [2, num_layers, num_tokens, heads, head_dim]
        in KV_2LTD format) is serialized with torch.save and sent as opaque
        bytes. Machine B doesn't inspect the bytes — it just stores them.
        """
        block_id = self._encode_key(key)
        tensor   = obj.get_tensor(0)           # [2, L, T, H, D] or similar
        data     = self._tensor_to_bytes(tensor)

        print(f"[GRPCBackend] PUT block={block_id[:32]}… size={len(data)//1024}KB")
        try:
            resp = self.stub.Store(
                evicpress_pb2.StoreRequest(
                    block_id=block_id,
                    data=data,
                    quality_score=self._quality_score,
                )
            )
            if not resp.success:
                print(f"[GRPCBackend] Store failed: {resp.message}")
            elif resp.tier == 1 and self.local_cpu_backend is not None:
                # Machine B decided this block belongs in Machine A RAM (Tier 1).
                # Mirror it into LocalCPUBackend. Machine B still holds the
                # canonical T3 copy, so A's LRU can drop this at will.
                print(f"[GRPCBackend] Tier1 promote {block_id[:32]}…")
                self.local_cpu_backend.submit_put_task(key, obj)

        except grpc.RpcError as e:
            print(f"[GRPCBackend] Store RPC error: {e.details()}")

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Fetch a KV block from Machine B and reconstruct a MemoryObj.
        Returns None on miss. Machine B handles tier selection internally
        (Tier 2 RAM preferred, falls back to Tier 3 disk).
        """
        block_id = self._encode_key(key)
        print(f"[GRPCBackend] GET block={block_id[:32]}…")
        try:
            resp = self.stub.Retrieve(
                evicpress_pb2.RetrieveRequest(block_id=block_id)
            )
        except grpc.RpcError as e:
            print(f"[GRPCBackend] Retrieve RPC error: {e.details()}")
            return None

        if not resp.found:
            return None

        tensor = self._bytes_to_tensor(resp.data)

        if self.local_cpu_backend is not None:
            # Allocate a MemoryObj via the local CPU allocator and copy tensor in.
            allocator  = self.local_cpu_backend.get_allocator_backend()
            memory_obj = allocator.allocate(
                tensor.shape,
                tensor.dtype,
                fmt=MemoryFormat.KV_2LTD,
            )
            if memory_obj is None:
                print("[GRPCBackend] allocator returned None — out of local CPU memory")
                return None
            memory_obj.tensor.copy_(tensor)
            return memory_obj
        else:
            # No local CPU backend available — wrap the tensor directly.
            from lmcache.v1.memory_management import TensorMemoryObj, MemoryObjMetadata
            raw_data = tensor.cpu().contiguous().view(torch.uint8)
            meta = MemoryObjMetadata(
                shape=tensor.shape,
                dtype=tensor.dtype,
                address=raw_data.data_ptr(),
                phy_size=raw_data.nbytes,
                ref_count=1,
                fmt=MemoryFormat.KV_2LTD,
            )
            return TensorMemoryObj(raw_data, meta, parent_allocator=None)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        block_id = self._encode_key(key)
        try:
            resp = self.stub.Delete(
                evicpress_pb2.DeleteRequest(block_id=block_id)
            )
            return resp.success
        except grpc.RpcError as e:
            print(f"[GRPCBackend] Delete RPC error: {e.details()}")
            return False

    def pin(self, key: CacheEngineKey) -> bool:
        return True   # Machine B manages eviction; no pin concept yet

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        self.channel.close()
