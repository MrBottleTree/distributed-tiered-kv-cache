import grpc
from typing import Optional, List, Sequence
from concurrent.futures import Future
import numpy as np
import torch
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.DummyMemoryObj import DummyMemoryObj
from lmcache.v1.storage_backend import kv_cache_pb2
from lmcache.v1.storage_backend import kv_cache_pb2_grpc

class GRPCBackend(StoragePluginInterface):

    def __init__(
    self,
    dst_device="cuda",
    config=None,
    metadata=None,
    local_cpu_backend=None,
    loop=None,
):
        super().__init__(dst_device, config, metadata, local_cpu_backend, loop)

        self.server_addr = config.extra_config.get("grpc_server")
        self.channel = grpc.insecure_channel(self.server_addr)
        self.stub = kv_cache_pb2_grpc.KVCacheServiceStub(self.channel)

    # ─────────────────────────────────────────────
    # 🔑 KEY TRANSLATION (CRITICAL)
    # ─────────────────────────────────────────────
    def _convert_key(self, key: CacheEngineKey):
        return kv_cache_pb2.KVCacheKey(
            model_name=key.model_name,
            world_size=key.world_size,
            worker_id=key.worker_id,
            chunk_hash=key.chunk_hash
        )
        

    #     def _convert_key(self, key: CacheEngineKey):
    # return kv_cache_pb2.KVCacheKey(
    #     model_id=str(key.model_id) if hasattr(key, "model_id") else "unknown",
    #     prefix_hash=str(hash(key)),   # TEMP fallback
    #     num_tokens=0,                 # TEMP
    #     prefix_chain=[]
    # )

    # ─────────────────────────────────────────────
    # REQUIRED METHODS
    # ─────────────────────────────────────────────

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        resp = self.stub.Fetch(
            kv_cache_pb2.FetchRequest(
                key=self._convert_key(key)
            )
        )
        return resp.found

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False  # we do synchronous puts

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec=None,
        on_complete_callback=None,
    ):
        for key, obj in zip(keys, objs):
            print("\n[DEBUG] PUT CALLED")
            print("Key:", key)
            print("Obj type:", type(obj))
            print("Obj contents:", obj)
            print("Obj attributes:", dir(obj))
            self._store_one(key, obj)
            print("\n[DEBUG] PUT DONE maybe")
            if on_complete_callback:
                on_complete_callback(key)

        return None  # synchronous

    def _store_one(self, key: CacheEngineKey, obj: MemoryObj):
        """
        Convert MemoryObj → KVCacheValue and store via gRPC
        """
        tensor = obj.get_tensor(0)

        k_all = tensor[0]
        print(tensor)
        v_all = tensor[0]

        layers = []

        for layer_id in range(k_all.shape[0]):
            k = k_all[layer_id]
            v = v_all[layer_id]

            k_np = k.cpu().numpy()
            v_np = v.cpu().numpy()

            k_proto = kv_cache_pb2.KVTensor(
                precision=kv_cache_pb2.PRECISION_FP32,
                data=k_np.tobytes(),
                shape=list(k_np.shape)
            )

            v_proto = kv_cache_pb2.KVTensor(
                precision=kv_cache_pb2.PRECISION_FP32,
                data=v_np.tobytes(),
                shape=list(v_np.shape)
            )

            layers.append(
                kv_cache_pb2.LayerKV(
                    layer_id=layer_id,
                    key_tensor=k_proto,
                    value_tensor=v_proto
                )
            )

        kv_value = kv_cache_pb2.KVCacheValue(
            layers=layers,
            num_tokens=obj.get_num_tokens()
        )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        print("\n[DEBUG] GET CALLED")
        print("Key type:", type(key))
        print("Key contents:", key)
        print("Key attributes:", dir(key))
        resp = self.stub.Fetch(
            kv_cache_pb2.FetchRequest(
                key=self._convert_key(key)
            )
        )

        if not resp.found:
            return None

        # ⚠️ YOU MUST CONVERT proto → MemoryObj
        return self._convert_to_memory_obj(resp.value)

    def _convert_to_memory_obj(self, value):

        k_list = []
        v_list = []

        for layer in value.layers:
            shape = list(layer.key_tensor.shape)

            k_np = np.frombuffer(layer.key_tensor.data, dtype=np.float32).reshape(shape)
            v_np = np.frombuffer(layer.value_tensor.data, dtype=np.float32).reshape(shape)

            k_list.append(torch.from_numpy(k_np))
            v_list.append(torch.from_numpy(v_np))

        k_all = torch.stack(k_list, dim=0)
        v_all = torch.stack(v_list, dim=0)

        tensor = torch.stack([k_all, v_all], dim=0)

        return DummyMemoryObj(tensor)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        resp = self.stub.Delete(
            kv_cache_pb2.DeleteRequest(
                key=self._convert_key(key)
            )
        )
        return resp.success

    def pin(self, key: CacheEngineKey) -> bool:
        return True  # no-op

    def unpin(self, key: CacheEngineKey) -> bool:
        return True  # no-op

    def get_allocator_backend(self):
        return None  # not needed for now

    def close(self):
        self.channel.close()