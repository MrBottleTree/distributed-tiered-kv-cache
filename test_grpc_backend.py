# import torch
from LMCache.lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj, MemoryFormat
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventManager
from lmcache.v1.storage_backend.DummyMemoryObj import DummyMemoryObj
import torch
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

# 1. Setup config
config = LMCacheEngineConfig()

config.storage_plugins = ["grpc"]
config.extra_config = {
    "storage_plugin.grpc.module_path": "lmcache.v1.storage_backend.grpc_backend",
    "storage_plugin.grpc.class_name": "GRPCBackend",
    "grpc_server": "127.0.0.1:8080",
}

config.enable_pd = False
config.local_cpu = False
config.local_disk = None
config.remote_url = None


#Configuring metadata with random stuff right now just minimum that will work for this test of the backend
metadata = LMCacheMetadata(
    model_name="test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.float32,
    kv_shape=(1, 2, 1, 1, 1),  # minimal dummy shape
    use_mla=False,
    role="worker",
)
event_manager = EventManager()

storage_manager = StorageManager(
    config=config,
    metadata=metadata,
    event_manager=event_manager,
)

print("\nStorageManager initialized")
print("Backends:", storage_manager.list_backends())

# ─────────────────────────────────────────────
# 3. Create fake key
# ─────────────────────────────────────────────
key = CacheEngineKey(
    model_name="test_model",
    world_size=1,
    worker_id=0,
    chunk_hash=123,              # MUST be int apparently
    dtype=torch.float32,
)

#creating random tensor for now to copy into memory obj
tensor = torch.randn(2, 8, 64, 64)  # fake KV-like tensor

# mem_obj = MemoryObj(
#     tensor=tensor,
#     fmt=MemoryFormat.KV_2LTD
# )
local_cpu_backend = list(storage_manager.storage_backends.values())[0]
allocator = local_cpu_backend.get_allocator_backend()
memory_obj = allocator.allocate(
    tensor.shape,
    tensor.dtype,
    fmt=MemoryFormat.KV_2LTD
)
memory_obj.tensor.copy_(tensor)

backend = list(storage_manager.storage_backends.values())[1]

print("\n DIRECT PUT")
print(memory_obj.get_tensor)
backend.batched_submit_put_task([key], [memory_obj])

print("\n DIRECT GET")
result = backend.get_blocking(key)
result.ref_count_down()
print("Result:", result)

print("\n TESTING GET")
result = storage_manager.get(key)

print("\nResult:", result)