# import torch
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj, MemoryFormat
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventManager

# ─────────────────────────────────────────────
# 1. Setup config
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 2. Setup metadata + manager
# ─────────────────────────────────────────────
metadata = LMCacheMetadata(
    model_name="test_model",
    worker_id=0,
    role="worker"
)

event_manager = EventManager()

storage_manager = StorageManager(
    config=config,
    metadata=metadata,
    event_manager=event_manager,
)

print("\n✅ StorageManager initialized")
print("Backends:", storage_manager.list_backends())

# ─────────────────────────────────────────────
# 3. Create fake key
# ─────────────────────────────────────────────
key = CacheEngineKey(
    model_id="test_model",
    block_hash="abc123",   # may need adjustment
    seq_len=64
)

# ─────────────────────────────────────────────
# 4. Create fake MemoryObj
# ─────────────────────────────────────────────
tensor = torch.randn(1, 8, 64, 64)  # fake KV-like tensor

mem_obj = MemoryObj(
    tensor=tensor,
    fmt=MemoryFormat.KV_2LTD
)

# ─────────────────────────────────────────────
# 5. TEST PUT
# ─────────────────────────────────────────────
print("\n🚀 TESTING PUT")
storage_manager.batched_put([key], [mem_obj])

# ─────────────────────────────────────────────
# 6. TEST GET
# ─────────────────────────────────────────────
print("\n🚀 TESTING GET")
result = storage_manager.get(key)

print("\nResult:", result)