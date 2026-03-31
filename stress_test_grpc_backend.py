# test-grpc-backend.py

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.utils import CacheEngineKey
from lmcache.v1.event_manager import EventManager

import torch
import time

# 1. CONFIG
config = LMCacheEngineConfig()

config.storage_plugins = ["grpc"]
config.extra_config = {
    "storage_plugin.grpc.module_path": "lmcache.v1.storage_backend.grpc_backend",
    "storage_plugin.grpc.class_name": "GRPCBackend",
    "grpc_server": "127.0.0.1:8080",
}

config.enable_pd = False
config.local_cpu = True
config.max_local_cpu_size = 2
config.local_disk = None
config.remote_url = None

metadata = LMCacheMetadata(
    model_name="test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.float32,
    kv_shape=(2, 2, 16, 16),
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

# 2. HELPERS
local_cpu_backend = storage_manager.storage_backends["LocalCPUBackend"]
allocator = local_cpu_backend.get_allocator_backend()


def create_fake_kv(shape=(2, 8, 64, 64)):
    tensor = torch.randn(*shape)

    memory_obj = allocator.allocate(
        tensor.shape,
        tensor.dtype,
        fmt=MemoryFormat.KV_2LTD
    )
    memory_obj.tensor.copy_(tensor)
    return memory_obj


def create_key(chunk_id):
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_id,
        dtype=torch.float32,
    )


# 3. CREATE CHUNKS (simulate vLLM)
NUM_CHUNKS = 5

keys = []
objs = []

for i in range(NUM_CHUNKS):
    key = create_key(i)
    obj = create_fake_kv()

    keys.append(key)
    objs.append(obj)

print(f"\n Created {NUM_CHUNKS} KV chunks")

# 4. PREFILL (STORE KV)
print("\nPREFILL: Storing KV chunks")
storage_manager.batched_put(keys, objs)

# 5. FULL PREFIX FETCH
print("\n FULL PREFIX FETCH (FORCED REMOTE)")

grpc_backend = storage_manager.storage_backends["grpc"]

for key in keys:
    result = grpc_backend.get_blocking(key)
    print(f"remote fetch chunk={key.chunk_hash}, found={result is not None}")
    if result:
        result.ref_count_down()

# 6. PARTIAL PREFIX TEST
print("\n PARTIAL PREFIX TEST (first 3 chunks)")

partial_keys = keys[:3]

for key in partial_keys:
    result = storage_manager.get(key)
    print(f"[PARTIAL FETCH] chunk={key.chunk_hash}, found={result is not None}")
    if result:
        result.ref_count_down()

# 7. PREFIX BREAK TEST
print("\n PREFIX BREAK TEST (missing chunk 2)")

for i, key in enumerate(keys):
    if i == 2:
        print(f"[FETCH] chunk={key.chunk_hash}, ❌ simulated miss → stop")
        break

    result = storage_manager.get(key)
    print(f"[FETCH] chunk={key.chunk_hash}, found={result is not None}")
    if result:
        result.ref_count_down()

# 8. REPEATED QUERY TEST
print("\n REPEATED QUERY TEST")

for round_id in range(2):
    print(f"\nRound {round_id + 1}")
    for key in keys:
        result = storage_manager.get(key)
        print(f"[REPEAT FETCH] chunk={key.chunk_hash}, found={result is not None}")
        if result:
            result.ref_count_down()

# 9. BATCHED CONTAINS TEST
print("\nBATCHED CONTAINS TEST")

hit_chunks, mapping = storage_manager.batched_contains(keys)

print("Hit chunks:", hit_chunks)
print("Mapping:", mapping)

# 10. TIMING TEST
print("\nTIMING TEST")

start = time.time()

for key in keys:
    result = storage_manager.get(key)
    if result:
        result.ref_count_down()

end = time.time()

print(f"⏱️ Fetch time for {NUM_CHUNKS} chunks: {end - start:.4f}s")
#should be zero right now because I am on local host.
#for implementations check whether server is getting fetch hits

# DONE  HOPEFULLY YAY
print("\n TEST COMPLETE")
