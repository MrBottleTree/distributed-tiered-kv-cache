import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "py_client"))

import grpc
import kv_cache_pb2
import kv_cache_pb2_grpc
import numpy as np
channel = grpc.insecure_channel(
    "3.111.32.60:8080"
)
stub = kv_cache_pb2_grpc.KVCacheServiceStub(channel)

fake_key = np.random.randn(2, 2).astype(np.float16)
fake_value = np.random.randn(2, 2).astype(np.float16)

block = kv_cache_pb2.KVCacheBlock(
    metadata=kv_cache_pb2.CacheMetadata(
        session_id="test_session",
        layer_id=1,
        sequence_index=0
    ),
    key_tensor=kv_cache_pb2.KVTensor(
        precision=kv_cache_pb2.PRECISION_FP16,
        data=fake_key.tobytes(),
        shape=list(fake_key.shape)
    ),
    value_tensor=kv_cache_pb2.KVTensor(
        precision=kv_cache_pb2.PRECISION_FP16,
        data=fake_value.tobytes(),
        shape=list(fake_value.shape)
    )
)

print("Sending Store request...")

try:
    response = stub.Store(block)
    print("Response:", response)
except Exception as e:
    print("Expected failure (no server yet):", e)




# After Store succeeds

metadata = kv_cache_pb2.CacheMetadata(
    session_id="test_session",
    layer_id=1,
    sequence_index=0
)

response = stub.Fetch(metadata)

retrieved_key = np.frombuffer(
    response.key_tensor.data,
    dtype=np.float16
).reshape(response.key_tensor.shape)

retrieved_value = np.frombuffer(
    response.value_tensor.data,
    dtype=np.float16
).reshape(response.value_tensor.shape)

print("Retrieved key:\n", retrieved_key)
print("Retrieved value:\n", retrieved_value)
