import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "py_client"))

import grpc
import kv_cache_pb2
import kv_cache_pb2_grpc
import numpy as np
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token=""

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)

# Connect to Machine B
channel = grpc.insecure_channel("3.111.32.60:8080")
stub = kv_cache_pb2_grpc.KVCacheServiceStub(channel)

prompt = "hello " * 1000
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_key_values = outputs.past_key_values
session_id = "stress_session"


for layer_id, (key, value) in enumerate(past_key_values):

    key_np = key.squeeze(0).cpu().numpy().astype(np.float16)
    value_np = value.squeeze(0).cpu().numpy().astype(np.float16)

    block = kv_cache_pb2.KVCacheBlock(
        metadata=kv_cache_pb2.CacheMetadata(
            session_id=session_id,
            layer_id=layer_id,
            sequence_index=0  # entire sequence block
        ),
        key_tensor=kv_cache_pb2.KVTensor(
            precision=kv_cache_pb2.PRECISION_FP16,
            data=key_np.tobytes(),
            shape=list(key_np.shape)
        ),
        value_tensor=kv_cache_pb2.KVTensor(
            precision=kv_cache_pb2.PRECISION_FP16,
            data=value_np.tobytes(),
            shape=list(value_np.shape)
        )
    )

    stub.Store(block)

print("All layers pushed to machine B")



retrieved = []

for layer_id in range(len(past_key_values)):
    metadata = kv_cache_pb2.CacheMetadata(
        session_id=session_id,
        layer_id=layer_id,
        sequence_index=0
    )

    response = stub.Get(metadata)

    key = np.frombuffer(
        response.key_tensor.data,
        dtype=np.float16
    ).reshape(response.key_tensor.shape)

    value = np.frombuffer(
        response.value_tensor.data,
        dtype=np.float16
    ).reshape(response.value_tensor.shape)

    retrieved.append((key, value))

print("All layers pulled back")


original_key = past_key_values[0][0].squeeze(0).cpu().numpy()
print(np.allclose(original_key, retrieved[0][0]))
