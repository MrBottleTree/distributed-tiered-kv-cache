import sys
import os
import time
import grpc
import numpy as np
import argparse
from typing import List, Tuple

# Ensure we can import the generated gRPC code
sys.path.append(os.path.join(os.path.dirname(__file__), "py_client"))

import kv_cache_pb2
import kv_cache_pb2_grpc

def generate_dummy_kv(layer_id: int, seq_len: int, num_heads: int, head_dim: int) -> kv_cache_pb2.KVCacheBlock:
    """Generates a dummy KV block for testing."""
    shape = (seq_len, num_heads, head_dim)
    key_np = np.random.randn(*shape).astype(np.float16)
    value_np = np.random.randn(*shape).astype(np.float16)

    return kv_cache_pb2.KVCacheBlock(
        metadata=kv_cache_pb2.CacheMetadata(
            session_id="stats_benchmark_session",
            layer_id=layer_id,
            sequence_index=0
        ),
        key_tensor=kv_cache_pb2.KVTensor(
            precision=kv_cache_pb2.PRECISION_FP16,
            data=key_np.tobytes(),
            shape=list(shape)
        ),
        value_tensor=kv_cache_pb2.KVTensor(
            precision=kv_cache_pb2.PRECISION_FP16,
            data=value_np.tobytes(),
            shape=list(shape)
        )
    )

def run_benchmark(target: str, num_layers: int, seq_len: int, num_heads: int, head_dim: int):
    print(f"Connecting to Machine B at {target}...")
    
    # Increase max message size to 128MB
    options = [
        ('grpc.max_send_message_length', 128 * 1024 * 1024),
        ('grpc.max_receive_message_length', 128 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(target, options=options)
    stub = kv_cache_pb2_grpc.KVCacheServiceStub(channel)

    store_latencies = []
    fetch_latencies = []
    total_bytes_sent = 0
    total_bytes_received = 0

    print(f"\n[Benchmarking] {num_layers} layers, {seq_len} tokens/layer")
    print("-" * 50)

    # 1. Store Benchmark
    for i in range(num_layers):
        block = generate_dummy_kv(i, seq_len, num_heads, head_dim)
        payload_size = len(block.key_tensor.data) + len(block.value_tensor.data)
        total_bytes_sent += payload_size

        start_time = time.perf_counter()
        try:
            response = stub.Store(block)
            end_time = time.perf_counter()
            if response.success:
                store_latencies.append(end_time - start_time)
                print(f"Layer {i:02d} Stored: {payload_size / 1024 / 1024:.2f} MB in {(end_time - start_time)*1000:.2f} ms")
            else:
                print(f"Layer {i:02d} Store failed: {response.message}")
        except Exception as e:
            print(f"Layer {i:02d} Store error: {e}")

    # 2. Fetch Benchmark
    for i in range(num_layers):
        signal = kv_cache_pb2.ContextSignal(
            session_id="stats_benchmark_session",
            request_sequence_id=0 # Matching the Store sequence_index logic in machine_b_server.cpp
        )

        start_time = time.perf_counter()
        try:
            # Note: Fetch implementation in machine_b_server.cpp uses ContextSignal
            response_block = stub.Fetch(signal)
            end_time = time.perf_counter()
            
            fetch_payload_size = len(response_block.key_tensor.data) + len(response_block.value_tensor.data)
            total_bytes_received += fetch_payload_size
            fetch_latencies.append(end_time - start_time)
            print(f"Layer {i:02d} Fetched: {fetch_payload_size / 1024 / 1024:.2f} MB in {(end_time - start_time)*1000:.2f} ms")
        except Exception as e:
            print(f"Layer {i:02d} Fetch error: {e}")

    # Statistics Calculation
    print("\n" + "="*50)
    print("KV CACHE PERFORMANCE STATISTICS")
    print("="*50)
    
    if store_latencies:
        avg_store = sum(store_latencies) / len(store_latencies)
        print(f"Avg Store Latency:  {avg_store*1000:.2f} ms")
        print(f"Store Throughput:   {(total_bytes_sent / sum(store_latencies)) / 1024 / 1024:.2f} MB/s")
    
    if fetch_latencies:
        avg_fetch = sum(fetch_latencies) / len(fetch_latencies)
        print(f"Avg Fetch Latency:  {avg_fetch*1000:.2f} ms")
        print(f"Fetch Throughput:   {(total_bytes_received / sum(fetch_latencies)) / 1024 / 1024:.2f} MB/s")

    print(f"Total Data Moved:   {(total_bytes_sent + total_bytes_received) / 1024 / 1024:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Benchmark & Statistics Tool")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of Machine B server")
    parser.add_argument("--port", type=str, default="8080", help="Port of Machine B server")
    parser.add_argument("--layers", type=int, default=10, help="Number of layers to simulate")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length per layer")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")

    args = parser.parse_args()
    target_addr = f"{args.ip}:{args.port}"
    
    run_benchmark(target_addr, args.layers, args.seq_len, args.heads, args.dim)
