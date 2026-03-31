"""
KV-Cache client  —  prefix-hash keyed, longest-prefix-match aware
==================================================================
Changes vs v1
  • compute_prefix_chain()   builds sub-prefix hashes at block boundaries
                             so the server can validate genuine-prefix-ness
  • build_store_request()    includes the full prefix_chain in KVCacheKey
  • build_fetch_request()    includes the chain so the server can validate
                             candidates on Fetch too
  • main()                   demonstrates a partial-hit scenario:
                             store 64 tokens, then fetch with 100 tokens
                             → server should return matched_tokens=64
"""

import hashlib
import sys
import os

import grpc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "py_client"))
import kv_cache_pb2
import kv_cache_pb2_grpc

# ───────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN     = ""
SERVER_ADDR  = "3.111.32.60:8080"
CHAIN_BLOCK  = 64   # compute a sub-prefix hash every N tokens

device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────────────────────
# Hashing utilities
# ───────────────────────────────────────────────────────────────────────────

def _hash(token_ids: np.ndarray) -> str:
    """sha256 of a flat int32 token-id array."""
    return hashlib.sha256(token_ids.astype(np.int32).tobytes()).hexdigest()


def compute_prefix_hash(input_ids: torch.Tensor) -> str:
    """Hash of the full token sequence (scalar string)."""
    return _hash(input_ids.squeeze(0).cpu().numpy())


def compute_prefix_chain(
    input_ids: torch.Tensor,
    block_size: int = CHAIN_BLOCK,
) -> list[str]:
    """
    Hashes at every block boundary up to (and including) the full length.

    Example with 200 tokens and block_size=64:
        hashes of tokens[:64], tokens[:128], tokens[:192], tokens[:200]

    The server stores these so it can later confirm "is entry E a genuine
    prefix of query Q?" by checking whether E.prefix_hash ∈ Q.prefix_chain.
    """
    ids = input_ids.squeeze(0).cpu().numpy()
    n   = len(ids)
    boundaries = list(range(block_size, n, block_size)) + [n]
    return [_hash(ids[:b]) for b in boundaries]


# ───────────────────────────────────────────────────────────────────────────
# Tensor serialisation helpers
# ───────────────────────────────────────────────────────────────────────────

def _tensor_to_proto(t: torch.Tensor) -> kv_cache_pb2.KVTensor:
    arr = t.squeeze(0).cpu().numpy().astype(np.float16)
    return kv_cache_pb2.KVTensor(
        precision=kv_cache_pb2.PRECISION_FP16,
        data=arr.tobytes(),
        shape=list(arr.shape),
    )


def _proto_to_numpy(proto: kv_cache_pb2.KVTensor) -> np.ndarray:
    return np.frombuffer(proto.data, dtype=np.float16).reshape(proto.shape)


# ───────────────────────────────────────────────────────────────────────────
# Request builders
# ───────────────────────────────────────────────────────────────────────────

def build_store_request(
    model_id: str,
    input_ids: torch.Tensor,
    past_key_values,
) -> kv_cache_pb2.StoreRequest:
    num_tokens   = input_ids.shape[-1]
    prefix_hash  = compute_prefix_hash(input_ids)
    prefix_chain = compute_prefix_chain(input_ids)

    layers = [
        kv_cache_pb2.LayerKV(
            layer_id=i,
            key_tensor=_tensor_to_proto(k),
            value_tensor=_tensor_to_proto(v),
        )
        for i, (k, v) in enumerate(past_key_values)
    ]

    return kv_cache_pb2.StoreRequest(
        key=kv_cache_pb2.KVCacheKey(
            model_id=model_id,
            prefix_hash=prefix_hash,
            num_tokens=num_tokens,
            prefix_chain=prefix_chain,      # ← new: enables server validation
        ),
        value=kv_cache_pb2.KVCacheValue(layers=layers, num_tokens=num_tokens),
    )


def build_fetch_request(
    model_id: str,
    input_ids: torch.Tensor,
) -> kv_cache_pb2.FetchRequest:
    """
    Build a Fetch for the given token sequence.
    We send the full prefix_chain so the server can validate partial hits.
    """
    return kv_cache_pb2.FetchRequest(
        key=kv_cache_pb2.KVCacheKey(
            model_id=model_id,
            prefix_hash=compute_prefix_hash(input_ids),
            num_tokens=input_ids.shape[-1],
            prefix_chain=compute_prefix_chain(input_ids),
        )
    )


def unpack_fetch_response(
    resp: kv_cache_pb2.FetchResponse,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Returns (matched_tokens, [(key_np, val_np), …]) sorted by layer_id.
    matched_tokens == 0 and empty list on miss.
    """
    if not resp.found:
        return 0, []
    layers = sorted(resp.value.layers, key=lambda l: l.layer_id)
    pairs  = [(_proto_to_numpy(l.key_tensor), _proto_to_numpy(l.value_tensor))
              for l in layers]
    return resp.matched_tokens, pairs


# ───────────────────────────────────────────────────────────────────────────
# Main pipeline — demonstrates partial-hit scenario
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )

    channel = grpc.insecure_channel(SERVER_ADDR)
    stub    = kv_cache_pb2_grpc.KVCacheServiceStub(channel)

    # ── 1. Store a short prefix (64 tokens) ────────────────────────────────
    short_prompt = "hello " * 32          # ≈ 64 tokens
    short_inputs = tokenizer(short_prompt, return_tensors="pt").to(device)
    actual_short = short_inputs["input_ids"].shape[-1]
    print(f"\nStoring short prefix  ({actual_short} tokens) …")

    with torch.no_grad():
        short_out = model(**short_inputs, use_cache=True)

    store_req  = build_store_request(MODEL_NAME, short_inputs["input_ids"],
                                     short_out.past_key_values)
    store_resp = stub.Store(store_req)
    assert store_resp.success, f"Store failed: {store_resp.message}"
    print(f"✓ Stored  ({actual_short} tokens)")

    # ── 2. Fetch with a longer prefix (100 tokens) — partial hit expected ──
    long_prompt  = "hello " * 50          # ≈ 100 tokens
    long_inputs  = tokenizer(long_prompt, return_tensors="pt").to(device)
    actual_long  = long_inputs["input_ids"].shape[-1]
    print(f"\nFetching longer prefix ({actual_long} tokens) …")

    fetch_req         = build_fetch_request(MODEL_NAME, long_inputs["input_ids"])
    fetch_resp        = stub.Fetch(fetch_req)
    matched, kv_pairs = unpack_fetch_response(fetch_resp)

    if not fetch_resp.found:
        print("✗ MISS — no usable prefix found")
    else:
        print(f"✓ HIT   matched_tokens={matched}  layers={len(kv_pairs)}  "
              f"reuse={matched/actual_long*100:.1f}%")

    # ── 3. Exact-match fetch (should also hit) ─────────────────────────────
    print(f"\nFetching exact match  ({actual_short} tokens) …")
    exact_req          = build_fetch_request(MODEL_NAME, short_inputs["input_ids"])
    exact_resp         = stub.Fetch(exact_req)
    matched_e, pairs_e = unpack_fetch_response(exact_resp)

    if not exact_resp.found:
        print("✗ MISS — unexpected!")
    else:
        # Sanity: layer-0 key should round-trip exactly
        orig_k = short_out.past_key_values[0][0].squeeze(0).cpu().numpy().astype(np.float16)
        ok     = np.allclose(orig_k, pairs_e[0][0], atol=1e-3)
        print(f"✓ HIT   matched_tokens={matched_e}  layer-0 key match={ok}")


if __name__ == "__main__":
    main()
