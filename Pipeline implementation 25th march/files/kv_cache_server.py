
# Run on Machine B:
#     python kv_cache_server.py
# """

import sys
import os
from concurrent import futures

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), "py_client"))
import kv_cache_pb2
import kv_cache_pb2_grpc

LISTEN_ADDR = "127.0.0.1:8080"


class KVCacheServicer(kv_cache_pb2_grpc.KVCacheServiceServicer):

    def __init__(self):
        self._store = {}

    # ─────────────────────────────────────────────
    # KEY
    # ─────────────────────────────────────────────
    def _key_tuple(self, key):
        return (
            key.model_name,
            key.world_size,
            key.worker_id,
            key.chunk_hash
        )

    # ─────────────────────────────────────────────
    # STORE
    # ─────────────────────────────────────────────
    def Store(self, request, context):
        k = self._key_tuple(request.key)

        self._store[k] = request.value

        print(
            f"[STORE] model={request.key.model_name} "
            f"chunk={request.key.chunk_hash} "
            f"layers={len(request.value.layers)}"
        )

        return kv_cache_pb2.StoreResponse(success=True, message="OK")

    # ─────────────────────────────────────────────
    # FETCH
    # ─────────────────────────────────────────────
    def Fetch(self, request, context):
        k = self._key_tuple(request.key)

        if k not in self._store:
            print(
                f"[FETCH] MISS model={request.key.model_name} "
                f"chunk={request.key.chunk_hash}"
            )
            return kv_cache_pb2.FetchResponse(found=False)

        value = self._store[k]

        print(
            f"[FETCH] HIT model={request.key.model_name} "
            f"chunk={request.key.chunk_hash}"
        )

        return kv_cache_pb2.FetchResponse(
            found=True,
            value=value,
            matched_tokens=value.num_tokens
        )

    # ─────────────────────────────────────────────
    # DELETE
    # ─────────────────────────────────────────────
    def Delete(self, request, context):
        k = self._key_tuple(request.key)

        removed = self._store.pop(k, None)

        return kv_cache_pb2.DeleteResponse(
            success=removed is not None,
            message="OK" if removed else "NOT_FOUND"
        )
# ─────────────────────────────────────────────────────────────────────────────

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    kv_cache_pb2_grpc.add_KVCacheServiceServicer_to_server(
        KVCacheServicer(), server
    )
    server.add_insecure_port(LISTEN_ADDR)
    server.start()
    print(f"KV-Cache server listening on {LISTEN_ADDR}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
