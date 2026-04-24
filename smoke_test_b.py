"""Quick gRPC smoke test: Machine A → Machine B. No vLLM needed."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                'LMCache/lmcache/v1/storage_backend'))
import grpc
import evicpress_pb2 as pb2
import evicpress_pb2_grpc as grpc2

addr = os.environ.get('MACHINE_B', '172.31.12.251') + ':50051'
print(f'[smoke] connecting to {addr}')
ch   = grpc.insecure_channel(addr)
stub = grpc2.EvicPressServiceStub(ch)

s = stub.GetStats(pb2.StatsRequest())
print(f'[stats]    tier2={s.tier2_blocks}blk  tier3={s.tier3_blocks}blk  '
      f'hit_rate={s.hit_rate:.3f}  promotions={s.tier1_promotions}')

# --- Test 1: miss before store ---
r = stub.Lookup(pb2.LookupRequest(block_id='smoke-test'))
assert not r.hit, 'expected miss before store'
print(f'[lookup]   hit={r.hit} tier={r.tier}  (miss ✓)')

# --- Test 2: store (quality computed by Machine B from data) ---
# Use random data so it's hard to compress → likely stays in T2, not promoted to T1
data = bytes(random.getrandbits(8) for _ in range(4096))
sr = stub.Store(pb2.StoreRequest(block_id='smoke-test', data=data, quality_score=0.5))
assert sr.success, f'store failed: {sr.message}'
print(f'[store]    success={sr.success} tier={sr.tier} msg={sr.message!r}')

# --- Test 3: hit after store ---
r2 = stub.Lookup(pb2.LookupRequest(block_id='smoke-test'))
assert r2.hit, 'expected hit after store'
print(f'[lookup2]  hit={r2.hit} tier={r2.tier}  (hit ✓)')

# --- Test 4: retrieve (only if Machine B holds the canonical copy: tier 2 or 3) ---
if sr.tier in (2, 3):
    ret = stub.Retrieve(pb2.RetrieveRequest(block_id='smoke-test'))
    assert ret.found and len(ret.data) == 4096
    print(f'[retrieve] found={ret.found} tier={ret.tier} size={len(ret.data)}B  ✓')
else:
    print(f'[retrieve] skipped — block promoted to tier=1 (Machine A holds it)')

# --- Test 5: delete ---
d = stub.Delete(pb2.DeleteRequest(block_id='smoke-test'))
assert d.success
print(f'[delete]   success={d.success}  ✓')

ch.close()
print('\nALL SMOKE TESTS PASSED')
