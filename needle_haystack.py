(venv) ubuntu@ip-172-31-11-107:~/distributed-tiered-kv-cache$ cat needle_haystack.py 
"""
Needle-in-a-Haystack cache verification script.

Inserts a unique fact ("needle") into a large text context ("haystack"),
then queries the model to retrieve it.  Re-using the same haystack exercises
LMCache prefix-caching, so you see real MISS→HIT latency reduction and can
verify retrieval accuracy across needle positions.

Each "trial" = one (position, repeat) pair.
  - repeat 0  → expected cache MISS  (first time this haystack is seen)
  - repeat 1+ → expected cache HIT   (same prefix, served from Machine B)

Statistics reported per position and overall:
  latency (s), accuracy (%), speedup (miss→hit), cache tier counts, hit_rate

Usage:
    python needle_haystack.py                       # defaults
    python needle_haystack.py --haystack-kb 32      # ~32 KB context
    python needle_haystack.py --positions 5         # 5 depth points
    python needle_haystack.py --repeats 4           # 1 miss + 3 hits per position
    python needle_haystack.py --output-json out.json
"""

import os, sys, time, argparse, textwrap, socket, random, json, statistics, string
os.environ.setdefault("LMCACHE_CONFIG_FILE", "lmcache_config.yaml")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LMCACHE_LOG_LEVEL", "WARNING")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# ── CLI ────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Needle-in-a-Haystack LMCache verification")
p.add_argument("--max-tokens",  type=int,   default=80)
p.add_argument("--temperature", type=float, default=0.0,   help="0 = greedy, best for accuracy")
p.add_argument("--machine-b",   default=os.environ.get("MACHINE_B", "172.31.7.166"))
p.add_argument("--haystack-kb", type=int,   default=16,    help="Target haystack size in KB")
p.add_argument("--positions",   type=int,   default=4,     help="Needle depth points to test")
p.add_argument("--repeats",     type=int,   default=3,     help="Queries per position (1 miss + N-1 hits)")
p.add_argument("--output-json", default=None, metavar="FILE")
p.add_argument("--no-cache-info", action="store_true",     help="Hide Machine B tier stats")
args = p.parse_args()

# ── Haystack corpus ────────────────────────────────────────────────────────────
# Varied sentences so the tokenizer sees realistic token diversity.
_CORPUS = [
    "The global economy experienced unprecedented disruption during the last fiscal quarter.",
    "Researchers at the institute published findings on climate adaptation strategies in coastal regions.",
    "Municipal authorities announced a new public transit expansion plan covering the eastern districts.",
    "The software engineering team completed a major refactor of the authentication subsystem.",
    "Migratory patterns of Arctic terns have shifted measurably over the past two decades.",
    "The board of directors approved the merger after months of regulatory review and negotiation.",
    "An unexpected solar flare disrupted satellite communications across the northern hemisphere.",
    "Archaeologists uncovered a previously unknown settlement dating back to the Bronze Age.",
    "The central bank raised interest rates by a quarter point to curb inflationary pressure.",
    "A new deep-learning architecture achieved state-of-the-art results on several benchmark tasks.",
    "The city council passed a resolution requiring all new buildings to meet green energy standards.",
    "Oceanographers deployed a network of buoys to monitor sea surface temperature anomalies.",
    "The marathon runner broke the course record despite difficult weather conditions at the start.",
    "Engineers successfully completed the stress tests for the new suspension bridge design.",
    "An international coalition agreed on a framework for regulating autonomous weapon systems.",
    "The pharmaceutical company announced positive phase-three trial results for its new vaccine.",
    "Historians debated the long-term consequences of the treaty signed in the eighteenth century.",
    "Local farmers adopted precision irrigation techniques, reducing water consumption by thirty percent.",
    "The telecommunications provider upgraded its backbone infrastructure to support next-generation traffic.",
    "A documentary filmmaker spent three years documenting the lives of deep-sea fishermen.",
    "Supply-chain disruptions continued to affect semiconductor availability across multiple industries.",
    "The conservation group reintroduced a dozen wolves to the national park's northern territory.",
    "An independent audit revealed discrepancies in the company's reported carbon-offset figures.",
    "The space agency confirmed that its latest Mars lander successfully deployed its solar panels.",
    "Linguists identified structural similarities between two languages previously thought unrelated.",
    "The stadium renovation project was completed two months ahead of schedule and under budget.",
    "Cybersecurity analysts detected a coordinated phishing campaign targeting financial institutions.",
    "Botanists catalogued over two hundred previously undescribed plant species in the rainforest.",
    "The court ruling set a significant precedent for intellectual property cases involving AI-generated work.",
    "Community volunteers restored the historic watermill to full working order over a single weekend.",
]

def _build_haystack(target_bytes: int) -> str:
    """Repeat corpus sentences until we reach approximately target_bytes."""
    rng = random.Random(42)          # deterministic so prefix cache keys are stable
    sentences = _CORPUS[:]
    buf = []
    total = 0
    while total < target_bytes:
        s = rng.choice(sentences)
        buf.append(s)
        total += len(s) + 1
    return "  ".join(buf)


def _insert_needle(haystack: str, needle: str, depth_frac: float) -> str:
    """Insert needle at approximately depth_frac (0.0–1.0) of the haystack."""
    idx = int(len(haystack) * depth_frac)
    # Snap to nearest sentence boundary (period + space)
    snap = haystack.rfind(". ", 0, idx)
    if snap == -1:
        snap = 0
    insert_at = snap + 2
    return haystack[:insert_at] + needle + "  " + haystack[insert_at:]


def _make_needle(secret: str) -> str:
    return f"IMPORTANT: The secret passphrase for this document is '{secret}'."


def _random_secret() -> str:
    """6-character alphanumeric secret that the model should be able to quote exactly."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


# ── Connectivity check ─────────────────────────────────────────────────────────
def _check_b():
    s = socket.socket(); s.settimeout(2)
    ok = s.connect_ex((args.machine_b, 50051)) == 0
    s.close()
    return ok

machine_b_ok = _check_b()
if not machine_b_ok:
    print(f"[warn] Machine B at {args.machine_b}:50051 unreachable — cache stats will be unavailable", flush=True)

# ── Machine B stats ────────────────────────────────────────────────────────────
def _b_stats():
    if not machine_b_ok:
        return None
    try:
        import grpc, sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                         "LMCache/lmcache/v1/storage_backend"))
        import evicpress_pb2 as pb2, evicpress_pb2_grpc as grpc2
        ch = grpc.insecure_channel(f"{args.machine_b}:50051")
        s  = grpc2.EvicPressServiceStub(ch).GetStats(pb2.StatsRequest(), timeout=1)
        ch.close()
        return s
    except Exception:
        return None

# ── Load model ─────────────────────────────────────────────────────────────────
print("[niah] Loading model…", flush=True)
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    enable_prefix_caching=True,
    max_model_len=16384,
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both"
       
    ),
)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)

# ── Accuracy check ─────────────────────────────────────────────────────────────
def _check_accuracy(reply: str, secret: str) -> bool:
    return secret.upper() in reply.upper()

# ── Prompt builder ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise document-analysis assistant. "
    "Answer questions using only information from the provided document. "
    "Quote exact values when asked for them."
)

def _build_prompt(document: str, question: str) -> str:
    turns = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"<document>\n{document}\n</document>\n\n{question}"},
    ]
    return tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
    )

# ── Print helpers ──────────────────────────────────────────────────────────────
SEP = "─" * 72

def _fmt_stats(s) -> str:
    if s is None:
        return "Machine B unreachable"
    return (f"T1={s.tier1_blocks}blk  T2={s.tier2_blocks}blk  T3={s.tier3_blocks}blk  "
            f"hit_rate={s.hit_rate:.3f}  promotions={s.tier1_promotions}")

def _verdict(correct: bool) -> str:
    return "CORRECT" if correct else "WRONG  "

# ── Main experiment ────────────────────────────────────────────────────────────
target_bytes = args.haystack_kb * 1024
base_haystack = _build_haystack(target_bytes)
actual_kb     = len(base_haystack) / 1024

depth_points = [round(i / (args.positions - 1), 4) if args.positions > 1 else 0.5
                for i in range(args.positions)]

print(f"\n{SEP}")
print(f"  Needle-in-a-Haystack  |  Model: Llama-3.1-8B-Instruct")
print(f"  Machine B  : {args.machine_b}:50051")
print(f"  Haystack   : {actual_kb:.1f} KB  ({len(base_haystack):,} chars)")
print(f"  Positions  : {args.positions}  depths={[f'{d:.0%}' for d in depth_points]}")
print(f"  Repeats    : {args.repeats}  (repeat-0 = expected MISS, rest = expected HIT)")
print(SEP + "\n")

results = []   # list of dicts — one per (position_idx, repeat)
position_summaries = []

for pos_idx, depth in enumerate(depth_points):
    secret   = _random_secret()
    needle   = _make_needle(secret)
    document = _insert_needle(base_haystack, needle, depth)
    question = (
        "What is the secret passphrase mentioned in the document? "
        "Reply with only the passphrase, nothing else."
    )
    prompt = _build_prompt(document, question)

    print(f"{'='*72}")
    print(f"  Position {pos_idx + 1}/{args.positions}  |  depth={depth:.0%}  |  secret='{secret}'")
    print(f"  Needle: \"{needle}\"")
    print(f"{'='*72}")

    pos_times  = []
    pos_correct = []

    for rep in range(args.repeats):
        label = "MISS (expected)" if rep == 0 else f"HIT  (expected, rep {rep})"
        b_before = _b_stats()
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed = time.time() - t0
        b_after  = _b_stats()

        reply   = outputs[0].outputs[0].text.strip()
        correct = _check_accuracy(reply, secret)

        pos_times.append(elapsed)
        pos_correct.append(correct)

        wrapped_reply = textwrap.fill(reply, width=60)
        print(f"  rep={rep}  [{label}]")
        print(f"    latency  : {elapsed:.3f}s")
        print(f"    accuracy : {_verdict(correct)}  (model said: {wrapped_reply!r})")
        if not args.no_cache_info:
            print(f"    cache    : {_fmt_stats(b_after)}")

        result_row = {
            "position_idx":  pos_idx,
            "depth_pct":     round(depth * 100, 1),
            "secret":        secret,
            "repeat":        rep,
            "expected_hit":  rep > 0,
            "latency_s":     round(elapsed, 4),
            "correct":       correct,
            "reply":         reply,
        }
        if b_after:
            result_row.update({
                "t1_blocks":       b_after.tier1_blocks,
                "t2_blocks":       b_after.tier2_blocks,
                "t3_blocks":       b_after.tier3_blocks,
                "hit_rate":        round(b_after.hit_rate, 4),
                "tier1_promotions": b_after.tier1_promotions,
            })
        results.append(result_row)
        print()

    # Per-position summary
    miss_time = pos_times[0]
    hit_times = pos_times[1:]
    speedup   = (miss_time / statistics.mean(hit_times)) if hit_times else float("nan")
    accuracy  = sum(pos_correct) / len(pos_correct) * 100

    pos_sum = {
        "position_idx": pos_idx,
        "depth_pct":    round(depth * 100, 1),
        "miss_latency_s":     round(miss_time, 4),
        "mean_hit_latency_s": round(statistics.mean(hit_times), 4) if hit_times else None,
        "speedup_x":          round(speedup, 2),
        "accuracy_pct":       round(accuracy, 1),
    }
    position_summaries.append(pos_sum)
    print(f"  >> pos summary: miss={miss_time:.3f}s  "
          + (f"mean_hit={statistics.mean(hit_times):.3f}s  speedup={speedup:.2f}x  " if hit_times else "")
          + f"accuracy={accuracy:.0f}%")
    print()

# ── Overall summary ────────────────────────────────────────────────────────────
print(SEP)
print("  OVERALL SUMMARY")
print(SEP)

all_latencies   = [r["latency_s"] for r in results]
miss_latencies  = [r["latency_s"] for r in results if not r["expected_hit"]]
hit_latencies   = [r["latency_s"] for r in results if r["expected_hit"]]
all_correct     = [r["correct"]   for r in results]
hit_correct     = [r["correct"]   for r in results if r["expected_hit"]]
miss_correct    = [r["correct"]   for r in results if not r["expected_hit"]]

overall_accuracy = sum(all_correct) / len(all_correct) * 100 if all_correct else 0

print(f"  Trials total        : {len(results)}")
print(f"  Overall accuracy    : {overall_accuracy:.1f}%  "
      f"({sum(all_correct)}/{len(all_correct)} correct)")
if miss_correct:
    print(f"  Miss accuracy       : {sum(miss_correct)/len(miss_correct)*100:.1f}%")
if hit_correct:
    print(f"  Hit  accuracy       : {sum(hit_correct)/len(hit_correct)*100:.1f}%")
print()
if miss_latencies:
    print(f"  Miss latency        : mean={statistics.mean(miss_latencies):.3f}s  "
          f"min={min(miss_latencies):.3f}s  max={max(miss_latencies):.3f}s")
if hit_latencies:
    print(f"  Hit  latency        : mean={statistics.mean(hit_latencies):.3f}s  "
          f"min={min(hit_latencies):.3f}s  max={max(hit_latencies):.3f}s")
if miss_latencies and hit_latencies:
    overall_speedup = statistics.mean(miss_latencies) / statistics.mean(hit_latencies)
    print(f"  Cache speedup       : {overall_speedup:.2f}x  "
          f"({statistics.mean(miss_latencies):.3f}s → {statistics.mean(hit_latencies):.3f}s)")

print()
print(f"  {'Pos':>3}  {'Depth':>6}  {'Secret':>6}  {'Miss (s)':>9}  {'Hit (s)':>9}  {'Speedup':>8}  {'Accuracy':>9}")
print(f"  {'---':>3}  {'------':>6}  {'------':>6}  {'---------':>9}  {'---------':>9}  {'--------':>8}  {'---------':>9}")
for ps in position_summaries:
    hit_str  = f"{ps['mean_hit_latency_s']:.3f}" if ps["mean_hit_latency_s"] is not None else "   n/a"
    spd_str  = f"{ps['speedup_x']:.2f}x"         if ps["mean_hit_latency_s"] is not None else "   n/a"
    print(f"  {ps['position_idx']+1:>3}  {ps['depth_pct']:>5.0f}%  "
          f"{results[ps['position_idx']*args.repeats]['secret']:>6}  "
          f"{ps['miss_latency_s']:>9.3f}  {hit_str:>9}  {spd_str:>8}  "
          f"{ps['accuracy_pct']:>8.1f}%")

# Final cache state
final_stats = _b_stats()
if final_stats and not args.no_cache_info:
    print()
    print(f"  Final cache state   : {_fmt_stats(final_stats)}")

print(SEP)

# ── JSON output ────────────────────────────────────────────────────────────────
if args.output_json:
    payload = {
        "config": {
            "haystack_kb":   args.haystack_kb,
            "actual_kb":     round(actual_kb, 2),
            "positions":     args.positions,
            "repeats":       args.repeats,
            "temperature":   args.temperature,
            "machine_b":     args.machine_b,
        },
        "trials":   results,
        "position_summaries": position_summaries,
        "overall": {
            "accuracy_pct":          round(overall_accuracy, 2),
            "miss_mean_latency_s":   round(statistics.mean(miss_latencies), 4) if miss_latencies else None,
            "hit_mean_latency_s":    round(statistics.mean(hit_latencies),  4) if hit_latencies  else None,
            "cache_speedup_x":       round(statistics.mean(miss_latencies) / statistics.mean(hit_latencies), 3)
                                     if (miss_latencies and hit_latencies) else None,
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[niah] Results saved → {args.output_json}")
