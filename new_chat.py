"""
Interactive chat with Llama-3.1-8B via vLLM + LMCache + EvicPress (Machine B).

The system prompt is shared across every turn — LMCache caches it after the
first message and serves it from Machine B on all subsequent turns, so you
can see real cache-hit / promotion behaviour while chatting.

Usage:
    python chat.py                  # interactive mode
    python chat.py --max-tokens 200 # longer replies
    python chat.py --no-cache-info  # hide cache status lines
"""

import os, sys, time, argparse, textwrap, socket
os.environ.setdefault("LMCACHE_CONFIG_FILE", "lmcache_config.yaml")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LMCACHE_LOG_LEVEL", "WARNING")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# ── Args ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="EvicPress-backed interactive chat")
p.add_argument("--max-tokens",   type=int,   default=300)
p.add_argument("--temperature",  type=float, default=0.7)
p.add_argument("--no-cache-info",action="store_true")
p.add_argument("--machine-b",    default=os.environ.get("MACHINE_B","172.31.7.166"))
args = p.parse_args()

SYSTEM_PROMPT = (
    "You are a helpful, concise, and friendly AI assistant. "
    "You answer questions clearly and directly. "
    "When writing code you use clean, readable style. "
    "You acknowledge uncertainty when you are not sure of an answer."
)

# ── Connectivity check ─────────────────────────────────────────────────────────
def _check_b():
    s = socket.socket(); s.settimeout(2)
    ok = s.connect_ex((args.machine_b, 50051)) == 0
    s.close()
    return ok

if not _check_b():
    print(f"[warn] Machine B at {args.machine_b}:50051 is unreachable — cache will be skipped", flush=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("[chat] Loading model…", flush=True)
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

# ── Cache stats helper ─────────────────────────────────────────────────────────
def _b_stats():
    try:
        import grpc, sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                         'LMCache/lmcache/v1/storage_backend'))
        import evicpress_pb2 as pb2, evicpress_pb2_grpc as grpc2
        ch = grpc.insecure_channel(f"{args.machine_b}:50051")
        s  = grpc2.EvicPressServiceStub(ch).GetStats(pb2.StatsRequest(), timeout=1)
        ch.close()
        return s
    except Exception:
        return None

# ── Chat loop ──────────────────────────────────────────────────────────────────
history = []   # list of {"role": ..., "content": ...}

def _build_prompt():
    """Build a Llama-3 chat prompt with the full conversation history."""
    turns = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    return tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
def _build_prompt():
    """Build a Mistral chat prompt."""
    if args.stateless and history:
        # ONLY send last user message (no history)
        turns = [{"role": "user", "content": history[-1]["content"]}]
    else:
        # normal behavior
        turns = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    return tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
    )   )

print("\n" + "="*60)
print("  EvicPress Chat  |  Model: Llama-3.1-8B-Instruct")
print(f"  Machine B: {args.machine_b}:50051")
print("  Type 'quit' or Ctrl-C to exit, 'clear' to reset history")
print("="*60 + "\n")

turn = 0
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[chat] bye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("[chat] bye!")
        break
    if user_input.lower() == "clear":
        history.clear()
        turn = 0
        print("[chat] history cleared\n")
        continue

    history.append({"role": "user", "content": user_input})
    prompt = _build_prompt()

    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.time() - t0

    reply = outputs[0].outputs[0].text.strip()
    history.append({"role": "assistant", "content": reply})

    # Wrap reply for readability
    wrapped = textwrap.fill(reply, width=72, subsequent_indent="       ")
    print(f"\nBot:   {wrapped}\n")

    turn += 1
    if not args.no_cache_info:
        s = _b_stats()
        if s:
            tier_str = (f"T1={s.tier1_blocks}blk  "
                        f"T2={s.tier2_blocks}blk  "
                        f"T3={s.tier3_blocks}blk  "
                        f"hit_rate={s.hit_rate:.2f}  "
                        f"promotions={s.tier1_promotions}")
        else:
            tier_str = "Machine B unreachable"
        print(f"       [{elapsed:.2f}s | {tier_str}]\n")
