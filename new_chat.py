"""
Interactive chat with Mistral-7B-Instruct-v0.3 via vLLM + LMCache + EvicPress.

The system prompt is prepended to the first user turn (Mistral's chat template
does not accept role="system"). LMCache caches the shared prefix so subsequent
turns hit Machine B's tiered cache instead of recomputing from scratch.

Usage:
    MACHINE_B=172.31.12.251 python new_chat.py
    python new_chat.py --max-tokens 200
    python new_chat.py --no-cache-info
"""

import argparse
import os
import socket
import sys
import textwrap
import time

# Resolve LMCACHE_CONFIG_FILE relative to this script so it works from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LMCACHE_CONFIG_FILE", os.path.join(_HERE, "lmcache_config.yaml"))
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LMCACHE_LOG_LEVEL", "WARNING")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# Make the Machine B proto stubs importable once (used by _b_stats).
sys.path.insert(0, os.path.join(_HERE, "LMCache", "lmcache", "v1", "storage_backend"))

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


# ── Args ───────────────────────────────────────────────────────────────────────
# Try to discover Machine B from the LMCache config if not set in env.
_default_b = os.environ.get("MACHINE_B")
if not _default_b:
    try:
        import yaml
        with open(os.environ["LMCACHE_CONFIG_FILE"], "r") as f:
            _y = yaml.safe_load(f)
            _s = _y.get("extra_config", {}).get("grpc_server", "localhost:50051")
            _default_b = _s.split(":")[0]
    except Exception:
        _default_b = "172.31.12.251"

p = argparse.ArgumentParser(description="EvicPress-backed interactive chat")
p.add_argument("--model",          default="mistralai/Mistral-7B-Instruct-v0.3")
p.add_argument("--max-tokens",     type=int,   default=300)
p.add_argument("--temperature",    type=float, default=0.7)
p.add_argument("--max-model-len",  type=int,   default=16384)
p.add_argument("--no-cache-info",  action="store_true")
p.add_argument("--machine-b",      default=_default_b)
p.add_argument("--stateless", action="store_true",
                                    help="Do not resend history (for KV cache demo)")
args = p.parse_args()

SYSTEM_PROMPT = (
    "You are a helpful, concise, and friendly AI assistant. "
    "You answer questions clearly and directly. "
    "When writing code you use clean, readable style. "
    "You acknowledge uncertainty when you are not sure of an answer."
)


# ── Connectivity check ─────────────────────────────────────────────────────────
def _check_b() -> bool:
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex((args.machine_b, 50051)) == 0
    finally:
        s.close()


if not _check_b():
    print(f"[warn] Machine B at {args.machine_b}:50051 is unreachable — "
          f"cache will still try to run but no hits will be served", flush=True)


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"[chat] Loading model {args.model}…", flush=True)
llm = LLM(
    model=args.model,
    enable_prefix_caching=True,
    max_model_len=args.max_model_len,
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ),
)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    # Because we build the Mistral prompt as a raw string, the model emits
    # the LITERAL "</s>" character sequence instead of the real EOS token —
    # which vLLM can't use as a stop signal. Stop explicitly on the string
    # pattern, plus "[INST]" in case the model tries to open a new turn.
    stop=["</s>", "[INST]"],
)


# ── Cache stats helper ─────────────────────────────────────────────────────────
try:
    import grpc
    import evicpress_pb2 as _pb2
    import evicpress_pb2_grpc as _pb2_grpc
    _stats_channel = grpc.insecure_channel(f"{args.machine_b}:50051")
    _stats_stub = _pb2_grpc.EvicPressServiceStub(_stats_channel)
except Exception as e:
    print(f"[chat] stats channel init failed: {e}", flush=True)
    _stats_stub = None


def _b_stats():
    if _stats_stub is None:
        return None
    try:
        return _stats_stub.GetStats(_pb2.StatsRequest(), timeout=1.0)
    except Exception:
        return None


# ── Prompt construction ────────────────────────────────────────────────────────
# We build Mistral's [INST]/[/INST] format manually. Reasons:
#   1. role="system" isn't accepted by Mistral's chat template — we fold the
#      system prompt into the first user turn.
#   2. Mistral-Common's validator rejects the implicit empty assistant turn
#      that `add_generation_prompt=True` produces, raising
#      InvalidAssistantMessageException. Building the string ourselves
#      sidesteps the validator entirely.
# The exact byte-for-byte format matches what vLLM's Mistral tokenizer emits
# for a valid conversation, so prefix caching (and LMCache) still hits across
# turns because the prefix is deterministic.
def _build_prompt(history: list[dict]) -> str:
    # --stateless: only the last user turn reaches the model. Useful for a
    # KV-cache eviction demo where you want an empty prefix every call.
    # Ordinary chat should NOT set this — the model needs history to remember
    # anything across turns.
    if args.stateless and history:
        last = history[-1]
        return f"<s>[INST] {SYSTEM_PROMPT}\n\n{last['content']} [/INST]"

    parts = ["<s>"]
    for i, msg in enumerate(history):
        if msg["role"] == "user":
            content = msg["content"]
            if i == 0:
                content = f"{SYSTEM_PROMPT}\n\n{content}"
            parts.append(f"[INST] {content} [/INST]")
        elif msg["role"] == "assistant":
            parts.append(f" {msg['content']}</s>")
    return "".join(parts)


# ── Chat loop ──────────────────────────────────────────────────────────────────
history: list[dict] = []

print("\n" + "=" * 60)
print(f"  EvicPress Chat  |  Model: {args.model}")
print(f"  Machine B: {args.machine_b}:50051")
print("  Type 'quit' or Ctrl-C to exit, 'clear' to reset history")
print("=" * 60 + "\n")

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
    prompt = _build_prompt(history)

    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.time() - t0

    reply = outputs[0].outputs[0].text.strip()
    history.append({"role": "assistant", "content": reply})

    wrapped = textwrap.fill(reply, width=72, subsequent_indent="       ")
    print(f"\nBot:   {wrapped}\n")

    turn += 1
    if not args.no_cache_info:
        s = _b_stats()
        if s:
            tier_str = (
                f"T1={s.tier1_blocks}blk  "
                f"T2={s.tier2_blocks}blk  "
                f"T3={s.tier3_blocks}blk  "
                f"hit_rate={s.hit_rate:.2f}  "
                f"promotions={s.tier1_promotions}"
            )
        else:
            tier_str = "Machine B unreachable"
        print(f"       [{elapsed:.2f}s | {tier_str}]\n")
