from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import time

# LMCache reads its config from this env var.
os.environ.setdefault("LMCACHE_CONFIG_FILE", "lmcache_config.yaml")

# Initialize vLLM with LMCacheConnectorV1 wired in.
# kv_transfer_config activates the connector — without it, LMCACHE_CONFIG_FILE
# is never read and LMCache is silently skipped.
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,
    max_model_len=16384,
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ),
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
)

# ─────────────────────────────────────────────
# Test prompts
# ─────────────────────────────────────────────

# Long shared prefix (>256 tokens) so LMCache forms at least one full chunk
# and sends it to Machine B over gRPC.
_context = (
    "You are a knowledgeable assistant with deep expertise in world geography, "
    "history, politics, and culture. When answering questions, you always provide "
    "detailed, accurate, and well-structured responses. You draw on a wide range of "
    "sources and consider multiple perspectives before giving your answer. "
    "Your goal is to educate and inform the user as thoroughly as possible, "
    "referencing relevant historical context, geographical facts, political systems, "
    "cultural traditions, and economic factors where applicable. "
    "You are patient, thorough, and always cite specific details to support your points. "
    "You also acknowledge when a topic is complex or when there are multiple valid "
    "viewpoints, and you strive to present a balanced and nuanced answer. "
    "Here is the user's question: "
)
prompt1 = _context + "What is the capital of India and why is it historically significant?"
prompt2 = _context + "What is the capital of India and why is it historically significant?"  # identical → cache hit

# ─────────────────────────────────────────────
# First run (MISS → STORE)
# ─────────────────────────────────────────────

print("\n FIRST RUN (expect MISS + STORE)")
start = time.time()

outputs1 = llm.generate([prompt1], sampling_params)

print(outputs1[0].outputs[0].text)
print("Time:", time.time() - start)

# ─────────────────────────────────────────────
# Second run (HIT → FETCH)
# ─────────────────────────────────────────────

print("\n SECOND RUN (expect HIT + FETCH)")
start = time.time()

outputs2 = llm.generate([prompt2], sampling_params)

print(outputs2[0].outputs[0].text)
print("Time:", time.time() - start)
