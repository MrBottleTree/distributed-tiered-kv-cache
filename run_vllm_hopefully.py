from vllm import LLM, SamplingParams
import os
import time

# LMCache reads its config from this env var.
# Can also be set before launching: LMCACHE_CONFIG_FILE=lmcache_config.yaml python run_vllm_hopefully.py
os.environ.setdefault("LMCACHE_CONFIG_FILE", "lmcache_config.yaml")

# Initialize vLLM. enable_prefix_caching lets vLLM reuse KV blocks
# for shared prompt prefixes — LMCache intercepts these via its storage hooks.
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
)

# ─────────────────────────────────────────────
# Test prompts
# ─────────────────────────────────────────────

prompt1 = "The capital of India is"
prompt2 = "The capital of India is"  # identical → should hit cache

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
