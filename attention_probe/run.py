"""One-request GPU smoke run; Machine B is optional."""

import argparse
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from model_settings import model_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe live KV-window attention mass")
    parser.add_argument("--probe", action="store_true", help="enable the extra Q/K pass")
    parser.add_argument("--remote", action="store_true", help="use the LMCache connector")
    parser.add_argument("--include-prefill", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=320)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--layer", default="0")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.decode_steps < 1 or args.chunk_size < 1:
        parser.error("prompt tokens, decode steps, and chunk size must be positive")
    if not args.layer.isdigit():
        parser.error("--layer must be a nonnegative layer number")
    if args.prompt_tokens + args.decode_steps + 1 > args.max_model_len:
        parser.error("prompt and generated tokens exceed max model length")
    if args.remote and not os.getenv("LMCACHE_CONFIG_FILE"):
        parser.error("--remote requires LMCACHE_CONFIG_FILE")

    root = Path(__file__).resolve().parents[1]
    output = args.output or (
        root / "benchmarks" / "results" / f"attention_probe_{uuid4().hex[:8]}.json"
    )
    if args.probe:
        installed_plugins = {
            plugin.name for plugin in entry_points(group="vllm.general_plugins")
        }
        if "dstn_attention_probe" not in installed_plugins:
            parser.error("install the local plugin first: python -m pip install -e . --no-deps")
        if output.exists():
            parser.error(f"probe output already exists: {output}")
    os.environ["DSTN_ATTENTION_PROBE"] = "1" if args.probe else "0"
    os.environ["DSTN_ATTENTION_PROBE_OUTPUT"] = str(output.resolve())
    os.environ["DSTN_ATTENTION_PROBE_CHUNK_SIZE"] = str(args.chunk_size)
    os.environ["DSTN_ATTENTION_PROBE_LAYER"] = args.layer
    os.environ["DSTN_ATTENTION_PROBE_STEPS"] = str(args.decode_steps)
    os.environ["DSTN_ATTENTION_PROBE_PREFILL"] = "1" if args.include_prefill else "0"

    # vLLM and its worker processes must see the environment before import.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    manifest = json.loads((root / "benchmarks" / "manifest.json").read_text())
    options = {
        **model_options(manifest),
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "attention_backend": "FLASH_ATTN",
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.remote,
    }
    if args.remote:
        options["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        )
    llm = LLM(**options)
    tokenizer = llm.get_tokenizer()
    filler = "A blue marker was placed inside the long document. "
    prompt = filler
    while len(tokenizer.encode(prompt)) < args.prompt_tokens:
        prompt += filler
    prompt += "\nContinue the document briefly."
    token_count = len(tokenizer.encode(prompt))
    if token_count + args.decode_steps + 1 > args.max_model_len:
        raise RuntimeError("generated prompt exceeds --max-model-len")

    start = time.perf_counter()
    result = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0,
            max_tokens=args.decode_steps + 1,
            min_tokens=args.decode_steps + 1,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - start
    print(f"prompt_tokens={token_count} generated={len(result[0].outputs[0].token_ids)} wall_s={elapsed:.3f}")
    if args.probe:
        if not output.exists():
            raise RuntimeError("probe produced no result; check plugin installation and worker logs")
        data = json.loads(output.read_text(encoding="utf-8"))
        print(f"probe_events={len(data['events'])} output={output}")


if __name__ == "__main__":
    main()
