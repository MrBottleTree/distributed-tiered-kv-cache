"""Shared model loading settings; weight precision is independent of KV precision.

Keep this module GPU-free so benchmarks and local tests can use the same settings
as the chat and attention-probe entry points.
"""

import json
import os
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "benchmarks" / "manifest.json"


def load_manifest() -> dict:
    """Legacy diagnostics inherit a custom manifest from the benchmark runner."""
    path = Path(os.environ.get("BENCH_MANIFEST", DEFAULT_MANIFEST))
    return json.loads(path.read_text(encoding="utf-8"))


def tokenizer_identity(manifest: dict) -> tuple[str, str]:
    # Frozen prompts belong to a tokenizer, not a particular weight checkpoint.
    return (
        manifest.get("tokenizer", manifest["model"]),
        manifest.get("tokenizer_revision", manifest["model_revision"]),
    )


def verify_input_tokenizer(manifest: dict, provenance: dict) -> None:
    """Accept old provenance keys without rewriting the frozen input files."""
    identity = (
        provenance.get("tokenizer", provenance.get("model")),
        provenance.get("tokenizer_revision", provenance.get("model_revision")),
    )
    if identity != tokenizer_identity(manifest):
        raise RuntimeError("Frozen inputs use a different tokenizer or tokenizer revision")


def validate_weight_config(manifest: dict, config: dict) -> None:
    """Fail before serving if the pinned checkpoint is not the expected INT4 model."""
    expected = manifest.get("weight_quantization")
    if expected is None:
        return  # Also support the previous, unquantized manifest.
    actual = config.get("quantization_config", {})
    if (actual.get("quant_method") != expected["method"]
            or actual.get("bits") != expected["bits"]):
        raise RuntimeError(f"Checkpoint weight quantization differs from manifest: {actual}")


def model_options(manifest: dict | None = None, *, model: str | None = None) -> dict:
    """vLLM LLM kwargs used by every project-owned model-loading entry point."""
    manifest = load_manifest() if manifest is None else manifest
    server = manifest.get("server", {})
    options = {
        "model": model or manifest["model"],
        "dtype": server.get("dtype", "float16"),
        "kv_cache_dtype": server.get("kv_cache_dtype", "float16"),
        "max_model_len": manifest.get("max_model_len", 8192),
        "gpu_memory_utilization": server.get("gpu_memory_utilization", 0.8),
        "max_num_seqs": server.get("max_num_seqs", 1),
    }
    if options["model"] == manifest["model"]:
        tokenizer, revision = tokenizer_identity(manifest)
        options.update(revision=manifest["model_revision"], tokenizer=tokenizer,
                       tokenizer_revision=revision)
    # For an explicit --model override, let that checkpoint supply its tokenizer
    # and quantization metadata; never apply Mistral's revision to another repo.
    # Do not force quantization="awq": auto-detection also permits Marlin kernels.
    return options


def model_server_args(manifest: dict) -> list[str]:
    """The server CLI and in-process LLM must receive identical model settings."""
    return [part for key, value in model_options(manifest).items()
            for part in ("--" + key.replace("_", "-"), str(value))]
