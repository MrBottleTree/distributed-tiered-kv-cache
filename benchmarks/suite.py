#!/usr/bin/env python3
"""Pinned, manifest-driven benchmark runner for Machine A.

Official task generation/scoring lives in pinned upstream checkouts. The
project-specific code here is only an adapter to our OpenAI-compatible server.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEPS = HERE / ".sources"
DATA = HERE / "data"
RESULTS = HERE / "results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checked(cmd: list[str], *, cwd: Path | None = None, env=None, capture=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True,
                              capture_output=capture)
    except subprocess.CalledProcessError as exc:
        # Captured scorer/workload/SSH errors otherwise disappear before their
        # result files are written. Surface them in the outer runner's stage log.
        if capture:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, flush=True)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, flush=True)
        raise


def git_rev(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def ensure_source(name: str, spec: dict) -> Path:
    path = DEPS / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "nltk_data":
            run_checked(["git", "clone", "--depth", "1", "--filter=blob:none",
                         "--sparse", spec["url"], str(path)])
            run_checked([
                "git", "sparse-checkout", "set", "--no-cone",
                "/packages/tokenizers/punkt.zip",
                "/packages/tokenizers/punkt_tab.zip"], cwd=path)
            if git_rev(path) != spec["commit"]:
                run_checked(["git", "fetch", "--depth", "1", "origin", spec["commit"]], cwd=path)
                run_checked(["git", "checkout", "--detach", spec["commit"]], cwd=path)
        else:
            run_checked(["git", "clone", "--filter=blob:none", "--no-checkout",
                         spec["url"], str(path)])
            run_checked(["git", "fetch", "--depth", "1", "origin", spec["commit"]], cwd=path)
            run_checked(["git", "checkout", "--detach", spec["commit"]], cwd=path)
    current = git_rev(path)
    if current != spec["commit"]:
        raise RuntimeError(f"{path} is at {current}, expected pinned {spec['commit']}; "
                           "move it aside manually before preparing")
    return path


def install_nltk_tokenizers(source: Path) -> Path:
    destination = HERE / ".nltk_data"
    tokenizers = destination / "tokenizers"
    for package in ("punkt", "punkt_tab"):
        if not (tokenizers / package).exists():
            archive = source / "packages" / "tokenizers" / (package + ".zip")
            if not archive.exists():
                raise RuntimeError(f"Official NLTK tokenizer archive missing: {archive}")
            tokenizers.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as data:
                data.extractall(tokenizers)
    return destination


def require_source(name: str, manifest: dict) -> Path:
    path = DEPS / name
    if not path.exists():
        raise RuntimeError(f"Missing official {name} checkout. Run: python benchmarks/suite.py prepare")
    current = git_rev(path)
    expected = manifest["sources"][name]["commit"]
    if current != expected:
        raise RuntimeError(f"{name} checkout differs from manifest: {current} != {expected}")
    return path


def tokenizer_for(model: str, revision: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = snapshot_download(
        repo_id=model, revision=revision,
        allow_patterns=["*.json", "*.model", "*.model.v3", "*.jinja", "*.txt"])
    return AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False), snapshot


def prepare_ruler(name: str, test: dict, manifest: dict, tokenizer_path: str,
                  force: bool) -> None:
    import yaml
    path = DATA / f"{name}.jsonl"
    if path.exists() and not force:
        print(f"Keep frozen {path}")
        return
    ruler = require_source("ruler", manifest)
    nltk_data = install_nltk_tokenizers(require_source("nltk_data", manifest))
    task = test["task"]
    tasks = yaml.safe_load((ruler / "scripts" / "synthetic.yaml").read_text(encoding="utf-8"))
    config = tasks[task]
    if config["args"].get("type_haystack") == "essay":
        corpus_dir = ruler / "scripts" / "data" / "synthetic" / "json"
        corpus = corpus_dir / "PaulGrahamEssays.json"
        if not corpus.exists():
            run_checked([sys.executable, "download_paulgraham_essay.py"],
                        cwd=corpus_dir)
        if not corpus.exists() or len(read_json(corpus).get("text", "")) < 100_000:
            raise RuntimeError(
                "Official RULER Paul Graham corpus download was incomplete. "
                "Check network access and rerun the upstream downloader in "
                f"{corpus_dir}; do not score a partial corpus.")
    constants_path = ruler / "scripts" / "data" / "synthetic" / "constants.py"
    spec = importlib.util.spec_from_file_location("ruler_data_constants", constants_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    base = module.TASKS[config["task"]]
    # RULER's own Mistral-compatible template. The generator splits its
    # answer_prefix from input; we put it back when sending prompts.
    template = "[INST] " + base["template"] + " [/INST]" + base.get("answer_prefix", "")
    dest = DATA / "_ruler_generated"
    cmd = [
        sys.executable, str(ruler / "scripts" / "data" / "synthetic" / "niah.py"),
        "--save_dir", str(dest), "--save_name", name, "--subset", "test",
        "--tokenizer_path", tokenizer_path, "--tokenizer_type", "hf",
        "--max_seq_length", str(test["length"]),
        "--tokens_to_generate", str(test["max_new_tokens"]),
        "--num_samples", str(test["samples"]), "--random_seed", "42",
        "--template", template,
    ]
    for key, value in config["args"].items():
        cmd.extend(["--" + key, str(value)])
    env = os.environ.copy()
    env["NLTK_DATA"] = str(nltk_data)
    run_checked(cmd, cwd=ruler / "scripts" / "data", env=env)
    src = dest / name / "test.jsonl"
    records = list(jsonl(src))
    if len(records) != test["samples"]:
        raise RuntimeError(f"RULER generated {len(records)} samples, expected {test['samples']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, path)
    print(f"Frozen official RULER task {name}: {len(records)} samples, sha256={file_sha256(path)}")


def prepare_longbench(name: str, test: dict, manifest: dict, tokenizer, force: bool) -> None:
    from datasets import load_dataset
    path = DATA / f"{name}.jsonl"
    if path.exists() and not force:
        print(f"Keep frozen {path}")
        return
    upstream = require_source("longbench", manifest) / "LongBench"
    dataset_name = test["dataset"]
    prompts = read_json(upstream / "config" / "dataset2prompt.json")
    max_outputs = read_json(upstream / "config" / "dataset2maxlen.json")
    max_new = max_outputs[dataset_name]
    limit = manifest["max_model_len"] - max_new - 32
    if limit <= 0:
        raise RuntimeError("max_model_len leaves no room for LongBench input")
    dataset_source = manifest["datasets"]["longbench"]
    source = load_dataset(
        dataset_source["repo"], dataset_name, split="test",
        revision=dataset_source["revision"], streaming=True,
        trust_remote_code=True)
    selected = []
    for row in source:
        prompt = prompts[dataset_name].format(**row)
        # No truncation: only exact official examples that fit this model are
        # selected, making this a clearly labelled subset, not a full score.
        if dataset_name not in {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens <= limit:
            selected.append({
                "id": row.get("_id", len(selected)),
                "input": prompt,
                "outputs": row["answers"],
                "all_classes": row.get("all_classes"),
                "source_length": row.get("length"),
                "input_tokens": tokens,
                "max_new_tokens": max_new,
            })
        if len(selected) >= test["samples"]:
            break
    if len(selected) < test["samples"]:
        raise RuntimeError(f"Only {len(selected)} fitting {dataset_name} items; "
                           f"requested {test['samples']}. Lower samples explicitly.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Frozen LongBench {dataset_name} subset: {len(selected)} samples, sha256={file_sha256(path)}")


def prepare(manifest: dict, force: bool, selected: list[str] | None = None,
            sources_only: bool = False) -> None:
    if sources_only:
        for name in ("ruler", "longbench"):
            ensure_source(name, manifest["sources"][name])
        provenance = read_json(DATA / "provenance.json")
        if provenance["model_revision"] != manifest["model_revision"]:
            raise RuntimeError("Frozen inputs use a different model/tokenizer revision")
        if provenance["longbench_dataset_revision"] != manifest["datasets"]["longbench"]["revision"]:
            raise RuntimeError("Frozen inputs use a different LongBench dataset revision")
        for name, spec in manifest["sources"].items():
            if provenance["sources"].get(name) != spec["commit"]:
                raise RuntimeError(f"Frozen inputs use a different {name} revision")
        for name, test in manifest["tests"].items():
            if test["kind"] not in ("ruler", "longbench"):
                continue
            path = DATA / f"{name}.jsonl"
            if not path.exists() or provenance["inputs"].get(name) != file_sha256(path):
                raise RuntimeError(f"Frozen input missing or changed: {name}")
        print("Official sources pinned; all included inputs verified")
        return
    for name, spec in manifest["sources"].items():
        print(f"Verifying official {name} source")
        ensure_source(name, spec)
    install_nltk_tokenizers(require_source("nltk_data", manifest))
    tokenizer, tokenizer_path = tokenizer_for(
        manifest["model"], manifest["model_revision"])
    for name, test in manifest["tests"].items():
        if selected is not None and name not in selected:
            continue
        if test["kind"] == "ruler":
            prepare_ruler(name, test, manifest, tokenizer_path, force)
        elif test["kind"] == "longbench":
            prepare_longbench(name, test, manifest, tokenizer, force)
    corpus = (DEPS / "ruler" / "scripts" / "data" / "synthetic" / "json"
              / "PaulGrahamEssays.json")
    write_json(DATA / "provenance.json", {
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "sources": {name: spec["commit"] for name, spec in manifest["sources"].items()},
        "longbench_dataset_revision": manifest["datasets"]["longbench"]["revision"],
        "ruler_essay_corpus_sha256": file_sha256(corpus) if corpus.exists() else None,
        "inputs": {
            name: file_sha256(DATA / f"{name}.jsonl")
            for name, test in manifest["tests"].items()
            if test["kind"] in ("ruler", "longbench")
            and (DATA / f"{name}.jsonl").exists()},
    })


def http_json(url: str, payload=None, timeout: int = 10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def b_state(dashboard_url: str):
    return http_json(dashboard_url.rstrip("/") + "/api/state")


def verify_b(state: dict, profile: dict, expected_data_dir: str | None = None):
    cfg = state["config"]
    if float(cfg["alpha"]) != float(profile["alpha"]):
        raise RuntimeError(f"Machine B alpha={cfg['alpha']}, profile requires {profile['alpha']}")
    if bool(cfg["quant_enabled"]) != bool(profile["quantization"]):
        raise RuntimeError("Machine B quantization setting differs from profile; "
                           "configure/restart B before running")
    if expected_data_dir and cfg["data_dir"] != expected_data_dir:
        raise RuntimeError(f"Machine B data_dir={cfg['data_dir']}, expected {expected_data_dir}")


def render_b_config(base_path: Path, profile: dict, data_dir: str, output: Path) -> None:
    import yaml
    if not profile["remote"]:
        raise ValueError("plain profile does not use Machine B")
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg["evicpress"]["alpha"] = float(profile["alpha"])
    cfg["quantization"]["enabled"] = bool(profile["quantization"])
    cfg["tier3"]["data_dir"] = data_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def configure_b_over_ssh(local_config: Path, ssh_target: str, remote_config: str,
                         restart_command: str, dashboard_url: str) -> None:
    # Explicit opt-in. The remote process/service is user-owned; this runner
    # only uploads one config and invokes the supplied service restart command.
    run_checked(["scp", str(local_config), f"{ssh_target}:{remote_config}"])
    run_checked(["ssh", ssh_target, restart_command])
    for _ in range(60):
        try:
            b_state(dashboard_url)
            return
        except (OSError, ValueError):
            time.sleep(1)
    raise RuntimeError("Machine B dashboard did not come up after service restart")


def make_a_config(b_host: str, output: Path) -> None:
    import yaml
    config = yaml.safe_load((ROOT / "lmcache_config.yaml").read_text(encoding="utf-8"))
    config["extra_config"]["grpc_server"] = f"{b_host}:50051"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def wait_for_server(base_url: str, timeout: int = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            http_json(base_url.rstrip("/") + "/models", timeout=2)
            return
        except (OSError, ValueError):
            time.sleep(2)
    raise RuntimeError("vLLM server did not become ready; inspect vllm.log")


@contextlib.contextmanager
def model_server(manifest: dict, profile: dict, run_dir: Path, b_host: str | None,
                 external_url: str | None):
    if external_url:
        wait_for_server(external_url)
        yield external_url.rstrip("/")
        return
    server = manifest["server"]
    url = f"http://{server['host']}:{server['port']}/v1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", manifest["model"], "--host", server["host"],
        "--revision", manifest["model_revision"],
        "--tokenizer-revision", manifest["model_revision"],
        "--port", str(server["port"]),
        "--max-model-len", str(manifest["max_model_len"]),
        "--gpu-memory-utilization", str(server["gpu_memory_utilization"]),
        "--dtype", server.get("dtype", "float16"),
        *server.get("extra_args", []),
    ]
    env = os.environ.copy()
    if profile["remote"]:
        if not b_host:
            raise RuntimeError("Remote profile requires --b-host")
        a_cfg = run_dir / "lmcache_config.yaml"
        make_a_config(b_host, a_cfg)
        env["LMCACHE_CONFIG_FILE"] = str(a_cfg)
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"})]
    else:
        env.pop("LMCACHE_CONFIG_FILE", None)
    with (run_dir / "vllm.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {proc.returncode}; inspect vllm.log")
                try:
                    http_json(url + "/models", timeout=2)
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise RuntimeError("vLLM failed to start in 240s; inspect vllm.log")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def completion(base_url: str, model: str, prompt: str, max_new_tokens: int) -> dict:
    payload = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_new_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base_url + "/completions", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    chunks = []
    first_token = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token_text = choice.get("text") or ""
                if token_text and first_token is None:
                    first_token = time.perf_counter() - started
                chunks.append(token_text)
    elapsed = time.perf_counter() - started
    if first_token is None:
        raise RuntimeError("Completion produced no text; inspect vLLM log and prompt")
    return {
        "prediction": "".join(chunks),
        "ttft_s": first_token,
        "latency_s": elapsed,
        "usage": usage,
    }


def percentile(values: list[float], percent: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def official_score(kind: str, test: dict, manifest: dict, predictions: list[str],
                   references: list[list[str]]) -> float:
    if kind == "ruler":
        path = require_source("ruler", manifest) / "scripts" / "eval" / "synthetic" / "constants.py"
        spec = importlib.util.spec_from_file_location("ruler_eval_constants", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.string_match_all(predictions, references)
    if kind == "longbench":
        upstream = require_source("longbench", manifest) / "LongBench"
        sys.path.insert(0, str(upstream))
        try:
            spec = importlib.util.spec_from_file_location("longbench_official_eval",
                                                          upstream / "eval.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            # Official scorer returns a 0-100 percentage, taking the best
            # matching ground truth for each question.
            return module.scorer(test["dataset"], predictions, references, None)
        finally:
            sys.path.pop(0)
    raise ValueError(kind)


def summarize_samples(rows: list[dict], test: dict, manifest: dict) -> dict:
    first = [r for r in rows if r["repeat"] == 0]
    warm = [r for r in rows if r["repeat"] > 0]
    first_by_sample = {r["sample"]: r["prediction"] for r in first}
    output = {
        "samples": len(first),
        "requests": len(rows),
        "cold_ttft_p50_s": percentile([r["ttft_s"] for r in first], 50),
        "cold_ttft_p95_s": percentile([r["ttft_s"] for r in first], 95),
        "warm_ttft_p50_s": percentile([r["ttft_s"] for r in warm], 50),
        "warm_ttft_p95_s": percentile([r["ttft_s"] for r in warm], 95),
        "cold_latency_p50_s": percentile([r["latency_s"] for r in first], 50),
        "warm_latency_p50_s": percentile([r["latency_s"] for r in warm], 50),
        "total_elapsed_s": sum(r["latency_s"] for r in rows),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0)
                                 for r in rows),
        "repeat_consistency_pct": (
            100 * sum(r["prediction"] == first_by_sample[r["sample"]] for r in warm) / len(warm)
            if warm else None),
    }
    if output["total_elapsed_s"]:
        output["completion_tokens_per_s"] = (
            output["completion_tokens"] / output["total_elapsed_s"])
    output["quality_score_pct"] = official_score(
        test["kind"], test, manifest,
        [r["prediction"] for r in first], [r["references"] for r in first])
    return output


def run_official(name: str, test: dict, manifest: dict, base_url: str,
                 run_dir: Path, repeats: int) -> dict:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise RuntimeError(f"Frozen {name} inputs absent; run prepare first")
    rows = []
    raw = run_dir / f"{name}.jsonl"
    with raw.open("w", encoding="utf-8") as stream:
        for idx, item in enumerate(jsonl(path)):
            prompt = item["input"] + item.get("answer_prefix", "")
            max_new = int(item.get("max_new_tokens", test.get("max_new_tokens", 128)))
            for repeat in range(repeats):
                try:
                    result = completion(base_url, manifest["model"], prompt, max_new)
                except Exception as exc:
                    failure = {"sample": idx, "repeat": repeat, "error": repr(exc)}
                    stream.write(json.dumps(failure) + "\n")
                    stream.flush()
                    raise
                row = {
                    "sample": idx,
                    "source_id": idx if test["kind"] == "ruler" else item.get("id", idx),
                    "repeat": repeat,
                    "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "references": item["outputs"],
                    **result,
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"{name}: {idx + 1} samples", flush=True)
    summary = summarize_samples(rows, test, manifest)
    summary["input_sha256"] = file_sha256(path)
    summary["kind"] = test["kind"]
    return summary


def run_long_doc(name: str, test: dict, manifest: dict, base_url: str,
                 run_dir: Path) -> dict:
    script = ROOT / "LMCache" / "benchmarks" / "long_doc_qa" / "long_doc_qa.py"
    cmd = [
        sys.executable, str(script), "--base-url", base_url,
        "--model", manifest["model"],
        "--document-length", str(test["document_length"]),
        "--num-documents", str(test["documents"]),
        "--repeat-count", str(test["repeat_count"]),
        "--output-len", str(test["output_len"]),
        "--max-inflight-requests", str(test["max_inflight_requests"]),
        "--repeat-mode", "tile", "--shuffle-seed", "42",
        "--hit-miss-ratio", test["hit_miss_ratio"], "--json-output",
    ]
    started = time.perf_counter()
    result = run_checked(cmd, cwd=run_dir, capture=True)
    (run_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("LMCache benchmark produced no JSON summary")
    upstream = json.loads(lines[-1])
    return {
        "kind": "lmcache",
        "wall_time_s": time.perf_counter() - started,
        "query_ttft_mean_s": upstream.get("query_ttft_per_prompt"),
        "query_time_per_prompt_s": upstream.get("query_round_time_per_prompt"),
        "warmup_time_per_prompt_s": upstream.get("warmup_round_time_per_prompt"),
        "upstream_metrics": upstream,
        "note": "Performance workload only; synthetic hi-token documents have no answer-quality score.",
    }


def run_legacy(name: str, test: dict, run_dir: Path, b_host: str | None,
               profile: dict, manifest: dict) -> dict:
    if test.get("remote_only") and not profile["remote"]:
        raise RuntimeError(f"{name} is an old remote-only diagnostic, not a plain-vLLM workload")
    env = os.environ.copy()
    if b_host:
        env["MACHINE_B"] = b_host
        cfg_path = run_dir / "lmcache_config.yaml"
        make_a_config(b_host, cfg_path)
        env["LMCACHE_CONFIG_FILE"] = str(cfg_path)
    env["BENCH_MODEL"] = manifest["model"]
    script = ROOT / test["script"]
    cmd = [sys.executable, str(script)]
    if name == "needle_haystack":
        cmd += ["--output-json", str(run_dir / "needle_haystack_output.json")]
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=3600)
    (run_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; inspect {name}.stdout.txt")
    return {"kind": "legacy", "wall_time_s": time.perf_counter() - started,
            "note": "Project diagnostic, not an official paper benchmark."}


def numeric_delta(after, before):
    if isinstance(after, dict) and isinstance(before, dict):
        return {key: numeric_delta(after[key], before[key])
                for key in before.keys() & after.keys()
                if numeric_delta(after[key], before[key]) is not None}
    if type(after) in (int, float) and type(before) in (int, float):
        return after - before
    return None


def git_info(path: Path):
    try:
        return {
            "commit": git_rev(path),
            "dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=path, text=True,
                encoding="utf-8", errors="replace").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions():
    names = ("vllm", "lmcache", "torch", "transformers", "grpcio", "datasets")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def run_suite(args, manifest: dict, manifest_path: Path) -> Path:
    profile = manifest["profiles"][args.profile]
    names = args.test or ["all"]
    if names == ["all"]:
        names = [n for n, t in manifest["tests"].items() if t["kind"] != "legacy"]
    elif names == ["all_legacy"]:
        names = [n for n, t in manifest["tests"].items() if t["kind"] == "legacy"]
    for name in names:
        if name not in manifest["tests"]:
            raise ValueError(f"Unknown test {name}; use 'list'")
        test = manifest["tests"][name]
        if test["kind"] in ("ruler", "longbench"):
            input_file = DATA / f"{name}.jsonl"
            if not input_file.exists():
                raise RuntimeError(
                    f"{input_file} is missing; run prepare --test {name} before GPU time")
            require_source(test["kind"], manifest)
            provenance_path = DATA / "provenance.json"
            if not provenance_path.exists():
                raise RuntimeError("Data provenance missing; rerun prepare before GPU time")
            provenance = read_json(provenance_path)
            if provenance["model_revision"] != manifest["model_revision"]:
                raise RuntimeError("Frozen prompts use a different model/tokenizer revision")
            if provenance["sources"][test["kind"]] != manifest["sources"][test["kind"]]["commit"]:
                raise RuntimeError(f"Frozen {name} was generated from a different official source")
            if test["kind"] == "longbench" and (
                    provenance["longbench_dataset_revision"]
                    != manifest["datasets"]["longbench"]["revision"]):
                raise RuntimeError("Frozen LongBench dataset revision differs from manifest")
            expected_hash = provenance["inputs"].get(name)
            if expected_hash != file_sha256(input_file):
                raise RuntimeError(f"Frozen input hash mismatch for {name}; rerun prepare")
        elif test["kind"] == "lmcache":
            if not (ROOT / "LMCache" / "benchmarks" / "long_doc_qa" / "long_doc_qa.py").exists():
                raise RuntimeError("Vendored LMCache long_doc_qa workload is missing")
    if any(manifest["tests"][n]["kind"] == "legacy" for n in names) and len(names) > 1:
        raise RuntimeError("Run legacy diagnostics one at a time; they manage their own model/process")
    if profile["remote"] and not args.b_host:
        raise RuntimeError("Remote profile requires --b-host")
    if profile["remote"] and any(
            manifest["tests"][n]["kind"] != "legacy" for n in names):
        spec = importlib.util.find_spec("lmcache")
        locations = list(spec.submodule_search_locations or []) if spec else []
        local = (ROOT / "LMCache" / "lmcache").resolve()
        if not locations or all(Path(p).resolve() != local for p in locations):
            raise RuntimeError(
                "Machine A must use this repo's LMCache checkout. "
                "Run: python -m pip install -e ./LMCache")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{args.profile}"
    run_dir = Path(args.output_dir).resolve() / run_id
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    b_url = args.b_dashboard_url or (f"http://{args.b_host}:8080" if args.b_host else None)
    b_data_dir = args.b_data_dir or f"/data/kv_cache/benchmark_{run_id}"
    metadata = {
        "run_id": run_id, "created_utc": timestamp,
        "profile_name": args.profile, "profile": profile, "tests": names,
        "manifest_sha256": file_sha256(manifest_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "manifest": manifest,
        "machine_a": {"platform": platform.platform(), "python": sys.version,
                      "git": git_info(ROOT), "packages": package_versions(),
                      "gpu": gpu_info()},
        "official_sources": {name: git_info(DEPS / name)
                             for name in manifest["sources"] if (DEPS / name).exists()},
        "data_provenance": (
            read_json(DATA / "provenance.json")
            if (DATA / "provenance.json").exists() else None),
        "b_host": args.b_host, "b_dashboard_url": b_url,
        "b_data_dir": b_data_dir if profile["remote"] else None,
        "external_server": args.external_server,
    }
    (run_dir / "machine_a.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=ROOT, text=True,
                                encoding="utf-8", errors="replace"),
        encoding="utf-8")
    write_json(run_dir / "run.json", metadata)
    if profile["remote"]:
        if args.b_ssh:
            if not args.b_restart_command or not args.b_config_path:
                raise RuntimeError("--b-ssh requires --b-config-path and --b-restart-command")
            config_path = run_dir / "machine_b_config.yaml"
            render_b_config(Path(args.b_base_config), profile, b_data_dir, config_path)
            configure_b_over_ssh(config_path, args.b_ssh, args.b_config_path,
                                 args.b_restart_command, b_url)
            metadata["machine_b_config_sha256"] = file_sha256(config_path)
            if args.b_repo_path:
                repo_q = shlex.quote(args.b_repo_path)
                metadata["machine_b_git_commit"] = run_checked([
                    "ssh", args.b_ssh, "git -C " + repo_q + " rev-parse HEAD"],
                    capture=True).stdout.strip()
                metadata["machine_b_git_status"] = run_checked([
                    "ssh", args.b_ssh, "git -C " + repo_q + " status --porcelain"],
                    capture=True).stdout.splitlines()
                (run_dir / "machine_b.patch").write_text(run_checked([
                    "ssh", args.b_ssh, "git -C " + repo_q + " diff"],
                    capture=True).stdout, encoding="utf-8")
        elif args.b_commit:
            metadata["machine_b_git_commit"] = args.b_commit
        try:
            with socket.create_connection((args.b_host, 50051), timeout=5):
                pass
        except OSError as exc:
            raise RuntimeError(f"Machine B gRPC {args.b_host}:50051 unreachable") from exc
        initial = b_state(b_url)
        verify_b(initial, profile, b_data_dir if args.b_ssh or args.b_data_dir else None)
        metadata["machine_b_initial"] = initial
        write_json(run_dir / "run.json", metadata)
    official = [n for n in names if manifest["tests"][n]["kind"] != "legacy"]
    legacy = [n for n in names if manifest["tests"][n]["kind"] == "legacy"]
    summaries = {}
    try:
        if official:
            with model_server(manifest, profile, run_dir, args.b_host,
                              args.external_server) as base_url:
                for name in official:
                    test = manifest["tests"][name]
                    before = b_state(b_url) if profile["remote"] else None
                    if test["kind"] in ("ruler", "longbench"):
                        summary = run_official(name, test, manifest, base_url,
                                               run_dir, args.repeats)
                    elif test["kind"] == "lmcache":
                        summary = run_long_doc(name, test, manifest, base_url, run_dir)
                    else:
                        raise ValueError(test["kind"])
                    if profile["remote"]:
                        after = b_state(b_url)
                        summary["machine_b_stats_delta"] = numeric_delta(
                            after["stats"], before["stats"])
                        delta = summary["machine_b_stats_delta"]
                        summary["remote_fetch_verified"] = (
                            delta.get("tier2_hits", 0) + delta.get("tier3_hits", 0) > 0)
                        if not summary["remote_fetch_verified"]:
                            summary["warning"] = (
                                "No Machine B Tier 2/3 hits during this workload. "
                                "Do not interpret warm latency as remote-fetch performance.")
                        summary["machine_b_tiers_before"] = {
                            tier: before[tier] for tier in ("tier1", "tier2", "tier3")}
                        summary["machine_b_tiers_after"] = {
                            tier: after[tier] for tier in ("tier1", "tier2", "tier3")}
                    summaries[name] = summary
                    write_json(run_dir / "summary.json", summaries)
        for name in legacy:
            test = manifest["tests"][name]
            before = b_state(b_url) if profile["remote"] else None
            summary = run_legacy(name, test, run_dir, args.b_host, profile, manifest)
            if profile["remote"]:
                after = b_state(b_url)
                summary["machine_b_stats_delta"] = numeric_delta(
                    after["stats"], before["stats"])
            summaries[name] = summary
            write_json(run_dir / "summary.json", summaries)
    except Exception as exc:
        write_json(run_dir / "error.json", {"error": repr(exc), "completed_tests": list(summaries)})
        raise
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "test", "kind", "quality_score_pct", "samples", "requests",
            "cold_ttft_p50_s", "cold_ttft_p95_s", "warm_ttft_p50_s",
            "warm_ttft_p95_s", "cold_latency_p50_s", "warm_latency_p50_s",
            "total_elapsed_s", "wall_time_s", "b_total_hits",
            "query_ttft_mean_s", "query_time_per_prompt_s",
            "warmup_time_per_prompt_s",
            "b_tier2_hits", "b_tier3_hits", "b_misses"])
        writer.writeheader()
        for name, summary in summaries.items():
            delta = summary.get("machine_b_stats_delta", {})
            writer.writerow({
                "test": name, "kind": summary["kind"],
                **{k: summary.get(k) for k in (
                    "quality_score_pct", "samples", "requests",
                    "cold_ttft_p50_s", "cold_ttft_p95_s", "warm_ttft_p50_s",
                    "warm_ttft_p95_s", "cold_latency_p50_s", "warm_latency_p50_s",
                    "total_elapsed_s", "wall_time_s")},
                "query_ttft_mean_s": summary.get("query_ttft_mean_s"),
                "query_time_per_prompt_s": summary.get("query_time_per_prompt_s"),
                "warmup_time_per_prompt_s": summary.get("warmup_time_per_prompt_s"),
                "b_total_hits": delta.get("total_hits"),
                "b_tier2_hits": delta.get("tier2_hits"),
                "b_tier3_hits": delta.get("tier3_hits"),
                "b_misses": delta.get("total_misses"),
            })
    print(f"Results: {run_dir}")
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    p_check = commands.add_parser("check", help="Preflight dependencies and Machine B")
    p_check.add_argument("--profile", required=True)
    p_check.add_argument("--b-host")
    p_check.add_argument("--b-dashboard-url")
    p_prepare = commands.add_parser("prepare", help="Pin official sources and freeze inputs")
    p_prepare.add_argument("--force", action="store_true", help="Regenerate frozen datasets")
    p_prepare.add_argument("--test", action="append",
                           help="Prepare selected official test; repeat for multiple")
    p_prepare.add_argument("--sources-only", action="store_true",
                           help="Fetch scorer code and verify included frozen inputs; no datasets download")
    p_render = commands.add_parser("render-b", help="Generate B config for a profile")
    p_render.add_argument("--profile", required=True)
    p_render.add_argument("--base-config", type=Path, default=HERE / "machine_b_base.yaml")
    p_render.add_argument("--data-dir", required=True, help="Unique B disk path for this run")
    p_render.add_argument("--output", type=Path, required=True)
    def add_run_options(option_parser):
        option_parser.add_argument("--test", action="append",
                                   help="Repeat to choose tests; default all official")
        option_parser.add_argument("--repeats", type=int, default=2,
                                   help="Exact prompt repeats for official quality tasks")
        option_parser.add_argument("--b-host", help="Machine B private IP/DNS, not a URL")
        option_parser.add_argument("--b-dashboard-url", help="Override B dashboard URL")
        option_parser.add_argument("--b-data-dir", help="Expected preconfigured B data dir")
        option_parser.add_argument("--b-ssh",
                                   help="SSH target, e.g. ubuntu@10.0.0.5; enables config upload")
        option_parser.add_argument("--b-config-path", help="Absolute path read by B service")
        option_parser.add_argument("--b-restart-command",
                                   help="Explicit remote service restart command")
        option_parser.add_argument("--b-base-config", type=Path,
                                   default=HERE / "machine_b_base.yaml")
        option_parser.add_argument("--b-repo-path",
                                   help="B repo path for SSH git revision capture")
        option_parser.add_argument("--b-commit",
                                   help="B git commit when configured manually")
        option_parser.add_argument("--external-server",
                                   help="Use an existing vLLM /v1 endpoint (no process isolation)")
        option_parser.add_argument("--output-dir", type=Path, default=RESULTS)

    p_run = commands.add_parser("run", help="Run one profile with selected tests")
    p_run.add_argument("--profile", required=True)
    add_run_options(p_run)
    p_matrix = commands.add_parser("matrix", help="Run selected tests across all five profiles")
    p_matrix.add_argument("--profile", action="append",
                          help="Limit to these profiles, in the order specified")
    add_run_options(p_matrix)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    if args.command == "list":
        print("Profiles:")
        for name, profile in manifest["profiles"].items():
            print(f"  {name}: {profile}")
        print("Tests:")
        for name, test in manifest["tests"].items():
            print(f"  {name}: {test['kind']}")
        return 0
    if args.command == "prepare":
        if args.test:
            for name in args.test:
                if name not in manifest["tests"] or manifest["tests"][name]["kind"] == "legacy":
                    parser.error(f"Cannot prepare {name}: not an official data task")
        if args.sources_only and (args.force or args.test):
            parser.error("--sources-only cannot be combined with --force or --test")
        prepare(manifest, args.force, args.test, args.sources_only)
        return 0
    if args.command == "matrix":
        names = args.profile or list(manifest["profiles"])
        for name in names:
            if name not in manifest["profiles"]:
                parser.error(f"Unknown profile {name}")
        if any(manifest["profiles"][n]["remote"] for n in names):
            if not all((args.b_host, args.b_ssh, args.b_config_path,
                        args.b_restart_command)):
                parser.error("Remote matrix profiles require --b-host, --b-ssh, "
                             "--b-config-path, and --b-restart-command")
            if args.b_data_dir:
                parser.error("Matrix uses a fresh B data directory per profile; "
                             "do not pass --b-data-dir")
        if args.external_server and len(names) > 1:
            parser.error("A multi-profile matrix must restart its managed vLLM server")
        if args.repeats < 1:
            parser.error("--repeats must be >= 1")
        matrix_path = (Path(args.output_dir).resolve() /
                       ("matrix_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".csv"))
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "profile", "test", "run_dir", "kind", "quality_score_pct",
            "cold_ttft_p50_s", "warm_ttft_p50_s",
            "query_ttft_mean_s", "query_time_per_prompt_s",
            "wall_time_s", "b_tier2_hits", "b_tier3_hits", "remote_fetch_verified",
        ]
        matrix_rows = []
        for name in names:
            item = copy.copy(args)
            item.profile = name
            output = run_suite(item, manifest, manifest_path)
            for test_name, summary in read_json(output / "summary.json").items():
                delta = summary.get("machine_b_stats_delta", {})
                matrix_rows.append({
                    "profile": name, "test": test_name, "run_dir": str(output),
                    "kind": summary["kind"],
                    "quality_score_pct": summary.get("quality_score_pct"),
                    "cold_ttft_p50_s": summary.get("cold_ttft_p50_s"),
                    "warm_ttft_p50_s": summary.get("warm_ttft_p50_s"),
                    "query_ttft_mean_s": summary.get("query_ttft_mean_s"),
                    "query_time_per_prompt_s": summary.get("query_time_per_prompt_s"),
                    "wall_time_s": summary.get("wall_time_s"),
                    "b_tier2_hits": delta.get("tier2_hits"),
                    "b_tier3_hits": delta.get("tier3_hits"),
                    "remote_fetch_verified": summary.get("remote_fetch_verified"),
                })
            with matrix_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(matrix_rows)
        print(f"Matrix comparison: {matrix_path}")
        return 0
    if args.profile not in manifest["profiles"]:
        parser.error(f"Unknown profile {args.profile}")
    if args.command == "render-b":
        render_b_config(args.base_config, manifest["profiles"][args.profile],
                        args.data_dir, args.output)
        print(args.output.resolve())
        return 0
    if args.command == "check":
        modules = ["yaml", "vllm", "lmcache", "transformers", "jieba",
                   "rouge", "grpc", "openai", "pandas", "matplotlib"]
        missing = [m for m in modules
                   if (importlib.util.find_spec(m) is None
                       or importlib.util.find_spec(m).origin is None)]
        print("Missing Python modules:", missing or "none")
        for name, spec in manifest["sources"].items():
            source = DEPS / name
            print(f"{name}: " + (git_rev(source) if source.exists() else "not prepared"))
        gpu = shutil.which("nvidia-smi")
        print("nvidia-smi:", gpu or "not found")
        profile = manifest["profiles"][args.profile]
        if profile["remote"]:
            if not args.b_host:
                parser.error("Remote profile needs --b-host")
            url = args.b_dashboard_url or f"http://{args.b_host}:8080"
            state = b_state(url)
            verify_b(state, profile)
            with socket.create_connection((args.b_host, 50051), timeout=5):
                pass
            print("Machine B:", state["config"])
        return 1 if missing or not gpu else 0
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    run_suite(args, manifest, manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
