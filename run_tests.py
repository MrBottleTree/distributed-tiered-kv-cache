#!/usr/bin/env python3
"""New-machine validation: unit tests first, then isolated GPU benchmark profiles.

This orchestrates benchmarks/suite.py; it does not implement new benchmarks or
provision EC2/Machine B. Running without arguments only runs GPU-free tests.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".validation-venv"
sys.path.insert(0, str(ROOT / "benchmarks"))
import suite
from model_settings import tokenizer_identity, validate_weight_config

UNIT_FILES = ("test_benchmark_suite.py", "test_attention_probe.py", "test_new_machine_runner.py", "test_head_segments.py")
# Config files use these same names, with underscores instead of CLI hyphens.
CONFIG_KEYS = ("mode", "manifest", "test", "remote_profile", "repeats",
               "b_host", "b_dashboard_url", "b_data_dir", "b_ssh",
               "b_config_path", "b_restart_command", "b_repo_path", "b_commit",
               "b_base_config", "timeout_minutes", "allow_no_remote_hits", "granularity")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON settings; CLI overrides them")
    parser.add_argument("--mode", choices=("unit", "single", "remote", "both"), default="unit")
    parser.add_argument("--granularity", choices=("chunk", "head"), default="chunk")
    parser.add_argument("--setup", action="store_true", help="Install into .validation-venv (Linux)")
    parser.add_argument("--manifest", type=Path, default=suite.DEFAULT_MANIFEST)
    parser.add_argument("--test", action="append", help="Official workload name; repeat to select several")
    parser.add_argument("--remote-profile", action="append", help="Default remote_fp16; use 'all' for four")
    parser.add_argument("--repeats", type=int, default=2, help="Cold/warm prompt repeats, at least 2")
    for key in ("b_host", "b_dashboard_url", "b_data_dir", "b_ssh", "b_config_path",
                "b_restart_command", "b_repo_path", "b_commit"):
        parser.add_argument("--" + key.replace("_", "-"))
    parser.add_argument("--b-base-config", type=Path, default=suite.HERE / "machine_b_base.yaml")
    parser.add_argument("--timeout-minutes", type=float, default=60, help="Timeout for each stage")
    parser.add_argument("--allow-no-remote-hits", action="store_true",
                        help="Keep remote results even without evidence of a Tier 2/3 fetch")
    parser.add_argument("--output-dir", type=Path, default=suite.RESULTS / "validation")
    parser.add_argument("--resume", type=Path, help="Retry a campaign; reuse saved settings and passed profiles")
    parser.add_argument("--dry-run", action="store_true", help="Print stages only; no installs/network/GPU work")
    parser.add_argument("--preflight-only", action="store_true", help="Stop before model download/benchmarks")
    parser.add_argument("--_check", choices=("runtime", "gpu", "model", "remote", "scorers"),
                        help=argparse.SUPPRESS)
    early, _ = parser.parse_known_args(argv)
    defaults = {}
    if early.resume:
        defaults.update(suite.read_json(early.resume / "state.json")["settings"])
    if early.config:
        config = suite.read_json(early.config)
        if not isinstance(config, dict) or set(config) - set(CONFIG_KEYS):
            parser.error("Config must be a JSON object using only: " + ", ".join(CONFIG_KEYS))
        defaults.update(config)
    for key, value in defaults.items():
        if key in ("test", "remote_profile") and (
                not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            parser.error(f"{key} must be a list of names")
    # append actions otherwise append CLI names to config lists, rather than override.
    tokens = list(sys.argv[1:] if argv is None else argv)
    for flag, key in (("--test", "test"), ("--remote-profile", "remote_profile")):
        if any(t == flag or t.startswith(flag + "=") for t in tokens):
            defaults.pop(key, None)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.manifest = Path(args.manifest).resolve()
    args.b_base_config = Path(args.b_base_config).resolve()
    if args.mode not in ("unit", "single", "remote", "both"):
        parser.error("Invalid mode in config")
    if args.granularity not in ("chunk", "head"):
        parser.error("Invalid granularity in config")
    if not isinstance(args.repeats, int) or args.repeats < 2:
        parser.error("--repeats must be >= 2 to measure cold/warm behavior")
    if not isinstance(args.timeout_minutes, (int, float)) or args.timeout_minutes <= 0:
        parser.error("--timeout-minutes must be positive")
    if not isinstance(args.allow_no_remote_hits, bool):
        parser.error("allow_no_remote_hits must be a JSON boolean")
    return args


def selected_profiles(args, manifest: dict) -> list[str]:
    """Keep the plain baseline independent of Machine B and its configuration."""
    names = ["plain"] if args.mode in ("single", "both") else []
    if args.mode in ("remote", "both"):
        remote = args.remote_profile or ["remote_fp16"]
        if remote == ["all"]:
            remote = [n for n, p in manifest["profiles"].items() if p["remote"]]
        for name in remote:
            if name not in manifest["profiles"] or not manifest["profiles"][name]["remote"]:
                raise ValueError(f"Not a remote profile: {name}")
        if not args.b_host:
            raise ValueError("Remote mode needs --b-host (B private IP/DNS)")
        if args.b_ssh:
            if not args.b_config_path or not args.b_restart_command:
                raise ValueError("--b-ssh needs --b-config-path and --b-restart-command")
            if not PurePosixPath(args.b_config_path).is_absolute():
                raise ValueError("--b-config-path must be an absolute path on B")
            if args.b_data_dir:
                raise ValueError("SSH runs choose a fresh B directory; omit --b-data-dir")
        elif len(remote) != 1 or not args.b_data_dir:
            raise ValueError("Without SSH, select one remote profile and provide --b-data-dir; "
                             "restart B manually with a fresh directory first")
        names.extend(remote)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate profiles selected")
    return names


def fingerprint(settings: dict) -> str:
    """Do not mix old successful runs with changed code, settings, or frozen data."""
    paths = [ROOT / "run_tests.py", ROOT / "model_settings.py", ROOT / "requirements.txt", ROOT / "pyproject.toml",
             ROOT / "lmcache_config.yaml", Path(settings["manifest"]),
             Path(settings["b_base_config"])]
    for directory in (ROOT / "tests", ROOT / "attention_probe", ROOT / "benchmarks"):
        paths.extend(directory.glob("*.py"))
    paths.extend((suite.DATA).glob("*.json*"))
    # Include vendored runtime/build sources, but never edit or reset that checkout.
    paths.extend(p for p in (ROOT / "LMCache" / "lmcache").rglob("*.py")
                 if p.name != "_version.py")  # Generated by the editable install.
    paths.extend([ROOT / "LMCache" / "setup.py", ROOT / "LMCache" / "pyproject.toml",
                  suite.HERE / "requirements.txt"])
    comparable = {k: v for k, v in settings.items()
                  if k not in ("timeout_minutes", "allow_no_remote_hits")}
    digest = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode())
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stop_owned_processes(proc) -> None:
    """Interrupt only our process group, giving suite.py time to close its server."""
    if os.name == "posix":
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=15 if sig == signal.SIGINT else 5)
            except subprocess.TimeoutExpired:
                continue
            # A worker may outlive the runner. Signal any remaining group members.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
    elif proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)


class Campaign:
    """Small stage runner: stream output, preserve failures, checkpoint profiles."""

    def __init__(self, path: Path, settings: dict, signature: str, dry_run=False):
        self.path, self.settings, self.dry_run = path, settings, dry_run
        self.env = os.environ.copy()
        self.env.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED="0", DSTN_ATTENTION_PROBE="0")
        self.env.pop("LMCACHE_CONFIG_FILE", None)  # suite generates the remote-only config.
        if settings.get("b_host"):
            self.env["MACHINE_B"] = settings["b_host"]
        state_path = path / "state.json"
        self.state = suite.read_json(state_path) if state_path.exists() else {
            "created_utc": utc_now(), "settings": settings, "fingerprint": signature, "stages": {}}
        if self.state["fingerprint"] != signature:
            raise RuntimeError("Code/config/input changed: start a new campaign instead of --resume")
        self.state["settings"] = settings

    def save(self):
        if not self.dry_run:
            suite.write_json(self.path / "state.json", self.state)

    def step(self, name: str, command: list[str], *, resume=False, verify=None):
        previous = self.state["stages"].get(name, {})
        if resume and previous.get("status") == "passed":
            if verify:
                verify()
            print(f"[SKIP] {name}: already passed", flush=True)
            return
        print(f"\n[STAGE] {name}\n+ {shlex.join(map(str, command))}", flush=True)
        if self.dry_run:
            return
        attempt = previous.get("attempt", 0) + 1
        log_path = self.path / "logs" / f"{name}.{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"status": "running", "started_utc": utc_now(), "attempt": attempt,
                  "command": list(map(str, command)), "log": str(log_path)}
        self.state["stages"][name] = record
        self.save()
        started, tail, proc = time.monotonic(), deque(maxlen=30), None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                        errors="replace", start_new_session=os.name == "posix")
                messages = queue.Queue()

                def read_output():
                    for line in proc.stdout:
                        messages.put(line)
                    messages.put(None)

                reader = threading.Thread(target=read_output, daemon=True)
                reader.start()
                heartbeat = started
                while True:
                    elapsed = time.monotonic() - started
                    if elapsed > self.settings["timeout_minutes"] * 60:
                        raise TimeoutError(f"{name} timed out; increase --timeout-minutes if appropriate")
                    try:
                        line = messages.get(timeout=1)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    if line:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        tail.append(line)
                    if time.monotonic() - heartbeat >= 30:
                        print(f"[RUNNING] {name}: {elapsed:.0f}s; log: {log_path}", flush=True)
                        heartbeat = time.monotonic()
                remaining = max(1, self.settings["timeout_minutes"] * 60 - elapsed)
                code = proc.wait(timeout=remaining)
                if code:
                    raise RuntimeError(f"{name} exited with code {code}")
                if verify:
                    verify()
            record["status"] = "passed"
            print(f"[PASS] {name}", flush=True)
        except BaseException as exc:
            if proc:
                stop_owned_processes(proc)
            record.update(status="failed", error=str(exc))
            (self.path / "failure.txt").write_text(
                traceback.format_exc() + "\nLast output:\n" + "".join(tail), encoding="utf-8")
            if name.startswith("benchmark-"):
                server_logs = list(self.path.rglob("vllm.log"))
                if server_logs:
                    latest = max(server_logs, key=lambda p: p.stat().st_mtime)
                    with latest.open(encoding="utf-8", errors="replace") as stream:
                        server_tail = "".join(deque(stream, maxlen=40))
                    print(f"\nLast server output ({latest}):\n{server_tail}", file=sys.stderr)
                    with (self.path / "failure.txt").open("a", encoding="utf-8") as stream:
                        stream.write(f"\nServer log: {latest}\n{server_tail}")
            print(f"[FAIL] {name}: {exc}\nLog: {log_path}", file=sys.stderr, flush=True)
            raise
        finally:
            if proc and proc.stdout:
                proc.stdout.close()
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            self.save()


def check_runtime(args, manifest):
    """Import real modules, not just find_spec: catch missing shared libraries too."""
    for name in ("yaml", "vllm", "transformers", "grpc", "openai", "jieba", "rouge",
                 "fuzzywuzzy", "pandas", "matplotlib"):
        module = importlib.import_module(name)
        print(f"IMPORT OK: {name}: {getattr(module, '__file__', '?')}", flush=True)
    for name in ("torch", "vllm", "transformers"):
        expected = next(line.split("==", 1)[1].strip() for line in
                        (ROOT / "requirements.txt").read_text().splitlines()
                        if line.startswith(name + "=="))
        actual = metadata.version(name)
        print(f"VERSION: {name}={actual}, expected={expected}")
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(f"{name} version differs from repository pins; rerun --setup")


def check_gpu(args, manifest):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA; check nvidia-smi, driver, and CUDA torch build")
    print(f"torch={torch.__version__}, CUDA runtime={torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}; "
              f"free={free / 2**30:.1f}GiB, total={total / 2**30:.1f}GiB")
        # Force a CUDA operation now; availability alone does not check kernel execution.
        value = torch.ones(8, device=f"cuda:{index}").sum().item()
        if value != 8:
            raise RuntimeError("CUDA arithmetic check failed")
    port = manifest["server"]["port"]
    with socket.socket() as probe:
        try:
            probe.bind((manifest["server"]["host"], port))
        except OSError as exc:
            raise RuntimeError(f"vLLM port {port} is busy; stop your old server first") from exc


def prepare_model(args, manifest):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    # Download outside vLLM's 240-second startup timeout; resumes the HF cache.
    snapshot = snapshot_download(repo_id=manifest["model"], revision=manifest["model_revision"],
                                 allow_patterns=["*.json", "model*.safetensors", "*.model",
                                                 "*.model.v3", "*.jinja", "*.txt"])
    if not list(Path(snapshot).glob("model*.safetensors")):
        raise RuntimeError("Pinned model has no model*.safetensors weights; review download patterns")
    validate_weight_config(manifest, suite.read_json(Path(snapshot) / "config.json"))
    # Weight quantization must not change token IDs or the benchmark chat template.
    tokenizer, tokenizer_revision = tokenizer_identity(manifest)
    AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision, trust_remote_code=False)
    print(f"Pinned model ready: {snapshot}; tokenizer={tokenizer}@{tokenizer_revision}")


def check_remote(args, manifest):
    import lmcache
    if Path(lmcache.__file__).resolve().parent != (ROOT / "LMCache" / "lmcache").resolve():
        raise RuntimeError("Wrong LMCache checkout; install this repo with --setup")
    importlib.import_module("lmcache.v1.storage_backend.grpc_backend")
    importlib.import_module("lmcache.integration.vllm.vllm_v1_adapter")
    import grpc
    from lmcache.v1.storage_backend import evicpress_pb2 as pb, evicpress_pb2_grpc as rpc
    # Read-only RPC: manual one-host campaigns must not alter the utility
    # denominator during preflight before the actual cold benchmark.
    with grpc.insecure_channel(f"{args.b_host}:50051") as channel:
        stats = rpc.EvicPressServiceStub(channel).GetStats(pb.StatsRequest(), timeout=10)
        if args.granularity == "head" and stats.head_schema_version != 1:
            raise RuntimeError("Machine B gRPC lacks head schema v1")
    with socket.create_connection((args.b_host, 50051), timeout=5):
        pass
    state = suite.b_state(args.b_dashboard_url or f"http://{args.b_host}:8080")
    print("Machine B configuration:", state["config"])
    if args.granularity == "head" and state["config"].get("head_schema_version") != 1:
        raise RuntimeError("Machine B lacks head schema v1; update both repositories")
    if not args.b_ssh:
        profile = manifest["profiles"][(args.remote_profile or ["remote_fp16"])[0]]
        suite.verify_b(state, profile, args.b_data_dir, args.granularity)
        print("Manual B run: ensure service was restarted with a fresh empty data directory")


def check_scorers(args, manifest):
    for name in args.test or manifest["tests"]:
        test = manifest["tests"][name]
        if test["kind"] in ("ruler", "longbench"):
            suite.official_score(test["kind"], test, manifest, ["check"], [["check"]])
            print(f"Official scorer import/call OK: {name}")


def verify_benchmark(path: Path, profiles: dict, name: str, tests: list[str], allow_no_hits: bool):
    """A zero process exit is not sufficient evidence of a completed remote test."""
    runs = sorted(path.glob("*/summary.json"))
    if not runs:
        raise RuntimeError(f"No benchmark summary produced under {path}")
    summary_path = runs[-1]
    if not (summary_path.parent / "summary.csv").exists():
        raise RuntimeError("Benchmark did not finish writing summary.csv")
    summary = suite.read_json(summary_path)
    if set(tests) != set(summary):
        raise RuntimeError(f"Incomplete benchmark summary: expected {tests}, found {list(summary)}")
    missing_hits = [test for test in tests if profiles[name]["remote"]
                    and not summary[test].get("remote_fetch_verified")]
    if missing_hits:
        message = (f"No measured B Tier 2/3 fetches for {missing_hits}. Inspect B counters, "
                   "A vllm.log, local cache residency, and B tier capacities. "
                   "Warm latency alone does not prove remote reuse.")
        if not allow_no_hits:
            raise RuntimeError(message + " Use --allow-no-remote-hits only to retain this diagnostic result.")
        print("WARNING:", message)
    print(f"Metrics: {summary_path.parent / 'summary.csv'}")


def write_comparison(campaign: Campaign, profiles: dict):
    """Collect the existing CSV metrics, without changing scoring or timing."""
    if campaign.dry_run:
        return
    rows = []
    for name, profile in profiles.items():
        if campaign.state["stages"].get("benchmark-" + name, {}).get("status") != "passed":
            continue
        group = "remote" if profile["remote"] else "single"
        metrics = sorted((campaign.path / group / name).glob("*/summary.csv"))[-1]
        summaries = suite.read_json(metrics.with_suffix(".json"))
        with metrics.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"profile": name, "run_dir": str(metrics.parent), **row,
                             "remote_fetch_verified": summaries[row["test"]].get("remote_fetch_verified")})
    if rows:
        with (campaign.path / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_campaign(args):
    manifest = suite.read_json(args.manifest)
    profiles = selected_profiles(args, manifest)
    tests = args.test or [n for n, t in manifest["tests"].items() if t["kind"] != "legacy"]
    for name in tests:
        if name not in manifest["tests"] or manifest["tests"][name]["kind"] == "legacy":
            raise ValueError(f"--test requires an official workload; unknown/legacy test: {name}")
    if len(tests) != len(set(tests)):
        raise ValueError("Duplicate tests selected")
    settings = {key: str(value) if isinstance(value, Path) else value
                for key in CONFIG_KEYS if (value := getattr(args, key)) is not None}
    signature = fingerprint(settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = (args.resume or args.output_dir / timestamp).resolve()
    campaign = Campaign(path, settings, signature, args.dry_run)
    # One campaign lock prevents two clients from starting the same GPU stages.
    lock = path / ".running"
    if not args.dry_run:
        path.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open("x") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError as exc:
            raise RuntimeError(f"Campaign already locked: {lock}. Check its PID; only remove "
                               "this lock if the previous runner is no longer running") from exc
    try:
        print(f"Campaign: {path}\nMode: {args.mode}; profiles: {profiles}; workloads: {tests}")
        print(f"Host: {platform.node()}, {platform.platform()}, Python {sys.version.split()[0]}")
        print(f"Free repository disk: {shutil.disk_usage(ROOT).free / 2**30:.1f}GiB")
        campaign.save()
        python = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists() and not args.setup:
            python = Path(sys.executable)
        if not args.dry_run:
            suite.write_json(path / "environment.json", {
                "recorded_utc": utc_now(), "host": platform.node(), "platform": platform.platform(),
                "launcher_python": sys.version, "test_python": str(python),
                "repo": suite.git_info(ROOT), "profiles": profiles, "tests": tests,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "free_disk_bytes": shutil.disk_usage(ROOT).free,
            })
        if args.setup:
            if not args.dry_run and (platform.system() != "Linux" or
                                    not (3, 10) <= sys.version_info[:2] < (3, 14)):
                raise RuntimeError("--setup requires Linux and Python 3.10-3.13")
            if not args.dry_run and (not shutil.which("git") or not shutil.which("g++")):
                raise RuntimeError("Setup needs git and g++ (Ubuntu: install git build-essential python3-venv)")
            campaign.step("setup-venv", [sys.executable, "-m", "venv", str(VENV)])
            # Avoid downloading a second editable LMCache from the Git requirement.
            requirements = path / "setup-requirements.txt"
            if not args.dry_run:
                requirements.write_text("\n".join(line for line in
                    (ROOT / "requirements.txt").read_text().splitlines()
                    if "#egg=lmcache" not in line) + "\n", encoding="utf-8")
            campaign.env.setdefault("MAX_JOBS", "2")
            campaign.step("setup-dependencies", [str(python), "-m", "pip", "install", "-r",
                          str(requirements), "-r", str(suite.HERE / "requirements.txt"), "wheel"])
            campaign.step("setup-local-lmcache", [str(python), "-m", "pip", "install", "-e",
                          str(ROOT / "LMCache"), "--no-deps", "--no-build-isolation"])
            campaign.step("setup-project", [str(python), "-m", "pip", "install", "-e",
                          str(ROOT), "--no-deps", "--no-build-isolation"])
        for filename in UNIT_FILES:
            campaign.step("unit-" + Path(filename).stem,
                          [str(python), "-m", "unittest", "discover", "-s", "tests", "-p", filename, "-v"])
        if not profiles:
            return path
        if platform.system() != "Linux" and not args.dry_run:
            raise RuntimeError("GPU benchmark serving is supported on Linux; use --mode unit here")
        if not shutil.which("git") and not args.dry_run:
            raise RuntimeError("Install git to fetch pinned official scorer sources")
        campaign.step("dependency-consistency", [str(python), "-m", "pip", "check"])
        check_cmd = [str(python), str(ROOT / "run_tests.py"), "--manifest", str(args.manifest),
                     "--granularity", args.granularity]
        campaign.step("runtime-imports", check_cmd + ["--_check", "runtime"])
        campaign.step("gpu-driver", ["nvidia-smi"])
        campaign.step("gpu-runtime-and-port", check_cmd + ["--_check", "gpu"])
        runner = [str(python), str(suite.HERE / "suite.py"), "--manifest", str(args.manifest)]
        campaign.step("official-sources-and-inputs", runner + ["prepare", "--sources-only"])
        selected = [arg for test in tests for arg in ("--test", test)]
        campaign.step("official-scorers", check_cmd + ["--_check", "scorers"] + selected)
        remote_args = []
        for key in CONFIG_KEYS:
            if key.startswith("b_") and settings.get(key):
                remote_args.extend(["--" + key.replace("_", "-"), str(settings[key])])
        if any(manifest["profiles"][n]["remote"] for n in profiles):
            if args.b_ssh:
                campaign.step("ssh-access", ["ssh", "-o", "BatchMode=yes", "-o",
                              "ConnectTimeout=10", args.b_ssh, "true"])
            campaign.step("remote-imports-and-connectivity", check_cmd + ["--_check", "remote"] +
                          remote_args + ["--remote-profile", next(n for n in profiles if n != "plain")])
            if args.b_ssh:
                # The profile runner restarts B afterwards, clearing smoke data
                # and counters. Manual/local runs use only the read-only ping.
                campaign.step("remote-grpc-roundtrip", [str(python), str(ROOT / "tests" / "smoke_test_b.py")])
            else:
                print("[INFO] Read-only gRPC ping passed; KV smoke omitted to keep B counters cold")
        if args.preflight_only:
            print("Preflight passed; model download and GPU benchmarks were not run")
            return path
        campaign.step("pinned-model-download", check_cmd + ["--_check", "model"])
        for name in profiles:
            # A stale server must never make the next profile silently use the
            # previous connector/configuration. Recheck the port before every run.
            campaign.step("profile-ready-" + name, check_cmd + ["--_check", "gpu"])
            group = "remote" if manifest["profiles"][name]["remote"] else "single"
            output = path / group / name
            # Retry a failed profile as a whole: partial cold/warm runs are not merged.
            command = runner + ["run", "--profile", name, "--repeats", str(args.repeats),
                                "--output-dir", str(output), "--granularity", args.granularity] + selected
            if group == "remote":
                command += remote_args
            campaign.step("benchmark-" + name, command, resume=bool(args.resume),
                          verify=lambda n=name, p=output: verify_benchmark(
                              p, manifest["profiles"], n, tests, args.allow_no_remote_hits))
            write_comparison(campaign, manifest["profiles"])
        return path
    finally:
        if not args.dry_run:
            lock.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    if args._check:
        checks = {"runtime": check_runtime, "gpu": check_gpu, "model": prepare_model,
                  "remote": check_remote, "scorers": check_scorers}
        checks[args._check](args, suite.read_json(args.manifest))
        return 0
    path = run_campaign(args)
    print(f"\n{'Dry-run plan only; nothing executed' if args.dry_run else 'Done'}. "
          f"{'Planned output' if args.dry_run else 'Logs, stage status, and metrics'}: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()  # Keep import/shared-library errors and their causes visible.
        print(f"\nSTOPPED: {exc or 'interrupted'}", file=sys.stderr)
        print("Fix the failing stage, then: python3 run_tests.py --resume CAMPAIGN_PATH\n"
              "After code/config/input changes, start a new campaign (omit --resume).\n"
              "Debug: stage logs + failure.txt; GPU failures: single/remote/PROFILE/RUN/vllm.log.\n"
              "CUDA/OOM: check driver and free VRAM; imports/build: inspect pip/local LMCache;\n"
              "downloads: check network/HF access and cache disk; B: check ports 50051/8080,\n"
              "security groups, B logs, alpha/quantization settings and tier capacities.", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1)
