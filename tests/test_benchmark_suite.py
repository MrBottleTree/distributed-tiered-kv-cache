"""Local, GPU-free tests for the benchmark harness."""
import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import suite
from model_settings import model_options, model_server_args, validate_weight_config, verify_input_tokenizer


class _FakeOpenAIServer(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        data = b'{"data":[{"id":"model"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] is True
        assert body["temperature"] == 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for part in (
            {"choices": [{"text": "blue"}]},
            {"choices": [{"text": "bird"}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ):
            self.wfile.write(("data: " + json.dumps(part) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


class SuiteTests(unittest.TestCase):
    def test_int4_weights_keep_fp16_kv_and_original_tokenizer(self):
        manifest = suite.read_json(suite.DEFAULT_MANIFEST)
        options = model_options(manifest)
        self.assertEqual(manifest["weight_quantization"], {"method": "awq", "bits": 4})
        self.assertEqual(options["dtype"], "float16")
        self.assertEqual(options["kv_cache_dtype"], "float16")
        self.assertEqual(options["tokenizer"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertNotEqual(options["revision"], options["tokenizer_revision"])
        self.assertNotIn("quantization", options)  # Checkpoint detection permits AWQ/Marlin.
        verify_input_tokenizer(manifest, suite.read_json(suite.DATA / "provenance.json"))
        with self.assertRaises(RuntimeError):
            verify_input_tokenizer(manifest, {"model": "wrong", "model_revision": "wrong"})
        validate_weight_config(manifest, {"quantization_config": {"quant_method": "awq", "bits": 4}})
        with self.assertRaises(RuntimeError):
            validate_weight_config(manifest, {"quantization_config": {"quant_method": "awq", "bits": 8}})
        self.assertNotIn("revision", model_options(manifest, model="another/model"))

        # Exercise the real server-command builder without launching a GPU worker.
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(suite.subprocess, "Popen") as process, \
             mock.patch.object(suite, "http_json", return_value={"data": []}):
            process.return_value.poll.return_value = None
            with suite.model_server(manifest, {"remote": False}, Path(temporary), None, None):
                command = process.call_args.args[0]
                expected = model_server_args(manifest)
                self.assertEqual(command[3:3 + len(expected)], expected)

    def test_manifest_profiles_and_sources(self):
        manifest = suite.read_json(suite.DEFAULT_MANIFEST)
        self.assertEqual(len(manifest["profiles"]), 5)
        self.assertEqual(len(manifest["sources"]["ruler"]["commit"]), 40)
        self.assertEqual(len(manifest["sources"]["longbench"]["commit"]), 40)
        self.assertIn("long_doc_qa", manifest["tests"])
        self.assertIn("needle_haystack", manifest["tests"])

    def test_render_and_verify_b_profile(self):
        manifest = suite.read_json(suite.DEFAULT_MANIFEST)
        profile = manifest["profiles"]["remote_alpha_5"]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "b.yaml"
            suite.render_b_config(
                suite.HERE / "machine_b_base.yaml", profile,
                "/data/kv_cache/test_run", target)
            import yaml
            cfg = yaml.safe_load(target.read_text())
            self.assertEqual(cfg["evicpress"]["alpha"], 5.0)
            self.assertTrue(cfg["quantization"]["enabled"])
            self.assertEqual(cfg["tier3"]["data_dir"], "/data/kv_cache/test_run")
            state = {"config": {"alpha": 5.0, "quant_enabled": True,
                                "data_dir": "/data/kv_cache/test_run"}}
            suite.verify_b(state, profile, "/data/kv_cache/test_run")
            with self.assertRaises(RuntimeError):
                suite.verify_b(state, manifest["profiles"]["remote_fp16"])

    def test_streaming_completion_timing_and_usage(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = suite.completion(
                f"http://127.0.0.1:{server.server_port}/v1",
                "model", "prompt", 8)
            self.assertEqual(result["prediction"], "bluebird")
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertGreaterEqual(result["latency_s"], result["ttft_s"])
        finally:
            server.shutdown()
            server.server_close()

    def test_summary_uses_first_reply_for_quality(self):
        rows = [
            {"sample": 0, "repeat": 0, "prediction": "right",
             "references": ["right"], "ttft_s": 2, "latency_s": 3,
             "usage": {"completion_tokens": 1}},
            {"sample": 0, "repeat": 1, "prediction": "wrong",
             "references": ["right"], "ttft_s": 1, "latency_s": 2,
             "usage": {"completion_tokens": 1}},
        ]
        with mock.patch.object(suite, "official_score", return_value=100) as scorer:
            summary = suite.summarize_samples(rows, {"kind": "ruler"}, {})
        self.assertEqual(scorer.call_args.args[3], ["right"])
        self.assertEqual(summary["warm_ttft_p50_s"], 1)
        self.assertEqual(summary["repeat_consistency_pct"], 0)
        self.assertAlmostEqual(summary["completion_tokens_per_s"], 0.4)

    def test_counter_delta(self):
        self.assertEqual(
            suite.numeric_delta(
                {"total_hits": 12, "quant_breakdown": {"fp16": 3}},
                {"total_hits": 5, "quant_breakdown": {"fp16": 1}}),
            {"total_hits": 7, "quant_breakdown": {"fp16": 2}})

    def test_external_server_end_to_end_writes_artifacts(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            data_dir = temp / "data"
            data_dir.mkdir()
            item = data_dir / "official.jsonl"
            item.write_text(json.dumps({"input": "Prompt", "outputs": ["bluebird"]}) + "\n")
            suite.write_json(data_dir / "provenance.json", {
                "model": "model",
                "inputs": {"official": suite.file_sha256(item)},
                "model_revision": "x" * 40,
                "sources": {"ruler": "x" * 40}})
            manifest = {
                "profiles": {"plain": {"remote": False}},
                "tests": {"official": {"kind": "ruler", "max_new_tokens": 8}},
                "sources": {"ruler": {"commit": "x" * 40}},
                "model": "model",
                "model_revision": "x" * 40,
                "server": {},
            }
            manifest_path = temp / "manifest.json"
            suite.write_json(manifest_path, manifest)
            args = argparse.Namespace(
                profile="plain", test=["official"], repeats=2, b_host=None,
                b_dashboard_url=None, b_data_dir=None, b_ssh=None,
                b_config_path=None, b_restart_command=None, b_base_config=None,
                b_repo_path=None, b_commit=None,
                external_server=f"http://127.0.0.1:{server.server_port}/v1",
                output_dir=temp / "results",
            )
            try:
                with mock.patch.object(suite, "DATA", data_dir), \
                     mock.patch.object(suite, "require_source", return_value=temp), \
                     mock.patch.object(suite, "official_score", return_value=100):
                    output = suite.run_suite(args, manifest, manifest_path)
                summary = suite.read_json(output / "summary.json")["official"]
                self.assertEqual(summary["quality_score_pct"], 100)
                self.assertEqual(summary["requests"], 2)
                self.assertTrue((output / "summary.csv").exists())
                self.assertTrue((output / "official.jsonl").exists())
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
