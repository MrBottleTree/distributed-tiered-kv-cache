"""One GPU-free orchestration check; no model download or remote writes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tests


class NewMachineRunnerTest(unittest.TestCase):
    def test_profile_isolation_and_resume(self):
        commands = []
        default_remote = run_tests.parse_args([
            "--mode", "remote", "--b-host", "storage.example", "--b-data-dir", "/fresh"])
        self.assertEqual(run_tests.selected_profiles(default_remote,
                         run_tests.suite.read_json(run_tests.suite.DEFAULT_MANIFEST)), ["remote_fp16"])
        args = run_tests.parse_args([
            "--mode", "both", "--test", "ruler_niah_single_4k", "--dry-run",
            "--b-host", "storage.example", "--b-ssh", "user@storage.example",
            "--b-config-path", "/opt/b/config.yaml", "--b-restart-command", "restart-b",
            "--remote-profile", "all",
        ])
        with mock.patch.object(run_tests.Campaign, "step", side_effect=lambda n, c, **kw:
                               commands.append((n, c))), mock.patch("builtins.print"):
            run_tests.run_campaign(args)
        names = [name for name, _ in commands]
        self.assertEqual(names[:3], ["unit-" + Path(f).stem for f in run_tests.UNIT_FILES])
        benchmarks = [(n, c) for n, c in commands if n.startswith("benchmark-")]
        self.assertEqual(len(benchmarks), 5)
        self.assertNotIn("--b-host", benchmarks[0][1])
        self.assertTrue(all("--b-host" in c for _, c in benchmarks[1:]))
        self.assertLess(names.index("remote-grpc-roundtrip"), names.index("pinned-model-download"))

        # Only a passed benchmark can be skipped. Failed stages run again.
        with tempfile.TemporaryDirectory() as temporary:
            campaign = run_tests.Campaign(Path(temporary), {}, "signature", dry_run=True)
            campaign.state["stages"]["passed"] = {"status": "passed"}
            with mock.patch("builtins.print") as output:
                campaign.step("passed", ["never-execute"], resume=True)
                self.assertIn("[SKIP]", output.call_args.args[0])
            campaign.state["stages"]["passed"] = {"status": "failed"}
            with mock.patch("builtins.print") as output:
                campaign.step("passed", ["never-execute"], resume=True)
                self.assertIn("[STAGE]", output.call_args.args[0])

            # Exercise real subprocess logging/fail-fast behavior without GPU work.
            campaign = run_tests.Campaign(Path(temporary), {"timeout_minutes": 1}, "signature")
            with mock.patch("builtins.print"):
                with self.assertRaises(RuntimeError):
                    campaign.step("broken", [sys.executable, "-c", "print('debug marker'); raise SystemExit(2)"])
            self.assertEqual(campaign.state["stages"]["broken"]["status"], "failed")
            self.assertIn("debug marker", (Path(temporary) / "failure.txt").read_text())

            # Completed metrics are preserved; missing remote-fetch evidence is a failure.
            metrics = Path(temporary) / "remote" / "remote_alpha_1" / "run1"
            metrics.mkdir(parents=True)
            run_tests.suite.write_json(metrics / "summary.json", {
                "task": {"kind": "ruler", "remote_fetch_verified": False}})
            (metrics / "summary.csv").write_text("test,quality_score_pct\ntask,100\n")
            profiles = {"remote_alpha_1": {"remote": True}}
            with self.assertRaises(RuntimeError):
                run_tests.verify_benchmark(metrics.parent, profiles, "remote_alpha_1", ["task"], False)
            with mock.patch("builtins.print"):
                run_tests.verify_benchmark(metrics.parent, profiles, "remote_alpha_1", ["task"], True)
            campaign.state["stages"]["benchmark-remote_alpha_1"] = {"status": "passed"}
            run_tests.write_comparison(campaign, profiles)
            self.assertIn("remote_alpha_1", (Path(temporary) / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
