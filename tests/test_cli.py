from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tiny_mlp.mlir"


class CliTests(unittest.TestCase):
    def run_cli(
        self, *arguments: str, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ngopt", *arguments],
            cwd=ROOT,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_and_version(self) -> None:
        help_result = self.run_cli("--help")
        version_result = self.run_cli("--version")

        self.assertEqual(help_result.returncode, 0)
        self.assertIn("MLIR-style neural graph optimizer", help_result.stdout)
        self.assertEqual(version_result.returncode, 0)
        self.assertRegex(version_result.stdout, r"ng-opt \d+\.\d+\.\d+")

    def test_optimize_example_and_write_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "optimized.mlir"
            stats_path = Path(directory) / "stats.json"
            result = self.run_cli(
                "optimize",
                str(EXAMPLE),
                "-o",
                str(output_path),
                "--stats",
                str(stats_path),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ng.linear_relu", output_path.read_text(encoding="utf-8"))
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertLess(stats["nodes_after"], stats["nodes_before"])

    def test_run_example_returns_json_outputs(self) -> None:
        result = self.run_cli("run", str(EXAMPLE), "--input", "%x=3,-2")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["outputs"][0]["shape"], [1, 2])
        self.assertEqual(payload["outputs"][0]["values"], [5.25, 0.0])

    def test_invalid_input_has_nonzero_exit_without_traceback(self) -> None:
        result = self.run_cli("optimize", "-", stdin="not a module")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("<stdin>:1:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_validate_reports_deterministic_equivalence(self) -> None:
        result = self.run_cli("validate", str(EXAMPLE), "--trials", "5", "--seed", "7")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["equivalent"])
        self.assertEqual(payload["trials"], 5)
        self.assertEqual(payload["seed"], 7)

    def test_lower_inspect_and_list_passes(self) -> None:
        lower_result = self.run_cli("lower", str(EXAMPLE), "--optimize")
        inspect_result = self.run_cli("inspect", str(EXAMPLE))
        passes_result = self.run_cli("--list-passes")

        self.assertEqual(lower_result.returncode, 0, lower_result.stderr)
        self.assertIn("linalg.generic", lower_result.stdout)
        self.assertNotIn("ng.linear_relu", lower_result.stdout)
        self.assertEqual(json.loads(inspect_result.stdout)["parameters"], 6)
        self.assertIn("const-fold", passes_result.stdout)

    def test_visualize_and_demo_generate_artifact_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.svg"
            demo_path = Path(directory) / "demo"
            visualize_result = self.run_cli(
                "visualize", str(EXAMPLE), "--optimize", "-o", str(graph_path)
            )
            demo_result = self.run_cli("demo", str(EXAMPLE), "--output-dir", str(demo_path))

            self.assertEqual(visualize_result.returncode, 0, visualize_result.stderr)
            self.assertIn("<svg", graph_path.read_text(encoding="utf-8"))
            self.assertEqual(demo_result.returncode, 0, demo_result.stderr)
            self.assertTrue((demo_path / "before.svg").exists())
            self.assertTrue((demo_path / "after.svg").exists())
            self.assertTrue((demo_path / "optimized.mlir").exists())
            self.assertTrue(json.loads(demo_result.stdout)["equivalent"])

    def test_unknown_pass_has_user_friendly_error(self) -> None:
        result = self.run_cli("optimize", str(EXAMPLE), "--passes", "magic")

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown optimizer pass", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_unknown_or_duplicate_cli_inputs_are_rejected(self) -> None:
        unknown = self.run_cli("run", str(EXAMPLE), "--input", "%x=3,-2", "--input", "%bogus=1")
        duplicate = self.run_cli("run", str(EXAMPLE), "--input", "%x=3,-2", "--input", "%x=1,2")

        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown input %bogus", unknown.stderr)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("duplicate input %x", duplicate.stderr)

    def test_validation_trial_limit_is_enforced(self) -> None:
        result = self.run_cli("validate", str(EXAMPLE), "--trials", "1001")

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot exceed 1000", result.stderr)

    def test_non_finite_tolerance_is_rejected(self) -> None:
        result = self.run_cli("validate", str(EXAMPLE), "--atol", "inf")

        self.assertEqual(result.returncode, 2)
        self.assertIn("finite and non-negative", result.stderr)

    def test_demo_directory_error_is_user_friendly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "already-a-file"
            file_path.write_text("content", encoding="utf-8")
            result = self.run_cli("demo", str(EXAMPLE), "--output-dir", str(file_path))

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot create demo directory", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_input_only_graph_cannot_bypass_validation_work_budget(self) -> None:
        module = """module @large_input {
  func.func @main(%x: tensor<1000000xf32>) -> tensor<1000000xf32> {
    ng.return %x : tensor<1000000xf32>
  }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "large.mlir"
            model_path.write_text(module, encoding="utf-8")
            result = self.run_cli("validate", str(model_path), "--trials", "1000")

        self.assertEqual(result.returncode, 2)
        self.assertIn("validation exceeds", result.stderr)

    def test_custom_passes_imply_optimization(self) -> None:
        result = self.run_cli("lower", str(EXAMPLE), "--passes", "fuse-linear-relu,dce")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ngopt_gemm_bias_relu", result.stdout)

    def test_optimize_rejects_colliding_output_and_stats_sinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.txt"
            result = self.run_cli(
                "optimize",
                str(EXAMPLE),
                "-o",
                str(output_path),
                "--stats",
                str(output_path),
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("different destinations", result.stderr)


if __name__ == "__main__":
    unittest.main()
