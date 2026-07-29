from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_full_model_acceptance.py")
SPEC = importlib.util.spec_from_file_location("run_full_model_acceptance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


class FullModelAcceptanceTests(unittest.TestCase):
    def test_accepts_exact_complete_payload_report(self) -> None:
        ACCEPTANCE.assert_authenticated_payload(
            {
                "schema_valid": True,
                "files": [
                    {
                        "logical_bytes": ACCEPTANCE.EXPECTED_LENGTH,
                        "allocated_bytes": ACCEPTANCE.EXPECTED_LENGTH + 4096,
                        "sparse": False,
                        "compressed": False,
                        "sha256": ACCEPTANCE.EXPECTED_SHA256,
                    }
                ],
            }
        )

    def test_rejects_sparse_payload_report(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-sparse"):
            ACCEPTANCE.assert_authenticated_payload(
                {
                    "schema_valid": True,
                    "files": [
                        {
                            "logical_bytes": ACCEPTANCE.EXPECTED_LENGTH,
                            "allocated_bytes": 64 * 1024 * 1024,
                            "sparse": True,
                            "compressed": False,
                            "sha256": ACCEPTANCE.EXPECTED_SHA256,
                        }
                    ],
                }
            )

    def test_accepts_authenticated_ntfs_compressed_payload(self) -> None:
        ACCEPTANCE.assert_authenticated_payload(
            {
                "schema_valid": True,
                "files": [
                    {
                        "logical_bytes": ACCEPTANCE.EXPECTED_LENGTH,
                        "allocated_bytes": ACCEPTANCE.EXPECTED_LENGTH - 4096,
                        "sparse": False,
                        "compressed": True,
                        "sha256": ACCEPTANCE.EXPECTED_SHA256,
                    }
                ],
            }
        )

    def test_runtime_command_is_greedy_and_rtk_wrapped(self) -> None:
        args = SimpleNamespace(
            model=Path("model.gguf"),
            context=512,
            cache_mib=2048,
            cpu_threads=12,
            memory_headroom_mib=512,
            max_tokens=2,
            prompt="hello",
        )
        command = ACCEPTANCE.common_runtime_args(args, Path("bridge.exe"))
        self.assertEqual(command[0], "rtk")
        self.assertEqual(command[1:3], ["bridge.exe", "chat"])
        self.assertEqual(command[command.index("--temperature") + 1], "0")
        self.assertEqual(command[command.index("--top-k") + 1], "1")
        self.assertIn("--json", command)

    def test_benchmark_command_uses_one_in_process_cold_warm_run(self) -> None:
        args = SimpleNamespace(
            model=Path("model.gguf"),
            output_dir=Path("acceptance"),
            context=512,
            cache_mib=2048,
            cpu_threads=12,
            memory_headroom_mib=512,
            max_tokens=2,
            prompt="Hi",
        )
        command = ACCEPTANCE.common_bench_args(args, Path("bridge.exe"))
        self.assertEqual(command[0], "rtk")
        self.assertIn("--cold-warm", command)
        self.assertNotIn("--cache-heat", command)

    def test_stage_runner_refuses_unwrapped_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = ACCEPTANCE.StageRunner(Path(directory), Path(directory))
            with self.assertRaisesRegex(RuntimeError, "must execute through rtk"):
                runner.run("bad", ["python", "--version"])


if __name__ == "__main__":
    unittest.main()
