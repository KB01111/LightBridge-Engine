from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("verify_full_model.py")
SPEC = importlib.util.spec_from_file_location("verify_full_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_FULL_MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY_FULL_MODEL)


def bridge_report(generated: list[int]) -> dict[str, object]:
    return {
        "prompt_token_ids": [1, 2],
        "token_ids": generated,
        "raw_text": "accepted",
        "total_milliseconds": 7,
    }


def oracle_report(second_prediction: int = 12) -> dict[str, object]:
    return {
        "format": "lightbridge-llama-full-model-oracle-v1",
        "llama_commit": VERIFY_FULL_MODEL.LLAMA_COMMIT,
        "steps": [
            {"position": 0, "input_id": 1, "greedy_id": 99, "margin": 1.0},
            {"position": 1, "input_id": 2, "greedy_id": 11, "margin": 0.5},
            {"position": 2, "input_id": 11, "greedy_id": second_prediction, "margin": 0.25},
        ],
    }


class VerifyFullModelTests(unittest.TestCase):
    def test_accepts_exact_direct_sidecar_and_llama_sequence(self) -> None:
        report = VERIFY_FULL_MODEL.verify(
            bridge_report([11, 12]),
            bridge_report([11, 12]),
            oracle_report(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["generated_token_ids"], [11, 12])
        self.assertEqual(report["llama_greedy_margins"], [0.5, 0.25])

    def test_rejects_sidecar_divergence(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "generated token IDs differ"):
            VERIFY_FULL_MODEL.verify(
                bridge_report([11, 12]),
                bridge_report([11, 13]),
                oracle_report(),
            )

    def test_rejects_llama_divergence(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "greedy token mismatch"):
            VERIFY_FULL_MODEL.verify(
                bridge_report([11, 12]),
                bridge_report([11, 12]),
                oracle_report(second_prediction=13),
            )

    def test_rejects_truncated_oracle(self) -> None:
        oracle = oracle_report()
        oracle["steps"] = oracle["steps"][:-1]
        with self.assertRaisesRegex(RuntimeError, "step count mismatch"):
            VERIFY_FULL_MODEL.verify(
                bridge_report([11, 12]),
                bridge_report([11, 12]),
                oracle,
            )


if __name__ == "__main__":
    unittest.main()
