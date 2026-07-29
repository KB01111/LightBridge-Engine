"""Compare the pinned llama.cpp graph with BRIDGE's llama-Q8_K mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    return parser.parse_args()


def load_bridge(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((root / "bridge-vectors.json").read_text("utf-8"))
    raw = np.memmap(root / manifest["data_file"], dtype="<f4", mode="r")
    arrays: dict[str, np.ndarray] = {}
    for entry in manifest["arrays"]:
        start = entry["offset_bytes"] // 4
        end = start + entry["element_count"]
        values = np.asarray(raw[start:end], dtype=np.float32)
        digest = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
        if digest != entry["sha256_f32le"]:
            raise RuntimeError(f"BRIDGE vector hash mismatch: {entry['name']}")
        arrays[entry["name"]] = values
    return arrays, manifest


def main() -> None:
    args = parse_args()
    arrays, manifest = load_bridge(args.bridge)
    report = json.loads(args.llama.read_text("utf-8"))
    mode = next(item for item in manifest["modes"] if item["mode"] == "llama_q8_k")
    failures: list[str] = []
    comparisons = []

    for index, (bridge_step, llama_step) in enumerate(
        zip(mode["steps"], report["steps"], strict=True)
    ):
        if bridge_step["selected_experts"] != llama_step["selected_experts"]:
            failures.append(
                f"step {index} routing: bridge={bridge_step['selected_experts']} "
                f"llama={llama_step['selected_experts']}"
            )
        if bridge_step["greedy_id"] != llama_step["greedy_id"]:
            failures.append(
                f"step {index} greedy: bridge={bridge_step['greedy_id']} "
                f"llama={llama_step['greedy_id']}"
            )
        for suffix, atol, rtol in [
            ("final.logits", 3.0e-4, 3.0e-4),
            ("final.probabilities", 5.0e-5, 5.0e-5),
        ]:
            key = f"llama_q8_k.step{index}.{suffix}"
            actual = arrays[key]
            expected = np.asarray(llama_step[suffix.split(".")[-1]], dtype=np.float32)
            absolute = np.abs(actual - expected)
            passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            comparisons.append(
                {
                    "name": f"step{index}.{suffix}",
                    "atol": atol,
                    "rtol": rtol,
                    "max_abs": float(absolute.max(initial=0.0)),
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"step {index} {suffix}: "
                    f"max_abs={float(absolute.max(initial=0.0)):.9g}"
                )

    result = {
        "format": "lightbridge-llama-differential-report-v1",
        "passed": not failures,
        "failures": failures,
        "comparisons": comparisons,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
