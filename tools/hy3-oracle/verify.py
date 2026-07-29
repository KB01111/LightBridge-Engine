#!/usr/bin/env python3
"""Compare pinned Transformers arrays to BRIDGE dequant-F32 vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--transformers", type=Path, required=True)
    return parser.parse_args()


def tolerance(name: str) -> tuple[float, float]:
    # Q/K values in later blocks include accumulated BLAS-vs-scalar rounding
    # from all preceding projections. Five micro-units remains tighter than
    # the downstream residual/logit bounds while covering the measured
    # float32 accumulation envelope.
    if name.endswith((".queries", ".keys")):
        return 5.0e-6, 5.0e-6
    if name.endswith(".attention_normalized"):
        return 2.0e-6, 2.0e-6
    if name.endswith((".values", ".attention_context")):
        return 2.0e-5, 2.0e-5
    if name.endswith(".final.logits"):
        return 3.0e-4, 3.0e-4
    if name.endswith(".final.probabilities"):
        return 5.0e-5, 5.0e-5
    return 2.0e-4, 2.0e-4


def load_bridge(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((root / "bridge-vectors.json").read_text("utf-8"))
    raw = np.memmap(root / manifest["data_file"], dtype="<f4", mode="r")
    arrays = {}
    for entry in manifest["arrays"]:
        start = entry["offset_bytes"] // 4
        end = start + entry["element_count"]
        values = np.asarray(raw[start:end], dtype=np.float32)
        actual_hash = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
        if actual_hash != entry["sha256_f32le"]:
            raise RuntimeError(f"BRIDGE vector hash mismatch: {entry['name']}")
        arrays[entry["name"]] = values
    return arrays, manifest


def main() -> None:
    args = parse_args()
    bridge, bridge_manifest = load_bridge(args.bridge)
    official = np.load(args.transformers.with_suffix(".npz"))
    official_report = json.loads(
        args.transformers.with_suffix(".json").read_text("utf-8")
    )

    failures = []
    comparisons = []
    for key in sorted(official.files):
        bridge_key = f"dequant_f32.{key}"
        if bridge_key not in bridge:
            failures.append(f"missing BRIDGE array {bridge_key}")
            continue
        expected = official[key].reshape(-1)
        actual = bridge[bridge_key].reshape(-1)
        if expected.shape != actual.shape:
            failures.append(
                f"shape mismatch {key}: official={expected.shape}, bridge={actual.shape}"
            )
            continue
        atol, rtol = tolerance(key)
        absolute = np.abs(actual - expected)
        denominator = np.maximum(np.abs(expected), np.float32(1.0e-30))
        max_abs = float(absolute.max(initial=0.0))
        max_rel = float((absolute / denominator).max(initial=0.0))
        passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        comparisons.append(
            {
                "name": key,
                "atol": atol,
                "rtol": rtol,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"{key}: max_abs={max_abs:.9g}, max_rel={max_rel:.9g}, "
                f"atol={atol}, rtol={rtol}"
            )

    dequant = next(
        mode for mode in bridge_manifest["modes"] if mode["mode"] == "dequant_f32"
    )
    for step, (bridge_step, official_step) in enumerate(
        zip(dequant["steps"], official_report["steps"], strict=True)
    ):
        if bridge_step["selected_experts"] != official_step["selected_experts"]:
            failures.append(
                f"step {step} routing IDs: bridge={bridge_step['selected_experts']}, "
                f"official={official_step['selected_experts']}"
            )
        if bridge_step["greedy_id"] != official_step["greedy_id"]:
            failures.append(
                f"step {step} greedy ID: bridge={bridge_step['greedy_id']}, "
                f"official={official_step['greedy_id']}"
            )

    report = {
        "format": "lightbridge-hy3-differential-report-v1",
        "comparison_count": len(comparisons),
        "passed": not failures,
        "failures": failures,
        "comparisons": comparisons,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
