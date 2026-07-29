#!/usr/bin/env python3
"""Verify direct/sidecar output and pinned llama.cpp greedy-token parity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


LLAMA_COMMIT = "b77d646751d01c0962bc203b6809e9d94f7d50b7"
MAX_REPORT_BYTES = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_REPORT_BYTES:
        raise RuntimeError(f"report exceeds {MAX_REPORT_BYTES} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"report root must be an object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def integer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise RuntimeError(f"{name} must be an array of non-negative integers")
    return value


def verify(
    direct: dict[str, Any],
    sidecar: dict[str, Any],
    llama: dict[str, Any],
) -> dict[str, Any]:
    direct_prompt = integer_list(direct.get("prompt_token_ids"), "direct.prompt_token_ids")
    sidecar_prompt = integer_list(sidecar.get("prompt_token_ids"), "sidecar.prompt_token_ids")
    direct_generated = integer_list(direct.get("token_ids"), "direct.token_ids")
    sidecar_generated = integer_list(sidecar.get("token_ids"), "sidecar.token_ids")
    if not direct_prompt:
        raise RuntimeError("the acceptance prompt must contain at least one token")
    if not direct_generated:
        raise RuntimeError("the acceptance run must generate at least one token")
    if direct_prompt != sidecar_prompt:
        raise RuntimeError("direct and sidecar prompt token IDs differ")
    if direct_generated != sidecar_generated:
        raise RuntimeError("direct and sidecar generated token IDs differ")
    if direct.get("raw_text") != sidecar.get("raw_text"):
        raise RuntimeError("direct and sidecar raw decoded text differs")
    if llama.get("format") != "lightbridge-llama-full-model-oracle-v1":
        raise RuntimeError("llama.cpp report format mismatch")
    if llama.get("llama_commit") != LLAMA_COMMIT:
        raise RuntimeError("llama.cpp report commit mismatch")

    steps = llama.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("llama.cpp steps must be an array")
    expected_inputs = direct_prompt + direct_generated[:-1]
    if len(steps) != len(expected_inputs):
        raise RuntimeError(
            f"llama.cpp step count mismatch: expected {len(expected_inputs)}, got {len(steps)}"
        )

    oracle_inputs: list[int] = []
    oracle_greedy: list[int] = []
    oracle_margins: list[float] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RuntimeError(f"llama.cpp step {index} must be an object")
        position = step.get("position")
        input_id = step.get("input_id")
        greedy_id = step.get("greedy_id")
        margin = step.get("margin")
        if position != index:
            raise RuntimeError(f"llama.cpp step {index} position mismatch")
        if not isinstance(input_id, int) or isinstance(input_id, bool) or input_id < 0:
            raise RuntimeError(f"llama.cpp step {index} has an invalid input ID")
        if not isinstance(greedy_id, int) or isinstance(greedy_id, bool) or greedy_id < 0:
            raise RuntimeError(f"llama.cpp step {index} has an invalid greedy ID")
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin < 0:
            raise RuntimeError(f"llama.cpp step {index} has an invalid greedy margin")
        oracle_inputs.append(input_id)
        oracle_greedy.append(greedy_id)
        oracle_margins.append(float(margin))

    if oracle_inputs != expected_inputs:
        raise RuntimeError("llama.cpp evaluated token IDs differ from the LightBridge sequence")
    prediction_start = len(direct_prompt) - 1
    predictions = oracle_greedy[prediction_start:]
    if predictions != direct_generated:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(predictions, direct_generated))
                if actual != expected
            ),
            min(len(predictions), len(direct_generated)),
        )
        raise RuntimeError(
            "greedy token mismatch at generated position "
            f"{mismatch}: LightBridge={direct_generated[mismatch]}, "
            f"llama.cpp={predictions[mismatch]}"
        )

    prediction_margins = oracle_margins[prediction_start:]
    return {
        "format": "lightbridge-full-model-parity-v1",
        "passed": True,
        "llama_commit": LLAMA_COMMIT,
        "prompt_token_ids": direct_prompt,
        "generated_token_ids": direct_generated,
        "llama_greedy_token_ids": predictions,
        "llama_greedy_margins": prediction_margins,
        "direct_raw_text": direct.get("raw_text"),
        "sidecar_raw_text": sidecar.get("raw_text"),
        "direct_total_milliseconds": direct.get("total_milliseconds"),
        "sidecar_total_milliseconds": sidecar.get("total_milliseconds"),
    }


def main() -> int:
    args = parse_args()
    report = verify(read_json(args.direct), read_json(args.sidecar), read_json(args.llama))
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"verify-full-model: {error}") from error
