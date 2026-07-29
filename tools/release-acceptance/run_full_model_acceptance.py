#!/usr/bin/env python3
"""Run every full-checkpoint LightBridge release acceptance gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


EXPECTED_LENGTH = 96_019_311_104
EXPECTED_SHA256 = "1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7"
LLAMA_COMMIT = "b77d646751d01c0962bc203b6809e9d94f7d50b7"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bridge",
        type=Path,
        default=workspace / "target" / "release" / "bridge.exe",
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=Path(r"C:\tmp\lightbridge-llama-b10153"),
    )
    parser.add_argument(
        "--llama-oracle",
        type=Path,
        default=(
            workspace
            / "target"
            / "hy3-llama-oracle-nmake"
            / "bin"
            / "bridge-hy3-full-model-oracle.exe"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Hi",
    )
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--cache-mib", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--memory-headroom-mib", type=int, default=512)
    parser.add_argument(
        "--resume-sidecar",
        action="store_true",
        help="Reuse an already completed sidecar and prepare report in the output directory.",
    )
    parser.add_argument(
        "--overwrite-sidecar",
        action="store_true",
        help="Explicitly permit bridge prepare to replace completed sidecar outputs.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use already-built release and oracle executables.",
    )
    return parser.parse_args()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(HASH_CHUNK_BYTES)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as source:
        while read_bytes := source.readinto(buffer):
            digest.update(view[:read_bytes])
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CAPTURE_BYTES:
        raise RuntimeError(f"JSON report exceeds {MAX_CAPTURE_BYTES} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON report root must be an object: {path}")
    return value


class StageRunner:
    def __init__(self, workspace: Path, output_dir: Path) -> None:
        self.workspace = workspace
        self.log_dir = output_dir / "logs"
        self.commands: list[dict[str, Any]] = []

    def run(self, stage: str, arguments: list[str]) -> bytes:
        if not arguments or arguments[0].lower() != "rtk":
            raise RuntimeError(f"stage {stage} must execute through rtk")
        print(f"[{stage}] {' '.join(arguments)}", flush=True)
        started = datetime.now(timezone.utc)
        completed = subprocess.run(
            arguments,
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        ended = datetime.now(timezone.utc)
        if len(completed.stdout) > MAX_CAPTURE_BYTES or len(completed.stderr) > MAX_CAPTURE_BYTES:
            raise RuntimeError(f"stage {stage} exceeded the bounded output capture")
        atomic_write(self.log_dir / f"{stage}.stdout.log", completed.stdout)
        atomic_write(self.log_dir / f"{stage}.stderr.log", completed.stderr)
        self.commands.append(
            {
                "stage": stage,
                "arguments": arguments,
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
                "exit_code": completed.returncode,
            }
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            tail = stderr[-4000:]
            raise RuntimeError(f"stage {stage} failed with exit code {completed.returncode}: {tail}")
        return completed.stdout

    def run_json(self, stage: str, arguments: list[str], output_path: Path) -> dict[str, Any]:
        stdout = self.run(stage, arguments)
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"stage {stage} did not emit one JSON value: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"stage {stage} JSON root must be an object")
        write_json(output_path, value)
        return value


def assert_authenticated_payload(report: dict[str, Any]) -> None:
    if report.get("schema_valid") is not True:
        raise RuntimeError("payload validation did not prove the selected schema")
    files = report.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise RuntimeError("payload validation must report exactly one selected file")
    file = files[0]
    if file.get("logical_bytes") != EXPECTED_LENGTH:
        raise RuntimeError("payload validation length mismatch")
    if file.get("sha256") != EXPECTED_SHA256:
        raise RuntimeError("payload validation SHA-256 mismatch")
    if file.get("sparse") is not False:
        raise RuntimeError("payload validation did not prove a non-sparse file")
    allocated = file.get("allocated_bytes")
    compressed = file.get("compressed")
    if not isinstance(allocated, int) or allocated <= 0:
        raise RuntimeError("payload validation reported an invalid physical allocation")
    if compressed not in (True, False):
        raise RuntimeError("payload validation omitted the compression state")
    if compressed is False and allocated < EXPECTED_LENGTH:
        raise RuntimeError("uncompressed payload validation did not prove complete allocation")


def common_runtime_args(args: argparse.Namespace, bridge: Path) -> list[str]:
    return [
        "rtk",
        str(bridge),
        "chat",
        "--model",
        str(args.model),
        "--context",
        str(args.context),
        "--cache-mib",
        str(args.cache_mib),
        "--backend",
        "cpu-q8-k",
        "--cpu-threads",
        str(args.cpu_threads),
        "--memory-headroom-mib",
        str(args.memory_headroom_mib),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--repetition-penalty",
        "1",
        "--seed",
        "0",
        "--prompt",
        args.prompt,
        "--json",
    ]


def common_bench_args(args: argparse.Namespace, bridge: Path) -> list[str]:
    return [
        "rtk",
        str(bridge),
        "bench",
        "--model",
        str(args.model),
        "--context",
        str(args.context),
        "--cache-mib",
        str(args.cache_mib),
        "--backend",
        "cpu-q8-k",
        "--cpu-threads",
        str(args.cpu_threads),
        "--memory-headroom-mib",
        str(args.memory_headroom_mib),
        "--sidecar-data",
        str(args.output_dir / "hy3-experts.bridge"),
        "--sidecar-manifest",
        str(args.output_dir / "hy3-experts.manifest.json"),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--repetition-penalty",
        "1",
        "--seed",
        "0",
        "--prompt",
        args.prompt,
        "--cold-warm",
        "--json",
    ]


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise RuntimeError("--max-tokens must be at least one")
    if args.context < 1:
        raise RuntimeError("--context must be at least one")
    if args.cache_mib < 1:
        raise RuntimeError("--cache-mib must be at least one")
    if args.cpu_threads < 0:
        raise RuntimeError("--cpu-threads cannot be negative")
    if args.resume_sidecar and args.overwrite_sidecar:
        raise RuntimeError("--resume-sidecar and --overwrite-sidecar are mutually exclusive")

    workspace = workspace_root()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    bridge = args.bridge.resolve()
    llama_source = args.llama_source.resolve()
    llama_oracle = args.llama_oracle.resolve()
    if not args.model.is_file():
        raise RuntimeError(f"selected model is missing: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = args.output_dir / "acceptance.json"
    if final_report_path.exists():
        raise RuntimeError(
            f"a completed acceptance report already exists; use a fresh output directory: "
            f"{final_report_path}"
        )
    runner = StageRunner(workspace, args.output_dir)

    try:
        if os.name == "nt":
            runner.run(
                "disable-output-compression",
                ["rtk", "compact.exe", "/U", "/I", "/Q", str(args.output_dir)],
            )
        if not args.skip_build:
            runner.run(
                "build-release",
                ["rtk", "cargo", "build", "--release", "-p", "bridge-cli"],
            )
            runner.run(
                "build-oracle",
                [
                    "rtk",
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(workspace / "tools" / "hy3-oracle" / "generate-llama-vectors.ps1"),
                    "-LlamaCppSource",
                    str(llama_source),
                ],
            )
        if not bridge.is_file():
            raise RuntimeError(f"release executable is missing: {bridge}")
        if not llama_oracle.is_file():
            raise RuntimeError(f"pinned llama.cpp full-model oracle is missing: {llama_oracle}")

        doctor = runner.run_json(
            "doctor",
            ["rtk", str(bridge), "doctor", "--json"],
            args.output_dir / "doctor.json",
        )
        plan = runner.run_json(
            "plan",
            [
                "rtk",
                str(bridge),
                "plan",
                "--model",
                str(args.model),
                "--context",
                str(args.context),
                "--cache-mib",
                str(args.cache_mib),
                "--memory-headroom-mib",
                str(args.memory_headroom_mib),
                "--json",
            ],
            args.output_dir / "plan.json",
        )
        if plan.get("memory_preflight_passes") is not True:
            raise RuntimeError("selected runtime plan does not pass the physical-memory preflight")
        routed_payload_bytes = plan.get("routed_expert_payload_bytes")
        if not isinstance(routed_payload_bytes, int) or routed_payload_bytes <= 0:
            raise RuntimeError("selected runtime plan omitted routed expert payload bytes")
        disk_free_bytes = shutil.disk_usage(args.output_dir).free
        sidecar_reserve_bytes = routed_payload_bytes + 256 * 1024 * 1024
        disk_required_bytes = 0 if args.resume_sidecar else sidecar_reserve_bytes
        disk_preflight = {
            "free_bytes": disk_free_bytes,
            "required_bytes": disk_required_bytes,
            "sidecar_reserve_bytes": sidecar_reserve_bytes,
            "passes": disk_free_bytes >= disk_required_bytes,
        }
        write_json(args.output_dir / "disk-preflight.json", disk_preflight)
        if disk_preflight["passes"] is not True:
            raise RuntimeError(
                f"sidecar disk preflight needs {disk_required_bytes} bytes, "
                f"only {disk_free_bytes} bytes are free"
            )
        validation = runner.run_json(
            "validate-payload",
            ["rtk", str(bridge), "validate", "--model", str(args.model), "--payload", "--json"],
            args.output_dir / "payload-validation.json",
        )
        assert_authenticated_payload(validation)

        direct_path = args.output_dir / "chat-direct.json"
        direct = runner.run_json(
            "chat-direct",
            common_runtime_args(args, bridge),
            direct_path,
        )

        sidecar_data = args.output_dir / "hy3-experts.bridge"
        sidecar_manifest = args.output_dir / "hy3-experts.manifest.json"
        prepare_report_path = args.output_dir / "prepare.json"
        if args.resume_sidecar:
            if not sidecar_data.is_file() or not sidecar_manifest.is_file() or not prepare_report_path.is_file():
                raise RuntimeError("--resume-sidecar requires completed data, manifest, and prepare report")
            prepare = read_json(prepare_report_path)
        else:
            prepare_args = [
                "rtk",
                str(bridge),
                "prepare",
                "--model",
                str(args.model),
                "--output",
                str(sidecar_data),
                "--manifest",
                str(sidecar_manifest),
                "--json",
            ]
            if args.overwrite_sidecar:
                prepare_args.append("--overwrite")
            prepare = runner.run_json("prepare-sidecar", prepare_args, prepare_report_path)

        sidecar_path = args.output_dir / "chat-sidecar.json"
        sidecar_args = common_runtime_args(args, bridge)
        sidecar_args.extend(
            [
                "--sidecar-data",
                str(sidecar_data),
                "--sidecar-manifest",
                str(sidecar_manifest),
            ]
        )
        sidecar = runner.run_json("chat-sidecar", sidecar_args, sidecar_path)

        prompt_ids = direct.get("prompt_token_ids")
        generated_ids = direct.get("token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(generated_ids, list):
            raise RuntimeError("direct chat report omitted token IDs")
        oracle_inputs = prompt_ids + generated_ids[:-1]
        if not oracle_inputs or len(oracle_inputs) > 2048:
            raise RuntimeError("oracle input sequence must contain between 1 and 2048 tokens")
        oracle_threads = args.cpu_threads or max(1, (os.cpu_count() or 1) // 2)
        llama_path = args.output_dir / "llama-full-model.json"
        runner.run(
            "llama-parity",
            [
                "rtk",
                str(llama_oracle),
                str(args.model),
                str(llama_path),
                str(oracle_threads),
                *[str(token_id) for token_id in oracle_inputs],
            ],
        )
        parity_path = args.output_dir / "parity.json"
        parity = runner.run_json(
            "verify-parity",
            [
                "rtk",
                "python",
                str(workspace / "tools" / "release-acceptance" / "verify_full_model.py"),
                "--direct",
                str(direct_path),
                "--sidecar",
                str(sidecar_path),
                "--llama",
                str(llama_path),
                "--output",
                str(parity_path),
            ],
            args.output_dir / "parity-verifier-stdout.json",
        )
        parity = read_json(parity_path)

        benchmark = runner.run_json(
            "bench-cold-warm",
            common_bench_args(args, bridge),
            args.output_dir / "bench-cold-warm.json",
        )
        if benchmark.get("mode") != "cold_admission_warm":
            raise RuntimeError("cold/warm benchmark omitted its execution mode")
        warm_report = benchmark.get("warm")
        if not isinstance(warm_report, dict):
            raise RuntimeError("cold/warm benchmark omitted the warm-state report")
        warm_cache_before = warm_report.get("cache_before")
        if (
            not isinstance(warm_cache_before, dict)
            or not isinstance(warm_cache_before.get("resident_entries"), int)
            or warm_cache_before["resident_entries"] <= 0
        ):
            raise RuntimeError("warm-state benchmark began without resident expert entries")

        final_report = {
            "format": "lightbridge-full-model-acceptance-v1",
            "passed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": {
                "path": str(args.model),
                "logical_bytes": EXPECTED_LENGTH,
                "sha256": EXPECTED_SHA256,
            },
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "doctor": doctor,
            "plan": plan,
            "disk_preflight": disk_preflight,
            "payload_validation": validation,
            "prepare": prepare,
            "direct": direct,
            "sidecar": sidecar,
            "parity": parity,
            "bench_cold": benchmark.get("cold"),
            "bench_admission": benchmark.get("admission"),
            "bench_warm": benchmark.get("warm"),
            "benchmark": benchmark,
            "provenance": {
                "llama_repository": "https://github.com/ggml-org/llama.cpp.git",
                "llama_release": "b10153",
                "llama_commit": LLAMA_COMMIT,
                "bridge_executable_sha256": sha256_file(bridge),
                "llama_oracle_executable_sha256": sha256_file(llama_oracle),
                "llama_oracle_source_sha256": sha256_file(
                    workspace / "tools" / "hy3-oracle" / "full-model-oracle.cpp"
                ),
                "acceptance_runner_sha256": sha256_file(Path(__file__).resolve()),
                "parity_verifier_sha256": sha256_file(
                    workspace / "tools" / "release-acceptance" / "verify_full_model.py"
                ),
            },
            "commands": runner.commands,
        }
        write_json(final_report_path, final_report)
        write_json(args.output_dir / "commands.json", runner.commands)
        print(json.dumps({"passed": True, "report": str(final_report_path)}))
        return 0
    except Exception:
        write_json(args.output_dir / "commands.json", runner.commands)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"run-full-model-acceptance: {error}") from error
