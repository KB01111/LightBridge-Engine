#!/usr/bin/env python3
"""Resumably acquire and authenticate LightBridge's pinned Hy3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


REPOSITORY = "satgeze/Hy3-1M-GGUF"
REVISION = "c29be1652dbe5addbca537e3060cbc523d336966"
FILENAME = "hy3-1M-IQ2_M.gguf"
EXPECTED_LENGTH = 96_019_311_104
EXPECTED_SHA256 = "1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the exact selected Hy3 GGUF through huggingface_hub/Xet, "
            "then authenticate its complete length and SHA-256."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory; model data and all Hugging Face/Xet state stay here.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Authenticate an existing destination without contacting Hugging Face.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Discard the Hub client's cached result and download again.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(HASH_CHUNK_BYTES)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as source:
        while read_bytes := source.readinto(buffer):
            digest.update(view[:read_bytes])
    return digest.hexdigest()


def authenticate(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"selected model is missing: {path}")

    actual_length = path.stat().st_size
    if actual_length != EXPECTED_LENGTH:
        raise RuntimeError(
            f"selected model length mismatch: expected {EXPECTED_LENGTH}, got {actual_length}"
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"selected model SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )

    return {
        "format": "lightbridge-selected-model-receipt-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "filename": FILENAME,
        "path": str(path),
        "logical_bytes": actual_length,
        "sha256": actual_sha256,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
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


def disable_ntfs_compression(path: Path) -> None:
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["compact.exe", "/U", "/I", "/Q", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"failed to disable NTFS compression for {path}: {stderr}")


def download(output_dir: Path, force_download: bool) -> Path:
    # huggingface_hub reads these values while importing constants. Keeping every
    # cache on the destination drive prevents a 96 GB transfer from filling C:.
    hub_home = output_dir / ".hf-home"
    os.environ["HF_HOME"] = str(hub_home)
    os.environ["HF_XET_CACHE"] = str(hub_home / "xet")
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=REPOSITORY,
        filename=FILENAME,
        revision=REVISION,
        local_dir=output_dir,
        force_download=force_download,
    )
    return Path(downloaded).resolve()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = (output_dir / FILENAME).resolve()

    if args.verify_only:
        downloaded = target
    else:
        disable_ntfs_compression(output_dir)
        downloaded = download(output_dir, args.force_download)
        if downloaded != target:
            raise RuntimeError(
                f"Hub client returned an unexpected destination: expected {target}, got {downloaded}"
            )
        disable_ntfs_compression(downloaded)

    receipt = authenticate(downloaded)
    receipt["downloader_sha256"] = sha256_file(Path(__file__).resolve())
    if not args.verify_only:
        receipt["huggingface_hub_version"] = importlib.metadata.version("huggingface_hub")
        receipt["hf_xet_version"] = importlib.metadata.version("hf_xet")
    receipt_path = output_dir / f"{FILENAME}.receipt.json"
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"download-selected-model: {error}") from error
