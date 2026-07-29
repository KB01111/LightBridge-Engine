# Full-model release acceptance

These tools close the checks that cannot be represented by the sparse 64 MiB
header mirror. They are deliberately bound to the selected artifact and pinned
llama.cpp release; they are not generic model download or benchmark wrappers.

## Acquire the selected checkpoint

The downloader pins the repository revision, places both Hugging Face and Xet
state under the destination directory, resumes interrupted transfers, and
publishes a receipt only after verifying the exact 96,019,311,104-byte length
and SHA-256. On Windows it also disables inherited NTFS compression on the
destination and final model so quantized payload reads do not spend CPU in the
filesystem decompressor.

```powershell
rtk python tools\release-acceptance\download-selected-model.py `
  --output-dir D:\LightBridge\Models
```

The final model path is
`D:\LightBridge\Models\hy3-1M-IQ2_M.gguf`. A partial download remains under
the destination's `.cache` tree and is never exposed as an authenticated
checkpoint.

## Run acceptance

Use a fresh output directory on a drive with room for the lossless expert
sidecar. On Windows the runner marks that directory uncompressed before
creating the sidecar. It builds the current release executable and the independently
pinned llama.cpp oracle, then performs:

1. host capability, RAM, and sidecar-disk admission, followed by non-sparse
   storage, schema, exact length, and SHA-256 validation;
2. deterministic direct-GGUF generation as an early executable gate;
3. verified lossless sidecar preparation and sidecar generation;
4. exact generated-token equality between the direct and sidecar paths;
5. exact greedy-token equality with llama.cpp b10153 over the same numeric
   prompt sequence;
6. measured cold, admission, and repeated warm-state sidecar runs in one
   authenticated process, including resident-cache counters and exact output
   equality;
7. a hash-bound `acceptance.json` report with every executed command.

```powershell
rtk python tools\release-acceptance\run_full_model_acceptance.py `
  --model D:\LightBridge\Models\hy3-1M-IQ2_M.gguf `
  --output-dir D:\LightBridge\Acceptance
```

If sidecar preparation completed but a later stage was interrupted, rerun with
`--resume-sidecar`. Replacing an existing sidecar requires the explicit
`--overwrite-sidecar` flag.

## Tool tests

```powershell
rtk python -m unittest discover `
  -s tools\release-acceptance `
  -p "test_*.py" `
  -v
```
