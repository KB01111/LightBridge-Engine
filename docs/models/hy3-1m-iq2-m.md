# Hy3 1M IQ2_M model provenance

## Selected artifact

- Repository: <https://huggingface.co/satgeze/Hy3-1M-GGUF>
- Filename: `hy3-1M-IQ2_M.gguf`
- Repository revision: `c29be1652dbe5addbca537e3060cbc523d336966`
- Complete-file logical length: `96_019_311_104` bytes
- Expected complete-file SHA-256:
  `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7`
- GGUF version: 3
- Tensor data offset: `5_160_192`
- Tensor count: 1,278
- Encoded tensor bytes: `96_014_150_912`

The embedded GGUF metadata reports `general.license = apache-2.0`. The
publisher repository currently has no root `LICENSE` file, so redistribution
and legal status must be verified from the complete provenance chain rather
than inferred solely from that metadata field.

This checkpoint belongs to the final
[`tencent/Hy3`](https://huggingface.co/tencent/Hy3) lineage. It is not the
separate Hy3-preview release and the preview license is not used as permission
for this artifact.

## Exact physical-type histogram

| GGML type | Tensor count | Logical elements | Encoded bytes |
|---|---:|---:|---:|
| F32 | 479 | 62,823,232 | 251,292,928 |
| IQ2_S | 627 | 284,841,476,096 | 91,238,285,312 |
| IQ3_S | 91 | 9,298,771,968 | 3,995,566,080 |
| Q4_K | 80 | 335,544,320 | 188,743,680 |
| Q5_K | 1 | 494,927,872 | 340,262,912 |
| **Total** | **1,278** | **295,033,543,488** | **96,014,150,912** |

## Local sparse header mirror

`C:\tmp\hy3-1M-IQ2_M.sparse.gguf` was constructed by copying the first
64 MiB of the remote artifact, marking the local file sparse, and extending
its logical EOF to the exact complete-file length. The copied extent contains
the complete GGUF metadata and tensor directory. The logical region after
that extent is a sparse hole, not model data.

The sparse mirror proves only bounded header, metadata, tensor-directory,
shape, type, offset, and range handling. It cannot prove payload completeness,
the complete-file SHA-256, dequantization correctness, or inference.

Never run a complete-file hash command against the sparse mirror and report
that result as the selected artifact's checksum.

## Complete-file verification

On 2026-07-29 the actual complete artifact was acquired at
`D:\LightBridge\Models\hy3-1M-IQ2_M.gguf`, outside the repository. The
acceptance workflow confirmed:

- logical and allocated length `96_019_311_104` bytes;
- SHA-256
  `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7`;
- `sparse = false`;
- `compressed = false`;
- schema validity and full payload readability.

The complete payload subsequently passed direct and sidecar inference plus
pinned llama.cpp b10153 greedy-token parity. See
[`../full-model-acceptance-2026-07-29.md`](../full-model-acceptance-2026-07-29.md).
The following check applies only to the complete file, never the sparse
mirror:

```powershell
rtk powershell -NoProfile -Command "$model = Get-Item -LiteralPath 'D:\LightBridge\Models\hy3-1M-IQ2_M.gguf'; if ($model.Length -ne 96019311104) { throw \"unexpected length: $($model.Length)\" }; $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model.FullName).Hash.ToLowerInvariant(); if ($hash -ne '1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7') { throw \"unexpected SHA-256: $hash\" }; 'complete model length and SHA-256 verified'"
```
