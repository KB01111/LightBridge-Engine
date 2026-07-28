# LightBridge Engine

LightBridge is a native Rust inference engine being built for the Hy3 mixture-
of-experts architecture. The project is deliberately targeting the smaller
Hy3-1M GGUF release instead of the original GLM-5.2 target so development and
local verification can iterate against a substantially smaller model.

> **Project status: pre-alpha.** Authoritative GGUF ingestion and the first
> bit-exact scalar quantization decoders are implemented. Tokenization, a
> complete forward pass, token generation, accelerated kernels, and full-model
> inference are not implemented yet.

The current non-MTP baseline is
[`satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf`](https://huggingface.co/satgeze/Hy3-1M-GGUF/blob/main/hy3-1M-IQ2_M.gguf).
Model weights are not distributed in this repository.

## What works today

- Bounded GGUF v2/v3 metadata and tensor-directory parsing.
- Strict discovery and validation of single-file and numbered-shard GGUF
  models.
- An authoritative Hy3 configuration and tensor-schema validator.
- Exact validation of the selected 80-block, 192-expert, top-8 Hy3 profile.
- Deterministic human-readable and JSON `inspect-gguf` reports.
- Checked aligned arenas and Windows host-topology discovery.
- A pinned llama.cpp b10153 scalar oracle with hash-bound offline fixtures.
- Safe Rust packed layouts and bit-exact scalar decoding for F32, Q4_K, and
  Q5_K, including adversarial length, scale, atomicity, and overflow tests.

The ingestion CLI does **not** read or execute tensor payload bytes. A sparse
metadata mirror can validate the header and tensor directory, but it is not a
model and cannot be used for inference.

## Selected model

| Property | Value |
|---|---|
| Repository | `satgeze/Hy3-1M-GGUF` |
| File | `hy3-1M-IQ2_M.gguf` |
| Logical size | `96,019,311,104` bytes |
| SHA-256 | `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7` |
| Blocks / experts / selected experts | `80 / 192 / 8` |
| MTP | Not present in the selected baseline |

See [the model profile](docs/models/hy3-1m-iq2-m.md) for the exact metadata and
tensor inventory.

## Build and test

Rust 1.82 or newer is required.

```powershell
cargo build --workspace
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

At the 2026-07-28 public checkpoint, the workspace test suite contains 152
passing tests across 17 suites. The scalar decoder suite also passes in a
release build.

## Inspect a GGUF model

Render the stable human-readable report:

```powershell
cargo run -p bridge-cli -- inspect-gguf --model C:\path\to\hy3-1M-IQ2_M.gguf
```

Render JSON with no stdout chatter:

```powershell
cargo run -p bridge-cli -- inspect-gguf --model C:\path\to\hy3-1M-IQ2_M.gguf --json
```

## Workspace

- `bridge-core`: GGML type descriptors, checked tensor ranges, arenas, and
  topology.
- `bridge-gguf`: bounded GGUF parsing.
- `bridge-gguf-split`: strict shard discovery and global tensor directories.
- `bridge-model-hy3`: Hy3 metadata and complete tensor-schema validation.
- `bridge-quant-layout`: packed scalar reference decoding.
- `bridge-cli`: deterministic inspection commands and reports.

## Documentation

- [Current implementation status and roadmap](docs/STATUS.md)
- [Hy3 engine architecture](docs/superpowers/specs/2026-07-27-hy3-engine-architecture-design.md)
- [Authoritative ingestion design](docs/superpowers/specs/2026-07-27-hy3-ingestion-design.md)
- [Reference-math implementation plan](docs/superpowers/plans/2026-07-28-hy3-reference-math.md)
- [Upstream source and license provenance](docs/UPSTREAM.md)

## License

LightBridge Engine is released under the [MIT License](LICENSE). Retained
third-party reference files keep their own upstream notices and provenance.
