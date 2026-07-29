# LightBridge Engine

LightBridge is a Windows-first native Rust inference engine for the Hy3
mixture-of-experts architecture. It targets the smaller Hy3-1M GGUF release
instead of the original GLM-5.2 target and executes routed experts directly
from disk under bounded RAM.

> **Project status: CPU baseline complete and full-model accepted.** The
> selected-model loader, full Hy3 forward path, tokenizer, generation loop,
> AVX2/parallel CPU backend, storage/cache layer, CLI, and OpenAI-compatible
> server are implemented. On 2026-07-29 the exact authenticated 96 GB payload
> completed direct and sidecar generation, matched pinned llama.cpp greedy
> tokens, and passed the bounded cold/admission/warm benchmark workflow.

The current non-MTP baseline is
[`satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf`](https://huggingface.co/satgeze/Hy3-1M-GGUF/blob/main/hy3-1M-IQ2_M.gguf).
Model weights are not distributed in this repository.

## What works today

- Bounded GGUF v2/v3 metadata and tensor-directory parsing.
- Strict discovery and validation of single-file and numbered-shard GGUF
  models.
- Exact validation of the selected 80-block, 192-expert, top-8, non-MTP Hy3
  schema and every executable physical type.
- Bit-exact F32, IQ2_S, IQ3_S, Q4_K, Q5_K, and Q8_K reference math against
  hash-bound llama.cpp oracle fixtures.
- Dense and routed/shared MoE blocks, exact routing, RMSNorm, SwiGLU, YaRN
  RoPE, grouped-query causal attention, lazy paged KV, sampling, and
  autoregressive generation.
- Embedded GPT-2 byte-level BPE, Hy3 chat formatting, reasoning modes,
  tool-call parsing, incremental decoding, token stops, and text stops.
- Authenticated selected-payload loading with early sparse-file refusal,
  conservative memory admission, direct expert reads, lossless expert-major
  sidecars, parallel route prefetch, and bounded deduplicating caches.
- Runtime-selected AVX2 integer dot kernels with bounded CPU row parallelism;
  scalar modes remain available for differential diagnosis.
- Model-bound, checksummed, size-bounded persistent chat/KV sessions and
  atomically persisted expert-cache heat.
- Product commands for inspect, doctor, plan, validate, prepare, tokenize,
  detokenize, chat, serve, bench, and cache management.
- A bounded OpenAI-compatible Chat Completions server with JSON/SSE responses,
  structured reasoning/tool calls, stop strings, usage chunks, concurrency
  limits, and request-body limits.

A sparse metadata mirror may validate the header and tensor directory, but it
is not a complete model. Inference commands reject it before hashing or
allocating model weights.

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
cargo fmt --all -- --check
cargo test -p bridge-quant-layout --release --all-targets
```

At the 2026-07-29 checkpoint, the workspace suite contains **287 passing tests
across 59 suites**. This includes malformed-input boundaries, pinned
quantization and reduced-logit oracles, scalar/AVX2 parity, direct/sidecar
equality, cache and session persistence, transactional failure rollback, and
real reduced-model HTTP routes.

## Use the engine

Inspect the selected schema without reading tensor payloads:

```powershell
cargo run -p bridge-cli -- inspect-gguf --model C:\path\to\hy3-1M-IQ2_M.gguf
```

Check the host and build a deterministic memory/storage plan:

```powershell
cargo run -p bridge-cli -- doctor
cargo run -p bridge-cli -- plan --model C:\path\to\hy3-1M-IQ2_M.gguf --json
```

Authenticate the complete payload, then chat with the default bounded AVX2
CPU backend:

```powershell
cargo run -p bridge-cli -- validate --model C:\path\to\hy3-1M-IQ2_M.gguf --payload
cargo run -p bridge-cli -- chat --model C:\path\to\hy3-1M-IQ2_M.gguf --prompt "Hello"
```

The resumable selected-model acquisition and one-command full release
acceptance workflow are documented in
[`tools/release-acceptance/README.md`](tools/release-acceptance/README.md).
It authenticates the complete artifact, proves direct/sidecar generated-token
equality, compares exact greedy token IDs with pinned llama.cpp b10153, and
records measured in-process cold/admission/warm-state benchmark reports.
The completed selected-checkpoint evidence is summarized in
[`docs/full-model-acceptance-2026-07-29.md`](docs/full-model-acceptance-2026-07-29.md).

Persist cache heat and a resumable, model-bound chat session:

```powershell
cargo run -p bridge-cli -- chat `
  --model C:\path\to\hy3-1M-IQ2_M.gguf `
  --chat-json C:\path\to\conversation.json `
  --cache-heat C:\path\to\expert-heat.json `
  --session-out C:\path\to\conversation.lbgs
```

Run the local API:

```powershell
cargo run -p bridge-cli -- serve `
  --model C:\path\to\hy3-1M-IQ2_M.gguf `
  --bind 127.0.0.1:8080
```

## Workspace

- `bridge-core`: GGML type descriptors, checked tensor ranges, arenas, and
  topology.
- `bridge-gguf`: bounded GGUF parsing.
- `bridge-gguf-split`: strict shard discovery and global tensor directories.
- `bridge-model-hy3`: Hy3 metadata and complete tensor-schema validation.
- `bridge-quant-layout`: exact packed decoders, Q8_K activation quantization,
  and scalar/AVX2 dot products.
- `bridge-kernels-reference`: complete allocation-explicit Hy3 reference
  kernels.
- `bridge-kernels-cpu`: bounded CPU pool, ISA detection, and runtime dispatch.
- `bridge-kv-gqa`: lazy paged grouped-query KV plus persistent snapshots.
- `bridge-tokenizer`: embedded tokenizer, chat templates, incremental decode,
  reasoning, and tool-call protocol.
- `bridge-io-windows`: cancellable bounded positioned reads and physical sparse
  storage inspection.
- `bridge-format`, `bridge-prepare`, and `bridge-cache`: bound expert sidecars,
  preparation, and compressed expert caching.
- `bridge-runtime`: authenticated model/session lifecycle and generation.
- `bridge-server`: bounded OpenAI-compatible HTTP/SSE service.
- `bridge-cli`: all user-facing commands and deterministic reports.
- `bridge-test-model`: deterministic reduced Hy3 model and GGUF fixtures.

## Explicit capability gates

- CUDA is disabled when no live NVIDIA device is available; CPU execution is
  fully functional and is the selected machine's production path.
- Grouped multi-token prefill and experimental iGPU placement are reported
  unavailable rather than silently falling back under an accelerated label.
- MTP is not applicable because the selected checkpoint has no MTP block.
- Full-model inference acceptance passed with the exact pinned length and
  SHA-256. The separate sparse header mirror remains metadata-only and is
  rejected by inference commands.

## Documentation

- [Current implementation status and roadmap](docs/STATUS.md)
- [Hy3 engine architecture](docs/superpowers/specs/2026-07-27-hy3-engine-architecture-design.md)
- [Authoritative ingestion design](docs/superpowers/specs/2026-07-27-hy3-ingestion-design.md)
- [Reference-math implementation plan](docs/superpowers/plans/2026-07-28-hy3-reference-math.md)
- [Upstream source and license provenance](docs/UPSTREAM.md)

## License

LightBridge Engine is released under the [MIT License](LICENSE). Retained
third-party reference files keep their own upstream notices and provenance.
