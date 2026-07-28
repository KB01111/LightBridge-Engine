# BRIDGE-LLM Hy3 Engine Architecture

**Date:** 2026-07-27

**Status:** Approved by the user's instruction to continue the supplied engine brief with either
Hy3 option and to proceed without clarification questions.

## Decision

BRIDGE-LLM will be a native, Windows-first Rust inference engine specialized for
`satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf`. The 96,019,311,104-byte, non-MTP checkpoint is the
baseline because it is the smallest recommended Hugging Face Hy3 GGUF, is materially smaller than
the original 239 GB GLM-5.2 target, has native mainline llama.cpp support for differential testing,
and avoids the extra MTP block during correctness bring-up.

The available Hy3-preview MXFP4 artifacts are not the baseline. They are 161-172 GB
MLX/compressed-safetensors distributions rather than directly runnable GGUF files and are a worse
fit for the target laptop.

llama.cpp remains a pinned format and differential-correctness oracle. It is not linked as the
runtime and BRIDGE-LLM is not a wrapper over its graph executor.

## Target Machine and Consequences

The inspected machine has:

- Windows 11 on an AMD Ryzen AI 9 HX 370, 12 cores and 24 logical processors;
- 33,412,722,688 bytes of physical RAM;
- an NVIDIA GeForce RTX 4070 Laptop GPU currently reported as disconnected;
- CUDA toolkit 13.1 and NVCC 13.1;
- approximately 117 GB free on `C:` and 395 GB free on `D:`.

The selected model cannot reside wholly in RAM or VRAM. Direct GGUF execution, bounded compressed
expert caches, and disk-to-CPU cold-expert execution are baseline behavior. CUDA is optional and
must degrade cleanly when the GPU is unavailable. A reduced deterministic Hy3 fixture is used for
fast development; the real model is used for header validation and, after its weights are made
available, release inference acceptance.

## Exact Model Profile

The selected GGUF header was range-fetched and parsed without downloading the weight payload:

- GGUF v3; data begins at byte 5,160,192;
- `general.architecture = hy_v3`;
- 80 transformer blocks and no MTP block in this artifact;
- context 1,048,576 via YaRN factor 4 from an original 262,144-token context;
- hidden size 4096;
- dense FFN width 13,312;
- 64 query heads, 8 KV heads, head/key/value dimension 128;
- one dense layer followed by 79 MoE layers;
- 192 routed experts, exact top-8, and one always-active shared expert;
- routed and shared expert width 1536;
- sigmoid routing with learned selection bias, normalized selected weights, and scale 2.826;
- per-head Q/K RMSNorm, RMS epsilon `1e-5`, and RoPE base 11,158,840;
- vocabulary 120,832 with embedded GPT-2 byte-level BPE metadata;
- BOS 120000, EOS 120025, PAD 120002;
- 1,278 tensors and 96,014,150,912 encoded tensor bytes.

Actual physical tensor types:

| Type | Tensors | Encoded bytes |
|---|---:|---:|
| F32 | 479 | 251,292,928 |
| IQ2_S | 627 | 91,238,285,312 |
| IQ3_S | 91 | 3,995,566,080 |
| Q4_K | 80 | 188,743,680 |
| Q5_K | 1 | 340,262,912 |

The filename `IQ2_M` is a quantization recipe, not a dispatch type. Only the stored type of each
tensor selects a decoder or kernel.

## Model Graph

The obsolete GLM-specific graph is removed from the active architecture. Hy3 uses:

1. token embedding;
2. pre-attention RMSNorm;
3. Q, K, and V projections;
4. per-head Q/K RMSNorm;
5. YaRN-scaled RoPE on Q and K;
6. causal grouped-query attention with a paged GQA KV cache;
7. attention output projection and residual;
8. pre-FFN RMSNorm;
9. either the layer-0 dense SwiGLU FFN or a MoE block;
10. residual;
11. final RMSNorm and untied LM head.

For a MoE block:

1. compute F32 router logits;
2. compute sigmoid scores;
3. add the learned expert bias only for expert selection;
4. select exactly eight experts with deterministic tie-breaking;
5. gather the unbiased sigmoid weights of those experts;
6. normalize their sum and multiply by 2.826;
7. execute each selected expert's gate, up, SiLU-multiply, and down projections;
8. add the always-active, ungated shared expert;
9. accumulate routed outputs in deterministic expert-ID order.

Placement may alter latency but never expert IDs, coefficients, quantization, attention semantics,
or sampling results.

## Workspace Boundaries

The recovered workspace uses focused crates:

- `bridge-core`: checked byte/shape arithmetic, tensor descriptors, hardware facts, errors, arenas;
- `bridge-gguf`: bounded GGUF v2/v3 metadata and tensor-directory parser;
- `bridge-gguf-split`: single-file and numbered split discovery plus global tensor directory;
- `bridge-model-hy3`: metadata resolution, tensor schema, graph configuration, exact routing;
- `bridge-quant-layout`: packed ABI definitions and scalar decoders for required GGML types;
- `bridge-kernels-reference`: allocation-explicit scalar operations and full reference layer;
- `bridge-kernels-cpu`: runtime-selected scalar/AVX2/AVX-512 kernels;
- `bridge-kernels-cuda`: NVCC-built kernels behind a small C ABI, optional at runtime;
- `bridge-io-windows`: bounded positioned reads, IOCP, aligned buffers, cancellation, shutdown;
- `bridge-format` and `bridge-prepare`: lossless expert-major sidecar and manifest validation;
- `bridge-cache`: bounded compressed RAM/VRAM caches and persisted heat;
- `bridge-scheduler`: measured CPU/CUDA/disk completion-time placement;
- `bridge-kv-gqa`: paged grouped-query KV state, placement, save, and restore;
- `bridge-mtp`: optional Hy3 MTP after baseline decode is correct;
- `bridge-tokenizer`: embedded GPT-2 byte-level BPE and Hy3 chat formatting;
- `bridge-runtime`: model/session lifecycle and generation loop;
- `bridge-cli`, `bridge-server`, and `bridge-bench`: user surfaces and measured reporting;
- `bridge-test-model`: deterministic reduced Hy3 GGUF fixtures and differential vectors.

Crates communicate through checked owned descriptors and explicit read/compute requests. No
component receives an unchecked file offset, raw tensor shape, or unbounded channel.

## Storage and Execution

### Direct GGUF

Both single-file and numbered split GGUF sets are supported. Each file has an independent handle.
The global directory records shard, absolute and relative offsets, dimensions, checked strides,
exact type, encoded length, role, layer, and expert range.

For Hy3 routed tensors, expert dimension `ne[2] = 192` is verified. Each expert slab is contiguous:
gate and up are 2,015,232 bytes each; down is either 2,015,232 or 2,703,360 bytes. A cold expert
therefore requires at most three positioned reads and can execute on CPU without an H2D transfer.

### Expert-major Sidecar

`bridge prepare` writes one aligned record per `(layer, expert)` with losslessly copied packed
gate/up/down bytes. Sequential and fused-gate-up layouts are supported. The manifest binds the
sidecar to source file hashes, tensor-directory hash, engine format version, quant ABI version,
and layout. Any mismatch rejects the sidecar.

### Caches and Scheduler

Compressed expert bytes remain compressed in RAM and VRAM. Fixed byte ceilings, in-flight pins,
deduplicated reads, batch union, and hysteretic admission prevent hot-path allocation and churn.
The scheduler compares measured completion times for:

- RAM hit plus CPU;
- RAM hit plus H2D plus CUDA;
- disk read plus CPU;
- disk read plus H2D plus CUDA;
- VRAM hit plus CUDA.

With the inspected hardware, disk-read-plus-CPU is the default cold path. GPU promotion is enabled
only when the GPU is live and measured reuse amortizes transfer and eviction.

## Quantization and Kernels

The baseline supports exactly F32, IQ2_S, IQ3_S, Q4_K, and Q5_K. Every packed decoder has:

- a scalar, bounds-checked reference;
- differential vectors derived from pinned llama.cpp;
- a CPU implementation selected by measured ISA and shape;
- a CUDA implementation where the GPU is available and benchmarking shows a benefit.

The decode hot path never dequantizes a complete matrix. Quant blocks are decoded inside dot or
tile loops. Batch-one decode uses GEMV; prefill groups rows by expert and uses batch union.

## Tokenizer, Chat, and Server

The tokenizer is constructed from embedded GGUF tokens, token types, merges, and special IDs.
Byte-level BPE behavior is differential-tested against the pinned Hugging Face/llama.cpp
tokenizer. The embedded Hy3 template is retained as metadata, while BRIDGE-LLM implements the
known Hy3 message, reasoning-effort, tool-call, and stop-token semantics directly.

`bridge chat` streams decoded text. `bridge serve` exposes health, model information, tokenization,
and OpenAI-compatible chat-completion endpoints with bounded request/session queues.

## Error Handling and Safety

All untrusted counts, offsets, lengths, products, alignments, and ranges use checked arithmetic.
Metadata string/array sizes and tensor rank are bounded before allocation. Tensor descriptors have
a checked constructor and cannot be deserialized into an invalid state.

Unsafe code is restricted to aligned allocation, Windows I/O, SIMD, CUDA FFI, and memory mapping.
Each unsafe module documents its invariants and has boundary tests. A corrupt model, stale
sidecar, unavailable GPU, cancelled read, or exhausted bounded resource produces an actionable
error; none silently changes model quality.

## Verification Strategy

Verification is layered:

1. malformed and synthetic GGUF unit tests;
2. exact real-header inspection of the selected Hy3 GGUF;
3. scalar quantization differential vectors;
4. reduced Hy3 layer and teacher-forced logits;
5. tokenizer and chat-template differential tests;
6. CPU SIMD versus scalar;
7. CUDA versus scalar on available hardware;
8. sidecar/direct and cold/warm equivalence;
9. greedy token equality against llama.cpp on the same full GGUF;
10. reproducible cold/warm benchmark reports with no fabricated numbers.

The current 64 MiB range copy is a sparse header mirror, suitable for real metadata and
tensor-directory validation but not inference. Full-model generation is not accepted until all
96,019,311,104 bytes are available and hash-verified.

## Delivery Slices

1. **Authoritative ingestion:** safe core, native GGUF/split parsing, Hy3 validation, inspect CLI.
2. **Reference math:** required quant decoders, exact routing, dense/GQA/MoE reference layers.
3. **Text generation:** tokenizer, chat formatter, KV cache, complete scalar runtime and sampling.
4. **Storage:** direct expert reads, sidecar, native Windows reader, bounded RAM cache.
5. **Optimized execution:** CPU SIMD, CUDA kernels, mixed scheduler, prefetch and batch union.
6. **Product surfaces:** doctor, plan, prepare, chat, serve, bench, cache and validation commands.
7. **Advanced features:** grouped prefill, MTP, persistent KV, optional experimental iGPU.

Each slice ends in executable, independently testable software. Unsupported later-slice features
are reported as unavailable and are never represented as complete.

## Provenance

Initial pins:

- llama.cpp release `b10153`,
  `b77d646751d01c0962bc203b6809e9d94f7d50b7` (MIT);
- llama.cpp Hy3 merge
  `2969d6d15d67a08e7b83f26164b15350c79c5248` (MIT);
- Tencent Hy3
  `a960ebc3da325ba167f069f76c41eb62c9280d22` (Apache-2.0);
- Transformers 5.6.0
  `3e80155a968c1080f11b2710e8b31741ac5ab0ed` (Apache-2.0);
- Colibri 1.1.1
  `81f08a09e5651ce52616dc720f68810f9021c0be` (Apache-2.0);
- SatGeze GGUF repository
  `c29be1652dbe5addbca537e3060cbc523d336966`;
- selected GGUF SHA-256
  `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7`.

`docs/UPSTREAM.md` records copied/adapted source paths, local hashes, notices, and modifications.
The SatGeze repository lacks a standalone license file, so the exact final-model provenance and
the official Tencent Hy3 Apache-2.0 license are retained together; Hy3-preview's separate custom
license is not treated as permission for this final-model artifact.
