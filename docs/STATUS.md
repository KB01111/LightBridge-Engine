# LightBridge Engine status

Snapshot date: **2026-07-28**

LightBridge is a pre-alpha native Rust Hy3 engine. This document separates
implemented and verified foundations from planned runtime work so a green
inspection or decoder test is not mistaken for complete inference.

## Current checkpoint

| Area | Status | Evidence and boundary |
|---|---|---|
| Target model | Selected | `hy3-1M-IQ2_M.gguf`, 96,019,311,104 logical bytes, non-MTP baseline |
| GGUF parsing | Implemented | Bounded GGUF v2/v3 metadata and tensor-directory parsing; payload bytes remain isolated |
| Split GGUF handling | Implemented | Strict numbered-shard discovery, common-field checks, and global tensor directory |
| Hy3 validation | Implemented | Exact configuration and 1,278-tensor schema validation for 80 blocks and 192 experts |
| Inspection CLI | Implemented | Stable human and JSON reports for validated model headers |
| Core memory/topology | Implemented | Checked tensor ranges, aligned arenas, and Windows topology discovery |
| Scalar oracle | Implemented and independently reviewed | Pinned llama.cpp b10153 development-only oracle; exact 16-binary fixture inventory; tamper checks |
| F32/Q4_K/Q5_K decoding | Implemented and locally verified | Safe Rust layouts, exact scalar decoding, prevalidation, atomic errors, and bit-exact fixture comparisons |
| IQ2_S/IQ3_S/Q8_K/dot math | Planned | Fixtures and implementation plan exist; Rust execution primitives are not implemented yet |
| Hy3 routing and MoE block | Planned | No complete layer or reduced-model forward pass yet |
| Tokenizer and chat templates | Not implemented | No text-to-token or token-to-text runtime |
| KV cache and attention | Not implemented | No causal generation state |
| Payload I/O and expert storage | Not implemented | No full GGUF payload loading, sidecar, direct I/O, or cache scheduler |
| SIMD/CUDA execution | Not implemented | No accelerated production kernels |
| CLI/server generation | Not implemented | The CLI inspects models only; it cannot generate text |

## Verified behavior

The current public checkpoint passed:

```text
cargo fmt --all -- --check
cargo check -p bridge-quant-layout --all-targets
cargo clippy -p bridge-quant-layout --all-targets -- -D warnings
cargo test -p bridge-quant-layout --all-targets
cargo test -p bridge-quant-layout --release --all-targets
cargo test --workspace --all-targets
```

Results:

- `bridge-quant-layout`: 19 passing tests across 3 suites in debug and release.
- `bridge-core`: 20 passing tests across 2 suites.
- Entire workspace: 152 passing tests across 17 suites.
- The frozen oracle fixture verifier still accepts the exact 16-binary
  inventory and all provenance cross-links.
- Task 1's oracle/provenance review concluded CLEAN with no Critical or
  Important findings.

Task 2's Rust decoder implementation has passed all local gates. A second
independent review is still a release-process checkpoint; local green tests do
not broaden the claim to complete model inference.

## Selected Hy3 profile

The selected model is
[`satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf`](https://huggingface.co/satgeze/Hy3-1M-GGUF/blob/main/hy3-1M-IQ2_M.gguf)
at revision `c29be1652dbe5addbca537e3060cbc523d336966`, with expected SHA-256
`1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7`.

The validated header reports:

- 45 metadata entries;
- 1,278 tensors;
- data offset 5,160,192;
- 80 transformer blocks;
- 192 routed experts with top-8 selection;
- no MTP tensors in this baseline.

The complete model payload has not been downloaded into or committed to this
repository. A sparse local header mirror used during development contains a
payload hole and is not inference evidence.

## Roadmap from this checkpoint

1. Finish IQ2_S and IQ3_S decoding, Q8_K activation quantization, and the four
   selected scalar dot products.
2. Implement deterministic Hy3 routing, packed GEMV, SwiGLU, routed/shared
   experts, normalization, YaRN RoPE, softmax, paged KV state, and causal
   attention.
3. Assemble and verify a reduced Hy3 model with one complete scalar forward
   block.
4. Add tokenizer/chat-template support and a complete scalar token-generation
   loop.
5. Add expert storage, persistent indexes, Windows async I/O, caches, and
   scheduling.
6. Add SIMD and CUDA kernels, mixed CPU/GPU execution, prefill, prefetch, and
   MTP acceleration where a selected model supports it.
7. Complete CLI/server/config/telemetry surfaces and run full-model and
   multi-token acceptance against an authenticated complete GGUF payload.

The detailed plans live under `docs/superpowers/plans/`. They are design and
execution records, not proof that every planned item is complete.

## What this repository must not claim yet

- It is not a usable chat server.
- It does not generate tokens.
- It has not executed the full 96 GB Hy3 model.
- Header validation does not prove payload integrity.
- Scalar reference math does not prove accelerated-kernel parity.
- Automated tests do not replace future full-model runtime acceptance.
