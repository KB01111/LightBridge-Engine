# LightBridge Engine status

Snapshot date: **2026-07-29**

The CPU baseline is implemented and accepted end to end. This document
separates shipped runtime behavior, locally verified acceptance, and
hardware/model-inapplicable optional acceleration.

## Current checkpoint

| Area | Status | Evidence and boundary |
|---|---|---|
| Target model | Selected and authenticated | `hy3-1M-IQ2_M.gguf`, 96,019,311,104 bytes, pinned SHA-256, non-MTP |
| GGUF parsing | Implemented | Bounded GGUF v2/v3 parsing, typed metadata, checked ranges, endian handling, and payload isolation during inspection |
| Split GGUF handling | Implemented | Strict numbered-shard discovery, common-field checks, and global tensor directory |
| Hy3 validation | Implemented | Exact configuration and 1,278-tensor schema validation for 80 blocks and 192 experts |
| Payload authentication | Implemented | Product loading checks physical completeness, exact 96,019,311,104-byte length, and pinned SHA-256 before execution |
| Quantization | Implemented and oracle-verified | F32, IQ2_S, IQ3_S, Q4_K, Q5_K, Q8_K, four exact packed dot paths, malformed-scale and atomic-output coverage |
| CPU execution | Implemented | Scalar diagnostics plus bounded row-parallel AVX2 integer dots selected at runtime; mixed F32 routers execute correctly |
| Hy3 forward path | Implemented and reduced-model verified | Exact routing, dense/SwiGLU, routed/shared MoE, RMSNorm, YaRN RoPE, GQA attention, teacher-forced logits |
| Tokenizer/chat protocol | Implemented and differential-tested | Embedded GPT-2 BPE, exact Hy3 formatting, incremental decode, reasoning, tools, stop tokens, and up to four stop strings |
| KV and generation | Implemented | Lazy million-token-capable paged KV, deterministic sampling, causal generation, model-bound persistent sessions, transactional rollback |
| Payload I/O and expert storage | Implemented | Windows positioned reads, sparse allocation detection, direct GGUF experts, verified sidecars, parallel route prefetch |
| Caching | Implemented | Fixed byte ceilings, pins, deduplicated concurrent loads, hysteretic admission, LRU eviction, atomically persisted heat |
| Resource safety | Implemented | Conservative startup memory preflight, checked allocation/arithmetic, bounded requests, cancellation between tokens, typed errors |
| CLI | Implemented | `inspect-gguf`, `doctor`, `plan`, `validate`, `prepare`, `tokenize`, `detokenize`, `chat`, `serve`, `bench`, `cache` |
| HTTP server | Implemented and route-tested | Health/model/tokenization plus OpenAI-compatible JSON/SSE chat completions, usage chunks, tools/reasoning, body/concurrency bounds |
| CUDA | Unavailable on selected host | No live NVIDIA device; the CPU path remains functional and CUDA is never advertised |
| Grouped prefill | Explicitly unavailable | Decode and prompt prefill are exact token-serial operations; no accelerated grouped-prefill claim |
| MTP | Not applicable | The selected checkpoint contains no MTP block |
| Experimental iGPU | Explicitly unavailable | No experimental placement is advertised |
| Full-model acceptance | Passed | Exact payload authentication, direct and sidecar generation, llama.cpp b10153 greedy parity, and bounded in-process benchmark completed on 2026-07-29 |

## Local verification

The 2026-07-29 checkpoint passes:

```text
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p bridge-quant-layout --release --all-targets
cargo build --workspace
cargo doc --workspace --no-deps
python -m unittest discover -s tools\release-acceptance -p "test_*.py" -v
powershell -File tools\hy3-oracle\generate-llama-vectors.ps1 -VerifyOnly
git diff --check
```

Current automated evidence:

- **287 tests across 59 suites** pass for the full workspace.
- **29 release-mode quantization tests** and **10 acceptance-tool unit tests**
  pass.
- Frozen llama.cpp b10153 quantization fixtures remain hash-bound and
  tamper-checked.
- Reduced two-block Hy3 teacher-forced hidden states, routes, logits,
  probabilities, and greedy IDs match the pinned oracle.
- Scalar and AVX2/parallel packed execution are bit-identical across all four
  selected quantized weight types.
- The real selected header validates 45 metadata values, 1,278 tensors, all
  executable types, the F32 router boundary, tokenizer metadata, and expert
  slab layout.
- Direct-GGUF and prepared-sidecar expert execution produce identical logits;
  cold/warm cache behavior and persisted heat round-trip.
- Corrupt routed weights fail with typed errors and rewind every KV layer;
  retry starts from a clean committed position.
- Persistent chat snapshots restore history, logits, model-bound KV, and
  cached-prefix continuation; wrong bindings, corruption, bounds, and shape
  mismatches are rejected atomically.
- A real reduced GGUF drives health, tokenize, detokenize, non-streaming
  completion, SSE completion, stop filtering, structured usage, and request
  limit routes.
- The authenticated 96,019,311,104-byte selected payload generated
  `Hello!` as token IDs `[16883, 0]` through both direct-GGUF and verified
  sidecar execution.
- Pinned llama.cpp b10153 produced the same two greedy IDs with margins
  `2.40599442` and `5.21518898`.
- One authenticated engine produced the same IDs in cold, admission, and warm
  phases while enforcing the 2 GiB expert-cache ceiling.

The reduced-model suite proves fast isolated boundaries; the separately
authenticated full-model run closes the selected-payload execution and
independent greedy-parity gate.

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

The complete model payload was downloaded to
`D:\LightBridge\Models\hy3-1M-IQ2_M.gguf`, outside this repository. Payload
validation confirmed the exact length and SHA-256, a non-sparse allocation,
and no NTFS compression. Model weights remain uncommitted. A separate sparse
local header mirror used during development still contains a payload hole and
is metadata-only evidence.

## Full-model release acceptance

The bounded acceptance workflow completed all selected-checkpoint gates:

1. authenticated the exact complete, non-sparse GGUF;
2. generated `[16883, 0]` (`Hello!`) through direct GGUF;
3. prepared and authenticated a 15,168-record lossless expert sidecar, then
   generated the same IDs and text through it;
4. matched both greedy IDs against pinned llama.cpp b10153;
5. emitted identical IDs from cold, admission, and warm phases in one process
   with measured cache counters and timings.

All five checks are automated by
`tools/release-acceptance/run_full_model_acceptance.py`. The resumable
`download-selected-model.py` keeps Hub/Xet state on the selected destination
drive and publishes a receipt only after the exact length and SHA-256 pass.
The passing `acceptance.json` has SHA-256
`95e7a890b4c2d1a2619f5743fee22730531047d8fdf9d13daf6452f4f95581a2`;
the runner cannot emit a passing report from the sparse mirror. Exact timings,
cache behavior, sidecar hashes, and executable/oracle provenance are recorded
in [the acceptance summary](full-model-acceptance-2026-07-29.md).

Optional future acceleration does not block the CPU baseline:

- CUDA kernels and CPU/GPU scheduling can be enabled and parity-tested if an
  NVIDIA device becomes live.
- Grouped multi-token prefill may improve prompt throughput; current prefill
  is correct and token-serial.
- MTP would require selecting a checkpoint that actually contains an MTP
  block.
- Experimental iGPU placement remains outside the supported baseline.

## Claim boundary

- LightBridge is a usable CPU chat CLI/server for the pinned checkpoint when
  the authenticated complete payload is supplied.
- The current machine executed and independently parity-checked the exact
  authenticated 96 GB checkpoint on 2026-07-29.
- The sparse mirror remains header/schema evidence only and cannot be confused
  with the authenticated payload.
- The measured cold/admission/warm timings are CPU-baseline observations on
  this host, not generalized performance claims.
- The 2 GiB cache remained bounded and resident, but the admission and warm
  phases recorded zero cross-run hits for this route sequence. The evidence
  therefore proves persistence, eviction bounds, and output equality—not a
  claimed warm-cache speedup.
