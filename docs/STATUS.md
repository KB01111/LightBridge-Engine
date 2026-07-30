# LightBridge Engine status

Snapshot date: **2026-07-30**

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
| Quantization | Implemented and oracle-verified | F32, IQ2_S, IQ3_S, Q4_K, Q5_K, Q8_K, four exact packed dot paths, malformed-scale and atomic-output coverage; validated matrix handles eliminate repeated common-row checks without weakening all-row-before-write atomicity |
| CPU execution | Implemented | Scalar diagnostics plus bounded row-parallel AVX2 integer dots selected at runtime; opt-in AVX-VNNI and AVX-512/VNNI dots are bit-exact on all selected formats and the reduced full route. Live mixed-format timing retains AVX2 as the accepted path |
| Hy3 forward path | Implemented and reduced-model verified | Exact routing, dense/SwiGLU, routed/shared MoE, RMSNorm, YaRN RoPE, GQA attention, teacher-forced logits |
| Tokenizer/chat protocol | Implemented and differential-tested | Embedded GPT-2 BPE, exact Hy3 formatting, incremental decode, reasoning, tools, stop tokens, and up to four stop strings |
| KV and generation | Implemented | Lazy million-token-capable paged KV, deterministic sampling, causal generation, model-bound persistent sessions, transactional rollback |
| Payload I/O and expert storage | Implemented | Direct GGUF/sidecar reads into lazy aligned generation-stamped cache slots, poison/recycle lifetime tests, parallel route prefetch, and tuner-only buffered/unbuffered IOCP batches |
| Caching | Implemented | Fixed byte ceilings, pins, deduplicated concurrent loads, hysteretic admission, LRU eviction, atomically persisted heat |
| Resource safety | Implemented | Conservative startup memory preflight, checked allocation/arithmetic, bounded requests, cancellation between tokens, typed errors |
| Hardware tuning | Implemented with qualification gates | Versioned hardware/artifact fingerprints, execution policies, bound/unbound persistent CPU worker tuning, 10% backend decisions, storage tuning, profile drift rejection, and Chrome/Perfetto aggregate spans |
| CLI | Implemented | `inspect-gguf`, `doctor`, `plan`, `validate`, `prepare`, `tokenize`, `detokenize`, `chat`, `serve`, `tune`, `bench`, `cache` |
| HTTP server | Implemented and route-tested | Health/model/tokenization plus OpenAI-compatible JSON/SSE chat completions, usage chunks, tools/reasoning, body/concurrency bounds |
| CUDA | Explicit streaming model backend implemented, opt-in | `--backend cuda-q8-k` uses runtime-compiled strict-FP32 packed kernels, reusable pinned double staging, batched Q/K/V and MoE projections, and ordered CPU reduction. Reduced full-route output is exact and deterministic; a forced CUDA failure rewinds every KV layer and retries AVX2 atomically. One authenticated full-model prompt preserved `[16883, 0]` and `Hello!`. Resident-spine ownership, asynchronous expert overlap, mixed scheduling, and the multi-prompt qualification corpus remain open |
| Grouped prefill and T=2 speculation | Implemented, opt-in | Layer-major chunks 2/4/8 use position route unions and one expert load per layer union; reduced logits/KV are exact. CPU-only chunk 8 improved complete time only 1.5%, while the explicit CUDA chunk-8 candidate completed the matching single prompt in 115,482 ms versus about 160,733 ms for its CPU control. Greedy T=2 supports accept, reject/replay, callback rewind, and per-position logits; neither path is automatic without corpus evidence |
| MTP | Not applicable | The selected checkpoint contains no MTP block |
| Experimental iGPU | Runtime detected, execution unavailable | Vulkan reports the Radeon 890M; no host-visible packed compute backend is compiled or advertised |
| Ryzen AI NPU | Feasibility reported, execution unavailable | Started AMD NPU is detected; authenticated GGUF conversion is prohibited and only a future advisory router predictor is in scope |
| Full-model acceptance | Passed | Exact payload authentication, direct and sidecar generation, llama.cpp b10153 greedy parity, and bounded in-process benchmark completed on 2026-07-29. CPU tuning preserved `[16883, 0]` and `Hello!` at 154,056-154,395 ms. The later explicit CUDA chunk-8 candidate also preserved them at 115,482 ms, but remains non-authoritative pending its deterministic multi-prompt corpus |

## Local verification

The 2026-07-30 checkpoint passes:

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

- **326 tests across 63 suites** pass for the full workspace.
- **34 release-mode quantization tests** and **10 acceptance-tool unit tests**
  pass.
- Frozen llama.cpp b10153 quantization fixtures remain hash-bound and
  tamper-checked.
- Reduced two-block Hy3 teacher-forced hidden states, routes, logits,
  probabilities, and greedy IDs match the pinned oracle.
- Scalar, AVX2/parallel, AVX-VNNI, and AVX-512/VNNI packed execution are
  bit-identical across all four selected quantized weight types; both VNNI
  reduced full-route logits are bit-identical to scalar Q8_K.
- The live RTX 4070 runtime compiler/Driver gate emits `compute_89` PTX,
  completes page-locked asynchronous H2D/kernel/D2H work, and matches the CPU
  scalar oracle bit-for-bit for 7x1024 GEMV in Q4_K, Q5_K, IQ2_S, and IQ3_S.
- The reusable CUDA executor preserves atomic caller output, passes two
  deterministic oracle passes across both staging arenas, and passes
  1,344x4,096 per-format tuning probes while retaining 1.25 GiB free VRAM.
- The explicit streaming CUDA backend batches Q/K/V, routed/shared gate/up,
  and down projections; its reduced-model full route is bit-exact and
  deterministic. Injected malformed expert data proves output atomicity,
  complete KV rewind, backend demotion, and a clean AVX2 retry.
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

Optional acceleration does not block the CPU baseline:

- The explicit CUDA streaming backend is executable and fail-closed. On the
  authenticated model, 12 CPU workers, a 512 MiB expert cache, chunk 8, and a
  4 GiB host reserve produced exact IDs `[16883, 0]` in 115,482 ms: 96,786 ms
  prefill and 18,693 ms decode. This is about 28% faster than the matching
  160,733 ms CPU control and about 43% faster than the original 202,981 ms
  baseline, but the interrupted multi-prompt repeat run is not acceptance
  evidence. CUDA remains explicit and non-authoritative.
- The measured CPU/storage tranche improved the accepted prompt by roughly
  24%, but 0.124–0.129 tok/s remains below the 0.5 tok/s target. Its zero cache
  hits and 6,775 evictions support replacing the churning 2 GiB default only
  after a multi-prompt cache-size qualification.
- Grouped multi-token prefill remains opt-in: chunk 8 reduced live expert loads
  from 11,376 to 6,777 and prefill from 142,493 ms to 136,733 ms, but total
  time improved only from 160,733 ms to 158,277 ms. Greedy T=2 n-gram
  verification still requires its full-model corpus.
- Any tuning profile produced before the current runtime/CUDA build is stale
  by construction because profiles bind the executable hash; regenerate it
  before relying on a policy decision.
- MTP would require selecting a checkpoint that actually contains an MTP
  block.
- Vulkan iGPU execution and an advisory NPU router remain research-only.

See [the hardware acceleration status](HARDWARE_ACCELERATION.md) for the exact
implemented/gated boundary and host-specific commands.

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
