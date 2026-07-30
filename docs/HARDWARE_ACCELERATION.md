# Hardware-tuned acceleration

Snapshot date: **2026-07-30**

This slice applies Deltafin's ownership and capability-gating principles to
LightBridge's Hy3 graph without importing KDA, Metal, MPS, or PyTorch-specific
mechanisms. The target host is an AMD Ryzen AI 9 HX 370 with 32 GiB RAM,
Radeon 890M, Ryzen AI NPU, and an RTX 4070 Laptop GPU with 8,188 MiB VRAM.

No accelerated path may change the authenticated IQ2_S, IQ3_S, Q4_K, or Q5_K
weights. CPU scalar/Q8_K remains the fallback. A candidate is automatic only
after its packed-dot oracle, routes, greedy tokens, repeated determinism, logit
limits, and at least 10% median complete-token improvement all pass.

## Implemented now

| Area | Status | Boundary |
|---|---|---|
| MoE transaction | Default | Each selected and shared expert executes once into an uncommitted candidate; validated output is published atomically. |
| Prompt output head | Default | Intermediate prompt tokens advance hidden/KV state without evaluating the 340 MiB Q5_K output head; only the final prompt position projects logits. |
| Gate/up preparation | Default | Packed gate and up projections quantize their common activation to Q8_K once. |
| Packed-dot validation | Default | `ValidatedQ8KMatrix` checks dimensions, packed scales, the common Q8_K row, and backend availability once before row execution. GEMV validates every row before mutating output, preserving atomic failure without rescanning the common activation per row. |
| Expert buffers | Default | Direct GGUF and sidecar spans read into lazy, 4 KiB-aligned, generation-stamped slots retained by cache leases and poisoned/recycled after eviction. Heterogeneous payload prefixes are charged by full physical slot size so eviction always precedes pool exhaustion. |
| Expert metadata | Default | Session-owned route and lease vectors plus fixed selected-expert storage avoid per-layer descriptor allocation. |
| Windows storage | Tuning candidate | Real overlapped IOCP batches support buffered and `FILE_FLAG_NO_BUFFERING` reads. Unbuffered requests query device, logical-sector, and physical-sector alignment and reject invalid offsets, sizes, or addresses. |
| AVX-VNNI and AVX-512/VNNI | Opt-in | Exact IQ2_S, IQ3_S, Q4_K, and Q5_K packed dots are runtime-dispatched with `--backend cpu-avx-vnni-q8-k` or `cpu-avx512-vnni-q8-k`. Quant and reduced full-route tests are bit-exact. The final bound HX 370 profile measured AVX2 at 0.410 ms, AVX-VNNI at 0.420 ms, and AVX-512/VNNI at 1.095 ms, so AVX2 remains accepted. |
| Grouped prefill | Opt-in | `--prefill-chunk 2`, `4`, or `8` runs the Hy3 graph layer-major, records one authoritative route per position, loads each unique routed expert once per layer, and preserves position/router reduction order. Reduced logits and KV lengths are bit-exact to token-serial execution. |
| T=2 n-gram speculation | Opt-in | `--prefill-chunk 2 --speculative-ngram-t 2` enables greedy-only lossless verification. Full accept, second-token reject/replay, callback interruption, per-position logits, and KV rewind are covered; stops, sampling, and unsupported shapes retain one-token decode. |
| CPU worker placement | Tuned, opt-in | The persistent worker pool accepts an exact logical-CPU assignment. `bridge tune` benchmarks OS placement and one-thread-per-physical-core candidates; `--cpu-set-ids` applies a validated selection. |
| Tuning profile | Implemented | `HardwareFingerprintV1`, `TuningProfileV1`, `BackendKind`, `ExecutionPolicy`, correctness evidence, measurements, rejections, and backend decisions are versioned and serialized. |
| Hardware tools | Implemented | `bridge doctor`, `bridge tune`, and `bridge bench --hardware-profile ... --trace ...` report capability reasons, benchmark candidates, reject drift, and write Chrome/Perfetto JSON atomically. `bench --prompt-corpus ... --corpus-repeats ...` adds bounded, versioned, deterministic multi-prompt reports. |
| CUDA arithmetic oracle | Diagnostic gate passed | A link-free NVRTC/Driver path compiles `compute_89` PTX at runtime, uses page-locked buffers, asynchronous H2D/kernel/D2H work, and proves 7-row by 1,024-element Q4_K, Q5_K, IQ2_S, and IQ3_S GEMV bit-exact against the CPU scalar oracle. |
| Reusable CUDA GEMV | Live primitive passed | A process-owned context validates complete matrices before output mutation, alternates between two grow-on-demand page-locked/device weight arenas, reuses Q8_K/output arenas and codebooks, caps weight staging at 512 MiB, preserves 1.25 GiB free VRAM, and reports host-copy and CUDA-event timing. Single, pair, and bounded batch submission preserve atomic output. |
| CUDA streaming model path | Implemented, opt-in | `--backend cuda-q8-k` sends packed projections through the reusable executor, batches Q/K/V plus routed/shared MoE gate/up and down matrices, keeps F32 routing and deterministic reduction on CPU, and publishes token/KV state transactionally. A CUDA error rewinds the session, permanently demotes that model instance to AVX2, and retries from the committed position. |

`bridge tune` authenticates the selected model and optional sidecar before
measuring CPU worker counts and bound/unbound placement,
scalar/AVX2/AVX-VNNI/AVX-512 mixed-format packed dots, buffered reads, and
buffered/unbuffered IOCP queue depths. These are structural and microbenchmark
measurements; the generated profile leaves new backends disabled until a
full-token qualification supplies the remaining evidence. Microbenchmark-only
worker affinity and storage queue-depth winners are also recorded as rejected
candidates rather than copied into the executable policy.

Example for the prepared D: artifacts:

```powershell
cargo run -p bridge-cli -- tune `
  --model D:\LightBridge\Models\hy3-1M-IQ2_M.gguf `
  --profile performance `
  --output D:\LightBridge\Profiles\hx370-rtx4070-performance.json

cargo run -p bridge-cli -- bench `
  --model D:\LightBridge\Models\hy3-1M-IQ2_M.gguf `
  --backend cuda-q8-k `
  --cpu-threads 12 `
  --prefill-chunk 8 `
  --prompt-corpus tools\hardware\hx370-qualification-corpus.json `
  --corpus-repeats 2 `
  --hardware-profile D:\LightBridge\Profiles\hx370-rtx4070-performance.json `
  --trace D:\LightBridge\Profiles\hx370-cuda-corpus-trace.json
```

Profiles bind the running executable hash, model and sidecar hashes and
lengths, OS/architecture, CPU topology, RAM, power state, device identities,
drivers, PCIe link, CUDA toolkit, MSVC, NVRTC/Driver canary, packed CUDA
oracle, and Vulkan probe. Any drift rejects the profile instead of silently
applying stale settings.

The runtime and CUDA implementation changed after the last local profile was
generated. That profile is intentionally stale under the executable-hash
binding and must be regenerated before it can support a policy decision.

## Live HX 370 CPU qualification result

A release-mode direct-GGUF run authenticated the exact 96,019,311,104-byte
checkpoint and used 24 unbound CPU workers, a 512 MiB expert cache, and an
enforced 4 GiB host-memory reserve. Two consecutive corrected runs completed
with 16 prompt tokens:

| Measurement | Accepted 12-thread baseline | Hardware candidate |
|---|---:|---:|
| Prefill | 181,662 ms | 137,977–138,907 ms |
| Two-token decode | 21,318 ms | 15,485–16,077 ms |
| Total | 202,981 ms | 154,056–154,395 ms |
| Decode throughput | ~0.094 tok/s | 0.124–0.129 tok/s |

The reporting run produced the exact accepted token IDs `[16883, 0]` and text
`Hello!`. Its cache recorded 11,376 misses, zero hits, and 6,775 evictions,
which is direct evidence against treating a large churning expert LRU as the
default hot tier for this prompt.

This is roughly a 24% complete-run reduction, not the required 4x TTFT or
0.5 tok/s result. It is also one prompt/configuration rather than the required
multi-prompt corpus. The 24-worker policy therefore remains explicit and is
not marked automatic.

Follow-up full-token qualification disproved two attractive micro-optimizations:

- One CPU microbenchmark selected 12 workers pinned one per physical core,
  while the final regenerated profile selected one unbound worker. The
  authenticated 12-pinned full run took 180,540 ms versus 160,733 ms for 24
  unbound workers on the same code path. That instability is why the profile
  records micro winners as rejected candidates and retains the accepted
  12-unbound policy until a full-token corpus qualifies a replacement.
- Chunk-8 grouped prefill preserved `[16883, 0]` and `Hello!`, reduced expert
  loads from 11,376 to 6,777 (40.4%), and reduced prefill from 142,493 ms to
  136,733 ms. Total time improved only from 160,733 ms to 158,277 ms (1.5%),
  below the 10% automatic threshold.
- An adaptive Rayon grain that kept the tiny 512x256 tuner probe on one task
  also preserved exact output, but regressed the authenticated 24-worker
  full-model control to 218,091 ms and 0.082 decode tok/s. It was removed;
  unrestricted worker fan-out remains in the accepted CPU path.

An attempted per-row interleaving of gate and up projection regressed the
same-binary serial control to 203,801 ms and a chunk-4 run to 217,119 ms. It
was removed; LightBridge retains one Q8_K activation quantization followed by
two contiguous matrix passes for better locality.

## Live RTX streaming candidate result

The explicit streaming CUDA backend was then measured against the same
authenticated 96,019,311,104-byte model with 12 CPU workers, a 512 MiB expert
cache, an enforced 4 GiB host-memory reserve, strict greedy sampling, two
generated tokens, and chunk-8 prefill:

| Measurement | Matching CPU control | CUDA chunk-8 candidate |
|---|---:|---:|
| Prefill | not separately retained | 96,786 ms |
| Two-token decode | not separately retained | 18,693 ms |
| Total | 160,733 ms | 115,482 ms |
| Decode throughput | not separately retained | 0.107 tok/s |
| Expert loads | 11,376 | 6,783 |
| Expert evictions | 6,771 | 2,178 |

The CUDA candidate produced exact token IDs `[16883, 0]` and `Hello!`, began
and ended as `cuda_streaming_q8_k`, and reported
`cuda_fallback_active=false`. Its complete time is about 28% lower than the
matching CPU control and about 43% lower than the original 202,981 ms
baseline. It therefore clears the 10% threshold for this one prompt only.

This does not make CUDA automatic or authoritative. The required
multi-prompt, repeated-determinism corpus was started but interrupted before
it emitted a report, so it supplies no acceptance evidence. Decode remains
well below the 0.5 tok/s target, and the requested 4x time-to-first-token goal
is not met.

## Accelerator gates

### RTX 4070

`bridge-kernels-cuda` defines a versioned ABI, strict-FP32 policy, `sm_89`
target, compute_89 PTX fallback, double-staging and 1.25 GiB VRAM-reserve
validation. The default workspace remains link-free.

The runtime-loaded NVRTC 12.9 and CUDA Driver path now passes two live RTX
gates without relying on the incompatible native host compiler:

- a 1,024-lane page-locked asynchronous transfer/kernel canary; and
- a packed Q8_K GEMV oracle covering all four authenticated weight formats,
  awkward row count 7, and 1,024 logical elements. NVRTC emitted 41,180 bytes
  of `compute_89` PTX; every output bit matched the scalar CPU oracle.

The same kernel has a reusable process-owned executor with two alternating
bounded staging arenas, persistent codebooks/Q8_K/output buffers, page-locked
copies, explicit VRAM reserve checks, deterministic timing records, and
atomic caller output. With the RTX started and its link reported as Gen4 x8,
the live gate passed eight operations: all four formats, two passes, both
arenas, bit-exact scalar parity, and deterministic output. Batch submission
validates every matrix and output range before any caller-visible mutation.

Two bound model-scale tuner passes measured bit-exact 1,344x4,096
caller-to-caller GEMV medians of 1.347-4.295 ms for Q4_K, 1.957-5.554 ms for
Q5_K, 0.711-1.363 ms for IQ2_S, and 0.673-1.538 ms for IQ3_S. The final
profile records the second pass. The spread after sustained compilation load
is additional evidence that the microbenchmarks are not stable full-token
throughput claims.

The runtime now exposes those kernels as `--backend cuda-q8-k`. Packed
attention Q/K/V projections are submitted as one batch, and every routed plus
shared expert uses one gate/up batch and one down batch per layer. F32 router
math and final ordered reductions stay on CPU. Model open reruns the live
bit-exact/deterministic canary. Any CUDA execution error rolls hidden/KV state
back to the committed position, atomically selects the authoritative AVX2
path for the rest of that model instance, and retries the token or group.
Reduced-model tests cover exact repeated logits plus a forced malformed
expert failure while work is in flight.

The authenticated single-prompt result above proves that the explicit model
path executes end to end, but it is still a streaming design: resident-spine
ownership, per-expert read/transfer overlap, completion-time CPU/GPU splitting,
and decode/prefill-specialized kernels remain open. The interrupted corpus
does not qualify repeated full-model correctness. CUDA therefore reports
`compiled=true`, `authoritative=false`, and `automatic=false` on Windows.

Native CUDA 13.1 still rejects the installed Visual Studio 2026 host compiler.
A future native build should use supported VS 2022 Build Tools; the working
NVRTC path is not used to overstate full-model readiness.

### Radeon 890M

Doctor confirms the 890M through Vulkan, but no packed Vulkan compute backend
is compiled. Host-visible device-local memory, integer-dot, subgroup-size,
timeline-semaphore, strict float-control, route, logit, and token gates remain
open. The 890M is therefore neither authoritative nor automatic.

### Ryzen AI NPU

Doctor confirms the started `NPU Compute Accelerator Device` and emits an
authoritative feasibility record:

- no Hy3 model backend;
- no conversion of the authenticated GGUF weights;
- no W4/BF16 substitution;
- only a future, separately trained and validated advisory next-router
  predictor may use the NPU.

The actual Hy3 router always remains authoritative.

### Grouped prefill and speculation

The runtime accepts only prompt chunks 1, 2, 4, or 8 and only T=2 speculation.
Grouped execution is layer-major: each position gets an exact route record,
the per-layer expert union is loaded once, and outputs are reduced in original
position/router order. The reduced model matches token-serial logits and KV
lengths bit-for-bit.

T=2 speculation is restricted to greedy decoding with no stop-token set. It
uses only the existing token history for an n-gram draft, executes two
positions through the grouped path, and keeps the Hy3 router authoritative.
Acceptance commits both positions; a second-token mismatch rewinds to the
first committed position and replays the authoritative token. Callback
interruption also removes the unobserved second position. No KV page is cloned.

Both paths remain explicit options until an authenticated full-model corpus
proves prompt/decode throughput, deterministic routes and tokens, numerical
limits, and the 10% automatic-selection threshold.

## Evidence still required

The structural changes and reduced oracles do not establish the requested
4x time-to-first-token or 0.5 token/s goals. Qualification still requires:

1. a completed authenticated full-model multi-prompt corpus for AVX2, the
   explicit CUDA backend, opt-in AVX-VNNI/AVX-512 paths, grouped chunks, and
   T=2 speculation;
2. exact route and greedy-ID comparison, deterministic repeats, and logit
   bounds;
3. cold, admission, warm, and long-session memory measurements with at least
   4 GiB host RAM free;
4. complete-token medians proving the 10% automatic threshold;
5. resident-spine CUDA ownership, asynchronous read/transfer overlap, and
   completion-time CPU/GPU split evidence before CUDA can become automatic,
   plus a Vulkan implementation before the iGPU can enter qualification.

Until those gates pass, the accepted CPU path remains authoritative.

## Design sources

- [Deltafin optimization design](https://github.com/gavamedia/deltafin/blob/main/OPTIMIZATIONS.md)
- [Windows file-buffering requirements](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [Windows I/O completion ports](https://learn.microsoft.com/en-us/windows/win32/fileio/i-o-completion-ports)
- [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [CUDA mapped-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Ryzen AI software](https://ryzenai.docs.amd.com/en/latest/)
- [Ryzen AI NPU operator support](https://ryzenai.docs.amd.com/en/latest/ops_support.html)
