# Hy3 Scalar Reference Math Implementation Plan

> **Execution:** Use the repository's subagent-driven workflow. Every task starts
> with failing tests, ends with fresh verification, receives an independent
> live-file review, and records results under
> `.superpowers/sdd/2026-07-28-hy3-reference-math/`.

**Goal:** Implement an allocation-explicit, scalar Rust correctness oracle for
the selected Hy3 checkpoint's exact packed tensor types, routing, FFN/MoE,
RMSNorm, YaRN RoPE, grouped-query attention, and one reduced complete Hy3 block.

**Boundary:** This slice consumes checked byte slices. It does not open payload
files, stream experts, tokenize text, schedule devices, or claim full-model
inference. The sparse 64 MiB mirror has no usable tensor payload.

**Pinned oracle:** llama.cpp release `b10153`, commit
`b77d646751d01c0962bc203b6809e9d94f7d50b7`.

**Selected physical types:** `F32`, `IQ2_S`, `IQ3_S`, `Q4_K`, `Q5_K`.

---

## Global constraints

- Prefix every project command with `rtk`.
- Use stable Rust and checked arithmetic for every shape, block, row, and byte
  calculation.
- Reject unsupported types and big-endian payload execution explicitly.
- Validate all input/output/scratch lengths before mutating caller buffers.
- Decode one packed block inside the dot loop; never expand a complete matrix.
- No heap allocation after caller-owned reference scratch is constructed.
- Preserve deterministic arithmetic order. Routed experts execute and
  accumulate in ascending expert ID.
- Do not link llama.cpp into the production engine. Oracle tooling is
  test/development-only.
- Record upstream URL, commit, source path, relevant lines, blob hash, license,
  and local modification for every derived table or algorithm.

Two scalar GEMV modes are intentionally distinct:

- `DequantF32` decodes packed weights and dots them with F32 activations. It is
  the high-precision mathematical/official-graph oracle.
- `LlamaQ8K` first quantizes activations to exact `block_q8_K` and invokes the
  pinned per-weight-type Q8_K vec-dot semantics. It is the llama.cpp-compatible
  GGUF execution oracle.

Neither mode may be described as the other.

Operation-specific acceptance tolerances:

| Operation | Acceptance |
|---|---|
| packed block decode fixtures | exact `f32::to_bits()` |
| Q8_K quantization and isolated vec-dot | exact IDs/integers; `atol 1e-6`, `rtol 1e-6` for F32 result |
| dequant-F32 scalar GEMV | `abs <= 1e-6 + 1e-5 * sum_abs_products` |
| routing coefficients | `atol 2e-6`, `rtol 2e-6`; IDs exact |
| RMSNorm and RoPE | `atol 2e-6`, `rtol 2e-6` |
| softmax and attention | `atol 2e-5`, `rtol 2e-5` |
| FFN/MoE and layer residuals | `atol 2e-4`, `rtol 2e-4` |
| final logits | `atol 3e-4`, `rtol 3e-4` |
| next-token probabilities | `atol 5e-5`, `rtol 5e-5`; greedy ID exact |

## Task 1: Establish reproducible quantization oracles and provenance

**Files**

- Create: `vendor/upstream/llama.cpp/LICENSE`
- Create: `vendor/upstream/llama.cpp/PINNED.toml`
- Create: `tools/quant-oracle/README.md`
- Create: `tools/quant-oracle/CMakeLists.txt`
- Create: `tools/quant-oracle/generate-vectors.ps1`
- Create: `tools/quant-oracle/oracle.cpp`
- Create: `crates/bridge-core/tests/quant_provenance.rs`
- Modify: `crates/bridge-core/Cargo.toml`
- Create: `crates/bridge-quant-layout/tests/fixtures/quant-vectors.json`
- Create: exact `.input.bin` and `.output-f32le.bin` fixtures for all five
  selected types
- Create: Q8_K activation-quantization and selected-type vec-dot fixtures
- Modify: `docs/UPSTREAM.md`
- Delete: `crates/bridge-core/src/glm.rs` after confirming it is unreferenced

**TDD and implementation**

1. Add `bridge-core/tests/quant_provenance.rs` first so it fails until the
   pinned source identities and fixture provenance are complete. Task 1 may
   create the future `bridge-quant-layout/tests/fixtures` directory, but must
   not add that crate to the workspace before Task 2.
2. Build the oracle helper only against an external checkout at the exact
   pinned commit. The helper may call upstream scalar dequantizers; production
   crates may not.
   Build a static helper by adding the pinned `ggml` directory with
   `GGML_CPU=OFF`, linking `ggml-base`, and compiling only
   `ggml/src/ggml-cpu/quants.c` into the helper. Do not compile `ggml-cpu.c`,
   link stock `ggml-cpu`, enable shared libraries, or define
   `GGML_CPU_GENERIC`.
   The helper defines `ggml_table_f32_f16[1 << 16]` and initializes every lane
   with upstream `ggml_fp16_to_fp32` before a generic dot call; this is the
   minimal non-dispatch dependency required by the pinned scalar file.
3. Generate at least three full blocks per quantized type:
   structural patterned bits, deterministic LCG bytes with finite scale fields,
   and zero scale. Generate representative F32 values including signed zero.
4. Generate exact `block_q8_K` activation bytes with
   `quantize_row_q8_K_ref` from `ggml/src/ggml-quants.c`, and scalar vec-dot
   outputs with the pinned
   `ggml_vec_dot_{q4_K,q5_K,iq2_s,iq3_s}_q8_K_generic` functions from
   `ggml/src/ggml-cpu/quants.c`. Do not use the normal x86 dispatch entry
   points as the scalar oracle.
5. Store expected output as little-endian F32 bit patterns. Normal tests read
   checked-in fixtures and never require C or network access.
6. Record hashes for the IQ tables and every fixture.
7. Add the llama.cpp MIT license and a machine-readable pin manifest. Do not
   claim provenance for the older untraceable vendored GLM graph subset.
8. Remove the unused GLM-only core source so later contributors cannot mistake
   it for an active model path.

The oracle validates `n > 0`, `n % 256 == 0`, `n <= INT_MAX`, exact byte
lengths, and little-endian execution before calling upstream functions. It
copies raw blocks into aligned typed storage with `memcpy`; it never casts an
unaligned byte buffer to a packed struct. Generic dot calls use `nrc = 1` and
zero strides.

**Verification**

```powershell
rtk cargo test -p bridge-core --lib --tests
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1 -VerifyOnly
```

## Task 2: Implement packed ABI and scalar F32/Q4_K/Q5_K decoding

**Files**

- Create: `crates/bridge-quant-layout/Cargo.toml`
- Create: `crates/bridge-quant-layout/src/lib.rs`
- Create: `crates/bridge-quant-layout/src/error.rs`
- Create: `crates/bridge-quant-layout/src/k_quants.rs`
- Create: `crates/bridge-quant-layout/tests/abi.rs`
- Create: `crates/bridge-quant-layout/tests/oracle_vectors.rs`
- Modify: workspace `Cargo.toml`

**Public API**

```rust
pub struct QuantLayout {
    pub ty: GgmlType,
    pub block_elements: usize,
    pub block_bytes: usize,
}

pub fn layout(ty: GgmlType) -> Result<QuantLayout, QuantError>;
pub fn decode_block_into(
    ty: GgmlType,
    encoded: &[u8],
    output: &mut [f32],
) -> Result<(), QuantError>;
pub fn decode_row_into(
    ty: GgmlType,
    encoded: &[u8],
    logical_elements: usize,
    output: &mut [f32],
) -> Result<(), QuantError>;
```

Expose focused F32, Q4_K, and Q5_K block functions for differential tests.

**Required ABI**

- F32: 1 element / 4 bytes.
- Q4_K: 256 elements / 144 bytes:
  `d[0..2] dmin[2..4] scales[4..16] qs[16..144]`.
- Q5_K: 256 elements / 176 bytes:
  `d[0..2] dmin[2..4] scales[4..16] qh[16..48] qs[48..176]`.

Use byte indexing plus `from_le_bytes`, never `repr(C)` casts. Validate exact
lengths, row divisibility, finite/bit-preserving F16 conversion behavior, and
unchanged output on all validation failures. Test every truncated block length.
Oracle decoder comparisons are F32-bit exact.

**Verification**

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-quant-layout --all-targets -- -D warnings
rtk cargo test -p bridge-quant-layout --all-targets
```

## Task 3: Implement IQ2_S and IQ3_S codebook decoding

**Files**

- Create: `crates/bridge-quant-layout/src/iq2_s.rs`
- Create: `crates/bridge-quant-layout/src/iq3_s.rs`
- Create: `crates/bridge-quant-layout/src/tables.rs`
- Modify: `crates/bridge-quant-layout/src/lib.rs`
- Modify: ABI/oracle tests and fixtures

**Required ABI**

- IQ2_S: 256 elements / 82 bytes:
  `d[0..2] qs[2..66] qh[66..74] scales[74..82]`.
- IQ3_S: 256 elements / 110 bytes:
  `d[0..2] qs[2..66] qh[66..74] signs[74..106] scales[106..110]`.

Transcribe the pinned `kmask_iq2xs[8]`, `iq2s_grid[1024]`, and
`iq3s_grid[512]` with source annotations. Represent grids as integer arrays and
expand lanes through `to_le_bytes` so host endianness cannot alter semantics.
Lock lengths and content hashes in tests. Test codebook boundary indices, signs,
scales, zero scale, patterned blocks, LCG blocks, and every truncated length.
Expected output must match the independent pinned C oracle bit-for-bit.

Also add the internal activation ABI used by the pinned dot kernels:

- Q8_K: 256 elements / 292 bytes:
  `d_f32[0..4] qs_i8[4..260] bsums_i16[260..292]`.

Implement exact pinned `quantize_row_q8_K` rounding, scale, signed-byte values,
and 16 block sums. Add scalar Q4_K×Q8_K, Q5_K×Q8_K, IQ2_S×Q8_K, and
IQ3_S×Q8_K dot functions. These are execution primitives, not new selected
weight types. Lock activation bytes and dot results to Task 1 fixtures.

**Verification**

```powershell
rtk cargo clippy -p bridge-quant-layout --all-targets -- -D warnings
rtk cargo test -p bridge-quant-layout --all-targets
```

## Task 4: Implement exact deterministic Hy3 routing

**Files**

- Create: `crates/bridge-model-hy3/src/routing.rs`
- Create: `crates/bridge-model-hy3/tests/routing.rs`
- Modify: `crates/bridge-model-hy3/src/lib.rs`

**Public API**

```rust
pub struct RouteCandidate {
    pub expert_id: u32,
    pub selection_score: f32,
    pub unbiased_weight: f32,
}

pub struct RoutedExpert {
    pub expert_id: u32,
    pub coefficient: f32,
}

pub fn route_experts_into(
    logits: &[f32],
    selection_bias: &[f32],
    expert_used_count: usize,
    weight_scale: f32,
    candidates: &mut [RouteCandidate],
    selected: &mut [RoutedExpert],
) -> Result<(), Hy3Error>;
```

Semantics:

1. Reject wrong lengths, invalid top-k, and non-finite logits/bias/scale before
   output mutation.
2. Compute unbiased sigmoid in F32.
3. Add learned bias only to the selection score.
4. Select score-descending with expert-ID-ascending ties.
5. Gather unbiased weights.
6. Clamp their sum to `2^-14`.
7. Normalize and multiply by the configured scale (`2.826` selected).
8. Return selected entries sorted by expert ID.

Add `ValidatedHy3Model::tensor_for_role` for later adapters. Test equal-score
ties, selection-only bias, underflow clamp, IDs/coefficient tolerances,
ascending output order, all invalid inputs, and unchanged outputs on error.
Document that deterministic ties are a BRIDGE policy; pinned llama.cpp's CPU
value-only `std::sort` does not define them.

**Verification**

```powershell
rtk cargo clippy -p bridge-model-hy3 --all-targets -- -D warnings
rtk cargo test -p bridge-model-hy3 --all-targets
```

## Task 5: Implement allocation-explicit packed GEMV and FFN/MoE

**Files**

- Create: `crates/bridge-kernels-reference/Cargo.toml`
- Create: `crates/bridge-kernels-reference/src/lib.rs`
- Create: `crates/bridge-kernels-reference/src/error.rs`
- Create: `crates/bridge-kernels-reference/src/matrix.rs`
- Create: `crates/bridge-kernels-reference/src/gemv.rs`
- Create: `crates/bridge-kernels-reference/src/activation.rs`
- Create: `crates/bridge-kernels-reference/src/moe.rs`
- Create: `crates/bridge-kernels-reference/tests/gemv.rs`
- Create: `crates/bridge-kernels-reference/tests/moe.rs`
- Create: `crates/bridge-kernels-reference/tests/allocation.rs`
- Modify: workspace `Cargo.toml`

Implement a non-owning checked encoded tensor view carrying physical type,
payload endianness, exact shape, and bytes. `PackedMatrix<'a>` is constructed
from that view, uses exact GGML `[input_width, output_width]` orientation, and
rejects non-little-endian payloads before interpreting bytes. Validate supported
type, block-aligned input width, checked row/total size, and exact byte-slice
length. Add a byte-swapped/big-endian view rejection test.

Implement:

- `ReferenceExecutionMode::{DequantF32, LlamaQ8K}`;
- `gemv_dequant_f32_into`;
- `gemv_llama_q8k_into`;
- `gemv_accumulate_scaled_into`;
- `swiglu_project_into`;
- `expert_swiglu_accumulate_into`;
- `moe_selected_into`.

Every composite GEMV, SwiGLU, routed/shared expert, MoE, layer, and model API
accepts one explicit `ReferenceExecutionMode` and threads it through every
projection. Mixing modes within one forward call is rejected.

The high-precision mode decodes one weight block into caller scratch inside the
dot loop. The llama-compatible mode quantizes each input row once into
caller-owned Q8_K scratch and calls the exact selected-type Q8_K dot primitive;
it must not dequantize the weight matrix. Accumulate rows in deterministic
input order. Reuse one caller-owned gate/hidden scratch region, split with
`split_at_mut`. Process routed experts in ascending ID, accumulate down
projections directly into the final output, then execute the always-active
shared expert at coefficient `1.0`. Reject out-of-order experts.

Tests cover rectangular F32 orientation, all five physical types in
dequant-F32 mode, all four quantized weight types in llama-Q8_K mode,
initialized scaled destinations, mixed IQ2_S gate/up with IQ3_S down,
shared/routed semantics, the tolerance table above, validation atomicity,
big-endian rejection, and zero allocations after scratch construction.

**Verification**

```powershell
rtk cargo clippy -p bridge-kernels-reference --all-targets -- -D warnings
rtk cargo test -p bridge-kernels-reference --all-targets
```

## Task 6: Implement scalar RMSNorm, residual, softmax, and Hy3 YaRN RoPE

**Files**

- Create: `crates/bridge-kernels-reference/src/norm.rs`
- Create: `crates/bridge-kernels-reference/src/rope.rs`
- Create: `crates/bridge-kernels-reference/src/softmax.rs`
- Create: `crates/bridge-kernels-reference/tests/norm.rs`
- Create: `crates/bridge-kernels-reference/tests/rope.rs`
- Create: `crates/bridge-kernels-reference/tests/softmax.rs`
- Add pinned oracle fixtures with provenance

Implement allocation-explicit:

- weighted RMSNorm with configured epsilon;
- weighted per-head Q/K RMSNorm over 128-element heads;
- residual add;
- causal stable softmax;
- Hy3's NeoX/non-interleaved RoPE through one immutable `Hy3RopeParams` record;
- YaRN factor `4`, original context `262_144`, selected effective context
  `1_048_576`, with explicit base, `freq_scale`, `ext_factor`, `attn_factor`,
  `beta_fast`, and `beta_slow` derived from validated metadata plus pinned
  defaults using exact llama.cpp scaling semantics.

Reject non-finite configuration and invalid dimensions before mutation.
Differential-test ordinary and scaled positions, boundary at original context,
Q/K head shapes, large finite logits, all-masked rows, and in-place aliasing
contracts. Do not approximate YaRN with simple position division.

**Verification**

```powershell
rtk cargo clippy -p bridge-kernels-reference --all-targets -- -D warnings
rtk cargo test -p bridge-kernels-reference --all-targets
```

## Task 7: Implement paged scalar GQA KV state and causal attention

**Files**

- Create: `crates/bridge-kv-gqa/Cargo.toml`
- Create: `crates/bridge-kv-gqa/src/lib.rs`
- Create: `crates/bridge-kv-gqa/src/error.rs`
- Create: `crates/bridge-kv-gqa/src/cache.rs`
- Create: `crates/bridge-kernels-reference/src/attention.rs`
- Create: `crates/bridge-kv-gqa/tests/cache.rs`
- Create: `crates/bridge-kernels-reference/tests/attention.rs`
- Modify: workspace `Cargo.toml`

Implement a bounded, caller-capacity paged F32 reference KV state for reduced
fixtures. The API must have explicit layer/head/token dimensions, checked
append/read ranges, reset, and deterministic page order. It must not allocate
per token after construction.

Implement causal GQA attention for 64 Q / 8 KV mapping generally as
`q_heads / kv_heads`, head dimension 128 for the selected profile, and scale
`1/sqrt(head_dim)`. Use caller-provided score/output scratch. This API consumes
Q and K that the layer integration has already weighted-normalized and
RoPE-transformed; it must not repeat either transform. It appends the already
rotated K plus V to the cache. Test single-token decode, multi-token causal
prefill, head sharing, page boundaries, cache exhaustion, reset, and
differential output against a straightforward dense oracle.

This task proves scalar semantics only; production compressed/offloaded KV
formats remain a later slice.

**Verification**

```powershell
rtk cargo clippy -p bridge-kv-gqa -p bridge-kernels-reference --all-targets -- -D warnings
rtk cargo test -p bridge-kv-gqa -p bridge-kernels-reference --all-targets
```

## Task 8: Build a reduced Hy3 test model and one complete reference block

**Files**

- Create: `crates/bridge-test-model/Cargo.toml`
- Create: `crates/bridge-test-model/src/lib.rs`
- Create: `crates/bridge-test-model/src/hy3.rs`
- Create: deterministic packed weight/vector fixtures
- Create: `tools/hy3-oracle/README.md`
- Create: `tools/hy3-oracle/generate.py`
- Create: `tools/hy3-oracle/llama-oracle.cpp`
- Create: `tools/hy3-oracle/generate-llama-vectors.ps1`
- Create: a pinned requirements/source manifest for Transformers 5.6.0 commit
  `3e80155a968c1080f11b2710e8b31741ac5ab0ed`
- Create: `crates/bridge-kernels-reference/src/layer.rs`
- Create: `crates/bridge-kernels-reference/tests/hy3_layer.rs`
- Create: `crates/bridge-kernels-reference/tests/hy3_logits.rs`
- Modify: workspace `Cargo.toml`
- Modify: `bridge-model-hy3` with a profile-parameterized validation core whose
  selected-profile public wrapper remains unchanged

Create a reduced deterministic Hy3 GGUF configuration that preserves:

- GQA head grouping;
- per-head Q/K normalization;
- YaRN/RoPE semantics;
- one dense block and one MoE block;
- sigmoid router with selection-only bias;
- exact top-k;
- one shared expert;
- mixed selected quant types.

The native parser/split layer must read the reduced GGUF, and a
profile-parameterized validator must authorize it without weakening or
bypassing `validate_selected_model` for the production checkpoint. The reduced
profile has an explicit tensor schema and type policy; private validated-model
construction remains impossible to callers.

`tools/hy3-oracle` is an offline, pinned official-Transformers graph oracle,
not an inference dependency. It loads the deterministic reduced weights (or
their exact dequantized F32 equivalents), captures named intermediates, final
logits, probabilities, and greedy IDs, writes hashes/provenance, and is never
required by normal Rust tests.

The same tool directory also contains a development-only executable built
against the exact pinned llama.cpp checkout. It loads the reduced GGUF through
llama.cpp, evaluates supplied token IDs through the normal graph, and writes
named routing IDs plus final logits/probabilities/greedy IDs for the
llama-Q8_K execution oracle. It records the exact checkout commit, local oracle
source hash, command, and output hashes. Production BRIDGE crates do not link
this tool or llama.cpp.

Implement a complete scalar reference model:

1. token embedding lookup from supplied token IDs;
2. attention weighted RMSNorm;
3. Q/K/V packed projections;
4. weighted per-head Q/K norm and NeoX YaRN RoPE exactly once;
5. causal GQA attention and KV append;
6. attention output plus residual;
7. FFN weighted RMSNorm;
8. dense SwiGLU or exact routed/shared MoE;
9. residual;
10. final weighted RMSNorm;
11. untied LM-head projection;
12. logits, probability distribution, and greedy token.

Check in independent oracle vectors for every intermediate and final residual.
Assert selected IDs exactly and the tolerance table above. Add at least a
two-step teacher-forced token-ID sequence covering dense and MoE blocks; assert
final logits, next-token probabilities, and greedy IDs. Tokenization remains
out of scope because tests supply IDs directly. Add a
zero-allocation-after-construction test.

Run that sequence twice:

- `DequantF32` against the pinned Transformers/dequantized-weight
  intermediates and logits;
- `LlamaQ8K` against the pinned llama.cpp reduced-GGUF routing IDs, logits,
  probabilities, and greedy IDs.

The two modes may produce different floating-point values or routing near a
decision boundary; each must match its own oracle and may never silently switch
to the other.

**Verification**

```powershell
rtk cargo fmt --all -- --check
rtk cargo check --workspace --all-targets
rtk cargo clippy --workspace --all-targets -- -D warnings
rtk cargo test --workspace --all-targets
```

## Task 9: Whole-slice acceptance and independent review

Run all gates fresh. Audit:

- exact selected block sizes and fixture hashes;
- bit-exact decoder vectors;
- routing IDs and coefficients;
- deterministic accumulation;
- no whole-matrix expansion;
- no hot-path allocation;
- both dequant-F32 and llama-Q8_K execution modes are named and tested without
  conflating their outputs;
- payload endianness rejection;
- complete reduced block intermediates;
- reduced native-GGUF/profile validation plus final logits, probabilities, and
  greedy IDs;
- provenance/license records;
- no active GLM-specific model API;
- no claims of real payload or full-model inference.

Request a whole-workspace independent code review. Fix all Critical and
Important findings in one bounded wave, re-run the full gate, and re-review
until clean.

Write:

`.superpowers/sdd/2026-07-28-hy3-reference-math/final-report.md`

The next slice after this plan is tokenizer/chat plus an ingestion-to-storage
adapter and complete scalar token-generation loop.
