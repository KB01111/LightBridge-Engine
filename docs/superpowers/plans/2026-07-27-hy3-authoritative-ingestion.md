# Hy3 Authoritative Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task, with a fresh implementation worker and a review pass for every task.

**Goal:** Turn the partial checkout into a safe, compiling Rust workspace whose `bridge inspect-gguf` command natively parses, indexes, validates, and reports the selected Hy3 checkpoint without reading tensor payloads.

**Architecture:** `bridge-core` owns checked tensor arithmetic and the GGML type ABI. `bridge-gguf` parses one bounded GGUF header. `bridge-gguf-split` discovers files and creates a globally checked tensor directory. `bridge-model-hy3` resolves the `hy_v3` graph and validates tensor schemas. `bridge-cli` turns those structures into one deterministic text/JSON inspection report. Dependencies point inward only: CLI -> Hy3/split -> GGUF -> core.

**Tech stack:** Rust 2021/MSRV 1.82, `thiserror`, `serde`, `serde_json`, `clap`, standard `Read + Seek`, Windows and Unix filesystem APIs used only through `std`.

## Global constraints

- Prefix every project command with `rtk`.
- Add a failing test before each behavior change, then make only that test pass, then refactor.
- Use checked integer operations at every file-controlled arithmetic boundary.
- Do not map, allocate, or read a tensor payload during inspection.
- Do not expose unchecked public constructors for tensor descriptors or byte ranges.
- Do not keep GLM-5.2, MLA, or DSA concepts in the active Hy3 dependency graph.
- Do not add placeholder inference, server, tokenizer, CUDA, or cache commands.
- The selected baseline is `satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf`, non-MTP:
  - logical file size `96_019_311_104`;
  - data offset `5_160_192`;
  - tensor count `1_278`;
  - tensor payload bytes `96_014_150_912`;
  - architecture `hy_v3`.
- Because this checkout has no Git metadata, each worker must report an exact changed-file list and commands run. Reviewers inspect the live files and do not claim commit hashes or diffs.

---

## Task 1: Restore a buildable, safe core

**Files:**

- Modify: `Cargo.toml`
- Modify: `crates/bridge-core/Cargo.toml`
- Create: `crates/bridge-core/src/lib.rs`
- Modify: `crates/bridge-core/src/arena.rs`
- Modify: `crates/bridge-core/src/error.rs`
- Modify: `crates/bridge-core/src/ggml_type.rs`
- Modify: `crates/bridge-core/src/tensor.rs`
- Modify: `crates/bridge-core/src/sys.rs`
- Leave unexported: `crates/bridge-core/src/glm.rs`
- Test inside: `crates/bridge-core/src/arena.rs`
- Test inside: `crates/bridge-core/src/ggml_type.rs`
- Test inside: `crates/bridge-core/src/tensor.rs`

### Step 1: Make the workspace describe only crates that exist in this slice

Replace the root member list and internal dependencies with:

```toml
[workspace]
resolver = "2"
members = [
    "crates/bridge-core",
    "crates/bridge-gguf",
    "crates/bridge-gguf-split",
    "crates/bridge-model-hy3",
    "crates/bridge-cli",
]
```

Keep third-party workspace dependencies needed by those crates. Remove active path dependencies for nonexistent GLM/MLA/DSA crates. Future slices add their crates when their implementations exist.

### Step 2: Add red tests for the GGML ABI and row strides

Tests must assert:

```rust
assert_eq!(GgmlType::Q1_0.block_size(), 128);
assert_eq!(GgmlType::Q1_0.type_size(), 18);
assert_eq!(GgmlType::Q2_0.block_size(), 64);
assert_eq!(GgmlType::Q2_0.type_size(), 18);
assert_eq!(GgmlType::Nvfp4.block_size(), 64);
assert_eq!(GgmlType::Nvfp4.type_size(), 36);

let strides = compute_strides([256, 3, 1, 1], 2, GgmlType::Q4K)?;
assert_eq!(strides[0], GgmlType::Q4K.type_size() as u64);
assert_eq!(strides[1], GgmlType::Q4K.row_size(256)?);
```

Also assert that a shape whose total element count is block-aligned but whose first dimension is not is rejected.

Run:

```powershell
rtk cargo test -p bridge-core ggml_type
rtk cargo test -p bridge-core tensor
```

Expected: failure before implementation.

### Step 3: Correct the type table and checked tensor arithmetic

Expose from `lib.rs`:

```rust
pub mod arena;
pub mod error;
pub mod ggml_type;
pub mod quantkey;
pub mod sys;
pub mod tensor;
```

Do not export `glm`.

Implement:

```rust
impl GgmlType {
    pub fn from_discriminant(value: u32) -> Result<Self, CoreError>;
    pub const fn discriminant(self) -> u32;
    pub const fn block_size(self) -> u64;
    pub const fn type_size(self) -> u64;
    pub fn row_size(self, ne0: u64) -> Result<u64, CoreError>;
}

pub fn compute_strides(
    ne: [u64; 4],
    n_dims: u32,
    ty: GgmlType,
) -> Result<[u64; 4], CoreError>;
```

Use GGML semantics:

- `nb[0] = type_size`;
- `nb[1] = type_size * (ne[0] / block_size)`;
- `nb[i] = nb[i - 1] * ne[i - 1]` for higher dimensions;
- `ne[0]` must be divisible by the block size;
- every multiply and conversion is checked.

Give `TensorDesc` private fields and one checked constructor:

```rust
pub fn new(
    name: impl Into<String>,
    shape: &[u64],
    ty: GgmlType,
    relative_offset: u64,
) -> Result<Self, CoreError>;
```

Add getters, `element_count`, `row_bytes`, `encoded_bytes`, and `checked_absolute_range(data_offset, file_len)`.

### Step 4: Add red tests for allocation overflow and alignment

Tests must prove:

- `alloc_f32(usize::MAX)` returns `None` and never wraps;
- default arena allocations up to 64-byte alignment are aligned;
- a request above the arena's base alignment is rejected;
- `AlignedBuffer::new(bytes, 4096)` returns a 4096-aligned allocation;
- invalid size/alignment returns an error instead of panicking.

Run:

```powershell
rtk cargo test -p bridge-core arena
```

Expected: failure before implementation.

### Step 5: Make arena construction and allocation fallible

Use `Layout::from_size_align` and checked `n.checked_mul(size_of::<f32>())`. Record the backing allocation's alignment in `Arena`. `alloc_bytes` may only satisfy alignments no greater than that recorded alignment.

Remove safe methods that return mutable slices from shared references. A mutable allocation requires `&mut self`; an explicitly unsafe raw allocation must document aliasing and lifetime requirements.

### Step 6: Verify the core

Run:

```powershell
rtk cargo fmt --all
rtk cargo check -p bridge-core --all-targets
rtk cargo clippy -p bridge-core --all-targets -- -D warnings
rtk cargo test -p bridge-core --all-targets
```

Expected: all green, no GLM module in rustdoc or the public API.

---

## Task 2: Build the bounded native GGUF reader

**Files:**

- Create: `crates/bridge-gguf/Cargo.toml`
- Create: `crates/bridge-gguf/src/lib.rs`
- Create: `crates/bridge-gguf/src/error.rs`
- Create: `crates/bridge-gguf/src/value.rs`
- Create: `crates/bridge-gguf/src/reader.rs`
- Create: `crates/bridge-gguf/src/testing.rs` behind `cfg(test)` or a `test-utils` feature
- Create: `crates/bridge-gguf/tests/reader.rs`

### Step 1: Define the public format model

Implement these owned structures:

```rust
pub enum Endianness { Little, Big }

pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), F32(f32), Bool(bool),
    String(String), Array(GgufArray),
    U64(u64), I64(i64), F64(f64),
}

pub struct GgufArray {
    pub element_type: GgufValueType,
    pub values: Vec<GgufValue>,
}

pub struct GgufFile {
    pub version: u32,
    pub endianness: Endianness,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorDesc>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_len: u64,
}
```

Provide typed metadata getters that distinguish missing keys from wrong stored types.

### Step 2: Write the malformed-input tests first

Add fixture builders that write bytes through normal Rust I/O. Cover:

- minimal valid v3;
- minimal valid v2;
- bad and byte-swapped magic;
- unsupported version;
- truncated scalar, string, array, and tensor record;
- excessive metadata/tensor/string/array counts;
- invalid UTF-8;
- invalid boolean byte;
- unknown metadata value type;
- unknown GGML type;
- dimension count zero and above the GGML maximum of four;
- row block mismatch;
- data-offset and tensor-range overflow;
- payload range beyond physical length.

Run:

```powershell
rtk cargo test -p bridge-gguf --test reader
```

Expected: compile or assertion failures before the parser exists.

### Step 3: Implement a limit-accounting reader

Expose:

```rust
pub struct ReaderLimits {
    pub max_dimensions: u32,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_tensors: u64,
    pub max_metadata_entries: u64,
    pub max_metadata_bytes: u64,
}

impl Default for ReaderLimits { /* ingestion design defaults */ }

pub struct GgufReader<R> { /* reader, limits, accounting */ }

impl<R: Read + Seek> GgufReader<R> {
    pub fn new(reader: R) -> Self;
    pub fn with_limits(reader: R, limits: ReaderLimits) -> Self;
    pub fn read(self) -> Result<GgufFile, GgufError>;
}

pub fn open(path: impl AsRef<Path>) -> Result<GgufFile, GgufError>;
```

Requirements:

- `max_dimensions` defaults to the GGML ABI maximum of 4 and cannot be configured above 4;
- GGUF v2/v3 count and string lengths are `u64`;
- all values honor detected endianness;
- array element types cannot recursively be arrays;
- every allocation is preceded by a checked limit and `usize` conversion;
- duplicate metadata keys are rejected;
- the default alignment is 32 and `general.alignment` must be a positive power of two;
- `data_offset = align_up(position_after_tensor_directory, alignment)`;
- reading stops at `data_offset`;
- tensor descriptor ranges are validated against `file_len` without seeking into them.

### Step 4: Verify parser isolation

Add a custom `Read + Seek` test double that fails any read whose starting position is at or beyond `data_offset`. A valid parse must succeed through that reader.

Run:

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-gguf --all-targets -- -D warnings
rtk cargo test -p bridge-gguf --all-targets
```

Expected: green and no payload read.

---

## Task 3: Discover split GGUFs and build the global directory

**Files:**

- Create: `crates/bridge-gguf-split/Cargo.toml`
- Create: `crates/bridge-gguf-split/src/lib.rs`
- Create: `crates/bridge-gguf-split/src/discovery.rs`
- Create: `crates/bridge-gguf-split/src/directory.rs`
- Create: `crates/bridge-gguf-split/tests/splits.rs`

### Step 1: Write split discovery tests first

Create temporary fixtures for:

- an ordinary one-file GGUF;
- entry through any member of `name-00001-of-00003.gguf`;
- a missing member;
- duplicate or zero shard number;
- changing total count;
- filename/`split.no`/`split.count` disagreement;
- aggregate `split.tensors.count` disagreement;
- duplicate tensor name across files;
- overlapping/out-of-bounds tensor range in one file.

Run:

```powershell
rtk cargo test -p bridge-gguf-split --test splits
```

Expected: red.

### Step 2: Implement canonical discovery

Expose:

```rust
pub struct GgufSet {
    pub files: Vec<GgufShard>,
    pub tensors: TensorDirectory,
}

pub struct GgufShard {
    pub path: PathBuf,
    pub parsed: GgufFile,
    pub ordinal: u32,
    pub count: u32,
}

pub struct TensorLocation {
    pub shard_index: usize,
    pub descriptor: TensorDesc,
    pub absolute_range: Range<u64>,
}

pub fn open_set(entry: impl AsRef<Path>) -> Result<GgufSet, SplitError>;
```

Store tensors in both stable file order and a name index. Reject duplicates. Canonicalize only existing paths and never discover outside the input file's parent directory.

For numbered files require exact current llama.cpp stored types: `split.no` and `split.count` are `U16`; `split.tensors.count` is a non-negative `I32`. Reject missing or wrong-typed split metadata rather than coercing numeric values.

### Step 3: Verify

Run:

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-gguf-split --all-targets -- -D warnings
rtk cargo test -p bridge-gguf-split --all-targets
```

Expected: green.

---

## Task 4: Resolve Hy3 metadata and validate tensor schemas

**Files:**

- Create: `crates/bridge-model-hy3/Cargo.toml`
- Create: `crates/bridge-model-hy3/src/lib.rs`
- Create: `crates/bridge-model-hy3/src/config.rs`
- Create: `crates/bridge-model-hy3/src/profile.rs`
- Create: `crates/bridge-model-hy3/src/tensor_role.rs`
- Create: `crates/bridge-model-hy3/src/schema.rs`
- Create: `crates/bridge-model-hy3/tests/config.rs`
- Create: `crates/bridge-model-hy3/tests/schema.rs`

### Step 1: Define the resolved graph model

Implement:

```rust
pub struct Hy3Config {
    pub block_count: u32,
    pub context_length: u64,
    pub embedding_length: u32,
    pub dense_ffn_length: u32,
    pub expert_ffn_length: u32,
    pub attention_head_count: u32,
    pub attention_kv_head_count: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    pub expert_weights_scale: f32,
    pub rope_base: f32,
    pub yarn_factor: f32,
    pub yarn_original_context: u64,
    pub vocabulary_size: u32,
}

pub enum Hy3TensorRole {
    TokenEmbedding,
    OutputNorm,
    Output,
    AttentionNorm { layer: u32 },
    AttentionQ { layer: u32 },
    AttentionQNorm { layer: u32 },
    AttentionK { layer: u32 },
    AttentionKNorm { layer: u32 },
    AttentionV { layer: u32 },
    AttentionOutput { layer: u32 },
    FfnNorm { layer: u32 },
    DenseGate { layer: u32 },
    DenseUp { layer: u32 },
    DenseDown { layer: u32 },
    RouterInput { layer: u32 },
    RouterSelectionBias { layer: u32 },
    RoutedGate { layer: u32 },
    RoutedUp { layer: u32 },
    RoutedDown { layer: u32 },
    SharedGate { layer: u32 },
    SharedUp { layer: u32 },
    SharedDown { layer: u32 },
}
```

`Hy3Profile::selected_iq2_m()` contains exact selected-checkpoint invariants. Config resolution and profile comparison are separate operations so future valid Hy3 checkpoints can be inspected without pretending they are the selected executable profile.

### Step 2: Write config mismatch tests first

Build an in-memory metadata map for the selected values. Mutate one required field per case:

- wrong architecture;
- missing block count;
- wrong hidden/head/KV dimensions;
- non-sigmoid gate;
- disabled expert-weight normalization;
- non-finite or wrong expert scale;
- wrong YaRN factor/original context;
- wrong vocabulary size;
- presence of an MTP-specific block count.

Run:

```powershell
rtk cargo test -p bridge-model-hy3 --test config
```

Expected: red.

### Step 3: Implement strict typed config resolution

Reject NaN and infinity before approximate floating-point comparison. Error messages include the metadata key, expected value, and actual value/type. The non-MTP selected profile requires exactly 80 layers.

### Step 4: Write schema and expert-slab tests first

Use a reduced two-layer fixture that preserves all role transitions: layer zero is dense; layer one is MoE. Cover:

- missing required tensor;
- unknown tensor name;
- exact shape mismatch;
- illegal physical type for the selected profile;
- expert tensor without third dimension;
- `ne[2] != expert_count`;
- expert byte count not divisible by expert count;
- checked `expert_slab(expert_index)` first/last/out-of-range;
- role classification for every naming pattern.

Run:

```powershell
rtk cargo test -p bridge-model-hy3 --test schema
```

Expected: red.

### Step 5: Implement schema generation and comparison

Generate expected tensor names/shapes from `Hy3Config`; do not store 1,278 handwritten entries. Validate exact set equality and physical-type eligibility. Expose:

```rust
pub struct ValidatedHy3Model {
    pub config: Hy3Config,
    pub tensors: Vec<Hy3Tensor>,
    pub has_mtp: bool,
}

pub struct ExpertSlab {
    pub expert: u32,
    pub relative_range: Range<u64>,
}

pub fn validate_selected_model(set: &GgufSet) -> Result<ValidatedHy3Model, Hy3Error>;
```

The expert slab range is relative to the tensor payload and is derived with checked division/multiplication.

### Step 6: Verify

Run:

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-model-hy3 --all-targets -- -D warnings
rtk cargo test -p bridge-model-hy3 --all-targets
```

Expected: green.

---

## Task 5: Build one deterministic inspection report

**Files:**

- Create: `crates/bridge-cli/Cargo.toml`
- Create: `crates/bridge-cli/src/lib.rs`
- Create: `crates/bridge-cli/src/report.rs`
- Create: `crates/bridge-cli/src/text.rs`
- Create: `crates/bridge-cli/tests/report.rs`
- Create: `crates/bridge-cli/tests/fixtures/expected-report.txt`
- Create: `crates/bridge-cli/tests/fixtures/expected-report.json`

### Step 1: Define serializable report records

The report owns deterministic `BTreeMap` summaries for:

- file identity and exact size;
- GGUF version/counts/data offset;
- general metadata and tokenizer counts/IDs;
- resolved Hy3 config;
- tensor count and bytes by exact GGML type;
- tensor count and bytes by semantic role;
- bytes by layer;
- dense/routed/shared expert totals;
- expert slab sizes by projection/type;
- MTP presence;
- inspectable-but-not-executable types;
- warnings.

Expose:

```rust
pub fn build_report(set: &GgufSet) -> Result<InspectionReport, ReportError>;
pub fn render_text(report: &InspectionReport) -> String;
```

JSON is `serde_json::to_string_pretty(&report)` and uses the same data.

### Step 2: Write snapshot tests first

Use a reduced valid Hy3 fixture. Assert byte-for-byte stable text and JSON. Also assert that changing tensor insertion order does not change output.

Run:

```powershell
rtk cargo test -p bridge-cli --test report
```

Expected: red.

### Step 3: Implement checked aggregation

All counts/bytes use checked addition. Sort type labels, roles, layers, shard paths, and warnings. Do not infer sparse-file status from logical length alone; on Windows query allocated ranges only when available and otherwise omit that warning.

### Step 4: Verify

Run:

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-cli --lib --tests -- -D warnings
rtk cargo test -p bridge-cli --lib --tests
```

Expected: green.

---

## Task 6: Wire the `bridge inspect-gguf` command

**Files:**

- Create: `crates/bridge-cli/src/main.rs`
- Create: `crates/bridge-cli/tests/cli.rs`
- Modify: `README.md` if it exists, otherwise create it

### Step 1: Write command integration tests first

With a valid temporary fixture, test:

- `bridge inspect-gguf --model <path>` exits zero and prints text;
- `--json` exits zero and emits parseable JSON only;
- missing path exits nonzero with a concise path-bearing error;
- malformed GGUF exits nonzero without a panic/backtrace;
- an unsupported subcommand is rejected by `clap`;
- help advertises only implemented commands.

Run:

```powershell
rtk cargo test -p bridge-cli --test cli
```

Expected: red.

### Step 2: Implement the CLI

Use:

```rust
#[derive(Parser)]
#[command(name = "bridge", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    InspectGguf {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        json: bool,
    },
}
```

The process:

1. opens the split set;
2. resolves and validates the selected Hy3 profile;
3. builds one report;
4. prints text or JSON;
5. returns a nonzero exit code on any error.

Errors go to stderr. Do not print progress on stdout in JSON mode.

### Step 3: Document exact status

README must state:

- inspection is complete for the selected header;
- tensor payload execution is not yet implemented in this slice;
- the sparse mirror validates metadata only;
- full-model execution requires the exact complete file and checksum;
- the next slice is reference dequantization/tokenization/inference.

### Step 4: Verify

Run:

```powershell
rtk cargo fmt --all
rtk cargo clippy -p bridge-cli --all-targets -- -D warnings
rtk cargo test -p bridge-cli --all-targets
rtk cargo run -p bridge-cli -- --help
```

Expected: green, and help contains no unimplemented command.

---

## Task 7: Prove the real Hy3 header profile

**Files:**

- Create: `crates/bridge-cli/tests/real_hy3_header.rs`
- Create: `docs/models/hy3-1m-iq2-m.md`
- Modify only if defects are found: ingestion crates from Tasks 1-6

### Step 1: Add an opt-in real-header test

The test returns early when `BRIDGE_HY3_HEADER` is unset. When set, it opens that path and asserts:

```rust
assert_eq!(report.gguf.tensor_count, 1_278);
assert_eq!(report.gguf.data_offset, 5_160_192);
assert_eq!(report.tensors.total_encoded_bytes, 96_014_150_912);
assert_eq!(report.types["F32"].count, 479);
assert_eq!(report.types["IQ2_S"].count, 627);
assert_eq!(report.types["IQ3_S"].count, 91);
assert_eq!(report.types["Q4_K"].count, 80);
assert_eq!(report.types["Q5_K"].count, 1);
assert_eq!(report.hy3.block_count, 80);
assert_eq!(report.hy3.expert_count, 192);
assert_eq!(report.hy3.expert_used_count, 8);
assert!(!report.hy3.has_mtp);
```

The test must not hash or read payload bytes.

### Step 2: Run the test red, diagnose, and fix the parser/schema

Run:

```powershell
$env:BRIDGE_HY3_HEADER = 'C:\tmp\hy3-1M-IQ2_M.sparse.gguf'
rtk cargo test -p bridge-cli --test real_hy3_header -- --nocapture
```

Expected initially: any real-format mismatch is exposed. Apply the smallest general format/schema fix, add a reduced regression fixture, and rerun until green.

### Step 3: Run both CLI modes against the real sparse mirror

Run:

```powershell
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf --json
```

Inspect output for the exact counts and totals above. JSON must parse successfully.

### Step 4: Record provenance without claiming payload verification

Document:

- Hugging Face repository and filename;
- repository revision `c29be1652dbe5addbca537e3060cbc523d336966`;
- expected complete-file SHA-256 `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7`;
- logical size and header data offset;
- sparse mirror construction and its metadata-only limitation;
- pinned llama.cpp reference release b10153, commit `b77d646751d01c0962bc203b6809e9d94f7d50b7`;
- Hy3 support merge commit `2969d6d15d67a08e7b83f26164b15350c79c5248`;
- licenses and the rule that no upstream runtime is wrapped.

### Step 5: Run full slice acceptance

Run fresh:

```powershell
rtk cargo fmt --all -- --check
rtk cargo check --workspace --all-targets
rtk cargo clippy --workspace --all-targets -- -D warnings
rtk cargo test --workspace --all-targets
$env:BRIDGE_HY3_HEADER = 'C:\tmp\hy3-1M-IQ2_M.sparse.gguf'
rtk cargo test -p bridge-cli --test real_hy3_header -- --nocapture
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf --json
```

Expected: every command succeeds. This proves authoritative ingestion only; it is not proof of tensor-payload integrity or inference.

---

## Completion handoff

After Task 7, report:

- exact files changed;
- the fresh acceptance output;
- the real-header counts and byte totals;
- that the sparse mirror was not read as tensor data;
- that no Git commit exists because the checkout has no repository metadata;
- the next implementation plan: tokenizer plus scalar reference dequantization/forward inference on a reduced Hy3 test model, followed by direct selected-model execution.
