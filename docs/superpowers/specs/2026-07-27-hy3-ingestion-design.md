# Hy3 Authoritative Ingestion Slice

**Date:** 2026-07-27

**Parent:** `2026-07-27-hy3-engine-architecture-design.md`

## Outcome

This slice turns the partial checkout into a compiling Rust workspace whose `bridge inspect-gguf`
command safely parses, discovers, indexes, validates, and reports a single or split Hy3 GGUF
without reading tensor payloads into RAM.

It ends when the command reproduces the selected checkpoint's 1,278-tensor histogram and exact
96,014,150,912 tensor-byte total from the sparse real-header mirror, and malformed fixtures are
rejected by focused tests.

## Scope

Included:

- repair the workspace and expose `bridge-core` as a library;
- remove GLM-specific types from the active core API;
- correct the GGML block table and tensor-stride semantics;
- make arena and tensor arithmetic overflow-safe;
- parse bounded GGUF v2/v3 scalar, string, and array metadata;
- parse tensor information without mapping or touching payload bytes;
- support one file and `-00001-of-000NN.gguf` split sets;
- build one checked global tensor directory;
- resolve and validate the exact Hy3 configuration;
- classify Hy3 tensor roles and expert slabs;
- print deterministic metadata, type/role/layer byte reports;
- create reduced and malformed GGUF fixtures in Rust tests;
- validate the real sparse header mirror when its path is supplied.

Excluded from this slice:

- dequantization and matrix math;
- tokenizer execution;
- model inference;
- expert sidecar generation;
- asynchronous I/O;
- caches, scheduling, CUDA, serving, and benchmarking.

The CLI reports those capabilities as absent rather than providing placeholder commands.

## Parsing Boundary

`bridge-gguf` owns file-format parsing. Its `GgufReader<R: Read + Seek>` consumes only the header,
metadata, tensor directory, and alignment padding. It never constructs tensor-data slices.

Configurable limits default to:

- 3 dimensions used by the selected model and the GGML ABI hard maximum of 4;
- 1 MiB per metadata string;
- 1,000,000 array elements;
- 4,000,000 tensors;
- 1,000,000 metadata entries;
- 64 MiB total metadata bytes before caller override.

The real Hy3 tokenizer metadata fits the total budget. All lengths are converted through checked
`u64 -> usize` conversion before allocation.

The parser returns:

- GGUF version and endianness;
- ordered metadata values with their exact stored types;
- tensor info with relative offsets;
- declared alignment and computed data offset;
- physical file length.

Tensor layout is validated in source directory order against the pinned GGUF
loader rule. The first relative offset is zero; every later offset equals the
prior running extent; each encoded length is rounded up to
`general.alignment` with checked arithmetic; and the complete final padded
extent must fit the physical file. Trailing file bytes are allowed. This
rejects aliases, partial overlap, gaps, and out-of-order layouts without
reading payload bytes.

No public constructor can create an invalid tensor descriptor.

## Split Discovery

`bridge-gguf-split` accepts any file:

- a filename without a numbered suffix is one shard;
- a numbered member discovers all siblings with the same stem/count;
- shard index/count metadata, when present, must agree with filenames;
- every expected index appears exactly once;
- tensor names are globally unique;
- every absolute range is aligned, checked, and within its owning file;
- GGUF version, endianness, and alignment agree across all shards;
- aggregate tensor count metadata, when present, must agree.

Independent file handles are retained by later slices, but this slice stores canonical paths and
checked descriptors only.

## Hy3 Validation

`bridge-model-hy3` reads values from GGUF metadata and compares them with the selected profile.
Its public schema generation and descriptor-validation APIs are explicitly
named for `selected_iq2_m` and authorize that exact profile before applying
checkpoint-specific physical types or layer transitions. General Hy3 configs
cannot enter those selected-schema APIs.
It requires:

- architecture `hy_v3`;
- 80 base blocks for the non-MTP baseline;
- hidden 4096, dense FFN 13,312, expert FFN 1536;
- 64 Q heads, 8 KV heads, Q/K/V head length 128;
- 192 experts and exact top-8;
- normalized sigmoid gating and scale 2.826;
- context 1,048,576 with YaRN factor 4 and original context 262,144;
- tokenizer/model vocabulary 120,832;
- all required tensor names and exact shapes for dense layer 0 and MoE layers 1-79.

It accepts only the actual baseline physical types for optimized eligibility. A known GGML type
with no decoder is reported as inspectable but not executable; an unknown type discriminant is a
hard format error.

Expert tensors must have `ne[2] = 192`, row-aligned `ne[0]`, exactly divisible encoded bytes, and
contiguous per-expert slabs. The descriptor exposes each slab only through a checked method.

## CLI Report

`bridge inspect-gguf --model <path> [--json]` prints:

- file(s), exact sizes, split identity, and each shard's GGUF version,
  endianness, metadata count, data offset, and alignment;
- validated common GGUF version, endianness, and alignment;
- explicitly authoritative shard-zero metadata count, tensor count,
  architecture, name, and license;
- resolved Hy3 configuration;
- tensor count and encoded bytes by exact type;
- tensor count and encoded bytes by semantic role;
- bytes per layer;
- dense, shared-expert, and routed-expert totals;
- gate/up/down per-expert slab sizes and contiguity;
- tokenizer model, vocabulary/merge counts, and special IDs;
- MTP presence;
- types unsupported by later execution kernels;
- warnings such as a sparse header mirror whose payload ranges read as holes.

Text and JSON modes are generated from the same serializable report structure.

## Tests

Tests follow red-green-refactor and use real parser behavior:

- minimal valid v3 file;
- v2 compatibility;
- bad magic/version;
- truncated scalar/string/array/tensor records;
- excessive counts and lengths;
- unknown metadata type and GGML type;
- arithmetic overflow and out-of-bounds ranges;
- per-row block misalignment;
- duplicate tensors;
- split filename and metadata mismatches;
- missing shard;
- valid reduced Hy3 fixture;
- each Hy3 profile mismatch;
- wrong/missing tensor shape;
- invalid expert dimension/slab;
- deterministic text and JSON report snapshots;
- optional real-header test via `BRIDGE_HY3_HEADER`.

The optional real-header test never claims payload correctness. Full inference later requires the
complete file and exact SHA-256.

## Acceptance Commands

The slice is accepted only after fresh successful runs of:

```powershell
rtk cargo fmt --all -- --check
rtk cargo check --workspace --all-targets
rtk cargo clippy --workspace --all-targets -- -D warnings
rtk cargo test --workspace --all-targets
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf
rtk cargo run -p bridge-cli -- inspect-gguf --model C:\tmp\hy3-1M-IQ2_M.sparse.gguf --json
```

The checkout currently has no Git metadata. The design and implementation can be verified locally,
but no commit claim is made until a repository is initialized or restored.
