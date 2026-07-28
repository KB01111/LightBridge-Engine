# Upstream provenance

LightBridge is a native Rust implementation. The authenticated source subset
under `vendor/upstream/llama.cpp` is pinned reference material for format, ABI,
and differential-correctness work; it is not compiled, linked, included by a
build script, or wrapped as the LightBridge runtime. `PINNED.toml` binds every
retained upstream file to the exact Git commit blob bytes.

## Pinned references

| Project | Pin | License | Use |
|---|---|---|---|
| llama.cpp | release `b10153`, commit `b77d646751d01c0962bc203b6809e9d94f7d50b7` | MIT | Primary GGUF/GGML ABI and differential-correctness reference |
| llama.cpp Hy3 support | merge `2969d6d15d67a08e7b83f26164b15350c79c5248` | MIT | `hy_v3` metadata, tensor mapping, graph, and optional MTP reference |
| Tencent Hy3 | final head `a960ebc3da325ba167f069f76c41eb62c9280d22` | Apache-2.0 | Official final-model architecture and tokenizer lineage |
| Transformers | v5.6.0 commit `3e80155a968c1080f11b2710e8b31741ac5ab0ed` | Apache-2.0 | Independent model/configuration behavior reference |
| Colibri | v1.1.1 commit `81f08a09e5651ce52616dc720f68810f9021c0be` | Apache-2.0 | Systems-design reference only |

llama.cpp's MIT terms require preservation of its copyright and permission
notices when material is copied or derived. Future changes that copy or adapt
upstream implementation material must retain the applicable notices and
record the affected files here. This document does not reproduce upstream
license text.

## Complete authenticated llama.cpp inventory

Repository: `https://github.com/ggml-org/llama.cpp.git`

Release: `b10153`

Revision: `b77d646751d01c0962bc203b6809e9d94f7d50b7`

All hashes in the table are full lowercase SHA-256 over local LF file bytes.
Every status is `exact`: `git hash-object` over each local file equals
`git rev-parse HEAD:<path>` in the authenticated checkout, and the local raw
bytes equal the corresponding Git commit blob bytes.

The authenticated Windows checkout materializes those text files with CRLF,
so its raw working-tree SHA-256 differs without representing content drift.
`PINNED.toml` records both that external working-tree SHA-256 and the
platform-independent upstream commit-blob SHA-256, plus both matching Git blob
object IDs, for every entry. The machine-readable paths, upstream paths,
revision, hashes, status, and purpose are therefore reproducible across the
checkout's EOL transformation.

| Live vendored path | Local SHA-256 | Status | Purpose |
|---|---|---|---|
| `ggml/include/ggml.h` | `c65c30fdb4dce95eac71c26bb38ae8423fbc80d79db91d2b2ffaea8c4e46276a` | exact | GGML type discriminants and public type-trait ABI reference |
| `ggml/src/ggml-common.h` | `af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0` | exact | Packed Q4_K, Q5_K, IQ2_S, and IQ3_S block-layout reference |
| `ggml/src/gguf.cpp` | `cc86dbbb4ea1a01b78e3aba879a8e0654dd23aa8d8d73ff56241443eb188bf6d` | exact | GGUF metadata, tensor-directory, and alignment behavior reference |
| `src/llama-arch.cpp` | `52d6e524cef2e015d6c88b880df697dffd3237b7ef79bc558f7634aa23c01c70` | exact | Architecture, metadata-key, and tensor-name mapping reference |
| `src/llama-arch.h` | `486f5b2136e29e39c59137f5795e0001fb8587f12c9b287244b5c816377ddce8` | exact | Architecture and tensor-role declarations, including `hy_v3` |
| `src/llama-graph.cpp` | `c83460b64090c19fb7737c9f1e8fea0e96af6179b5c5ce2de1aced1f8c1c55fe` | exact | Shared attention, MoE, and expert-routing graph-helper reference |
| `src/llama-graph.h` | `7010cc58f4d776d31b8c0f491125082ca50bb611f6e02fc98a264cb90cb46549` | exact | Shared graph-helper declarations |
| `src/llama-kv-cache-dsa.cpp` | `871e08b668b9af88e2139e0ee3ac305e3f7e0cc218a43e9277edadbf8830f95e` | exact | Inactive legacy DSA KV-cache implementation reference; not used by Hy3 ingestion |
| `src/llama-kv-cache-dsa.h` | `4ce577e70a0d9c2621966f95ffcc94d61cd2de0f6f8acaab0b5ebdd70f93a234` | exact | Inactive legacy DSA KV-cache declarations; not used by Hy3 ingestion |
| `src/llama-kv-cache.cpp` | `4ee0076fb49c2e3b08bd004c4da085eabe85c7fdaaa21ad5fc607b08eddc78df` | exact | Inactive general KV-cache systems reference; not used by Hy3 ingestion |
| `src/llama-model.cpp` | `3d55155fee0fdcbb7ded920e71dfba6eb3d820559295c5f4da8824d5cc8c8c78` | exact | Architecture selection and shared model-loading behavior reference |
| `src/models/glm-dsa.cpp` | `cbe84e8988d931739a017ba5e2b064454572693f1fbdcd66dc73d44e28118d25` | exact | Inactive legacy GLM/DSA historical-comparison reference; not used by Hy3 ingestion |
| `LICENSE` | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` | exact | Preserved upstream MIT license and copyright notice |

`PINNED.toml` is local provenance metadata, not an upstream file, and is
excluded from its own inventory to avoid a circular self-hash.

The pinned Hy3 merge additionally identifies `src/models/hy-v3.cpp` as the
architecture-specific tensor-loading and graph reference. That file is not
part of the local vendored subset; it is authenticated in the external-only
oracle-source inventory below.

## External-only scalar quantization oracle sources

Task 1 of the scalar reference-math slice authenticates ten additional files
in the clean checkout at `C:\tmp\lightbridge-llama-b10153`. They are not copied
under `vendor/`, are not production dependencies, and do not change the
13-file exact local-copy inventory above. `PINNED.toml` records, for every
entry, the revision, Git blob OID, canonical commit-blob SHA-256, Windows
worktree SHA-256, external-only status, and purpose.

| External upstream path | Git blob OID | Canonical blob SHA-256 | Oracle use |
|---|---|---|---|
| `ggml/src/ggml.c` | `a7d1fe7d94be4bee3df47f0d710fbfdb62087d1f` | `84b5c2608cf70f7beacdaa67d3cc4b58d34654d57d3d50268ff4b9eb83a643e0` | F32 type-trait and bit-preserving identity ABI evidence |
| `ggml/src/ggml-quants.c` | `1ebc50a763f16db909de37090da38cc8c0fdde94` | `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d` | Q4_K/Q5_K/IQ dequantizers and Q8_K reference quantizer |
| `ggml/src/ggml-quants.h` | `75188f1af180e3592f3c94bc4077989fb817c359` | `28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da` | Reference function declarations |
| `ggml/src/ggml-impl.h` | `62b76abbcec9e71c860ba1a99d79b501bad26b93` | `2ed56e264202906d107e26d08eabb242d3107b026ebfb78096fa1e5f94bdbbb8` | FP16 conversion and low-level helper declarations |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `491316f7491252248d6f74a60440d3efa7aa6177` | `f2abcaf7f627a2d8a4b7744a7128b210dad0d147fc92cb94ce9cbaed2945e84a` | Dispatch/type-trait evidence only; excluded from the build |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | `5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4` | `e7008069e3e46f1db5e3d2eaafb4ddec3c7d0ece5c0454f99c5a8e33a50f20ba` | Scalar CPU ABI and helper evidence |
| `ggml/src/ggml-cpu/quants.c` | `5e36459f8cbc5900b375d2189414307393471a6b` | `a61f1011e49d05b5f99d352b158d5b8e36cf008294bb0db309f72bdd7f1d4e35` | Directly compiled four generic scalar Q8_K dot kernels |
| `ggml/src/ggml-cpu/quants.h` | `93ea7eeffe5b00ad2c612aac49b7983c12949525` | `918d6755b3e601ec7bb83c7dbf1d73304490651ccbb4072b0d31a2b45df751da` | Generic scalar dot declarations |
| `src/models/hy-v3.cpp` | `47a0beaf217f19219e3e8fb8d5c35664625d7c73` | `fcc0822f3291db653a3a723614f525bf96204161517abec5a56a3f5d1d8ac6c3` | Hy3 graph and packed-tensor execution reference |
| `tests/test-quantize-fns.cpp` | `9510ac14ce00805e1689a8c8b16b6dd6c329911c` | `851f302b1f9338f2cc259f765cc36d702f13f564ee0fb6050043c7041a55c13b` | Differential quantization-test methodology |

The CMake project authenticates the exact HEAD, annotated release target,
origin, clean tracked/untracked state, all ten oracle-source identities, the
retained `ggml-common.h`, and both loaded GGML CMake files before
`add_subdirectory`. It forces static `ggml-base`, disables `ggml-cpu`, CPU
dispatch, OpenMP, shared libraries, and native tuning, and directly compiles
only `ggml/src/ggml-cpu/quants.c`. Strict floating-point flags apply to both
`ggml-base` and the local executable. The harness supplies and initializes all
65,536 `ggml_table_f32_f16` lanes before generic dot calls. It never builds
`ggml-cpu.c`, defines `GGML_CPU_GENERIC`, invokes SIMD/dispatch entry points,
or reads model weights.

## Quantization fixture provenance

The development-only harness under `tools/quant-oracle` generates an exact
16-binary aggregate fixture set plus deterministic JSON at
`crates/bridge-quant-layout/tests/fixtures`. Each quantized input contains
three full blocks in fixed order: structural pattern, deterministic LCG, and
zero scale. Q8_K contains structural, LCG, and zero activation blocks. F32 is
a 16-value bit-preserving IEEE-754 identity fixture, including signed zero; it
does not claim a nonexistent upstream F32 dequantizer.

| Fixture | Bytes | SHA-256 |
|---|---:|---|
| `decode-f32.input.bin` | 64 | `b3e8edc79e70091d2fdd3e4226e03c49bcc85219ac664ac909673ac9305a4e15` |
| `decode-f32.output-f32le.bin` | 64 | `b3e8edc79e70091d2fdd3e4226e03c49bcc85219ac664ac909673ac9305a4e15` |
| `decode-q4-k.input.bin` | 432 | `35ba5f01848598c81886481ff6cb083c1532b3a77bef2242c5494ab9a1e25a40` |
| `decode-q4-k.output-f32le.bin` | 3,072 | `bf26c424906a978f2149b00ec054c35ed9ab93af384223b2dbabb20920b30a73` |
| `decode-q5-k.input.bin` | 528 | `28d94a77d63d940e5ab4e8c615853124243522946da85b09df27c4a75e211f6d` |
| `decode-q5-k.output-f32le.bin` | 3,072 | `ae63040be7218f4d8ae5ec779197525f754576616812aa823bca294aaaadd14b` |
| `decode-iq2-s.input.bin` | 246 | `f727805b3cf3335de8b8d83df88704ac978378280e757d1fa4713c394957bb70` |
| `decode-iq2-s.output-f32le.bin` | 3,072 | `8e2f82c79fa36b241213529daf7fe456a0cac3efb6158b17743ed3e0de2020a4` |
| `decode-iq3-s.input.bin` | 330 | `496ac30534619d9d722e9e9e88ccc9300473891b2a3d7a7a116061660bec6d37` |
| `decode-iq3-s.output-f32le.bin` | 3,072 | `62417424adbafdc6fdb1d524d9cd6cb14f4b8746c7c7d5dff0d7bebbfbdc6512` |
| `q8-k-activations.input-f32le.bin` | 3,072 | `6d17fd2842f5d70b2fba6dcda516917878d780a4993097964780cf50b4a9636a` |
| `q8-k-activations.output-q8-k.bin` | 876 | `627fc80a95b7629f06686c36606a6abaef0f2a9904ed8af9667b16befc853bce` |
| `dot-q4-k-q8-k.output-f32le.bin` | 4 | `f37567ae0e2607aea21c66e6dca6b1782c47132feec32efdd412b732907b82cb` |
| `dot-q5-k-q8-k.output-f32le.bin` | 4 | `accac79c7b269652fa5a3934b850f561f499842f52a555d432145fa0ade362b4` |
| `dot-iq2-s-q8-k.output-f32le.bin` | 4 | `3d9151660ed213affc191f750e8ff02bdd8ce81e78d508defed06bb321fd10a7` |
| `dot-iq3-s-q8-k.output-f32le.bin` | 4 | `b1ebe1b3f6e5789b8360ac7d157d992f617bb084d493d34e360c51cb2d2cbc5c` |

The four dot fixtures use `n = 768`, `nrc = 1`, and zero strides. The helper
copies bytes into aligned typed storage, rejects invalid lengths, invalid
`n`, non-finite scales/activations/Q8 scales, and inconsistent Q8 block sums
before upstream entry, stages output bytes, and atomically promotes a
same-directory temporary only after successful computation. The all-zero Q8_K
block is value-initialized because the pinned reference path leaves its
`bsums` field unwritten.

The IQ table locks serialize integer values, not source text:

| Table | Canonical little-endian bytes | SHA-256 | Source |
|---|---:|---|---|
| `kmask_iq2xs` | 8 | `5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5` | `ggml-common.h:509-511` |
| `iq2s_grid` | 8,192 | `e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88` | `ggml-common.h:758-1015` |
| `iq3s_grid` | 2,048 | `bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f` | `ggml-common.h:1052-1117` |

Each IQ table record also carries its canonical source URL, revision, table
name and line range, blob OID, canonical and Windows worktree source hashes,
exact local vendored-file hash, generation command, endianness, license, and
local oracle, generator-script, and CMake-source hashes. The same three local
tool hashes are bound into every decode, Q8_K, and dot record.

Generate and verify with:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1 -VerifyOnly
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1 -VerifyTamperMatrix
```

`-VerifyOnly` branches before all build, temporary, generation, directory, and
write operations. It rejects source, pin, license, schema, inventory, length,
hash, cross-link, F32 identity, Q8 sum, activation, and dot drift. The separate
tamper matrix works only on disposable `C:\tmp` copies.

## Local reimplementation and modifications

No vendored C or C++ source is part of the active Rust build. The ingestion
slice independently implements:

- checked GGML type IDs, block sizes, tensor shape/stride arithmetic, and
  encoded-length calculation in `bridge-core`;
- a bounded endian-aware GGUF v2/v3 metadata and tensor-directory parser in
  `bridge-gguf`, with stricter allocation and count limits than the reference;
- canonical single/split-file discovery and a checked global tensor directory
  in `bridge-gguf-split`;
- selected-profile Hy3 metadata resolution, exact tensor-schema generation,
  semantic roles, and expert-slab validation in `bridge-model-hy3`;
- deterministic text/JSON inspection reports and the `inspect-gguf` command in
  `bridge-cli`.

Those Rust modules express only the behavior needed by the current ingestion
slice. They do not copy an upstream graph executor, quantization kernel,
tokenizer runtime, server, or model payload. Later mathematical kernels and
runtime behavior will be recorded here when they are implemented and
differentially validated.

The obsolete unexported `crates/bridge-core/src/glm.rs` file has been removed.
The checked-in quantization artifacts are data produced by the development
oracle, not linked C/C++ code. Normal Rust tests read only those authenticated
bytes and their JSON provenance; they do not require CMake, a compiler, the
external checkout, or network access.
