# Pinned scalar quantization oracle

This directory contains a development-only fixture generator for the exact
llama.cpp release `b10153`, revision
`b77d646751d01c0962bc203b6809e9d94f7d50b7`. It is not part of the LightBridge
workspace build and it never loads model weights.

The helper links the pinned static `ggml-base` target and directly compiles only
`ggml/src/ggml-cpu/quants.c`. It deliberately disables the `ggml-cpu` target,
CPU dispatch, SIMD dispatch names, shared libraries, OpenMP, and native tuning.
The four dot products call the pinned `_generic` scalar symbols. Strict
floating-point flags disable contraction and reassociation.

`oracle.cpp` is a local MIT-licensed harness. It creates deterministic packed
blocks, copies their bytes into naturally aligned upstream block types with
`memcpy`, and validates host endianness, ABI sizes, lengths, finite activation
values, and `n` before every upstream entry. Its F32 fixture is a
bit-preserving ABI identity fixture; it does not claim an upstream F32
dequantizer. Q8_K output storage is value-initialized because the pinned
all-zero path leaves its block sums unwritten.

Generate and then independently verify the checked-in vectors:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1
rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1 -VerifyOnly
```

The authenticated checkout must be at
`C:\tmp\lightbridge-llama-b10153`, exactly clean, at both tag `b10153` and the
revision above. Build output is confined to
`C:\tmp\lightbridge-quant-oracle-build`; generated binaries and their
deterministic manifest are written under
`crates/bridge-quant-layout/tests/fixtures`.

`-VerifyOnly` takes its branch before any directory creation, build, temporary
file, oracle execution, or write. It authenticates the checkout and every
source, enforces a hard-coded fixture inventory and schema, rehashes every
binary, cross-checks all manifest references, verifies F32 identity bytes,
recomputes Q8_K block sums, and rejects non-finite activation/dot values. It is
safe to use as the normal offline provenance gate.

The upstream MIT notice is preserved at
`vendor/upstream/llama.cpp/LICENSE`. Exact source identities and the distinction
between external oracle sources and retained local copies are recorded in
`vendor/upstream/llama.cpp/PINNED.toml` and `docs/UPSTREAM.md`.
