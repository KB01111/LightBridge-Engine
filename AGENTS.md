# Repository Guidelines

## Project Structure & Module Organization

This repository is a Rust 2021 workspace. Production code lives under `crates/`:

- `bridge-core`: checked tensor, arena, GGML type, and host-topology primitives.
- `bridge-gguf` and `bridge-gguf-split`: bounded GGUF parsing and shard discovery.
- `bridge-model-hy3`: Hy3 configuration, tensor roles, and schema validation.
- `bridge-quant-layout`: packed layouts and scalar reference decoding.
- `bridge-cli`: the `bridge` inspection binary and deterministic reports.

Keep integration tests in each crate’s `tests/` directory and small unit tests beside the relevant module. Binary fixtures belong in `crates/bridge-quant-layout/tests/fixtures/`. Architecture notes and verified status live in `docs/`; `vendor/upstream/llama.cpp/` is pinned reference material, not runtime code.

## Build, Test, and Development Commands

Rust 1.82 or newer is required.

```powershell
cargo build --workspace
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run -p bridge-cli -- inspect-gguf --model C:\path\to\model.gguf
```

Use `cargo test -p bridge-quant-layout --release --all-targets` when changing quantization math. Regenerate or verify oracle fixtures with `tools/quant-oracle/generate-vectors.ps1`; prefer `-VerifyOnly` when no fixture update is intended.

## Coding Style & Naming Conventions

Follow `rustfmt.toml` (110-column limit, field-init shorthand) and four-space Rust indentation. Use `snake_case` for modules, functions, and tests; `UpperCamelCase` for types and traits; and `SCREAMING_SNAKE_CASE` for constants. Keep parsing bounded, arithmetic checked, output deterministic, and failures represented by typed errors. Do not panic on model-controlled input.

## Testing Guidelines

Name integration files after the behavior under test, such as `reader.rs`, `schema.rs`, or `oracle_vectors.rs`; give tests descriptive `snake_case` names. Cover malformed inputs, boundary sizes, overflow, and atomic failure behavior, not only happy paths. Run the focused crate suite while iterating, then the full workspace gates above before submitting.

## Commit & Pull Request Guidelines

History currently uses short descriptive subjects, for example `Initial public Hy3 engine checkpoint`; no Conventional Commit scheme is established. Keep each commit scoped and explain non-obvious correctness or provenance decisions in its body.

Pull requests should summarize the change, identify affected crates, link relevant issues or design notes, and list exact validation commands and results. Include sample text/JSON output for CLI changes. Never claim full inference support from parser or scalar-test success.

## Security & Provenance

Do not commit model weights, `.env` files, keys, or generated caches. Changes derived from vendored upstream material must preserve licenses and update `vendor/upstream/llama.cpp/PINNED.toml` plus `docs/UPSTREAM.md`.
