# Reduced Hy3 graph oracles

This directory regenerates independent development-only vectors for the
deterministic model in `bridge-test-model`. Neither Transformers nor llama.cpp
is linked into a production BRIDGE crate.

## Official Transformers graph

Use the exact Transformers checkout recorded in `SOURCE.json`. The generator
verifies its Git commit and package version before importing it, loads only the
local deterministic F32 bundle, forces CPU/eager/F32 execution, and records
named block intermediates, routing IDs, logits, probabilities, and greedy IDs.
It never calls `from_pretrained` and performs no network access.

```powershell
rtk cargo run -p bridge-test-model --example export_oracle_bundle -- .oracle\weights
rtk python tools\hy3-oracle\generate.py `
  --bundle .oracle\weights `
  --transformers-source C:\path\to\transformers-at-pinned-commit `
  --output .oracle\transformers-hy3
rtk cargo run -p bridge-test-model --example export_run_vectors -- .oracle\bridge
rtk python tools\hy3-oracle\verify.py `
  --bridge .oracle\bridge `
  --transformers .oracle\transformers-hy3
```

`generate.py` writes `.json` provenance/hashes and a companion `.npz` with the
actual F32 arrays. The checked Rust fixture uses compact SHA-256 locks; release
acceptance compares arrays with the tolerances in the reference-math plan
rather than expecting PyTorch and scalar Rust accumulation to be bit-identical.

## llama.cpp graph

`llama-oracle.cpp` is built only against the exact pinned llama.cpp checkout.
`generate-llama-vectors.ps1` verifies the checkout commit, local source hash,
command, and output hashes before accepting vectors. No normal Cargo command
builds or runs it.

`full-model-oracle.cpp` is a compact release-only companion. It accepts exact
numeric token IDs, evaluates them token-serially through the same pinned
llama.cpp graph with CPU/F32 KV state, and records the greedy and runner-up
logits after every input token. The full-model acceptance runner uses those
predictions to compare LightBridge's generated IDs without depending on text
decoding or two independently formatted chat prompts. The oracle remains
development tooling and is never linked into the production engine.
