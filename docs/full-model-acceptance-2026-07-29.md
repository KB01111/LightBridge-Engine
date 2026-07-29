# Full-model acceptance: 2026-07-29

## Result

The CPU baseline passed the complete selected-checkpoint workflow on the
Windows target host. The final machine-local report is
`D:\LightBridge\Acceptance\acceptance.json` with SHA-256
`95e7a890b4c2d1a2619f5743fee22730531047d8fdf9d13daf6452f4f95581a2`.
The report is command-recorded and hash-binds the executed engine, acceptance
runner, parity verifier, oracle source, and oracle executable.

Model weights and the 92 GB expert sidecar remain outside the repository.

## Artifact and execution

| Property | Accepted value |
|---|---|
| Model | `satgeze/Hy3-1M-GGUF/hy3-1M-IQ2_M.gguf` |
| Revision | `c29be1652dbe5addbca537e3060cbc523d336966` |
| Logical and allocated bytes | `96,019,311,104` |
| SHA-256 | `1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7` |
| Physical storage | Non-sparse, uncompressed |
| Backend | `cpu_parallel_avx2_q8_k`, 12 threads |
| Prompt | `Hi`, 16 formatted tokens |
| Generated IDs | `[16883, 0]` |
| Decoded text | `Hello!` |

Direct GGUF execution completed in 202,981 ms: 181,662 ms prefill and
21,318 ms decode. The authenticated sidecar completed in 189,085 ms:
165,723 ms prefill and 23,360 ms decode. Both paths emitted the exact same
prompt IDs, generated IDs, raw text, and stop reason.

## Independent parity

The full-model oracle was built from llama.cpp release `b10153`, commit
`b77d646751d01c0962bc203b6809e9d94f7d50b7`. For the same numeric input
sequence it selected `[16883, 0]`, exactly matching LightBridge. The greedy
runner-up margins were `2.40599442` and `5.21518898`; neither match depended
on a near-tie.

Relevant provenance hashes:

| Component | SHA-256 |
|---|---|
| LightBridge release executable | `586d17b2de499f76734fa4f512675dfadaddec30f61a37f2aa5a63de318118d6` |
| Acceptance runner | `c9e693d8465922e26eb6569d78cfcace45d5342220fa9e68568310874261a116` |
| Parity verifier | `22bd0692c9fea5546b98e45acd23662b462da89ec99273e9d0a3ebf1ab2fa4d4` |
| llama.cpp oracle source | `30d59101130fc522f598e6ec6616f1a7cb9f6c1fa34c0667cbceaadc67295232` |
| llama.cpp oracle executable | `ca5e6c4e4253579bec7a5cd381d4d7d203fcaea000fd91ebcef2d2989222da9b` |

## Sidecar

Preparation authenticated all 96,019,311,104 source bytes and copied
92,361,719,808 expert payload bytes into 15,168 records. The resulting
92,361,723,904-byte sidecar has SHA-256
`ff99fae26659ec298d5b1ce8bb69fcf5d7627762b24acf34390c24467ce04361`.
Its tensor-directory binding is
`6e81ce480586c2e90f16ac9baf5e658203b6bb0f33a6fb8313049a8303bf5c6b`.

## Cold, admission, and warm measurements

All phases ran in one authenticated process with the same prompt and produced
the same `[16883, 0]` output:

| Phase | Prefill | Decode | Total | Decode tokens/s |
|---|---:|---:|---:|---:|
| Cold | 190,871 ms | 22,302 ms | 213,175 ms | 0.089676 |
| Admission | 185,223 ms | 22,101 ms | 207,325 ms | 0.090490 |
| Warm | 187,418 ms | 22,506 ms | 209,925 ms | 0.088861 |

The cache ceiling was 2,147,483,648 bytes. Cold execution began with zero
resident entries and ended with 351 entries using 2,122,039,296 bytes.
Admission began with 351 and ended with 355 entries; warm began and ended with
355 entries using 2,146,222,080 bytes.

The cold phase recorded 211 within-run hits. Admission and warm each recorded
zero cross-run hits and 11,376 loads because the routed working set and
hysteretic admission policy exceed the 2 GiB ceiling for this sequence. This
is valid bounded-cache evidence, but it is not presented as a warm-cache
speedup.

## Capability boundary

This closes the supported CPU path and selected-checkpoint release gate.
CUDA was unavailable because the host had no live NVIDIA device. Grouped
multi-token prefill and experimental iGPU placement remain explicitly
unavailable optional acceleration, and MTP is inapplicable because this
checkpoint has no MTP block.
