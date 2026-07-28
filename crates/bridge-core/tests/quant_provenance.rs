use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const REPOSITORY: &str = "https://github.com/ggml-org/llama.cpp.git";
const SOURCE_URL_PREFIX: &str = "https://github.com/ggml-org/llama.cpp/blob";
const RELEASE: &str = "b10153";
const REVISION: &str = "b77d646751d01c0962bc203b6809e9d94f7d50b7";
const GENERATION_COMMAND: &str =
    "rtk powershell -NoProfile -ExecutionPolicy Bypass -File tools/quant-oracle/generate-vectors.ps1";

const EXPECTED_FILES: [(&str, u64); 16] = [
    ("decode-f32.input.bin", 64),
    ("decode-f32.output-f32le.bin", 64),
    ("decode-iq2-s.input.bin", 246),
    ("decode-iq2-s.output-f32le.bin", 3_072),
    ("decode-iq3-s.input.bin", 330),
    ("decode-iq3-s.output-f32le.bin", 3_072),
    ("decode-q4-k.input.bin", 432),
    ("decode-q4-k.output-f32le.bin", 3_072),
    ("decode-q5-k.input.bin", 528),
    ("decode-q5-k.output-f32le.bin", 3_072),
    ("dot-iq2-s-q8-k.output-f32le.bin", 4),
    ("dot-iq3-s-q8-k.output-f32le.bin", 4),
    ("dot-q4-k-q8-k.output-f32le.bin", 4),
    ("dot-q5-k-q8-k.output-f32le.bin", 4),
    ("q8-k-activations.input-f32le.bin", 3_072),
    ("q8-k-activations.output-q8-k.bin", 876),
];

const EXPECTED_DECODE_IDS: [&str; 5] = [
    "decode-f32",
    "decode-q4-k",
    "decode-q5-k",
    "decode-iq2-s",
    "decode-iq3-s",
];

const EXPECTED_DOT_IDS: [&str; 4] = [
    "dot-q4-k-q8-k",
    "dot-q5-k-q8-k",
    "dot-iq2-s-q8-k",
    "dot-iq3-s-q8-k",
];

const EXPECTED_ORACLE_SOURCES: [&str; 10] = [
    "ggml/src/ggml.c",
    "ggml/src/ggml-quants.c",
    "ggml/src/ggml-quants.h",
    "ggml/src/ggml-impl.h",
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu-impl.h",
    "ggml/src/ggml-cpu/quants.c",
    "ggml/src/ggml-cpu/quants.h",
    "src/models/hy-v3.cpp",
    "tests/test-quantize-fns.cpp",
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("bridge-core must be two levels below the workspace root")
        .to_path_buf()
}

fn object<'a>(value: &'a Value, context: &str) -> &'a serde_json::Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{context} must be a JSON object"))
}

fn array<'a>(value: &'a Value, context: &str) -> &'a [Value] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{context} must be a JSON array"))
}

fn string<'a>(value: &'a Value, context: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{context} must be a JSON string"))
}

fn integer(value: &Value, context: &str) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{context} must be an unsigned JSON integer"))
}

fn require_sha256(value: &Value, context: &str) {
    let digest = string(value, context);
    assert_eq!(digest.len(), 64, "{context} must contain 32 bytes");
    assert!(
        digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{context} must be lowercase hexadecimal"
    );
}

fn require_provenance(value: &Value, context: &str) {
    let record = object(value, context);
    assert_eq!(
        string(&record["source_url"], &format!("{context}.source_url")),
        format!(
            "{SOURCE_URL_PREFIX}/{REVISION}/{}",
            string(&record["source_path"], "source_path")
        )
    );
    assert_eq!(string(&record["commit"], "commit"), REVISION);
    assert!(!string(&record["source_path"], "source_path").is_empty());
    assert!(!string(&record["source_function"], "source_function").is_empty());
    assert!(!string(&record["source_lines"], "source_lines").is_empty());
    assert_eq!(
        string(&record["generation_command"], "generation_command"),
        GENERATION_COMMAND
    );
    assert_eq!(string(&record["endianness"], "endianness"), "little");
    assert_eq!(string(&record["license"], "license"), "MIT");
    assert_eq!(
        string(&record["upstream_git_blob_oid"], "upstream_git_blob_oid").len(),
        40
    );
    require_sha256(&record["upstream_blob_sha256"], "upstream_blob_sha256");
    require_sha256(&record["local_oracle_sha256"], "local_oracle_sha256");
}

#[test]
fn quantization_vectors_are_complete_hash_bound_and_pinned() {
    let root = workspace_root();
    let fixtures = root.join("crates/bridge-quant-layout/tests/fixtures");
    let manifest_path = fixtures.join("quant-vectors.json");
    let manifest_text = fs::read_to_string(&manifest_path).unwrap_or_else(|error| {
        panic!(
            "read quantization provenance manifest {}: {error}",
            manifest_path.display()
        )
    });
    assert!(
        !manifest_text.starts_with('\u{feff}'),
        "manifest must be UTF-8 without a BOM"
    );
    assert!(
        manifest_text.ends_with('\n') && !manifest_text.ends_with("\n\n"),
        "manifest must have exactly one final newline"
    );

    let manifest: Value =
        serde_json::from_str(&manifest_text).expect("quantization manifest must be strict JSON");
    let top = object(&manifest, "manifest");
    assert_eq!(integer(&top["schema_version"], "schema_version"), 1);

    let upstream = object(&top["upstream"], "upstream");
    assert_eq!(string(&upstream["repository"], "repository"), REPOSITORY);
    assert_eq!(string(&upstream["release"], "release"), RELEASE);
    assert_eq!(string(&upstream["commit"], "commit"), REVISION);
    assert_eq!(string(&upstream["license"], "license"), "MIT");

    let generator = object(&top["generator"], "generator");
    assert_eq!(
        string(&generator["generation_command"], "generation_command"),
        GENERATION_COMMAND
    );
    assert_eq!(string(&generator["endianness"], "endianness"), "little");
    require_sha256(&generator["local_oracle_sha256"], "generator.local_oracle_sha256");

    let expected_files: BTreeMap<&str, u64> = EXPECTED_FILES.into_iter().collect();
    let file_records = array(&top["files"], "files");
    assert_eq!(file_records.len(), expected_files.len());

    let mut declared_files = BTreeMap::new();
    for value in file_records {
        let file = object(value, "file record");
        let path = string(&file["path"], "file.path");
        assert_eq!(
            Path::new(path).file_name().and_then(|name| name.to_str()),
            Some(path),
            "fixture paths must be plain basenames"
        );
        assert!(!path.contains(".."), "fixture paths may not traverse");
        let bytes = integer(&file["bytes"], "file.bytes");
        let prior = declared_files.insert(path, bytes);
        assert!(prior.is_none(), "duplicate fixture record for {path}");
        require_sha256(&file["sha256"], "file.sha256");
        assert!(!string(&file["role"], "file.role").is_empty());

        let actual = fs::metadata(fixtures.join(path))
            .unwrap_or_else(|error| panic!("stat fixture {path}: {error}"))
            .len();
        assert_eq!(actual, bytes, "actual length drift for {path}");
    }
    assert_eq!(declared_files, expected_files);

    let live_names: BTreeSet<String> = fs::read_dir(&fixtures)
        .expect("read fixture directory")
        .map(|entry| {
            let entry = entry.expect("read fixture entry");
            assert!(
                entry.file_type().expect("read fixture type").is_file(),
                "fixture directory may contain files only"
            );
            entry
                .file_name()
                .into_string()
                .expect("fixture filenames must be Unicode")
        })
        .collect();
    let expected_names: BTreeSet<String> = EXPECTED_FILES
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .chain(["quant-vectors.json".to_owned()])
        .collect();
    assert_eq!(live_names, expected_names, "fixture inventory drift");

    let decode_vectors = array(&top["decode_vectors"], "decode_vectors");
    assert_eq!(decode_vectors.len(), EXPECTED_DECODE_IDS.len());
    let decode_ids: BTreeSet<&str> = decode_vectors
        .iter()
        .map(|value| {
            require_provenance(value, "decode vector");
            string(&object(value, "decode vector")["id"], "decode.id")
        })
        .collect();
    assert_eq!(
        decode_ids,
        EXPECTED_DECODE_IDS.into_iter().collect::<BTreeSet<_>>()
    );

    let q8_vectors = array(&top["q8_k_vectors"], "q8_k_vectors");
    assert_eq!(q8_vectors.len(), 1);
    require_provenance(&q8_vectors[0], "Q8_K vector");
    let q8 = object(&q8_vectors[0], "Q8_K vector");
    assert_eq!(string(&q8["id"], "q8.id"), "q8-k-activations");
    assert_eq!(integer(&q8["block_elements"], "q8.block_elements"), 256);
    assert_eq!(integer(&q8["block_bytes"], "q8.block_bytes"), 292);
    assert_eq!(integer(&q8["block_count"], "q8.block_count"), 3);
    assert_eq!(array(&q8["block_sums"], "q8.block_sums").len(), 48);

    let dot_vectors = array(&top["dot_vectors"], "dot_vectors");
    assert_eq!(dot_vectors.len(), EXPECTED_DOT_IDS.len());
    let dot_ids: BTreeSet<&str> = dot_vectors
        .iter()
        .map(|value| {
            require_provenance(value, "dot vector");
            let record = object(value, "dot vector");
            assert_eq!(integer(&record["n"], "dot.n"), 768);
            string(&record["id"], "dot.id")
        })
        .collect();
    assert_eq!(dot_ids, EXPECTED_DOT_IDS.into_iter().collect::<BTreeSet<_>>());

    let tables = array(&top["iq_tables"], "iq_tables");
    let table_names: BTreeSet<&str> = tables
        .iter()
        .map(|value| {
            let table = object(value, "IQ table");
            require_sha256(&table["sha256"], "IQ table sha256");
            assert_eq!(
                string(&table["serialization"], "IQ table serialization"),
                "little-endian integers"
            );
            string(&table["name"], "IQ table name")
        })
        .collect();
    assert_eq!(
        table_names,
        ["kmask_iq2xs", "iq2s_grid", "iq3s_grid"].into_iter().collect()
    );

    let license = fs::read_to_string(root.join("vendor/upstream/llama.cpp/LICENSE"))
        .expect("read retained llama.cpp MIT license");
    assert!(license.contains("Copyright (c) 2023-2026 The ggml authors"));
    assert!(license.contains("Permission is hereby granted, free of charge"));

    let pinned = fs::read_to_string(root.join("vendor/upstream/llama.cpp/PINNED.toml"))
        .expect("read llama.cpp pin manifest");
    assert!(pinned.contains("oracle_source_count = 10"));
    assert_eq!(pinned.matches("[[oracle_sources]]").count(), 10);
    for source in EXPECTED_ORACLE_SOURCES {
        assert!(
            pinned.contains(&format!("upstream_path = \"{source}\"")),
            "missing authenticated oracle source {source}"
        );
    }
    assert!(
        !root.join("crates/bridge-core/src/glm.rs").exists(),
        "inactive GLM-only source must not survive the Hy3 oracle cutover"
    );
}
