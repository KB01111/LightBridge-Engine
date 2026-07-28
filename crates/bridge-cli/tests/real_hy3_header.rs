use std::path::PathBuf;

use bridge_cli::build_report;
use bridge_gguf_split::open_set;

const EXPECTED_LOGICAL_BYTES: u64 = 96_019_311_104;
const EXPECTED_DATA_OFFSET: u64 = 5_160_192;
const EXPECTED_TENSOR_BYTES: u64 = 96_014_150_912;

#[test]
fn selected_hy3_real_header_matches_the_ingestion_oracle() {
    let Some(path) = std::env::var_os("BRIDGE_HY3_HEADER").map(PathBuf::from) else {
        println!("skipped: BRIDGE_HY3_HEADER is not set");
        return;
    };

    // This is deliberately the same public production path used by `bridge inspect-gguf`.
    // Neither this test nor the report builder opens a payload handle or reads tensor bytes.
    let set = open_set(&path)
        .unwrap_or_else(|error| panic!("failed to open Hy3 GGUF header {}: {error}", path.display()));
    let report = build_report(&set)
        .unwrap_or_else(|error| panic!("failed to validate Hy3 GGUF header {}: {error}", path.display()));

    assert_eq!(report.files.len(), 1);
    assert_eq!(report.files[0].version, 3);
    assert_eq!(report.files[0].endianness, "little");
    assert_eq!(report.files[0].metadata_count, 45);
    assert_eq!(report.files[0].alignment, 32);
    assert_eq!(report.files[0].logical_size, EXPECTED_LOGICAL_BYTES);
    assert_eq!(report.files[0].data_offset, EXPECTED_DATA_OFFSET);

    assert_eq!(report.gguf.version, 3);
    assert_eq!(report.gguf.authoritative_metadata_count, 45);
    assert_eq!(report.gguf.tensor_count, 1_278);
    assert_eq!(report.gguf.encoded_tensor_bytes, EXPECTED_TENSOR_BYTES);
    assert_eq!(report.tensors.total.encoded_bytes, EXPECTED_TENSOR_BYTES);

    assert_eq!(report.types["F32"].count, 479);
    assert_eq!(report.types["F32"].encoded_bytes, 251_292_928);
    assert_eq!(report.types["IQ2_S"].count, 627);
    assert_eq!(report.types["IQ2_S"].encoded_bytes, 91_238_285_312);
    assert_eq!(report.types["IQ3_S"].count, 91);
    assert_eq!(report.types["IQ3_S"].encoded_bytes, 3_995_566_080);
    assert_eq!(report.types["Q4_K"].count, 80);
    assert_eq!(report.types["Q4_K"].encoded_bytes, 188_743_680);
    assert_eq!(report.types["Q5_K"].count, 1);
    assert_eq!(report.types["Q5_K"].encoded_bytes, 340_262_912);

    assert_eq!(report.hy3.block_count, 80);
    assert_eq!(report.hy3.expert_count, 192);
    assert_eq!(report.hy3.expert_used_count, 8);
    assert!(!report.hy3.has_mtp);
}
