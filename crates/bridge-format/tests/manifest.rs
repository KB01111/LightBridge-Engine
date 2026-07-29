use bridge_format::{
    align_up, ExpertKey, ExpertLayout, ExpertRecord, FormatError, Segment, SidecarFileIdentity,
    SidecarHeader, SidecarManifest, SourceFileIdentity, QUANT_ABI_VERSION, SIDECAR_FORMAT,
    SIDECAR_FORMAT_VERSION, SIDECAR_HEADER_BYTES,
};

fn sha() -> String {
    "a".repeat(64)
}

fn manifest(layout: ExpertLayout) -> SidecarManifest {
    let offset = 64;
    let (up_offset, down_offset) = match layout {
        ExpertLayout::Sequential => (96, 128),
        ExpertLayout::FusedGateUp => (80, 128),
    };
    SidecarManifest {
        format: SIDECAR_FORMAT.into(),
        format_version: SIDECAR_FORMAT_VERSION,
        engine_version: "0.1.0".into(),
        quant_abi_version: QUANT_ABI_VERSION,
        alignment: 32,
        source_files: vec![SourceFileIdentity {
            ordinal: 0,
            path: "model.gguf".into(),
            length: 1024,
            sha256: sha(),
        }],
        tensor_directory_sha256: sha(),
        sidecar: SidecarFileIdentity {
            length: 160,
            sha256: sha(),
        },
        layout,
        records: vec![ExpertRecord {
            key: ExpertKey { layer: 1, expert: 0 },
            offset,
            length: 96,
            gate: Segment {
                offset,
                length: 16,
                ggml_type: "IQ2_S".into(),
            },
            up: Segment {
                offset: up_offset,
                length: 16,
                ggml_type: "IQ2_S".into(),
            },
            down: Segment {
                offset: down_offset,
                length: 16,
                ggml_type: "IQ3_S".into(),
            },
        }],
    }
}

#[test]
fn validates_both_supported_layouts() {
    manifest(ExpertLayout::Sequential).validate().unwrap();
    manifest(ExpertLayout::FusedGateUp).validate().unwrap();
}

#[test]
fn header_round_trips_and_rejects_reserved_bytes() {
    let header = SidecarHeader::new(ExpertLayout::FusedGateUp, 4096, 15_168);
    assert_eq!(SidecarHeader::decode(&header.encode()).unwrap(), header);
    let mut bytes = header.encode();
    bytes[63] = 1;
    assert!(matches!(
        SidecarHeader::decode(&bytes),
        Err(FormatError::NonZeroReservedHeader)
    ));
}

#[test]
fn manifest_rejects_overlap_and_bad_hashes() {
    let mut invalid = manifest(ExpertLayout::Sequential);
    invalid.records[0].up.offset = 64;
    assert!(matches!(
        invalid.validate(),
        Err(FormatError::OverlappingSegments(_))
    ));

    let mut invalid = manifest(ExpertLayout::Sequential);
    invalid.tensor_directory_sha256 = "ABC".into();
    assert!(matches!(
        invalid.validate(),
        Err(FormatError::InvalidSha256 { .. })
    ));
}

#[test]
fn checked_alignment_rejects_invalid_values_and_overflow() {
    assert_eq!(align_up(SIDECAR_HEADER_BYTES as u64, 4096).unwrap(), 4096);
    assert!(matches!(align_up(1, 3), Err(FormatError::InvalidAlignment(3))));
    assert!(matches!(
        align_up(u64::MAX, 4096),
        Err(FormatError::ArithmeticOverflow)
    ));
}
