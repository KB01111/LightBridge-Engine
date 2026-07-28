//! Small format fixtures for crate-local tests and opt-in downstream parser tests.

/// Minimal little-endian GGUF v3 with no metadata or tensors and 32-byte alignment padding.
pub fn minimal_v3() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(32, 0);
    bytes
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::GgufReader;

    #[test]
    fn minimal_fixture_is_parseable() {
        let parsed = GgufReader::new(Cursor::new(super::minimal_v3())).read().unwrap();
        assert_eq!(parsed.data_offset, 32);
    }
}
