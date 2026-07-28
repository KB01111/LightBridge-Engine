//! Opt-in construction helpers for downstream validation tests.

use std::path::PathBuf;

use bridge_gguf::{GgufFile, MetadataError};

use crate::{directory, discovery, GgufSet, GgufShard, SplitError};

/// Assemble one already-parsed shard through the real tensor-directory builder.
///
/// This bypasses filesystem discovery only; descriptor ranges, alignment, duplicate names,
/// and optional aggregate counts still pass through the production directory checks.
pub fn from_file(parsed: GgufFile) -> Result<GgufSet, SplitError> {
    let path = PathBuf::from("<in-memory-test-shard>");
    discovery::validate_optional_split_metadata(&path, &parsed, 0, 1)?;
    let files = vec![GgufShard {
        path,
        parsed,
        ordinal: 0,
        count: 1,
    }];
    let tensors = directory::build(&files)?;
    Ok(GgufSet { files, tensors })
}

/// Assemble already-parsed numbered shards through production split and directory checks.
pub fn from_files(parsed_files: Vec<GgufFile>) -> Result<GgufSet, SplitError> {
    if parsed_files.len() == 1 {
        return from_file(parsed_files.into_iter().next().expect("length checked"));
    }
    let count = u32::try_from(parsed_files.len())
        .map_err(|_| SplitError::ArithmeticOverflow("test fixture shard count"))?;
    let mut files = Vec::new();
    files
        .try_reserve_exact(parsed_files.len())
        .map_err(|_| SplitError::AllocationFailed("test fixture shards"))?;
    for (index, parsed) in parsed_files.into_iter().enumerate() {
        let ordinal =
            u32::try_from(index).map_err(|_| SplitError::ArithmeticOverflow("test fixture shard ordinal"))?;
        let path = PathBuf::from(format!("<in-memory-test-shard-{ordinal}>"));
        discovery::validate_required_split_metadata(&path, &parsed, ordinal, count)?;
        files.push(GgufShard {
            path,
            parsed,
            ordinal,
            count,
        });
    }
    let tensors = directory::build(&files)?;
    Ok(GgufSet { files, tensors })
}

/// Assemble explicitly named parsed shards after canonicalizing them by declared ordinal.
///
/// This preserves synthetic shard identities for downstream tests while retaining the
/// production split-metadata and tensor-directory validation boundaries.
pub fn from_explicit_files(mut parsed_files: Vec<(PathBuf, GgufFile)>) -> Result<GgufSet, SplitError> {
    if parsed_files.len() == 1 {
        let (path, parsed) = parsed_files.pop().expect("length checked");
        discovery::validate_optional_split_metadata(&path, &parsed, 0, 1)?;
        let files = vec![GgufShard {
            path,
            parsed,
            ordinal: 0,
            count: 1,
        }];
        let tensors = directory::build(&files)?;
        return Ok(GgufSet { files, tensors });
    }

    let count = u32::try_from(parsed_files.len())
        .map_err(|_| SplitError::ArithmeticOverflow("explicit test fixture shard count"))?;
    for (path, parsed) in &parsed_files {
        parsed.get_u16("split.no").map_err(|error| match error {
            MetadataError::Missing { .. } => SplitError::MissingSplitMetadata {
                path: path.clone(),
                key: "split.no",
            },
            other => SplitError::Metadata(other),
        })?;
    }
    parsed_files.sort_by_key(|(_, parsed)| {
        parsed
            .get_u16("split.no")
            .expect("split.no type validated before sorting")
    });

    let mut files = Vec::new();
    files
        .try_reserve_exact(parsed_files.len())
        .map_err(|_| SplitError::AllocationFailed("explicit test fixture shards"))?;
    for (index, (path, parsed)) in parsed_files.into_iter().enumerate() {
        let ordinal = u32::try_from(index)
            .map_err(|_| SplitError::ArithmeticOverflow("explicit test fixture shard ordinal"))?;
        discovery::validate_required_split_metadata(&path, &parsed, ordinal, count)?;
        files.push(GgufShard {
            path,
            parsed,
            ordinal,
            count,
        });
    }
    let tensors = directory::build(&files)?;
    Ok(GgufSet { files, tensors })
}

#[cfg(test)]
mod tests {
    use bridge_gguf::{Endianness, GgufValue};

    use super::*;

    fn file_with(key: &str, value: GgufValue) -> GgufFile {
        GgufFile {
            version: 3,
            endianness: Endianness::Little,
            metadata: vec![(key.to_owned(), value)],
            tensors: Vec::new(),
            alignment: 32,
            data_offset: 0,
            file_len: 0,
        }
    }

    fn numbered_file(ordinal: u16, count: u16) -> GgufFile {
        GgufFile {
            version: 3,
            endianness: Endianness::Little,
            metadata: vec![
                ("split.no".to_owned(), GgufValue::U16(ordinal)),
                ("split.count".to_owned(), GgufValue::U16(count)),
                ("split.tensors.count".to_owned(), GgufValue::I32(0)),
            ],
            tensors: Vec::new(),
            alignment: 32,
            data_offset: 0,
            file_len: 0,
        }
    }

    #[test]
    fn from_file_rejects_a_contradictory_optional_split_ordinal() {
        let error = from_file(file_with("split.no", GgufValue::U16(1))).unwrap_err();

        assert!(matches!(
            error,
            SplitError::SplitMetadataDisagreement { key: "split.no", .. }
        ));
    }

    #[test]
    fn from_file_rejects_a_contradictory_optional_split_count() {
        let error = from_file(file_with("split.count", GgufValue::U16(2))).unwrap_err();

        assert!(matches!(
            error,
            SplitError::SplitMetadataDisagreement {
                key: "split.count",
                ..
            }
        ));
    }

    #[test]
    fn from_explicit_files_orders_shards_by_declared_ordinal_and_preserves_paths() {
        let set = from_explicit_files(vec![
            (PathBuf::from("second.gguf"), numbered_file(1, 2)),
            (PathBuf::from("first.gguf"), numbered_file(0, 2)),
        ])
        .unwrap();

        assert_eq!(set.files()[0].path(), PathBuf::from("first.gguf"));
        assert_eq!(set.files()[0].ordinal(), 0);
        assert_eq!(set.files()[1].path(), PathBuf::from("second.gguf"));
        assert_eq!(set.files()[1].ordinal(), 1);
    }
}
