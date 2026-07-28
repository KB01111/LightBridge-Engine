use std::path::{Path, PathBuf};

use bridge_gguf::{open, GgufFile, MetadataError};

use crate::{GgufShard, SplitError};

pub(crate) fn discover(entry: &Path) -> Result<Vec<GgufShard>, SplitError> {
    if !entry.is_file() {
        return Err(SplitError::NotAFile(entry.to_path_buf()));
    }
    let lexical_parent = entry
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()?;
    let entry = entry.canonicalize()?;
    if entry.parent() != Some(lexical_parent.as_path()) {
        return Err(SplitError::ShardEscapesParent(entry));
    }

    match NumberedName::parse(&entry)? {
        Some(numbered) => discover_numbered(&lexical_parent, numbered),
        None => discover_single(entry),
    }
}

fn discover_single(path: PathBuf) -> Result<Vec<GgufShard>, SplitError> {
    let parsed = open(&path)?;
    validate_optional_split_metadata(&path, &parsed, 0, 1)?;
    Ok(vec![GgufShard {
        path,
        parsed,
        ordinal: 0,
        count: 1,
    }])
}

fn discover_numbered(parent: &Path, numbered: NumberedName) -> Result<Vec<GgufShard>, SplitError> {
    let mut files = Vec::new();
    for one_based_ordinal in 1..=numbered.total {
        let filename = numbered.filename(one_based_ordinal);
        let expected = parent.join(filename);
        if !expected.is_file() {
            return Err(SplitError::MissingShard(expected));
        }
        let path = expected.canonicalize()?;
        if path.parent() != Some(parent) {
            return Err(SplitError::ShardEscapesParent(path));
        }
        let parsed = open(&path)?;
        let ordinal = one_based_ordinal
            .checked_sub(1)
            .ok_or(SplitError::ArithmeticOverflow("zero-based shard ordinal"))?;
        validate_required_split_metadata(&path, &parsed, ordinal, numbered.total)?;
        files.push(GgufShard {
            path,
            parsed,
            ordinal,
            count: numbered.total,
        });
    }
    Ok(files)
}

pub(crate) fn validate_required_split_metadata(
    path: &Path,
    parsed: &GgufFile,
    ordinal: u32,
    count: u32,
) -> Result<(), SplitError> {
    let metadata_ordinal = required_u16(path, parsed, "split.no")?;
    let metadata_count = required_u16(path, parsed, "split.count")?;
    if count == 1 {
        return Err(SplitError::NumberedSingleShard(path.to_path_buf()));
    }
    if u32::from(metadata_ordinal) != ordinal {
        return Err(SplitError::SplitMetadataDisagreement {
            path: path.to_path_buf(),
            key: "split.no",
        });
    }
    if u32::from(metadata_count) != count {
        return Err(SplitError::SplitMetadataDisagreement {
            path: path.to_path_buf(),
            key: "split.count",
        });
    }
    Ok(())
}

pub(crate) fn validate_optional_split_metadata(
    path: &Path,
    parsed: &GgufFile,
    ordinal: u32,
    count: u32,
) -> Result<(), SplitError> {
    if let Some(metadata_ordinal) = optional_u16(parsed, "split.no")? {
        if u32::from(metadata_ordinal) != ordinal {
            return Err(SplitError::SplitMetadataDisagreement {
                path: path.to_path_buf(),
                key: "split.no",
            });
        }
    }
    if let Some(metadata_count) = optional_u16(parsed, "split.count")? {
        if u32::from(metadata_count) != count {
            return Err(SplitError::SplitMetadataDisagreement {
                path: path.to_path_buf(),
                key: "split.count",
            });
        }
    }
    Ok(())
}

fn required_u16(path: &Path, parsed: &GgufFile, key: &'static str) -> Result<u16, SplitError> {
    parsed.get_u16(key).map_err(|error| match error {
        MetadataError::Missing { .. } => SplitError::MissingSplitMetadata {
            path: path.to_path_buf(),
            key,
        },
        other => SplitError::Metadata(other),
    })
}

fn optional_u16(parsed: &GgufFile, key: &str) -> Result<Option<u16>, SplitError> {
    match parsed.get_u16(key) {
        Ok(value) => Ok(Some(value)),
        Err(MetadataError::Missing { .. }) => Ok(None),
        Err(error) => Err(SplitError::Metadata(error)),
    }
}

#[derive(Debug)]
struct NumberedName {
    prefix: String,
    total: u32,
}

impl NumberedName {
    fn parse(path: &Path) -> Result<Option<Self>, SplitError> {
        let filename = match path.file_name().and_then(|name| name.to_str()) {
            Some(filename) => filename,
            None => return Ok(None),
        };
        let Some(stem) = filename.strip_suffix(".gguf") else {
            return Ok(None);
        };
        let Some((before_total, total_text)) = stem.rsplit_once("-of-") else {
            return Ok(None);
        };
        let Some((prefix, ordinal_text)) = before_total.rsplit_once('-') else {
            return Ok(None);
        };
        if ordinal_text.len() != 5
            || total_text.len() != 5
            || !ordinal_text.bytes().all(|byte| byte.is_ascii_digit())
            || !total_text.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Ok(None);
        }
        let ordinal = ordinal_text
            .parse::<u32>()
            .map_err(|_| SplitError::InvalidNumberedFilename(filename.to_owned()))?;
        let total = total_text
            .parse::<u32>()
            .map_err(|_| SplitError::InvalidNumberedFilename(filename.to_owned()))?;
        if ordinal == 0 || total == 0 || ordinal > total || total > u32::from(u16::MAX) {
            return Err(SplitError::InvalidNumberedFilename(filename.to_owned()));
        }
        Ok(Some(Self {
            prefix: prefix.to_owned(),
            total,
        }))
    }

    fn filename(&self, ordinal: u32) -> String {
        format!(
            "{}-{:0width$}-of-{:0width$}.gguf",
            self.prefix,
            ordinal,
            self.total,
            width = 5
        )
    }
}
