use std::collections::BTreeMap;

use bridge_gguf::MetadataError;

use crate::{GgufShard, SplitError, TensorDirectory, TensorLocation};

pub(crate) fn build(files: &[GgufShard]) -> Result<TensorDirectory, SplitError> {
    validate_common_fields(files)?;

    let mut ordered = Vec::new();
    let mut by_name = BTreeMap::new();
    let mut declared_count = None;

    for (shard_index, shard) in files.iter().enumerate() {
        if let Some(count) = optional_aggregate_count(shard)? {
            match declared_count {
                Some(previous) if previous != count => {
                    return Err(SplitError::AggregateTensorCountDisagreement)
                }
                None => declared_count = Some(count),
                Some(_) => {}
            }
        }
        for descriptor in &shard.parsed.tensors {
            let offset = descriptor.relative_offset();
            if offset % shard.parsed.alignment != 0 {
                return Err(SplitError::UnalignedTensorOffset {
                    path: shard.path.clone(),
                    name: descriptor.name().to_owned(),
                    offset,
                    alignment: shard.parsed.alignment,
                });
            }
            let absolute_range =
                descriptor.checked_absolute_range(shard.parsed.data_offset, shard.parsed.file_len)?;
            let name = descriptor.name().to_owned();
            if by_name.contains_key(&name) {
                return Err(SplitError::DuplicateTensorName(name));
            }
            let location_index = ordered.len();
            by_name.insert(name, location_index);
            ordered.push(TensorLocation {
                shard_index,
                descriptor: descriptor.clone(),
                absolute_range,
            });
        }
    }

    if let Some(declared_count) = declared_count {
        let actual = i32::try_from(ordered.len())
            .map_err(|_| SplitError::ArithmeticOverflow("aggregate tensor count"))?;
        if declared_count != actual {
            return Err(SplitError::AggregateTensorDirectoryDisagreement);
        }
    }
    Ok(TensorDirectory { ordered, by_name })
}

fn validate_common_fields(files: &[GgufShard]) -> Result<(), SplitError> {
    let Some(primary) = files.first() else {
        return Ok(());
    };
    for shard in &files[1..] {
        if shard.parsed.version != primary.parsed.version {
            return Err(SplitError::HeterogeneousVersion {
                path: shard.path.clone(),
                expected: primary.parsed.version,
                actual: shard.parsed.version,
            });
        }
        if shard.parsed.endianness != primary.parsed.endianness {
            return Err(SplitError::HeterogeneousEndianness {
                path: shard.path.clone(),
                expected: primary.parsed.endianness,
                actual: shard.parsed.endianness,
            });
        }
        if shard.parsed.alignment != primary.parsed.alignment {
            return Err(SplitError::HeterogeneousAlignment {
                path: shard.path.clone(),
                expected: primary.parsed.alignment,
                actual: shard.parsed.alignment,
            });
        }
    }
    Ok(())
}

fn optional_aggregate_count(shard: &GgufShard) -> Result<Option<i32>, SplitError> {
    match shard.parsed.get_i32("split.tensors.count") {
        Ok(value) if value >= 0 => Ok(Some(value)),
        Ok(_) => Err(SplitError::NegativeAggregateTensorCount(shard.path.clone())),
        Err(MetadataError::Missing { .. }) => Ok(None),
        Err(error) => Err(SplitError::Metadata(error)),
    }
}
