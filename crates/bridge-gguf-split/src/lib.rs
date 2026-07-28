mod directory;
mod discovery;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use bridge_gguf::{Endianness, GgufFile, MetadataError, TensorDesc};

#[derive(Debug)]
pub struct GgufSet {
    files: Vec<GgufShard>,
    tensors: TensorDirectory,
}

impl GgufSet {
    pub fn files(&self) -> &[GgufShard] {
        &self.files
    }

    pub fn tensors(&self) -> &TensorDirectory {
        &self.tensors
    }
}

#[derive(Debug)]
pub struct GgufShard {
    path: PathBuf,
    parsed: GgufFile,
    ordinal: u32,
    count: u32,
}

impl GgufShard {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn parsed(&self) -> &GgufFile {
        &self.parsed
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLocation {
    shard_index: usize,
    descriptor: TensorDesc,
    absolute_range: Range<u64>,
}

impl TensorLocation {
    pub const fn shard_index(&self) -> usize {
        self.shard_index
    }

    pub fn descriptor(&self) -> &TensorDesc {
        &self.descriptor
    }

    pub fn absolute_range(&self) -> &Range<u64> {
        &self.absolute_range
    }
}

#[derive(Debug)]
pub struct TensorDirectory {
    ordered: Vec<TensorLocation>,
    by_name: BTreeMap<String, usize>,
}

impl TensorDirectory {
    pub fn ordered(&self) -> &[TensorLocation] {
        &self.ordered
    }

    pub fn get(&self, name: &str) -> Option<&TensorLocation> {
        self.by_name.get(name).and_then(|&index| self.ordered.get(index))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SplitError {
    #[error("I/O error while discovering GGUF shards: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gguf(#[from] bridge_gguf::GgufError),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("input path is not a regular file: {0:?}")]
    NotAFile(PathBuf),
    #[error("invalid numbered GGUF filename {0:?}")]
    InvalidNumberedFilename(String),
    #[error("numbered GGUF shard {path:?} is missing required metadata {key:?}")]
    MissingSplitMetadata { path: PathBuf, key: &'static str },
    #[error("numbered GGUF shard {path:?} metadata {key:?} does not match its filename")]
    SplitMetadataDisagreement { path: PathBuf, key: &'static str },
    #[error("numbered GGUF shard {0:?} declares only one shard")]
    NumberedSingleShard(PathBuf),
    #[error("expected GGUF shard is missing or is not a regular file: {0:?}")]
    MissingShard(PathBuf),
    #[error("canonical GGUF shard escapes its input parent directory: {0:?}")]
    ShardEscapesParent(PathBuf),
    #[error("duplicate tensor name {0:?} across GGUF shards")]
    DuplicateTensorName(String),
    #[error("GGUF shard {path:?} has version {actual}, expected common version {expected}")]
    HeterogeneousVersion {
        path: PathBuf,
        expected: u32,
        actual: u32,
    },
    #[error("GGUF shard {path:?} has endianness {actual:?}, expected common endianness {expected:?}")]
    HeterogeneousEndianness {
        path: PathBuf,
        expected: Endianness,
        actual: Endianness,
    },
    #[error("GGUF shard {path:?} has alignment {actual}, expected common alignment {expected}")]
    HeterogeneousAlignment {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("tensor {name:?} in {path:?} has offset {offset} not aligned to {alignment}")]
    UnalignedTensorOffset {
        path: PathBuf,
        name: String,
        offset: u64,
        alignment: u64,
    },
    #[error("aggregate split.tensors.count is negative in {0:?}")]
    NegativeAggregateTensorCount(PathBuf),
    #[error("split.tensors.count disagrees between GGUF shards")]
    AggregateTensorCountDisagreement,
    #[error("split.tensors.count does not match the aggregate tensor directory")]
    AggregateTensorDirectoryDisagreement,
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("allocation failed while reserving {0}")]
    AllocationFailed(&'static str),
    #[error(transparent)]
    Core(#[from] bridge_core::error::CoreError),
}

pub fn open_set(entry: impl AsRef<Path>) -> Result<GgufSet, SplitError> {
    let files = discovery::discover(entry.as_ref())?;
    let tensors = directory::build(&files)?;
    Ok(GgufSet { files, tensors })
}
