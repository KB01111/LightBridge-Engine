use std::io;

use crate::GgufValueType;

pub type Result<T, E = GgufError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("I/O error while inspecting GGUF: {0}")]
    Io(#[source] io::Error),
    #[error("GGUF is truncated while reading {context}")]
    Truncated { context: &'static str },
    #[error("bad GGUF magic bytes {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("{kind} count {actual} exceeds configured limit {limit}")]
    LimitExceeded {
        kind: &'static str,
        limit: u64,
        actual: u64,
    },
    #[error("{kind} count {value} does not fit this platform's address space")]
    CountDoesNotFit { kind: &'static str, value: u64 },
    #[error("allocation failed while reserving {kind}")]
    AllocationFailed { kind: &'static str },
    #[error("invalid UTF-8 in {context}: {source}")]
    InvalidUtf8 {
        context: &'static str,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("invalid GGUF boolean byte {0}")]
    InvalidBoolean(u8),
    #[error("unknown GGUF metadata value type {0}")]
    UnknownValueType(u32),
    #[error("GGUF arrays may not contain arrays")]
    NestedArray,
    #[error("tensor dimension count {0} is outside GGML's supported 1..=4 dimensions")]
    InvalidDimensionCount(u32),
    #[error("configured dimension limit {0} exceeds GGML's maximum of four")]
    InvalidDimensionLimit(u32),
    #[error("duplicate GGUF metadata key {0:?}")]
    DuplicateMetadataKey(String),
    #[error("general.alignment must be a positive u32 power of two, got {0}")]
    InvalidAlignment(u32),
    #[error("general.alignment must have type U32, got {0:?}")]
    AlignmentWrongType(GgufValueType),
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("GGUF data offset {data_offset} exceeds physical file length {file_len}")]
    DataOffsetBeyondFile { data_offset: u64, file_len: u64 },
    #[error(
        "tensor {name:?} has relative payload offset {actual_offset}, expected {expected_offset} for {alignment}-byte alignment"
    )]
    TensorOffsetMismatch {
        name: String,
        actual_offset: u64,
        expected_offset: u64,
        alignment: u64,
    },
    #[error("tensor {name:?} encoded byte length {encoded_bytes} overflows {alignment}-byte padding")]
    TensorPaddingOverflow {
        name: String,
        encoded_bytes: u64,
        alignment: u64,
    },
    #[error("tensor {name:?} padded extent overflows while accumulating {expected_offset} + {padded_bytes}")]
    TensorExtentOverflow {
        name: String,
        expected_offset: u64,
        padded_bytes: u64,
    },
    #[error(
        "GGUF tensor data end overflows while adding data offset {data_offset} and padded extent {padded_extent}"
    )]
    TensorDataEndOverflow { data_offset: u64, padded_extent: u64 },
    #[error(
        "GGUF tensor data section requires padded range {data_offset}..{required_end}, but physical file length is {file_len}"
    )]
    TensorDataSectionTruncated {
        data_offset: u64,
        required_end: u64,
        file_len: u64,
    },
    #[error(transparent)]
    Core(#[from] bridge_core::error::CoreError),
}

impl GgufError {
    pub(crate) fn from_io(error: io::Error, context: &'static str) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Truncated { context }
        } else {
            Self::Io(error)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetadataError {
    #[error("missing GGUF metadata key {key:?}")]
    Missing { key: String },
    #[error("GGUF metadata key {key:?} has type {actual:?}, expected {expected:?}")]
    WrongType {
        key: String,
        expected: GgufValueType,
        actual: GgufValueType,
    },
}
