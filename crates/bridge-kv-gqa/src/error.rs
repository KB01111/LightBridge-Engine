use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvError {
    #[error("KV parameter {field} must be greater than zero")]
    ZeroParameter { field: &'static str },
    #[error("KV arithmetic overflow while computing {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("KV allocation failed for {field} with {elements} F32 elements")]
    AllocationFailed { field: &'static str, elements: usize },
    #[error("KV layer index {layer} is outside layer count {layer_count}")]
    LayerOutOfRange { layer: usize, layer_count: usize },
    #[error("KV head index {head} is outside head count {head_count}")]
    HeadOutOfRange { head: usize, head_count: usize },
    #[error("KV token index {token} is outside stored token count {stored_tokens}")]
    TokenOutOfRange { token: usize, stored_tokens: usize },
    #[error("KV {field} has actual length {actual}, expected {expected}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "KV capacity exhausted for layer {layer}: stored {stored}, appending {additional}, capacity {capacity}"
    )]
    CapacityExhausted {
        layer: usize,
        stored: usize,
        additional: usize,
        capacity: usize,
    },
    #[error("cannot rewind KV layer {layer} to {requested} tokens because only {stored} are stored")]
    RewindBeyondStored {
        layer: usize,
        requested: usize,
        stored: usize,
    },
    #[error("KV {field} index {index} is non-finite (F32 bits {bits:#010x})")]
    NonFiniteValue {
        field: &'static str,
        index: usize,
        bits: u32,
    },
    #[error("KV snapshot limit must be greater than zero")]
    ZeroSnapshotLimit,
    #[error("KV snapshot is {actual} bytes, maximum is {maximum}")]
    SnapshotTooLarge { actual: usize, maximum: usize },
    #[error("KV snapshot is truncated at byte {offset}; need {needed} more bytes")]
    SnapshotTruncated { offset: usize, needed: usize },
    #[error("KV snapshot has an invalid format marker")]
    SnapshotMagic,
    #[error("KV snapshot version {actual} is unsupported; expected {expected}")]
    SnapshotVersion { expected: u32, actual: u32 },
    #[error("KV snapshot checksum does not match its payload")]
    SnapshotChecksum,
    #[error("KV snapshot belongs to a different model")]
    SnapshotBinding,
    #[error("KV snapshot {field} is {actual}, expected {expected}")]
    SnapshotConfiguration {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("KV snapshot layer {layer} stores {actual} tokens, expected uniform length {expected}")]
    SnapshotLayerLength {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error("KV snapshot has {actual} trailing payload bytes")]
    SnapshotTrailingBytes { actual: usize },
}

pub(crate) type Result<T> = std::result::Result<T, KvError>;
