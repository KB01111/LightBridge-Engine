use bridge_core::ggml_type::GgmlType;
use bridge_kv_gqa::KvError;
use bridge_quant_layout::QuantError;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KernelError {
    #[error("tensor payload is big-endian; packed execution requires little-endian bytes")]
    BigEndianPayload,
    #[error("tensor rank has actual value {actual}, expected {expected}")]
    TensorRank { expected: usize, actual: usize },
    #[error("tensor dimension {dimension} is zero")]
    ZeroDimension { dimension: usize },
    #[error("tensor shape rank {actual} exceeds the supported maximum {maximum}")]
    ShapeRankTooLarge { maximum: usize, actual: usize },
    #[error("physical type {ty:?} is unsupported by the scalar packed matrix")]
    UnsupportedType { ty: GgmlType },
    #[error("checked arithmetic overflow while computing {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("tensor encoded length has actual value {actual}, expected {expected}")]
    EncodedLengthMismatch { expected: usize, actual: usize },
    #[error("{field} has actual length {actual}, expected {expected}")]
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} scratch has actual length {actual}, expected at least {required}")]
    ScratchTooSmall {
        field: &'static str,
        required: usize,
        actual: usize,
    },
    #[error("{field} index {index} is non-finite (F32 bits {bits:#010x})")]
    NonFiniteValue {
        field: &'static str,
        index: usize,
        bits: u32,
    },
    #[error("routed experts are out of order: expert {current} follows expert {previous}")]
    RoutedExpertOrder { previous: u32, current: u32 },
    #[error("duplicate routed expert ID {expert_id}")]
    DuplicateRoutedExpert { expert_id: u32 },
    #[error("invalid parameter {field}: {reason}")]
    InvalidParameter {
        field: &'static str,
        reason: &'static str,
    },
    #[error("allocation failed while reserving {requested} entries for {context}")]
    AllocationFailed { context: &'static str, requested: usize },
    #[error("Hy3 routing failed: {message}")]
    Routing { message: String },
    #[error("position {position} is outside configured context length {context_length}")]
    PositionOutOfRange { position: u64, context_length: u64 },
    #[error(transparent)]
    Quant(#[from] QuantError),
    #[error(transparent)]
    Kv(#[from] KvError),
}

pub(crate) type Result<T> = std::result::Result<T, KernelError>;
