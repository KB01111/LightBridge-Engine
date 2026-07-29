use bridge_core::ggml_type::GgmlType;
use thiserror::Error;

/// A packed-layout validation or scalar-decoding failure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// The GGML type is outside this crate's current decoding boundary.
    #[error("GGML type {ty:?} is not supported by the packed reference decoder")]
    UnsupportedType { ty: GgmlType },

    /// A row decode was requested with no logical elements.
    #[error("a packed row must contain at least one logical element")]
    ZeroLogicalElements,

    /// A row's logical element count is not an exact number of blocks.
    #[error(
        "{ty:?} logical element count {logical_elements} is not divisible by block size {block_elements}"
    )]
    LogicalElementsNotDivisible {
        ty: GgmlType,
        logical_elements: usize,
        block_elements: usize,
    },

    /// Checked length arithmetic overflowed.
    #[error("packed length arithmetic overflow while computing {operation}")]
    ArithmeticOverflow { operation: &'static str },

    /// A core ABI size could not be represented by the host pointer width.
    #[error("core ABI value {value} for {field} cannot be represented as usize")]
    IntegerConversionOverflow { field: &'static str, value: u64 },

    /// The encoded byte slice is not exactly the required size.
    #[error("{ty:?} encoded length mismatch: expected {expected} bytes, received {actual} bytes")]
    EncodedLengthMismatch {
        ty: GgmlType,
        expected: usize,
        actual: usize,
    },

    /// The caller-owned output slice is not exactly the required size.
    #[error("{ty:?} output length mismatch: expected {expected} lanes, received {actual} lanes")]
    OutputLengthMismatch {
        ty: GgmlType,
        expected: usize,
        actual: usize,
    },

    /// A packed binary16 scale is infinite or NaN.
    #[error("{ty:?} block {block_index} has non-finite {field} binary16 scale bits {bits:#06x}")]
    NonFiniteScale {
        ty: GgmlType,
        block_index: usize,
        field: &'static str,
        bits: u16,
    },

    /// A caller-provided activation contains an infinite or NaN value.
    #[error("Q8_K activation lane {index} is non-finite (F32 bits {bits:#010x})")]
    NonFiniteActivation { index: usize, bits: u32 },

    /// An encoded Q8_K activation block has an infinite or NaN scale.
    #[error("Q8_K block {block_index} has non-finite F32 scale bits {bits:#010x}")]
    NonFiniteQ8Scale { block_index: usize, bits: u32 },
}

pub(crate) type Result<T> = std::result::Result<T, QuantError>;
