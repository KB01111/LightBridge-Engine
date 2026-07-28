//! Errors for safe GGUF ingestion primitives.

pub type Result<T, E = CoreError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("unknown ggml type discriminant {0}")]
    UnknownGgmlType(u32),
    #[error("{ty} requires a leading dimension divisible by {block}, got {ne}")]
    NotBlockAligned { ty: &'static str, ne: u64, block: u64 },
    #[error("tensor rank {0} is outside GGML's supported 1..=4 dimensions")]
    InvalidTensorRank(u32),
    #[error("tensor dimension {dimension} is zero")]
    ZeroTensorDimension { dimension: usize },
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("invalid allocation layout: size {size}, alignment {align}")]
    InvalidAllocationLayout { size: usize, align: usize },
    #[error("allocation failed: size {size}, alignment {align}")]
    AllocationFailed { size: usize, align: usize },
    #[error("tensor byte range {start}..{end} exceeds file length {file_len}")]
    TensorOutOfBounds { start: u64, end: u64, file_len: u64 },
}

/// Compatibility alias for code which still imports the former error name.
pub type BridgeError = CoreError;
