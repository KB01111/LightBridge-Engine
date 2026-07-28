//! Bounded, payload-isolated inspection of GGUF v2 and v3 files.

mod error;
mod reader;
mod value;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use bridge_core::tensor::TensorDesc;
pub use error::{GgufError, MetadataError};
pub use reader::{open, Endianness, GgufFile, GgufReader, ReaderLimits};
pub use value::{GgufArray, GgufValue, GgufValueType};
