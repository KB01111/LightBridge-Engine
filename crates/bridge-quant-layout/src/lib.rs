//! Safe packed GGML layout validation and scalar reference decoding.

mod error;
mod k_quants;

pub use bridge_core::ggml_type::GgmlType;
pub use error::QuantError;
pub use k_quants::{
    decode_block_into, decode_f32_block_into, decode_q4_k_block_into, decode_q5_k_block_into,
    decode_row_into, layout, QuantLayout,
};
