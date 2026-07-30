//! Safe packed GGML validation, exact scalar decoding, Q8_K quantization, and CPU dot dispatch.

mod error;
mod iq2_s;
mod iq3_s;
mod k_quants;
mod q8_k;
mod tables;

pub use bridge_core::ggml_type::GgmlType;
pub use error::QuantError;
pub use k_quants::{
    decode_block_into, decode_f32_block_into, decode_iq2_s_block_into, decode_iq3_s_block_into,
    decode_q4_k_block_into, decode_q5_k_block_into, decode_row_into, layout, QuantLayout,
};
pub use q8_k::{
    iq2_s_grid_table, iq3_s_grid_table, quantize_row_q8_k_into, validate_vec_dot_q8_k, vec_dot_iq2_s_q8_k,
    vec_dot_iq3_s_q8_k, vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q8_k, vec_dot_q8_k_cpu,
    vec_dot_q8_k_cpu_backend, CpuDotBackend, ValidatedQ8KMatrix, Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS,
};
