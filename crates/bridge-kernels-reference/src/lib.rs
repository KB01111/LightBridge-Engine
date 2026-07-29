//! Allocation-explicit scalar reference kernels for checked packed tensors.

mod activation;
mod attention;
mod error;
mod gemv;
mod layer;
mod matrix;
mod moe;
mod norm;
mod rope;
mod softmax;

pub use activation::{expert_swiglu_accumulate_into, swiglu_project_into, SwiGluExpert, SwiGluScratch};
pub use attention::{causal_gqa_attention_into, AttentionInput, AttentionScratch};
pub use error::KernelError;
pub use gemv::{
    gemv_accumulate_scaled_into, gemv_cpu_parallel_q8k_into, gemv_dequant_f32_into, gemv_into,
    gemv_llama_q8k_into, required_q8_k_bytes, ReferenceExecutionMode,
};
pub use layer::{
    hy3_block_forward_token, hy3_moe_finish_token, hy3_moe_route_token, Hy3AttentionWeights,
    Hy3BlockExecution, Hy3BlockScratch, Hy3BlockWeights, Hy3FeedForwardWeights, Hy3MoeWeights,
    Hy3StreamingMoeWeights,
};
pub use matrix::{EncodedTensorView, PackedMatrix, PayloadEndian};
pub use moe::{moe_routed_by_id_into, moe_selected_into, RoutedMoeSelection, SelectedExpert};
pub use norm::{
    residual_add_in_place, weighted_head_rms_norm_in_place, weighted_rms_norm_in_place,
    weighted_rms_norm_into,
};
pub use rope::{apply_neox_yarn_rope_in_place, Hy3RopeParams};
pub use softmax::{causal_softmax_into, softmax_into};
