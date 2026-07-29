//! A deterministic, small-but-structurally-complete Hy3 model.
//!
//! The fixture preserves quantized projections, GQA, Q/K normalization, YaRN,
//! a dense block, routed/shared MoE, and an untied output head. It exists only
//! for differential tests and never substitutes for selected-model validation.

mod hy3;

pub use hy3::{
    reduced_config, DequantizedTensor, ReducedHy3Model, ReducedHy3Session, ReducedTokenOutput,
    TestModelError, BLOCK_COUNT, CONTEXT_LENGTH, EXPERT_COUNT, EXPERT_USED_COUNT, HEAD_DIMENSION,
    HIDDEN_WIDTH, KV_HEAD_COUNT, QUERY_HEAD_COUNT, VOCABULARY_SIZE,
};
