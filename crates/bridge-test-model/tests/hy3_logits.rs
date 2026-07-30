use std::collections::BTreeMap;

use bridge_kernels_reference::{Hy3BlockScratch, ReferenceExecutionMode};
use bridge_test_model::{
    ReducedHy3Model, BLOCK_COUNT, EXPERT_USED_COUNT, HIDDEN_WIDTH, KV_HEAD_COUNT, QUERY_HEAD_COUNT,
    VOCABULARY_SIZE,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Oracle {
    format: String,
    token_ids: Vec<u32>,
    modes: Vec<ModeOracle>,
}

#[derive(Deserialize)]
struct ModeOracle {
    mode: String,
    steps: Vec<StepOracle>,
}

#[derive(Deserialize)]
struct StepOracle {
    token_id: u32,
    selected_experts: [u32; 2],
    greedy_id: u32,
    hashes: BTreeMap<String, String>,
}

fn f32_hash(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn block_hashes(prefix: &str, scratch: &Hy3BlockScratch, hashes: &mut BTreeMap<String, String>) {
    for (name, values) in [
        ("attention_normalized", scratch.attention_normalized()),
        ("queries", scratch.queries()),
        ("keys", scratch.keys()),
        ("values", scratch.values()),
        ("attention_context", scratch.attention_context()),
        ("attention_delta", scratch.attention_delta()),
        ("attention_residual", scratch.attention_residual()),
        ("ffn_normalized", scratch.ffn_normalized()),
        ("ffn_delta", scratch.ffn_delta()),
    ] {
        hashes.insert(format!("{prefix}.{name}"), f32_hash(values));
    }
}

#[test]
fn reduced_model_preserves_the_required_hy3_structure() {
    let model = ReducedHy3Model::new().unwrap();
    let config = model.config();
    assert_eq!(config.block_count as usize, BLOCK_COUNT);
    assert_eq!(config.embedding_length as usize, HIDDEN_WIDTH);
    assert_eq!(config.attention_head_count as usize, QUERY_HEAD_COUNT);
    assert_eq!(config.attention_kv_head_count as usize, KV_HEAD_COUNT);
    assert_eq!(config.expert_used_count as usize, EXPERT_USED_COUNT);
    assert_eq!(config.vocabulary_size as usize, VOCABULARY_SIZE);
    assert_eq!(config.rope_scaling_type, "yarn");
}

#[test]
fn two_step_teacher_forced_sequence_has_stable_named_outputs() {
    let model = ReducedHy3Model::new().unwrap();
    let oracle: Oracle = serde_json::from_str(include_str!("fixtures/hy3-oracle-v1.json")).unwrap();
    assert_eq!(oracle.format, "lightbridge-reduced-hy3-oracle-v1");
    assert_eq!(oracle.token_ids, [3, 7]);
    for (expected_mode, mode) in oracle.modes.iter().zip([
        ReferenceExecutionMode::DequantF32,
        ReferenceExecutionMode::LlamaQ8K,
    ]) {
        assert_eq!(
            expected_mode.mode,
            match mode {
                ReferenceExecutionMode::DequantF32 => "dequant_f32",
                ReferenceExecutionMode::LlamaQ8K => "llama_q8_k",
                ReferenceExecutionMode::CpuParallelQ8K => "cpu_parallel_q8_k",
                ReferenceExecutionMode::CpuParallelAvxVnni => {
                    "cpu_parallel_avx_vnni_q8_k"
                }
                ReferenceExecutionMode::CpuParallelAvx512Vnni => {
                    "cpu_parallel_avx512_vnni_q8_k"
                }
                ReferenceExecutionMode::CudaQ8K => "cuda_q8_k",
            }
        );
        let mut session = model.new_session().unwrap();
        for expected in &expected_mode.steps {
            let output = model
                .evaluate_token(&mut session, mode, expected.token_id)
                .unwrap();
            let probability_sum: f32 = output.probabilities.iter().sum();
            assert!((probability_sum - 1.0).abs() <= 1.0e-5);
            assert_eq!(output.logits.len(), VOCABULARY_SIZE);
            assert_eq!(output.hidden.len(), HIDDEN_WIDTH);
            assert_eq!(output.selected_experts, expected.selected_experts);
            assert_eq!(output.greedy_id, expected.greedy_id);
            let mut hashes = BTreeMap::new();
            hashes.insert("final.hidden".into(), f32_hash(output.hidden));
            hashes.insert("final.normalized".into(), f32_hash(output.final_normalized));
            hashes.insert("final.logits".into(), f32_hash(output.logits));
            hashes.insert("final.probabilities".into(), f32_hash(output.probabilities));
            block_hashes("block0", session.dense_scratch(), &mut hashes);
            block_hashes("block1", session.moe_scratch(), &mut hashes);
            assert_eq!(hashes, expected.hashes);
        }
        assert_eq!(session.position(), 2);
        assert_eq!(session.cache().stored_tokens(0).unwrap(), 2);
        assert_eq!(session.cache().stored_tokens(1).unwrap(), 2);
    }
}

#[test]
fn execution_modes_are_explicit_and_numerically_distinct() {
    let model = ReducedHy3Model::new().unwrap();
    let mut dequant = model.new_session().unwrap();
    let mut llama = model.new_session().unwrap();
    let dequant_output = model
        .evaluate_token(&mut dequant, ReferenceExecutionMode::DequantF32, 3)
        .unwrap();
    let llama_output = model
        .evaluate_token(&mut llama, ReferenceExecutionMode::LlamaQ8K, 3)
        .unwrap();
    assert_ne!(f32_hash(dequant_output.logits), f32_hash(llama_output.logits));
}

#[test]
fn reset_restores_a_fresh_deterministic_session() {
    let model = ReducedHy3Model::new().unwrap();
    let mut session = model.new_session().unwrap();
    let first_hash = {
        let output = model
            .evaluate_token(&mut session, ReferenceExecutionMode::DequantF32, 5)
            .unwrap();
        f32_hash(output.logits)
    };
    model
        .evaluate_token(&mut session, ReferenceExecutionMode::DequantF32, 9)
        .unwrap();
    session.reset();
    let repeated_hash = {
        let output = model
            .evaluate_token(&mut session, ReferenceExecutionMode::DequantF32, 5)
            .unwrap();
        f32_hash(output.logits)
    };
    assert_eq!(first_hash, repeated_hash);
}

#[test]
fn invalid_token_is_rejected_before_mutating_session_state() {
    let model = ReducedHy3Model::new().unwrap();
    let mut session = model.new_session().unwrap();
    assert!(model
        .evaluate_token(
            &mut session,
            ReferenceExecutionMode::DequantF32,
            VOCABULARY_SIZE as u32,
        )
        .is_err());
    assert_eq!(session.position(), 0);
    assert_eq!(session.cache().stored_tokens(0).unwrap(), 0);
    assert_eq!(session.cache().stored_tokens(1).unwrap(), 0);
}
