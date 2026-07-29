use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bridge_test_model::{reduced_config, ReducedHy3Model};
use serde::Serialize;
use sha2::{Digest, Sha256};

const TRANSFORMERS_COMMIT: &str = "3e80155a968c1080f11b2710e8b31741ac5ab0ed";

#[derive(Serialize)]
struct Manifest {
    format: &'static str,
    transformers_commit: &'static str,
    config: OracleConfig,
    gguf_file: &'static str,
    gguf_sha256: String,
    weights_file: &'static str,
    tensors: Vec<TensorEntry>,
}

#[derive(Serialize)]
struct OracleConfig {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    max_position_embeddings: u64,
    rms_norm_eps: f32,
    num_experts: u32,
    num_experts_per_tok: u32,
    num_shared_experts: u32,
    moe_intermediate_size: u32,
    router_scaling_factor: f32,
    rope_theta: f32,
    rope_factor: f32,
    rope_original_context: u64,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
}

#[derive(Serialize)]
struct TensorEntry {
    name: String,
    shape: Vec<u64>,
    offset_bytes: u64,
    element_count: usize,
    sha256_f32le: String,
}

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: export_oracle_bundle <output-directory>");
    fs::create_dir_all(&output).unwrap();

    let model = ReducedHy3Model::new().unwrap();
    let gguf = model.gguf_bytes().unwrap();
    fs::write(output.join("reduced-hy3.gguf"), &gguf).unwrap();

    let file = File::create(output.join("weights.f32le")).unwrap();
    let mut writer = BufWriter::new(file);
    let mut offset_bytes = 0_u64;
    let mut entries = Vec::new();
    for tensor in model.dequantized_tensors().unwrap() {
        let mut hasher = Sha256::new();
        for value in &tensor.values {
            let encoded = value.to_bits().to_le_bytes();
            writer.write_all(&encoded).unwrap();
            hasher.update(encoded);
        }
        entries.push(TensorEntry {
            name: tensor.name,
            shape: tensor.shape,
            offset_bytes,
            element_count: tensor.values.len(),
            sha256_f32le: format!("{:x}", hasher.finalize()),
        });
        offset_bytes += (tensor.values.len() * 4) as u64;
    }
    writer.flush().unwrap();

    let config = reduced_config();
    let manifest = Manifest {
        format: "lightbridge-reduced-hy3-weights-v1",
        transformers_commit: TRANSFORMERS_COMMIT,
        config: OracleConfig {
            vocab_size: config.vocabulary_size,
            hidden_size: config.embedding_length,
            intermediate_size: config.dense_ffn_length,
            num_hidden_layers: config.block_count,
            num_attention_heads: config.attention_head_count,
            num_key_value_heads: config.attention_kv_head_count,
            head_dim: config.key_length,
            max_position_embeddings: config.context_length,
            rms_norm_eps: config.rms_epsilon,
            num_experts: config.expert_count,
            num_experts_per_tok: config.expert_used_count,
            num_shared_experts: 1,
            moe_intermediate_size: config.expert_ffn_length,
            router_scaling_factor: config.expert_weights_scale,
            rope_theta: config.rope_base,
            rope_factor: config.yarn_factor,
            rope_original_context: config.yarn_original_context,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
        },
        gguf_file: "reduced-hy3.gguf",
        gguf_sha256: format!("{:x}", Sha256::digest(&gguf)),
        weights_file: "weights.f32le",
        tensors: entries,
    };
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}
