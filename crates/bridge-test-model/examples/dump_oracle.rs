use std::collections::BTreeMap;

use bridge_kernels_reference::{Hy3BlockScratch, ReferenceExecutionMode};
use bridge_test_model::ReducedHy3Model;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct Oracle {
    format: &'static str,
    token_ids: [u32; 2],
    modes: Vec<ModeOracle>,
}

#[derive(Serialize)]
struct ModeOracle {
    mode: &'static str,
    steps: Vec<StepOracle>,
}

#[derive(Serialize)]
struct StepOracle {
    token_id: u32,
    selected_experts: [u32; 2],
    greedy_id: u32,
    hashes: BTreeMap<String, String>,
}

fn hash(values: &[f32]) -> String {
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
        hashes.insert(format!("{prefix}.{name}"), hash(values));
    }
}

fn main() {
    let model = ReducedHy3Model::new().unwrap();
    let token_ids = [3_u32, 7];
    let mut modes = Vec::new();
    for (mode, mode_name) in [
        (ReferenceExecutionMode::DequantF32, "dequant_f32"),
        (ReferenceExecutionMode::LlamaQ8K, "llama_q8_k"),
    ] {
        let mut session = model.new_session().unwrap();
        let mut steps = Vec::new();
        for token_id in token_ids {
            let output = model.evaluate_token(&mut session, mode, token_id).unwrap();
            let mut hashes = BTreeMap::new();
            hashes.insert("final.hidden".into(), hash(output.hidden));
            hashes.insert("final.normalized".into(), hash(output.final_normalized));
            hashes.insert("final.logits".into(), hash(output.logits));
            hashes.insert("final.probabilities".into(), hash(output.probabilities));
            let selected_experts = output.selected_experts;
            let greedy_id = output.greedy_id;
            block_hashes("block0", session.dense_scratch(), &mut hashes);
            block_hashes("block1", session.moe_scratch(), &mut hashes);
            steps.push(StepOracle {
                token_id,
                selected_experts,
                greedy_id,
                hashes,
            });
        }
        modes.push(ModeOracle {
            mode: mode_name,
            steps,
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Oracle {
            format: "lightbridge-reduced-hy3-oracle-v1",
            token_ids,
            modes,
        })
        .unwrap()
    );
}
