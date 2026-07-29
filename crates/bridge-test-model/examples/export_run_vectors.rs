use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use bridge_kernels_reference::{Hy3BlockScratch, ReferenceExecutionMode};
use bridge_test_model::ReducedHy3Model;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct Manifest {
    format: &'static str,
    data_file: &'static str,
    modes: Vec<ModeEntry>,
    arrays: Vec<ArrayEntry>,
}

#[derive(Serialize)]
struct ModeEntry {
    mode: &'static str,
    steps: Vec<StepEntry>,
}

#[derive(Serialize)]
struct StepEntry {
    token_id: u32,
    selected_experts: [u32; 2],
    greedy_id: u32,
}

#[derive(Serialize)]
struct ArrayEntry {
    name: String,
    offset_bytes: u64,
    element_count: usize,
    sha256_f32le: String,
}

fn write_array(
    writer: &mut BufWriter<File>,
    arrays: &mut Vec<ArrayEntry>,
    offset: &mut u64,
    name: String,
    values: &[f32],
) {
    let mut hasher = Sha256::new();
    for value in values {
        let encoded = value.to_bits().to_le_bytes();
        writer.write_all(&encoded).unwrap();
        hasher.update(encoded);
    }
    arrays.push(ArrayEntry {
        name,
        offset_bytes: *offset,
        element_count: values.len(),
        sha256_f32le: format!("{:x}", hasher.finalize()),
    });
    *offset += (values.len() * 4) as u64;
}

fn write_block(
    writer: &mut BufWriter<File>,
    arrays: &mut Vec<ArrayEntry>,
    offset: &mut u64,
    prefix: &str,
    scratch: &Hy3BlockScratch,
) {
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
        write_array(writer, arrays, offset, format!("{prefix}.{name}"), values);
    }
}

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: export_run_vectors <output-directory>");
    fs::create_dir_all(&output).unwrap();
    let mut writer = BufWriter::new(File::create(output.join("bridge-vectors.f32le")).unwrap());
    let mut arrays = Vec::new();
    let mut modes = Vec::new();
    let mut offset = 0_u64;
    let model = ReducedHy3Model::new().unwrap();

    for (mode, mode_name) in [
        (ReferenceExecutionMode::DequantF32, "dequant_f32"),
        (ReferenceExecutionMode::LlamaQ8K, "llama_q8_k"),
    ] {
        let mut session = model.new_session().unwrap();
        let mut steps = Vec::new();
        for (step, token_id) in [3_u32, 7].into_iter().enumerate() {
            let output = model.evaluate_token(&mut session, mode, token_id).unwrap();
            let prefix = format!("{mode_name}.step{step}");
            write_array(
                &mut writer,
                &mut arrays,
                &mut offset,
                format!("{prefix}.final.hidden"),
                output.hidden,
            );
            write_array(
                &mut writer,
                &mut arrays,
                &mut offset,
                format!("{prefix}.final.normalized"),
                output.final_normalized,
            );
            write_array(
                &mut writer,
                &mut arrays,
                &mut offset,
                format!("{prefix}.final.logits"),
                output.logits,
            );
            write_array(
                &mut writer,
                &mut arrays,
                &mut offset,
                format!("{prefix}.final.probabilities"),
                output.probabilities,
            );
            let selected_experts = output.selected_experts;
            let greedy_id = output.greedy_id;
            write_block(
                &mut writer,
                &mut arrays,
                &mut offset,
                &format!("{prefix}.block0"),
                session.dense_scratch(),
            );
            write_block(
                &mut writer,
                &mut arrays,
                &mut offset,
                &format!("{prefix}.block1"),
                session.moe_scratch(),
            );
            steps.push(StepEntry {
                token_id,
                selected_experts,
                greedy_id,
            });
        }
        modes.push(ModeEntry {
            mode: mode_name,
            steps,
        });
    }
    writer.flush().unwrap();
    fs::write(
        output.join("bridge-vectors.json"),
        serde_json::to_vec_pretty(&Manifest {
            format: "lightbridge-reduced-hy3-runtime-v1",
            data_file: "bridge-vectors.f32le",
            modes,
            arrays,
        })
        .unwrap(),
    )
    .unwrap();
}
