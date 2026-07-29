use bridge_kernels_reference::ReferenceExecutionMode;
use bridge_test_model::ReducedHy3Model;
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    format: String,
    llama_commit: String,
    steps: Vec<OracleStep>,
    provenance: Provenance,
}

#[derive(Deserialize)]
struct OracleStep {
    token_id: u32,
    selected_experts: [u32; 2],
    greedy_id: u32,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

#[derive(Deserialize)]
struct Provenance {
    repository: String,
    release: String,
    commit: String,
    license: String,
    local_oracle_sha256: String,
    gguf_sha256: String,
    command: String,
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn llama_q8_k_matches_pinned_normal_graph_oracle() {
    let oracle: Oracle = serde_json::from_str(include_str!("fixtures/hy3-llama-oracle-v1.json")).unwrap();
    assert_eq!(oracle.format, "lightbridge-llama-hy3-oracle-v1");
    assert_eq!(oracle.llama_commit, "b77d646751d01c0962bc203b6809e9d94f7d50b7");
    assert_eq!(
        oracle.provenance.repository,
        "https://github.com/ggml-org/llama.cpp.git"
    );
    assert_eq!(oracle.provenance.release, "b10153");
    assert_eq!(oracle.provenance.commit, oracle.llama_commit);
    assert_eq!(oracle.provenance.license, "MIT");
    assert_eq!(oracle.provenance.local_oracle_sha256.len(), 64);
    assert_eq!(oracle.provenance.gguf_sha256.len(), 64);
    assert!(oracle.provenance.command.contains("generate-llama-vectors.ps1"));

    let model = ReducedHy3Model::new().unwrap();
    let mut session = model.new_session().unwrap();
    for expected in oracle.steps {
        let actual = model
            .evaluate_token(&mut session, ReferenceExecutionMode::LlamaQ8K, expected.token_id)
            .unwrap();
        assert_eq!(actual.selected_experts, expected.selected_experts);
        assert_eq!(actual.greedy_id, expected.greedy_id);
        assert_close(actual.logits, &expected.logits, 3.0e-4, 3.0e-4);
        assert_close(actual.probabilities, &expected.probabilities, 5.0e-5, 5.0e-5);
    }
}
