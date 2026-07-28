use std::collections::BTreeMap;

use bridge_core::ggml_type::GgmlType;
use bridge_core::tensor::TensorDesc;
use bridge_gguf::{Endianness, GgufArray, GgufFile, GgufValue, GgufValueType};
use bridge_gguf_split::testing::{from_file, from_files};
use bridge_model_hy3::{
    checked_expert_slab, generate_selected_iq2_m_schema, validate_selected_iq2_m_tensor_descriptors,
    validate_selected_model, Hy3Config, Hy3TensorRole,
};

fn reduced_config() -> Hy3Config {
    Hy3Config {
        block_count: 2,
        context_length: 4_096,
        embedding_length: 256,
        dense_ffn_length: 512,
        expert_ffn_length: 256,
        shared_expert_ffn_length: 256,
        attention_head_count: 2,
        attention_kv_head_count: 1,
        key_length: 128,
        value_length: 128,
        rms_epsilon: 1.0e-5,
        expert_count: 2,
        expert_used_count: 1,
        expert_weights_norm: true,
        expert_gating_func: 2,
        expert_weights_scale: 2.826,
        rope_base: 11_158_840.0,
        rope_scaling_type: "yarn".into(),
        yarn_factor: 4.0,
        yarn_original_context: 4_096,
        vocabulary_size: 512,
    }
}

fn selected_config() -> Hy3Config {
    Hy3Config {
        block_count: 80,
        context_length: 1_048_576,
        embedding_length: 4_096,
        dense_ffn_length: 13_312,
        expert_ffn_length: 1_536,
        shared_expert_ffn_length: 1_536,
        attention_head_count: 64,
        attention_kv_head_count: 8,
        key_length: 128,
        value_length: 128,
        rms_epsilon: 1.0e-5,
        expert_count: 192,
        expert_used_count: 8,
        expert_weights_norm: true,
        expert_gating_func: 2,
        expert_weights_scale: 2.826,
        rope_base: 11_158_840.0,
        rope_scaling_type: "yarn".into(),
        yarn_factor: 4.0,
        yarn_original_context: 262_144,
        vocabulary_size: 120_832,
    }
}

#[test]
fn selected_iq2_m_schema_api_rejects_a_self_consistent_non_selected_profile() {
    let config = reduced_config();
    let generation_error = generate_selected_iq2_m_schema(&config).unwrap_err().to_string();
    let validation_error = validate_selected_iq2_m_tensor_descriptors(&config, &[])
        .unwrap_err()
        .to_string();

    for error in [generation_error, validation_error] {
        assert!(error.contains("hy_v3.block_count"), "{error}");
        assert!(error.contains("80"), "{error}");
        assert!(error.contains("2"), "{error}");
    }
}

fn selected_metadata(tensors: Vec<TensorDesc>) -> GgufFile {
    let mut metadata = vec![
        ("general.architecture".into(), GgufValue::String("hy_v3".into())),
        ("hy_v3.block_count".into(), GgufValue::U32(80)),
        ("hy_v3.context_length".into(), GgufValue::U32(1_048_576)),
        ("hy_v3.embedding_length".into(), GgufValue::U32(4_096)),
        ("hy_v3.feed_forward_length".into(), GgufValue::U32(13_312)),
        ("hy_v3.expert_feed_forward_length".into(), GgufValue::U32(1_536)),
        (
            "hy_v3.expert_shared_feed_forward_length".into(),
            GgufValue::U32(1_536),
        ),
        ("hy_v3.attention.head_count".into(), GgufValue::U32(64)),
        ("hy_v3.attention.head_count_kv".into(), GgufValue::U32(8)),
        ("hy_v3.attention.key_length".into(), GgufValue::U32(128)),
        ("hy_v3.attention.value_length".into(), GgufValue::U32(128)),
        (
            "hy_v3.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(0.000_01),
        ),
        ("hy_v3.expert_count".into(), GgufValue::U32(192)),
        ("hy_v3.expert_used_count".into(), GgufValue::U32(8)),
        ("hy_v3.expert_weights_norm".into(), GgufValue::Bool(true)),
        ("hy_v3.expert_gating_func".into(), GgufValue::U32(2)),
        ("hy_v3.expert_weights_scale".into(), GgufValue::F32(2.826)),
        ("hy_v3.rope.freq_base".into(), GgufValue::F32(11_158_840.0)),
        ("hy_v3.rope.scaling.type".into(), GgufValue::String("yarn".into())),
        ("hy_v3.rope.scaling.factor".into(), GgufValue::F32(4.0)),
        (
            "hy_v3.rope.scaling.original_context_length".into(),
            GgufValue::U32(262_144),
        ),
    ];
    metadata.push((
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![GgufValue::String(String::new()); 120_832],
        }),
    ));
    let file_len = tensors
        .iter()
        .map(|tensor| {
            tensor
                .relative_offset()
                .checked_add(tensor.encoded_bytes().unwrap())
                .unwrap()
        })
        .max()
        .unwrap_or(0);
    GgufFile {
        version: 3,
        endianness: Endianness::Little,
        metadata,
        tensors,
        alignment: 32,
        data_offset: 0,
        file_len,
    }
}

fn with_split(mut file: GgufFile, ordinal: u16, count: u16) -> GgufFile {
    file.metadata.push(("split.no".into(), GgufValue::U16(ordinal)));
    file.metadata.push(("split.count".into(), GgufValue::U16(count)));
    file
}

fn set_metadata(file: &mut GgufFile, key: &str, value: GgufValue) {
    file.metadata
        .iter_mut()
        .find(|(candidate, _)| candidate == key)
        .unwrap()
        .1 = value;
}

fn selected_two_shard_set(second: GgufFile) -> bridge_gguf_split::GgufSet {
    let first = with_split(selected_metadata(descriptors(&selected_config())), 0, 2);
    let second = with_split(second, 1, 2);
    from_files(vec![first, second]).unwrap()
}

fn descriptors(config: &Hy3Config) -> Vec<TensorDesc> {
    let mut next_offset = 0_u64;
    generate_selected_iq2_m_schema(config)
        .unwrap()
        .into_iter()
        .map(|spec| {
            let descriptor = TensorDesc::new(spec.name(), spec.shape(), spec.ty(), next_offset).unwrap();
            next_offset += descriptor.encoded_bytes().unwrap();
            next_offset = (next_offset + 31) & !31;
            descriptor
        })
        .collect()
}

fn replace(tensors: &mut [TensorDesc], name: &str, shape: &[u64], ty: GgmlType) {
    let index = tensors.iter().position(|tensor| tensor.name() == name).unwrap();
    let offset = tensors[index].relative_offset();
    tensors[index] = TensorDesc::new(name, shape, ty, offset).unwrap();
}

#[test]
fn classifies_every_exact_complete_tensor_name_pattern() {
    use Hy3TensorRole::*;
    let cases = [
        ("token_embd.weight", TokenEmbedding),
        ("output_norm.weight", OutputNorm),
        ("output.weight", Output),
        ("blk.0.attn_norm.weight", AttentionNorm { layer: 0 }),
        ("blk.1.attn_q.weight", AttentionQ { layer: 1 }),
        ("blk.1.attn_q_norm.weight", AttentionQNorm { layer: 1 }),
        ("blk.1.attn_k.weight", AttentionK { layer: 1 }),
        ("blk.1.attn_k_norm.weight", AttentionKNorm { layer: 1 }),
        ("blk.1.attn_v.weight", AttentionV { layer: 1 }),
        ("blk.1.attn_output.weight", AttentionOutput { layer: 1 }),
        ("blk.1.ffn_norm.weight", FfnNorm { layer: 1 }),
        ("blk.0.ffn_gate.weight", DenseGate { layer: 0 }),
        ("blk.0.ffn_up.weight", DenseUp { layer: 0 }),
        ("blk.0.ffn_down.weight", DenseDown { layer: 0 }),
        ("blk.1.ffn_gate_inp.weight", RouterInput { layer: 1 }),
        ("blk.1.exp_probs_b", RouterSelectionBias { layer: 1 }),
        ("blk.1.ffn_gate_exps.weight", RoutedGate { layer: 1 }),
        ("blk.1.ffn_up_exps.weight", RoutedUp { layer: 1 }),
        ("blk.1.ffn_down_exps.weight", RoutedDown { layer: 1 }),
        ("blk.1.ffn_gate_shexp.weight", SharedGate { layer: 1 }),
        ("blk.1.ffn_up_shexp.weight", SharedUp { layer: 1 }),
        ("blk.1.ffn_down_shexp.weight", SharedDown { layer: 1 }),
    ];

    for (name, expected) in cases {
        assert_eq!(Hy3TensorRole::classify(name, 2).unwrap(), expected);
    }
}

#[test]
fn rejects_alias_suffix_out_of_range_leading_zero_and_wrong_regime_names() {
    let invalid = [
        "tok_embeddings.weight",
        "token_embd.weight.extra",
        "blk.2.attn_q.weight",
        "blk.01.attn_q.weight",
        "blk.+1.attn_q.weight",
        "blk.0.exp_probs_b",
        "blk.0.ffn_gate_exps.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
        "blk.1.attn_q",
        "blk.1.attn_q.weight.suffix",
    ];

    for name in invalid {
        let error = Hy3TensorRole::classify(name, 2).unwrap_err().to_string();
        assert!(error.contains(name), "{error:?} lacks {name:?}");
        assert!(error.contains("expected"), "{error:?} lacks expected");
    }
}

#[test]
fn selected_iq2_m_schema_is_exact_and_valid() {
    let config = selected_config();
    let schema = generate_selected_iq2_m_schema(&config).unwrap();
    assert_eq!(schema.len(), 1_278);
    assert_eq!(
        schema
            .iter()
            .filter(|spec| spec.name().starts_with("blk.0."))
            .count(),
        11
    );
    assert_eq!(
        schema
            .iter()
            .filter(|spec| spec.name().starts_with("blk.1."))
            .count(),
        16
    );

    validate_selected_iq2_m_tensor_descriptors(&config, &descriptors(&config)).unwrap();
}

#[test]
fn schema_validation_rejects_missing_and_unexpected_tensors() {
    let config = selected_config();
    let mut missing = descriptors(&config);
    missing.retain(|tensor| tensor.name() != "blk.1.attn_q.weight");
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &missing)
        .unwrap_err()
        .to_string();
    assert!(error.contains("blk.1.attn_q.weight"));
    assert!(error.contains("missing"));

    let mut unexpected = descriptors(&config);
    unexpected.push(TensorDesc::new("extra.weight", &[1], GgmlType::F32, 1_000_000).unwrap());
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &unexpected)
        .unwrap_err()
        .to_string();
    assert!(error.contains("extra.weight"));
    assert!(error.contains("unexpected"));
}

#[test]
fn schema_validation_rejects_wrong_shape_and_physical_type() {
    let config = selected_config();
    let mut shape = descriptors(&config);
    replace(&mut shape, "blk.1.attn_q.weight", &[4_096, 128], GgmlType::IQ2_S);
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("blk.1.attn_q.weight"));
    assert!(error.contains("[4096, 8192]"));
    assert!(error.contains("[4096, 128]"));

    let mut ty = descriptors(&config);
    replace(&mut ty, "blk.1.attn_q.weight", &[4_096, 8_192], GgmlType::IQ3_S);
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &ty)
        .unwrap_err()
        .to_string();
    assert!(error.contains("blk.1.attn_q.weight"));
    assert!(error.contains("IQ2_S"));
    assert!(error.contains("IQ3_S"));
}

#[test]
fn routed_expert_validation_requires_rank_three_and_matching_expert_dimension() {
    let config = selected_config();
    let mut rank = descriptors(&config);
    replace(
        &mut rank,
        "blk.1.ffn_gate_exps.weight",
        &[4_096, 1_536],
        GgmlType::IQ2_S,
    );
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &rank)
        .unwrap_err()
        .to_string();
    assert!(error.contains("blk.1.ffn_gate_exps.weight"));
    assert!(error.contains("rank 3"));
    assert!(error.contains("rank 2"));

    let mut experts = descriptors(&config);
    replace(
        &mut experts,
        "blk.1.ffn_gate_exps.weight",
        &[4_096, 1_536, 193],
        GgmlType::IQ2_S,
    );
    let error = validate_selected_iq2_m_tensor_descriptors(&config, &experts)
        .unwrap_err()
        .to_string();
    assert!(error.contains("blk.1.ffn_gate_exps.weight"));
    assert!(error.contains("dimension 2"));
    assert!(error.contains("192"));
    assert!(error.contains("193"));
}

#[test]
fn checked_expert_slabs_reject_non_divisible_payloads_and_invalid_indices() {
    let name = "blk.1.ffn_gate_exps.weight";
    let error = checked_expert_slab(name, &[4096, 1536, 192], 100, 192, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains(name));
    assert!(error.contains("divisible"));
    assert!(error.contains("100"));
    assert!(error.contains("192"));

    let payload_bytes = 2_015_232_u64 * 192;
    assert_eq!(
        checked_expert_slab(name, &[4096, 1536, 192], payload_bytes, 192, 0)
            .unwrap()
            .relative_range,
        0..2_015_232
    );
    assert_eq!(
        checked_expert_slab(name, &[4096, 1536, 192], payload_bytes, 192, 191)
            .unwrap()
            .relative_range,
        (2_015_232 * 191)..payload_bytes
    );
    let error = checked_expert_slab(name, &[4096, 1536, 192], payload_bytes, 192, 192)
        .unwrap_err()
        .to_string();
    assert!(error.contains(name));
    assert!(error.contains("less than 192"));
    assert!(error.contains("192"));
}

#[test]
fn checked_expert_slabs_require_rank_three_and_matching_dimension_two() {
    let name = "blk.1.ffn_down_exps.weight";
    let rank = checked_expert_slab(name, &[1536, 4096], 2_703_360, 192, 0)
        .unwrap_err()
        .to_string();
    assert!(rank.contains(name));
    assert!(rank.contains("rank 3"));
    assert!(rank.contains("rank 2"));

    let dimension = checked_expert_slab(name, &[1536, 4096, 191], 2_703_360 * 191, 192, 0)
        .unwrap_err()
        .to_string();
    assert!(dimension.contains(name));
    assert!(dimension.contains("dimension 2"));
    assert!(dimension.contains("192"));
    assert!(dimension.contains("191"));
}

#[test]
fn selected_schema_has_1278_entries_and_exact_type_transition_boundaries() {
    let schema = generate_selected_iq2_m_schema(&selected_config()).unwrap();
    assert_eq!(schema.len(), 1_278);
    let by_name = schema
        .iter()
        .map(|spec| (spec.name(), spec.ty()))
        .collect::<BTreeMap<_, _>>();

    assert_eq!(by_name["blk.5.ffn_down_exps.weight"], GgmlType::IQ3_S);
    assert_eq!(by_name["blk.6.ffn_down_exps.weight"], GgmlType::IQ2_S);
    assert_eq!(by_name["blk.4.ffn_down_shexp.weight"], GgmlType::IQ3_S);
    assert_eq!(by_name["blk.5.ffn_down_shexp.weight"], GgmlType::IQ2_S);
}

#[test]
fn selected_schema_derives_the_oracle_histogram_and_stored_byte_totals() {
    let schema = generate_selected_iq2_m_schema(&selected_config()).unwrap();
    let mut aggregates = BTreeMap::<GgmlType, (usize, u64)>::new();
    let mut total = 0_u64;

    for spec in schema {
        let descriptor = TensorDesc::new(spec.name(), spec.shape(), spec.ty(), 0).unwrap();
        let bytes = descriptor.encoded_bytes().unwrap();
        let entry = aggregates.entry(spec.ty()).or_default();
        entry.0 += 1;
        entry.1 += bytes;
        total += bytes;
    }

    assert_eq!(aggregates[&GgmlType::F32], (479, 251_292_928));
    assert_eq!(aggregates[&GgmlType::IQ2_S], (627, 91_238_285_312));
    assert_eq!(aggregates[&GgmlType::IQ3_S], (91, 3_995_566_080));
    assert_eq!(aggregates[&GgmlType::Q4_K], (80, 188_743_680));
    assert_eq!(aggregates[&GgmlType::Q5_K], (1, 340_262_912));
    assert_eq!(total, 96_014_150_912);
}

#[test]
fn selected_model_rejects_an_oversized_directory_before_descriptor_cloning() {
    let mut tensors = descriptors(&selected_config());
    let offset = tensors
        .last()
        .map(|tensor| {
            let end = tensor.relative_offset() + tensor.encoded_bytes().unwrap();
            (end + 31) & !31
        })
        .unwrap();
    tensors.push(TensorDesc::new("extra.weight", &[1], GgmlType::F32, offset).unwrap());
    let set = from_file(selected_metadata(tensors)).unwrap();

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("tensor directory count"));
    assert!(error.contains("1278"));
    assert!(error.contains("1279"));
}

#[test]
fn validates_a_complete_selected_gguf_set_into_read_only_semantic_tensors() {
    let file = selected_metadata(descriptors(&selected_config()));
    let set = from_file(file).unwrap();

    let model = validate_selected_model(&set).unwrap();

    assert_eq!(model.config().block_count, 80);
    assert_eq!(model.tensors().len(), 1_278);
    assert!(!model.has_mtp());
    assert_eq!(
        model.tensors()[0].location().descriptor().name(),
        "token_embd.weight"
    );
    let routed = model
        .tensors()
        .iter()
        .find(|tensor| tensor.location().descriptor().name() == "blk.1.ffn_gate_exps.weight")
        .unwrap();
    assert_eq!(
        routed.expert_slab(192, 191).unwrap().relative_range,
        (2_015_232 * 191)..(2_015_232 * 192)
    );
}

#[test]
fn selected_model_accepts_an_exact_metadata_replica_on_a_later_shard() {
    let set = selected_two_shard_set(selected_metadata(Vec::new()));

    let model = validate_selected_model(&set).unwrap();

    assert_eq!(model.tensors().len(), 1_278);
}

#[test]
fn selected_model_accepts_complete_selected_metadata_omission_on_a_later_shard() {
    let mut omitted = selected_metadata(Vec::new());
    omitted.metadata.clear();
    let set = selected_two_shard_set(omitted);

    let model = validate_selected_model(&set).unwrap();

    assert_eq!(model.tensors().len(), 1_278);
}

#[test]
fn selected_model_rejects_partial_selected_metadata_on_a_later_shard() {
    let mut partial = selected_metadata(Vec::new());
    partial.metadata.retain(|(key, _)| key == "hy_v3.block_count");
    let set = selected_two_shard_set(partial);

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("shard 1"));
    assert!(error.contains("partial"));
    assert!(error.contains("general.architecture"));
}

#[test]
fn selected_model_rejects_a_value_conflict_on_a_later_shard() {
    let mut conflicting = selected_metadata(Vec::new());
    set_metadata(&mut conflicting, "hy_v3.expert_used_count", GgufValue::U32(7));
    let set = selected_two_shard_set(conflicting);

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("shard 1"));
    assert!(error.contains("hy_v3.expert_used_count"));
    assert!(error.contains('8'));
    assert!(error.contains('7'));
}

#[test]
fn selected_model_rejects_a_stored_type_conflict_on_a_later_shard() {
    let mut conflicting = selected_metadata(Vec::new());
    set_metadata(
        &mut conflicting,
        "hy_v3.context_length",
        GgufValue::U64(1_048_576),
    );
    let set = selected_two_shard_set(conflicting);

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("shard 1"));
    assert!(error.contains("hy_v3.context_length"));
    assert!(error.contains("U32"));
    assert!(error.contains("U64"));
}

#[test]
fn selected_model_rejects_an_architecture_conflict_on_a_later_shard() {
    let mut conflicting = selected_metadata(Vec::new());
    set_metadata(
        &mut conflicting,
        "general.architecture",
        GgufValue::String("hy_v2".into()),
    );
    let set = selected_two_shard_set(conflicting);

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("shard 1"));
    assert!(error.contains("general.architecture"));
    assert!(error.contains("hy_v3"));
    assert!(error.contains("hy_v2"));
}

#[test]
fn selected_model_rejects_a_token_count_conflict_on_a_later_shard() {
    let mut conflicting = selected_metadata(Vec::new());
    set_metadata(
        &mut conflicting,
        "tokenizer.ggml.tokens",
        GgufValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![GgufValue::String(String::new()); 120_831],
        }),
    );
    let set = selected_two_shard_set(conflicting);

    let error = validate_selected_model(&set).unwrap_err().to_string();

    assert!(error.contains("shard 1"));
    assert!(error.contains("tokenizer.ggml.tokens"));
    assert!(error.contains("120832"));
    assert!(error.contains("120831"));
}
