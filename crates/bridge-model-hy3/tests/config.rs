use bridge_gguf::{Endianness, GgufArray, GgufFile, GgufValue, GgufValueType};
use bridge_model_hy3::{resolve_config, Hy3Profile};

fn selected_metadata() -> GgufFile {
    let entries = [
        ("general.architecture", GgufValue::String("hy_v3".into())),
        ("hy_v3.block_count", GgufValue::U32(80)),
        ("hy_v3.context_length", GgufValue::U32(1_048_576)),
        ("hy_v3.embedding_length", GgufValue::U32(4_096)),
        ("hy_v3.feed_forward_length", GgufValue::U32(13_312)),
        ("hy_v3.expert_feed_forward_length", GgufValue::U32(1_536)),
        ("hy_v3.expert_shared_feed_forward_length", GgufValue::U32(1_536)),
        ("hy_v3.attention.head_count", GgufValue::U32(64)),
        ("hy_v3.attention.head_count_kv", GgufValue::U32(8)),
        ("hy_v3.attention.key_length", GgufValue::U32(128)),
        ("hy_v3.attention.value_length", GgufValue::U32(128)),
        ("hy_v3.attention.layer_norm_rms_epsilon", GgufValue::F32(0.000_01)),
        ("hy_v3.expert_count", GgufValue::U32(192)),
        ("hy_v3.expert_used_count", GgufValue::U32(8)),
        ("hy_v3.expert_weights_norm", GgufValue::Bool(true)),
        ("hy_v3.expert_gating_func", GgufValue::U32(2)),
        ("hy_v3.expert_weights_scale", GgufValue::F32(2.826)),
        ("hy_v3.rope.freq_base", GgufValue::F32(11_158_840.0)),
        ("hy_v3.rope.scaling.type", GgufValue::String("yarn".into())),
        ("hy_v3.rope.scaling.factor", GgufValue::F32(4.0)),
        (
            "hy_v3.rope.scaling.original_context_length",
            GgufValue::U32(262_144),
        ),
    ];
    let mut metadata = entries
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect::<Vec<_>>();
    metadata.push((
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![GgufValue::String(String::new()); 120_832],
        }),
    ));
    GgufFile {
        version: 3,
        endianness: Endianness::Little,
        metadata,
        tensors: Vec::new(),
        alignment: 32,
        data_offset: 0,
        file_len: 0,
    }
}

fn set(metadata: &mut GgufFile, key: &str, value: GgufValue) {
    metadata
        .metadata
        .iter_mut()
        .find(|(candidate, _)| candidate == key)
        .unwrap()
        .1 = value;
}

fn remove(metadata: &mut GgufFile, key: &str) {
    metadata.metadata.retain(|(candidate, _)| candidate != key);
}

fn selected_error(metadata: &GgufFile) -> String {
    match resolve_config(metadata) {
        Ok(config) => Hy3Profile::selected_iq2_m()
            .validate(&config)
            .unwrap_err()
            .to_string(),
        Err(error) => error.to_string(),
    }
}

#[test]
fn resolves_the_exact_typed_selected_metadata() {
    let config = resolve_config(&selected_metadata()).unwrap();
    Hy3Profile::selected_iq2_m().validate(&config).unwrap();

    assert_eq!(config.block_count, 80);
    assert_eq!(config.context_length, 1_048_576);
    assert_eq!(config.vocabulary_size, 120_832);
}

#[test]
fn architecture_is_required_exact_and_stored_as_string() {
    let mut missing = selected_metadata();
    remove(&mut missing, "general.architecture");
    let mut wrong = selected_metadata();
    set(
        &mut wrong,
        "general.architecture",
        GgufValue::String("hy_v2".into()),
    );
    let mut wrong_type = selected_metadata();
    set(&mut wrong_type, "general.architecture", GgufValue::U32(3));

    let cases = [
        (selected_error(&missing), ["general.architecture", "missing"]),
        (selected_error(&wrong), ["general.architecture", "hy_v3"]),
        (selected_error(&wrong_type), ["general.architecture", "String"]),
    ];
    for (error, fragments) in cases {
        for fragment in fragments {
            assert!(error.contains(fragment), "{error:?} lacks {fragment:?}");
        }
    }
    let missing_error = selected_error(&missing);
    assert!(missing_error.contains("String"));
    assert!(missing_error.contains("actual"));
}

#[test]
fn selected_profile_rejects_every_wrong_integer_dimension_or_expert_setting() {
    let cases = [
        ("hy_v3.block_count", 79, "80"),
        ("hy_v3.context_length", 65_536, "1048576"),
        ("hy_v3.embedding_length", 2_048, "4096"),
        ("hy_v3.feed_forward_length", 8_192, "13312"),
        ("hy_v3.expert_feed_forward_length", 2_048, "1536"),
        ("hy_v3.expert_shared_feed_forward_length", 2_048, "1536"),
        ("hy_v3.attention.head_count", 32, "64"),
        ("hy_v3.attention.head_count_kv", 4, "8"),
        ("hy_v3.attention.key_length", 64, "128"),
        ("hy_v3.attention.value_length", 64, "128"),
        ("hy_v3.expert_count", 128, "192"),
        ("hy_v3.expert_used_count", 4, "8"),
        ("hy_v3.expert_gating_func", 1, "2"),
        ("hy_v3.rope.scaling.original_context_length", 131_072, "262144"),
    ];

    for (key, actual, expected) in cases {
        let mut metadata = selected_metadata();
        set(&mut metadata, key, GgufValue::U32(actual));
        let error = selected_error(&metadata);
        assert!(error.contains(key), "{error:?} lacks {key:?}");
        assert!(error.contains(expected), "{error:?} lacks {expected:?}");
        assert!(error.contains(&actual.to_string()), "{error:?} lacks {actual}");
    }
}

#[test]
fn selected_profile_rejects_missing_block_count_and_the_81_block_mtp_profile() {
    let mut missing = selected_metadata();
    remove(&mut missing, "hy_v3.block_count");
    let missing_error = selected_error(&missing);
    assert!(missing_error.contains("hy_v3.block_count"));
    assert!(missing_error.contains("missing"));
    assert!(missing_error.contains("U32"));
    assert!(missing_error.contains("actual"));

    let mut mtp = selected_metadata();
    set(&mut mtp, "hy_v3.block_count", GgufValue::U32(81));
    let mtp_error = selected_error(&mtp);
    assert!(mtp_error.contains("hy_v3.block_count"));
    assert!(mtp_error.contains("80"));
    assert!(mtp_error.contains("81"));
}

#[test]
fn scalar_metadata_type_errors_name_the_key_and_expected_and_actual_types() {
    let mut metadata = selected_metadata();
    set(&mut metadata, "hy_v3.context_length", GgufValue::U64(1_048_576));

    let error = selected_error(&metadata);
    assert!(error.contains("hy_v3.context_length"));
    assert!(error.contains("expected U32"));
    assert!(error.contains("actual U64"));
}

#[test]
fn finite_float_metadata_is_checked_before_approximate_profile_comparison() {
    let keys = [
        ("hy_v3.attention.layer_norm_rms_epsilon", 1.0e-5_f32),
        ("hy_v3.expert_weights_scale", 2.826_f32),
        ("hy_v3.rope.freq_base", 11_158_840.0_f32),
        ("hy_v3.rope.scaling.factor", 4.0_f32),
    ];

    for (key, selected) in keys {
        for actual in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut metadata = selected_metadata();
            set(&mut metadata, key, GgufValue::F32(actual));
            let error = selected_error(&metadata);
            assert!(error.contains(key), "{error:?} lacks {key:?}");
            assert!(error.contains("finite"), "{error:?} lacks finite");
        }

        let mut wrong = selected_metadata();
        set(&mut wrong, key, GgufValue::F32(selected * 1.25));
        let error = selected_error(&wrong);
        assert!(error.contains(key), "{error:?} lacks {key:?}");
        assert!(error.contains("expected"), "{error:?} lacks expected");
        assert!(error.contains("actual"), "{error:?} lacks actual");
    }
}

#[test]
fn selected_profile_rejects_false_normalization_and_wrong_rope_scaling_type() {
    let mut normalization = selected_metadata();
    set(
        &mut normalization,
        "hy_v3.expert_weights_norm",
        GgufValue::Bool(false),
    );
    let error = selected_error(&normalization);
    assert!(error.contains("hy_v3.expert_weights_norm"));
    assert!(error.contains("true"));
    assert!(error.contains("false"));

    let mut rope = selected_metadata();
    set(
        &mut rope,
        "hy_v3.rope.scaling.type",
        GgufValue::String("linear".into()),
    );
    let error = selected_error(&rope);
    assert!(error.contains("hy_v3.rope.scaling.type"));
    assert!(error.contains("yarn"));
    assert!(error.contains("linear"));
}

#[test]
fn token_array_requires_stored_string_elements_and_the_selected_count() {
    let mut wrong_element_type = selected_metadata();
    set(
        &mut wrong_element_type,
        "tokenizer.ggml.tokens",
        GgufValue::Array(GgufArray {
            element_type: GgufValueType::I32,
            values: vec![GgufValue::I32(1); 120_832],
        }),
    );
    let error = selected_error(&wrong_element_type);
    assert!(error.contains("tokenizer.ggml.tokens"));
    assert!(error.contains("String"));
    assert!(error.contains("I32"));

    let mut wrong_count = selected_metadata();
    set(
        &mut wrong_count,
        "tokenizer.ggml.tokens",
        GgufValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![GgufValue::String(String::new()); 120_831],
        }),
    );
    let error = selected_error(&wrong_count);
    assert!(error.contains("tokenizer.ggml.tokens"));
    assert!(error.contains("120832"));
    assert!(error.contains("120831"));
}
