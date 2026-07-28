use std::collections::BTreeMap;
#[cfg(any(unix, windows))]
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;
#[cfg(windows)]
use std::os::windows::ffi::OsStringExt;
use std::path::PathBuf;

use bridge_cli::{build_report, render_json, render_text, Aggregate, InspectionReport, ReportError};
use bridge_core::tensor::TensorDesc;
use bridge_gguf::{Endianness, GgufArray, GgufFile, GgufValue, GgufValueType};
use bridge_gguf_split::testing::{from_explicit_files, from_file};
use bridge_model_hy3::{generate_selected_iq2_m_schema, Hy3Profile};

const DATA_OFFSET: u64 = 5_160_192;

fn aggregate(count: u64, logical_elements: u64, encoded_bytes: u64) -> Aggregate {
    Aggregate {
        count,
        logical_elements,
        encoded_bytes,
    }
}

fn array(element_type: GgufValueType, value: GgufValue, count: usize) -> GgufValue {
    GgufValue::Array(GgufArray {
        element_type,
        values: vec![value; count],
    })
}

fn selected_metadata(include_presentation: bool) -> Vec<(String, GgufValue)> {
    let mut metadata: Vec<(String, GgufValue)> = vec![
        ("general.architecture".into(), GgufValue::String("hy_v3".into())),
        ("general.type".into(), GgufValue::String("model".into())),
        ("general.sampling.top_k".into(), GgufValue::I32(-1)),
        ("general.sampling.top_p".into(), GgufValue::F32(1.0)),
        ("general.sampling.temp".into(), GgufValue::F32(0.9)),
        ("general.name".into(), GgufValue::String("Hy3 Src".into())),
        ("general.size_label".into(), GgufValue::String("192x10B".into())),
        ("general.license".into(), GgufValue::String("apache-2.0".into())),
        (
            "general.tags".into(),
            GgufValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: ["hunyuan", "hy3", "moe", "text-generation", "text-generation"]
                    .into_iter()
                    .map(|value| GgufValue::String(value.into()))
                    .collect(),
            }),
        ),
        ("general.quantization_version".into(), GgufValue::U32(2)),
        ("general.file_type".into(), GgufValue::U32(29)),
        ("hy_v3.block_count".into(), GgufValue::U32(80)),
        ("hy_v3.context_length".into(), GgufValue::U32(1_048_576)),
        ("hy_v3.embedding_length".into(), GgufValue::U32(4_096)),
        ("hy_v3.feed_forward_length".into(), GgufValue::U32(13_312)),
        ("hy_v3.attention.head_count".into(), GgufValue::U32(64)),
        ("hy_v3.attention.head_count_kv".into(), GgufValue::U32(8)),
        ("hy_v3.attention.key_length".into(), GgufValue::U32(128)),
        ("hy_v3.attention.value_length".into(), GgufValue::U32(128)),
        (
            "hy_v3.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(0.000_01),
        ),
        ("hy_v3.rope.freq_base".into(), GgufValue::F32(11_158_840.0)),
        ("hy_v3.rope.scaling.type".into(), GgufValue::String("yarn".into())),
        ("hy_v3.rope.scaling.factor".into(), GgufValue::F32(4.0)),
        (
            "hy_v3.rope.scaling.original_context_length".into(),
            GgufValue::U32(262_144),
        ),
        ("hy_v3.expert_count".into(), GgufValue::U32(192)),
        ("hy_v3.expert_used_count".into(), GgufValue::U32(8)),
        ("hy_v3.expert_feed_forward_length".into(), GgufValue::U32(1_536)),
        (
            "hy_v3.expert_shared_feed_forward_length".into(),
            GgufValue::U32(1_536),
        ),
        ("hy_v3.expert_weights_norm".into(), GgufValue::Bool(true)),
        ("hy_v3.expert_weights_scale".into(), GgufValue::F32(2.826)),
        ("hy_v3.expert_gating_func".into(), GgufValue::U32(2)),
        ("tokenizer.ggml.model".into(), GgufValue::String("gpt2".into())),
        (
            "tokenizer.ggml.pre".into(),
            GgufValue::String("hunyuan-dense".into()),
        ),
        ("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(120_000)),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(120_025)),
        ("tokenizer.ggml.padding_token_id".into(), GgufValue::U32(120_002)),
        (
            "tokenizer.ggml.seperator_token_id".into(),
            GgufValue::U32(120_007),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            array(GgufValueType::String, GgufValue::String(String::new()), 120_832),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            array(GgufValueType::I32, GgufValue::I32(1), 120_832),
        ),
        (
            "tokenizer.ggml.merges".into(),
            array(GgufValueType::String, GgufValue::String(String::new()), 119_758),
        ),
        (
            "tokenizer.chat_template".into(),
            GgufValue::String("fixture template".into()),
        ),
        (
            "quantize.imatrix.file".into(),
            GgufValue::String("/fixture/hy3.imatrix".into()),
        ),
        (
            "quantize.imatrix.dataset".into(),
            GgufValue::String("/fixture/calibration.txt".into()),
        ),
        ("quantize.imatrix.entries_count".into(), GgufValue::U32(876)),
        ("quantize.imatrix.chunks_count".into(), GgufValue::U32(40)),
    ];
    assert_eq!(metadata.len(), 45);
    if !include_presentation {
        metadata.retain(|(key, _)| {
            !matches!(
                key.as_str(),
                "general.name"
                    | "general.license"
                    | "general.size_label"
                    | "general.quantization_version"
                    | "general.file_type"
            )
        });
    }
    metadata
}

fn selected_file(include_presentation: bool, reverse_tensors: bool) -> GgufFile {
    let config = Hy3Profile::selected_iq2_m();
    let mut next_offset = 0_u64;
    let mut tensors = generate_selected_iq2_m_schema(config.config())
        .unwrap()
        .into_iter()
        .map(|spec| {
            let tensor = TensorDesc::new(spec.name(), spec.shape(), spec.ty(), next_offset).unwrap();
            next_offset = next_offset.checked_add(tensor.encoded_bytes().unwrap()).unwrap();
            next_offset = next_offset.checked_add(31).unwrap() & !31;
            tensor
        })
        .collect::<Vec<_>>();
    if reverse_tensors {
        tensors.reverse();
    }
    GgufFile {
        version: 3,
        endianness: Endianness::Little,
        metadata: selected_metadata(include_presentation),
        tensors,
        alignment: 32,
        data_offset: DATA_OFFSET,
        file_len: DATA_OFFSET.checked_add(next_offset).unwrap(),
    }
}

fn add_split_metadata(file: &mut GgufFile, ordinal: u16, count: u16) {
    file.metadata.extend([
        ("split.no".into(), GgufValue::U16(ordinal)),
        ("split.count".into(), GgufValue::U16(count)),
        ("split.tensors.count".into(), GgufValue::I32(1_278)),
    ]);
}

fn selected_explicit_shards(reverse_insertion: bool) -> Vec<(PathBuf, GgufFile)> {
    let mut first = selected_file(true, false);
    let mut second = GgufFile {
        version: first.version,
        endianness: first.endianness,
        metadata: Vec::new(),
        tensors: Vec::new(),
        alignment: first.alignment,
        data_offset: first.data_offset,
        file_len: first.file_len,
    };
    let tensors = std::mem::take(&mut first.tensors);
    for (index, tensor) in tensors.into_iter().enumerate() {
        if index % 2 == 0 {
            first.tensors.push(tensor);
        } else {
            second.tensors.push(tensor);
        }
    }
    add_split_metadata(&mut first, 0, 2);
    add_split_metadata(&mut second, 1, 2);

    let mut shards = vec![
        (PathBuf::from("<hy3-00001-of-00002.gguf>"), first),
        (PathBuf::from("<hy3-00002-of-00002.gguf>"), second),
    ];
    if reverse_insertion {
        shards.reverse();
    }
    shards
}

fn selected_report() -> InspectionReport {
    let set = from_file(selected_file(true, false)).unwrap();
    build_report(&set).unwrap()
}

#[test]
fn full_selected_profile_has_exact_checked_aggregates() {
    let report = selected_report();

    assert_eq!(report.files.len(), 1);
    assert_eq!(report.files[0].path, "<in-memory-test-shard>");
    assert_eq!(report.files[0].ordinal, 0);
    assert_eq!(report.files[0].count, 1);
    assert_eq!(report.files[0].version, 3);
    assert_eq!(report.files[0].endianness, "little");
    assert_eq!(report.files[0].metadata_count, 45);
    assert_eq!(report.files[0].alignment, 32);
    assert_eq!(report.files[0].logical_size, 96_019_311_104);
    assert_eq!(report.files[0].data_offset, DATA_OFFSET);

    assert_eq!(report.gguf.version, 3);
    assert_eq!(report.gguf.endianness, "little");
    assert_eq!(report.gguf.authoritative_metadata_count, 45);
    assert_eq!(report.gguf.tensor_count, 1_278);
    assert_eq!(report.gguf.alignment, 32);
    assert_eq!(report.gguf.encoded_tensor_bytes, 96_014_150_912);

    assert_eq!(report.general.architecture, "hy_v3");
    assert_eq!(report.general.name.as_deref(), Some("Hy3 Src"));
    assert_eq!(report.general.license.as_deref(), Some("apache-2.0"));
    assert_eq!(report.general.size_label.as_deref(), Some("192x10B"));
    assert_eq!(report.general.quantization_version, Some(2));
    assert_eq!(report.general.file_type, Some(29));

    assert_eq!(report.tokenizer.model.as_deref(), Some("gpt2"));
    assert_eq!(report.tokenizer.pretokenizer.as_deref(), Some("hunyuan-dense"));
    assert_eq!(report.tokenizer.token_count, 120_832);
    assert_eq!(report.tokenizer.merge_count, Some(119_758));
    assert_eq!(report.tokenizer.token_type_count, Some(120_832));
    assert_eq!(report.tokenizer.bos_token_id, Some(120_000));
    assert_eq!(report.tokenizer.eos_token_id, Some(120_025));
    assert_eq!(report.tokenizer.padding_token_id, Some(120_002));
    assert_eq!(report.tokenizer.separator_token_id, Some(120_007));
    assert!(report.tokenizer.has_chat_template);

    assert_eq!(report.hy3.block_count, 80);
    assert_eq!(report.hy3.context_length, 1_048_576);
    assert_eq!(report.hy3.embedding_length, 4_096);
    assert_eq!(report.hy3.dense_ffn_length, 13_312);
    assert_eq!(report.hy3.expert_ffn_length, 1_536);
    assert_eq!(report.hy3.shared_expert_ffn_length, 1_536);
    assert_eq!(report.hy3.attention_head_count, 64);
    assert_eq!(report.hy3.attention_kv_head_count, 8);
    assert_eq!(report.hy3.key_length, 128);
    assert_eq!(report.hy3.value_length, 128);
    assert_eq!(report.hy3.rms_epsilon, 0.000_01);
    assert_eq!(report.hy3.expert_count, 192);
    assert_eq!(report.hy3.expert_used_count, 8);
    assert!(report.hy3.expert_weights_norm);
    assert_eq!(report.hy3.expert_gating_func, 2);
    assert_eq!(report.hy3.expert_weights_scale, 2.826);
    assert_eq!(report.hy3.rope_base, 11_158_840.0);
    assert_eq!(report.hy3.rope_scaling_type, "yarn");
    assert_eq!(report.hy3.yarn_factor, 4.0);
    assert_eq!(report.hy3.yarn_original_context, 262_144);
    assert_eq!(report.hy3.vocabulary_size, 120_832);
    assert!(!report.hy3.has_mtp);

    assert_eq!(
        report.types,
        BTreeMap::from([
            ("F32".into(), aggregate(479, 62_823_232, 251_292_928)),
            ("IQ2_S".into(), aggregate(627, 284_841_476_096, 91_238_285_312),),
            ("IQ3_S".into(), aggregate(91, 9_298_771_968, 3_995_566_080),),
            ("Q4_K".into(), aggregate(80, 335_544_320, 188_743_680)),
            ("Q5_K".into(), aggregate(1, 494_927_872, 340_262_912)),
        ])
    );
    assert_eq!(
        report.roles,
        BTreeMap::from([
            ("attention_k".into(), aggregate(80, 335_544_320, 107_479_040),),
            ("attention_k_norm".into(), aggregate(80, 10_240, 40_960),),
            ("attention_norm".into(), aggregate(80, 327_680, 1_310_720),),
            (
                "attention_output".into(),
                aggregate(80, 2_684_354_560, 1_153_433_600),
            ),
            ("attention_q".into(), aggregate(80, 2_684_354_560, 859_832_320),),
            ("attention_q_norm".into(), aggregate(80, 10_240, 40_960),),
            ("attention_v".into(), aggregate(80, 335_544_320, 188_743_680),),
            ("dense_down".into(), aggregate(1, 54_525_952, 23_429_120),),
            ("dense_gate".into(), aggregate(1, 54_525_952, 17_465_344),),
            ("dense_up".into(), aggregate(1, 54_525_952, 17_465_344),),
            ("ffn_norm".into(), aggregate(80, 327_680, 1_310_720)),
            ("output".into(), aggregate(1, 494_927_872, 340_262_912),),
            ("output_norm".into(), aggregate(1, 4_096, 16_384)),
            (
                "routed_down".into(),
                aggregate(79, 95_428_804_608, 31_227_641_856),
            ),
            (
                "routed_gate".into(),
                aggregate(79, 95_428_804_608, 30_567_038_976),
            ),
            ("routed_up".into(), aggregate(79, 95_428_804_608, 30_567_038_976),),
            ("router_input".into(), aggregate(79, 62_128_128, 248_512_512),),
            ("router_selection_bias".into(), aggregate(79, 15_168, 60_672),),
            ("shared_down".into(), aggregate(79, 497_025_024, 161_955_840),),
            ("shared_gate".into(), aggregate(79, 497_025_024, 159_203_328),),
            ("shared_up".into(), aggregate(79, 497_025_024, 159_203_328),),
            ("token_embedding".into(), aggregate(1, 494_927_872, 212_664_320),),
        ])
    );

    assert_eq!(
        report.tensors.total,
        aggregate(1_278, 295_033_543_488, 96_014_150_912)
    );
    assert_eq!(
        report.tensors.dense_layer_0_ffn,
        aggregate(3, 163_577_856, 58_359_808)
    );
    assert_eq!(
        report.tensors.routed_experts,
        aggregate(237, 286_286_413_824, 92_361_719_808)
    );
    assert_eq!(
        report.tensors.shared_experts,
        aggregate(237, 1_491_075_072, 480_362_496)
    );
    assert_eq!(
        report.tensors.attention,
        aggregate(320, 6_039_797_760, 2_309_488_640)
    );
    assert_eq!(report.tensors.routers, aggregate(158, 62_143_296, 248_573_184));
    assert_eq!(report.tensors.embeddings, aggregate(1, 494_927_872, 212_664_320));
    assert_eq!(report.tensors.norms, aggregate(321, 679_936, 2_719_744));
    assert_eq!(report.tensors.output, aggregate(1, 494_927_872, 340_262_912));

    assert_eq!(report.layers.len(), 80);
    assert_eq!(report.layers[&0], aggregate(11, 239_083_776, 87_262_208));
    for layer in 1..=4 {
        assert_eq!(report.layers[&layer], aggregate(16, 3_719_045_568, 1_331_676_928));
    }
    assert_eq!(report.layers[&5], aggregate(16, 3_719_045_568, 1_330_988_800));
    for layer in 6..80 {
        assert_eq!(report.layers[&layer], aggregate(16, 3_719_045_568, 1_198_868_224));
    }

    assert_eq!(report.expert_storage.expert_count, 192);
    assert_eq!(
        report.expert_storage.routed_banks,
        aggregate(237, 286_286_413_824, 92_361_719_808)
    );
    assert_eq!(
        report.expert_storage.shared_experts,
        aggregate(237, 1_491_075_072, 480_362_496)
    );
    let projections = &report.expert_storage.routed_projections;
    assert_eq!(projections.len(), 4);
    assert_eq!(projections["routed_gate/IQ2_S"].tensor_count, 79);
    assert_eq!(projections["routed_gate/IQ2_S"].slab_logical_elements, 6_291_456);
    assert_eq!(projections["routed_gate/IQ2_S"].slab_bytes, 2_015_232);
    assert_eq!(projections["routed_up/IQ2_S"].tensor_count, 79);
    assert_eq!(projections["routed_up/IQ2_S"].slab_bytes, 2_015_232);
    assert_eq!(projections["routed_down/IQ3_S"].tensor_count, 5);
    assert_eq!(projections["routed_down/IQ3_S"].slab_bytes, 2_703_360);
    assert_eq!(projections["routed_down/IQ2_S"].tensor_count, 74);
    assert_eq!(projections["routed_down/IQ2_S"].slab_bytes, 2_015_232);

    assert_eq!(
        report.unsupported_execution_types,
        ["F32", "IQ2_S", "IQ3_S", "Q4_K", "Q5_K"]
    );
    assert_eq!(
        report.warnings,
        ["Tensor payload bytes were not read or verified."]
    );
    assert!(report.warnings.iter().all(|warning| {
        let warning = warning.to_ascii_lowercase();
        !warning.contains("sparse")
            && !warning.contains("sparsity")
            && !warning.contains("physical allocation")
            && !warning.contains("allocated range")
    }));
}

#[test]
fn optional_presentation_metadata_remains_optional() {
    let set = from_file(selected_file(false, false)).unwrap();
    let report = build_report(&set).unwrap();

    assert_eq!(report.general.architecture, "hy_v3");
    assert_eq!(report.general.name, None);
    assert_eq!(report.general.license, None);
    assert_eq!(report.general.size_label, None);
    assert_eq!(report.general.quantization_version, None);
    assert_eq!(report.general.file_type, None);
}

#[cfg(unix)]
fn non_unicode_path() -> PathBuf {
    PathBuf::from(OsString::from_vec(vec![0xFF]))
}

#[cfg(windows)]
fn non_unicode_path() -> PathBuf {
    PathBuf::from(OsString::from_wide(&[0xD800]))
}

#[cfg(any(unix, windows))]
#[test]
fn report_construction_rejects_a_non_unicode_shard_path_before_rendering() {
    let invalid_path = non_unicode_path();
    let set = from_explicit_files(vec![(invalid_path.clone(), selected_file(true, false))]).unwrap();

    let error = build_report(&set).unwrap_err();

    assert!(matches!(
        error,
        ReportError::NonUnicodePath(path) if path == invalid_path
    ));
}

#[test]
fn descriptor_insertion_order_does_not_change_report_serialization() {
    let forward = from_file(selected_file(true, false)).unwrap();
    let reverse = from_file(selected_file(true, true)).unwrap();
    let forward = build_report(&forward).unwrap();
    let reverse = build_report(&reverse).unwrap();

    assert_eq!(forward, reverse);
    assert_eq!(render_text(&forward), render_text(&reverse));
    assert_eq!(render_json(&forward).unwrap(), render_json(&reverse).unwrap());
}

#[test]
fn explicit_shard_insertion_order_produces_one_canonical_report() {
    let forward = from_explicit_files(selected_explicit_shards(false)).unwrap();
    let reverse = from_explicit_files(selected_explicit_shards(true)).unwrap();
    let forward = build_report(&forward).unwrap();
    let reverse = build_report(&reverse).unwrap();

    assert_eq!(
        forward
            .files
            .iter()
            .map(|file| (file.path.clone(), file.ordinal, file.count))
            .collect::<Vec<_>>(),
        [
            ("<hy3-00001-of-00002.gguf>".to_owned(), 0, 2,),
            ("<hy3-00002-of-00002.gguf>".to_owned(), 1, 2,),
        ]
    );
    assert_eq!(forward, reverse);
    assert_eq!(forward.types, reverse.types);
    assert_eq!(forward.roles, reverse.roles);
    assert_eq!(forward.layers, reverse.layers);
    assert_eq!(
        forward.unsupported_execution_types,
        reverse.unsupported_execution_types
    );
    assert_eq!(forward.warnings, reverse.warnings);
    assert_eq!(
        forward.expert_storage.routed_projections,
        reverse.expert_storage.routed_projections
    );
    assert_eq!(render_text(&forward), render_text(&reverse));
    assert_eq!(render_json(&forward).unwrap(), render_json(&reverse).unwrap());
}

#[test]
fn heterogeneous_metadata_counts_are_truthful_per_file_and_authoritative_for_shard_zero() {
    let set = from_explicit_files(selected_explicit_shards(false)).unwrap();
    let report = build_report(&set).unwrap();
    let json = serde_json::to_value(&report).unwrap();
    let text = render_text(&report);

    assert_eq!(json["files"][0]["version"], 3);
    assert_eq!(json["files"][0]["endianness"], "little");
    assert_eq!(json["files"][0]["metadata_count"], 48);
    assert_eq!(json["files"][0]["alignment"], 32);
    assert_eq!(json["files"][1]["version"], 3);
    assert_eq!(json["files"][1]["endianness"], "little");
    assert_eq!(json["files"][1]["metadata_count"], 3);
    assert_eq!(json["files"][1]["alignment"], 32);
    assert_eq!(json["gguf"]["authoritative_metadata_count"], 48);
    assert!(json["gguf"].get("metadata_count").is_none());

    assert!(text.contains(
        "  <hy3-00001-of-00002.gguf>\n\
         \x20   shard ordinal: 0\n\
         \x20   shard count: 2\n\
         \x20   version: 3\n\
         \x20   endianness: little\n\
         \x20   metadata count: 48\n\
         \x20   alignment: 32 bytes (32 B)\n"
    ));
    assert!(text.contains(
        "  <hy3-00002-of-00002.gguf>\n\
         \x20   shard ordinal: 1\n\
         \x20   shard count: 2\n\
         \x20   version: 3\n\
         \x20   endianness: little\n\
         \x20   metadata count: 3\n\
         \x20   alignment: 32 bytes (32 B)\n"
    ));
    assert!(text.contains("\nGGUF\n  version: 3\n  endianness: little\n  authoritative metadata count: 48\n"));
    assert!(!text.contains("\nGGUF\n  version: 3\n  endianness: little\n  metadata count:"));
}

#[test]
fn aggregate_count_overflow_is_atomic() {
    let mut aggregate = Aggregate {
        count: u64::MAX,
        logical_elements: 2,
        encoded_bytes: 3,
    };
    let original = aggregate.clone();

    let error = aggregate.checked_add(1, 1, 1).unwrap_err();

    assert!(matches!(
        error,
        ReportError::ArithmeticOverflow("aggregate tensor count")
    ));
    assert_eq!(aggregate, original);
}

#[test]
fn aggregate_logical_element_overflow_is_atomic() {
    let mut aggregate = Aggregate {
        count: 1,
        logical_elements: u64::MAX,
        encoded_bytes: 3,
    };
    let original = aggregate.clone();

    let error = aggregate.checked_add(1, 1, 1).unwrap_err();

    assert!(matches!(
        error,
        ReportError::ArithmeticOverflow("aggregate logical element count")
    ));
    assert_eq!(aggregate, original);
}

#[test]
fn aggregate_encoded_byte_overflow_is_atomic() {
    let mut aggregate = Aggregate {
        count: 1,
        logical_elements: 2,
        encoded_bytes: u64::MAX,
    };
    let original = aggregate.clone();

    let error = aggregate.checked_add(1, 1, 1).unwrap_err();

    assert!(matches!(
        error,
        ReportError::ArithmeticOverflow("aggregate encoded byte count")
    ));
    assert_eq!(aggregate, original);
}

#[test]
fn text_and_pretty_json_are_stable_snapshots_of_the_same_owned_report() {
    let report = selected_report();
    let text = render_text(&report);
    let json = render_json(&report).unwrap();

    assert_eq!(text, include_str!("fixtures/expected-report.txt"));
    assert_eq!(json, include_str!("fixtures/expected-report.json"));
    assert_eq!(serde_json::from_str::<InspectionReport>(&json).unwrap(), report);

    let mut without_warnings = report;
    without_warnings.warnings.clear();
    assert!(!render_text(&without_warnings).contains("\nWarnings\n"));
}
