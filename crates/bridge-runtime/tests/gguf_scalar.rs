use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bridge_gguf_split::open_set;
use bridge_io_windows::ReadCancellation;
use bridge_kernels_cpu::CpuCapabilities;
use bridge_kernels_reference::ReferenceExecutionMode;
use bridge_model_hy3::validate_model_with_profile;
use bridge_prepare::{prepare_sidecar, PrepareOptions};
use bridge_runtime::{
    CancellationToken, CausalModel, ExpertSourceOptions, Hy3ChatEngine, Hy3MemoryBudget, Hy3ScalarError,
    Hy3ScalarModel, Hy3ScalarOptions, SamplingConfig,
};
use bridge_test_model::{ReducedHy3Model, CONTEXT_LENGTH, VOCABULARY_SIZE};
use bridge_tokenizer::{ChatMessage, ChatTemplateOptions, Hy3Tokenizer};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[test]
fn memory_budget_is_checked_before_selected_weight_allocation() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let set = open_set(&file.path).unwrap();
    let validated = validate_model_with_profile(&set, &profile).unwrap();
    let options = Hy3ScalarOptions {
        context_capacity: CONTEXT_LENGTH,
        kv_page_tokens: 8,
        expert_cache_bytes: 2 * 1024 * 1024,
        memory_headroom_bytes: 16 * 1024 * 1024,
        ..Hy3ScalarOptions::default()
    };
    let budget = Hy3MemoryBudget::for_validated(&validated, &options).unwrap();
    assert_eq!(
        budget.required_available_bytes,
        budget.resident_weight_bytes
            + budget.expert_cache_bytes
            + budget.first_kv_page_bytes
            + budget.headroom_bytes
    );
    budget.ensure_available(0).unwrap();
    budget.ensure_available(budget.required_available_bytes).unwrap();
    assert!(matches!(
        budget.ensure_available(budget.required_available_bytes - 1),
        Err(Hy3ScalarError::InsufficientPhysicalMemory { .. })
    ));
}

struct TemporaryModel {
    path: PathBuf,
}

impl TemporaryModel {
    fn write(bytes: &[u8]) -> Self {
        let ordinal = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "lightbridge-runtime-{}-{ordinal}.gguf",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        Self { path }
    }
}

impl Drop for TemporaryModel {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct TemporaryOutputs {
    paths: Vec<PathBuf>,
}

impl Drop for TemporaryOutputs {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_file(path);
        }
    }
}

#[test]
fn validated_gguf_streaming_runtime_matches_in_memory_reference_logits() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let options = Hy3ScalarOptions {
        context_capacity: CONTEXT_LENGTH,
        execution_mode: ReferenceExecutionMode::LlamaQ8K,
        ..Hy3ScalarOptions::default()
    };
    let loaded = Hy3ScalarModel::open_profile_for_testing(&file.path, &profile, options).unwrap();
    assert_eq!(loaded.vocabulary_size(), VOCABULARY_SIZE);
    assert_eq!(loaded.context_length(), CONTEXT_LENGTH);
    assert!(loaded.resident_weight_bytes() > 0);

    let mut expected_session = reference.new_session().unwrap();
    let mut actual_session = loaded.new_session().unwrap();
    let mut actual_logits = vec![0.0_f32; VOCABULARY_SIZE];
    for token in [3_u32, 11, 7] {
        let expected = reference
            .evaluate_token(&mut expected_session, ReferenceExecutionMode::LlamaQ8K, token)
            .unwrap();
        loaded
            .evaluate_token(&mut actual_session, token, &mut actual_logits)
            .unwrap();
        assert_eq!(
            actual_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(loaded.position(&actual_session), 3);
    assert!(loaded.cache_stats().unwrap().loads > 0);
    let heat = loaded.export_cache_heat(128).unwrap();
    let restored = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            execution_mode: ReferenceExecutionMode::LlamaQ8K,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    assert!(restored.import_cache_heat(&heat, 64 * 1024, 128).unwrap() > 0);
    assert!(restored.cache_stats().unwrap().heat_entries > 0);

    let snapshot = loaded
        .export_kv_snapshot(&actual_session, 4 * 1024 * 1024)
        .unwrap();
    let mut restored_session = restored.new_session().unwrap();
    restored
        .restore_kv_snapshot(&mut restored_session, &snapshot, 4 * 1024 * 1024)
        .unwrap();
    assert_eq!(restored.position(&restored_session), 3);
    for layer in 0..restored.config().block_count as usize {
        assert_eq!(restored_session.kv_stored_tokens(layer).unwrap(), 3);
    }
    let mut restored_logits = vec![0.0_f32; VOCABULARY_SIZE];
    loaded
        .evaluate_token(&mut actual_session, 13, &mut actual_logits)
        .unwrap();
    restored
        .evaluate_token(&mut restored_session, 13, &mut restored_logits)
        .unwrap();
    assert_eq!(
        actual_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        restored_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    loaded.reset_session(&mut actual_session);
    assert_eq!(loaded.position(&actual_session), 0);
}

#[test]
fn explicit_vnni_full_routes_are_bit_exact_to_scalar_q8_k_when_available() {
    let capabilities = CpuCapabilities::detect();
    let candidates = [
        (
            capabilities.avx_vnni_dot_kernel_available(),
            ReferenceExecutionMode::CpuParallelAvxVnni,
        ),
        (
            capabilities.avx512_dot_kernel_available(),
            ReferenceExecutionMode::CpuParallelAvx512Vnni,
        ),
    ];
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    for (available, mode) in candidates {
        if !available {
            continue;
        }
        let scalar = Hy3ScalarModel::open_profile_for_testing(
            &file.path,
            &profile,
            Hy3ScalarOptions {
                context_capacity: CONTEXT_LENGTH,
                execution_mode: ReferenceExecutionMode::LlamaQ8K,
                ..Hy3ScalarOptions::default()
            },
        )
        .unwrap();
        let candidate = Hy3ScalarModel::open_profile_for_testing(
            &file.path,
            &profile,
            Hy3ScalarOptions {
                context_capacity: CONTEXT_LENGTH,
                execution_mode: mode,
                cpu_threads: 2,
                ..Hy3ScalarOptions::default()
            },
        )
        .unwrap();
        let mut scalar_session = scalar.new_session().unwrap();
        let mut candidate_session = candidate.new_session().unwrap();
        let mut scalar_logits = vec![0.0_f32; VOCABULARY_SIZE];
        let mut candidate_logits = vec![0.0_f32; VOCABULARY_SIZE];
        for token in [3_u32, 11, 7] {
            scalar
                .evaluate_token(&mut scalar_session, token, &mut scalar_logits)
                .unwrap();
            candidate
                .evaluate_token(&mut candidate_session, token, &mut candidate_logits)
                .unwrap();
            assert_eq!(
                candidate_logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                scalar_logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{mode:?}"
            );
        }
    }
}

#[cfg(windows)]
#[test]
fn explicit_cuda_full_route_is_bit_exact_deterministic_and_can_continue_on_cpu() {
    if bridge_kernels_cuda::runtime_reusable_packed_q8k_canary().is_err() {
        return;
    }
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let scalar = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            execution_mode: ReferenceExecutionMode::LlamaQ8K,
            prefill_chunk: 2,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let cuda = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            execution_mode: ReferenceExecutionMode::CudaQ8K,
            cpu_threads: 2,
            prefill_chunk: 2,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    assert_eq!(cuda.backend_name(), "cuda_streaming_q8_k");
    assert!(!cuda.cuda_fallback_active());

    let mut scalar_session = scalar.new_session().unwrap();
    let mut cuda_session = cuda.new_session().unwrap();
    let mut repeat_session = cuda.new_session().unwrap();
    let mut scalar_logits = vec![0.0_f32; VOCABULARY_SIZE];
    let mut cuda_logits = vec![0.0_f32; VOCABULARY_SIZE];
    let mut repeat_logits = vec![0.0_f32; VOCABULARY_SIZE];
    scalar
        .evaluate_tokens_with_projection(&mut scalar_session, &[3, 11], &mut scalar_logits, true)
        .unwrap();
    cuda.evaluate_tokens_with_projection(&mut cuda_session, &[3, 11], &mut cuda_logits, true)
        .unwrap();
    cuda.evaluate_tokens_with_projection(&mut repeat_session, &[3, 11], &mut repeat_logits, true)
        .unwrap();
    assert_eq!(
        cuda_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        scalar_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        repeat_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        cuda_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(cuda.position(&cuda_session), 2);
    for layer in 0..cuda.config().block_count as usize {
        assert_eq!(cuda_session.kv_stored_tokens(layer).unwrap(), 2);
    }

    assert!(cuda.fall_back_to_cpu());
    assert!(cuda.cuda_fallback_active());
    assert_eq!(cuda.backend_name(), "cpu_parallel_avx2_q8_k");
    scalar
        .evaluate_token(&mut scalar_session, 7, &mut scalar_logits)
        .unwrap();
    cuda.evaluate_token(&mut cuda_session, 7, &mut cuda_logits)
        .unwrap();
    assert_eq!(
        cuda_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        scalar_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(cuda.position(&cuda_session), 3);
}

#[test]
fn grouped_prefill_matches_token_serial_logits_and_kv_exactly() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let serial = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            prefill_chunk: 1,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let grouped = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            prefill_chunk: 4,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let tokens = [3_u32, 11, 7, 5];
    let mut serial_session = serial.new_session().unwrap();
    let mut grouped_session = grouped.new_session().unwrap();
    let mut serial_logits = vec![0.0_f32; VOCABULARY_SIZE];
    let mut grouped_logits = vec![0.0_f32; VOCABULARY_SIZE];
    for (index, &token) in tokens.iter().enumerate() {
        serial
            .evaluate_token_with_projection(
                &mut serial_session,
                token,
                &mut serial_logits,
                index + 1 == tokens.len(),
            )
            .unwrap();
    }
    grouped
        .evaluate_tokens_with_projection(&mut grouped_session, &tokens, &mut grouped_logits, true)
        .unwrap();
    assert_eq!(
        grouped_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        serial_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(grouped.position(&grouped_session), tokens.len());
    for layer in 0..grouped.config().block_count as usize {
        assert_eq!(
            grouped_session.kv_stored_tokens(layer).unwrap(),
            serial_session.kv_stored_tokens(layer).unwrap()
        );
    }
}

#[test]
fn speculative_group_returns_every_serial_logit_and_rewinds_kv_exactly() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let serial = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            prefill_chunk: 1,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let speculative = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            prefill_chunk: 2,
            speculative_ngram_t: Some(2),
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();

    let mut serial_session = serial.new_session().unwrap();
    let mut serial_rows = vec![0.0_f32; 2 * VOCABULARY_SIZE];
    for (&token, row) in [3_u32, 11]
        .iter()
        .zip(serial_rows.chunks_exact_mut(VOCABULARY_SIZE))
    {
        serial.evaluate_token(&mut serial_session, token, row).unwrap();
    }
    let mut speculative_session = speculative.new_session().unwrap();
    let mut speculative_rows = vec![0.0_f32; 2 * VOCABULARY_SIZE];
    speculative
        .evaluate_speculative_tokens(&mut speculative_session, &[3, 11], &mut speculative_rows)
        .unwrap()
        .unwrap();
    assert_eq!(
        speculative_rows
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        serial_rows
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    speculative
        .rewind_speculative(&mut speculative_session, 1)
        .unwrap()
        .unwrap();
    assert_eq!(speculative.position(&speculative_session), 1);
    for layer in 0..speculative.config().block_count as usize {
        assert_eq!(speculative_session.kv_stored_tokens(layer).unwrap(), 1);
    }

    let mut reference_after_rewind = serial.new_session().unwrap();
    let mut expected = vec![0.0_f32; VOCABULARY_SIZE];
    serial
        .evaluate_token(&mut reference_after_rewind, 3, &mut expected)
        .unwrap();
    serial
        .evaluate_token(&mut reference_after_rewind, 7, &mut expected)
        .unwrap();
    let mut actual = vec![0.0_f32; VOCABULARY_SIZE];
    speculative
        .evaluate_token(&mut speculative_session, 7, &mut actual)
        .unwrap();
    assert_eq!(
        actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn prepared_sidecar_and_direct_gguf_experts_produce_identical_logits() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let ordinal = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let data_path = std::env::temp_dir().join(format!(
        "lightbridge-runtime-{}-{ordinal}.experts",
        std::process::id()
    ));
    let manifest_path = data_path.with_extension("experts.json");
    let _outputs = TemporaryOutputs {
        paths: vec![data_path.clone(), manifest_path.clone()],
    };
    let set = open_set(&file.path).unwrap();
    let validated = validate_model_with_profile(&set, &profile).unwrap();
    prepare_sidecar(
        &set,
        &validated,
        &data_path,
        &manifest_path,
        PrepareOptions::default(),
        &ReadCancellation::new(),
    )
    .unwrap();

    let common = Hy3ScalarOptions {
        context_capacity: CONTEXT_LENGTH,
        execution_mode: ReferenceExecutionMode::LlamaQ8K,
        ..Hy3ScalarOptions::default()
    };
    let direct = Hy3ScalarModel::open_profile_for_testing(&file.path, &profile, common.clone()).unwrap();
    let sidecar = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            expert_source: ExpertSourceOptions::Sidecar {
                data_path,
                manifest_path,
                verify_data_hash: true,
                verify_source_bindings: true,
            },
            ..common
        },
    )
    .unwrap();
    let mut direct_session = direct.new_session().unwrap();
    let mut sidecar_session = sidecar.new_session().unwrap();
    let mut direct_logits = vec![0.0_f32; VOCABULARY_SIZE];
    let mut sidecar_logits = vec![0.0_f32; VOCABULARY_SIZE];

    for token in [5_u32, 9] {
        direct
            .evaluate_token(&mut direct_session, token, &mut direct_logits)
            .unwrap();
        sidecar
            .evaluate_token(&mut sidecar_session, token, &mut sidecar_logits)
            .unwrap();
        assert_eq!(
            direct_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            sidecar_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn failed_expert_execution_rewinds_every_layer_and_allows_a_clean_retry() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let set = open_set(&file.path).unwrap();
    let validated = validate_model_with_profile(&set, &profile).unwrap();
    let mut bytes = fs::read(&file.path).unwrap();
    for tensor in validated
        .tensors()
        .iter()
        .filter(|tensor| tensor.role().is_routed_expert())
    {
        let block = bridge_quant_layout::layout(tensor.location().descriptor().ty()).unwrap();
        let range = tensor.location().absolute_range();
        let start = usize::try_from(range.start).unwrap();
        let end = usize::try_from(range.end).unwrap();
        for offset in (start..end).step_by(block.block_bytes) {
            bytes[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        }
    }
    fs::write(&file.path, bytes).unwrap();

    let model = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            execution_mode: ReferenceExecutionMode::CpuParallelQ8K,
            cpu_threads: 2,
            prefill_chunk: 2,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let mut session = model.new_session().unwrap();
    let mut logits = vec![0.0_f32; VOCABULARY_SIZE];

    for _ in 0..2 {
        assert!(model
            .evaluate_tokens_with_projection(&mut session, &[3, 11], &mut logits, true)
            .is_err());
        assert_eq!(model.position(&session), 0);
        for layer in 0..model.config().block_count as usize {
            assert_eq!(session.kv_stored_tokens(layer).unwrap(), 0);
        }
    }
}

#[cfg(windows)]
#[test]
fn cuda_failure_rewinds_all_kv_switches_to_cpu_and_reports_failed_retry() {
    if bridge_kernels_cuda::runtime_reusable_packed_q8k_canary().is_err() {
        return;
    }
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes().unwrap());
    let set = open_set(&file.path).unwrap();
    let validated = validate_model_with_profile(&set, &profile).unwrap();
    let mut bytes = fs::read(&file.path).unwrap();
    for tensor in validated
        .tensors()
        .iter()
        .filter(|tensor| tensor.role().is_routed_expert())
    {
        let block = bridge_quant_layout::layout(tensor.location().descriptor().ty()).unwrap();
        let range = tensor.location().absolute_range();
        let start = usize::try_from(range.start).unwrap();
        let end = usize::try_from(range.end).unwrap();
        for offset in (start..end).step_by(block.block_bytes) {
            bytes[offset..offset + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        }
    }
    fs::write(&file.path, bytes).unwrap();

    let model = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: CONTEXT_LENGTH,
            execution_mode: ReferenceExecutionMode::CudaQ8K,
            cpu_threads: 2,
            prefill_chunk: 2,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let mut session = model.new_session().unwrap();
    let mut logits = vec![0.0_f32; VOCABULARY_SIZE];
    let error = model
        .evaluate_tokens_with_projection(&mut session, &[3, 11], &mut logits, true)
        .unwrap_err();
    assert!(matches!(error, Hy3ScalarError::CudaFallbackFailed { .. }));
    assert!(model.cuda_fallback_active());
    assert_eq!(model.backend_name(), "cpu_parallel_avx2_q8_k");
    assert_eq!(model.position(&session), 0);
    for layer in 0..model.config().block_count as usize {
        assert_eq!(session.kv_stored_tokens(layer).unwrap(), 0);
    }

    let retry = model
        .evaluate_tokens_with_projection(&mut session, &[3, 11], &mut logits, true)
        .unwrap_err();
    assert!(!matches!(retry, Hy3ScalarError::CudaFallbackFailed { .. }));
    assert_eq!(model.position(&session), 0);
    for layer in 0..model.config().block_count as usize {
        assert_eq!(session.kv_stored_tokens(layer).unwrap(), 0);
    }
}

#[test]
fn chat_session_snapshot_restores_history_logits_and_model_bound_kv() {
    let reference = ReducedHy3Model::new().unwrap();
    let profile = reference.profile().unwrap();
    let file = TemporaryModel::write(&reference.gguf_bytes_with_chat_tokenizer().unwrap());
    let parsed = bridge_gguf::open(&file.path).unwrap();
    let tokenizer = Hy3Tokenizer::from_gguf(&parsed).unwrap();
    let model = Hy3ScalarModel::open_profile_for_testing(
        &file.path,
        &profile,
        Hy3ScalarOptions {
            context_capacity: 64,
            kv_page_tokens: 8,
            expert_cache_bytes: 2 * 1024 * 1024,
            execution_mode: ReferenceExecutionMode::LlamaQ8K,
            ..Hy3ScalarOptions::default()
        },
    )
    .unwrap();
    let engine = Hy3ChatEngine::from_parts(model, tokenizer).unwrap();
    let mut session = engine.new_session().unwrap();
    let completion = engine
        .complete_in_session(
            &mut session,
            &[ChatMessage::user("a")],
            &ChatTemplateOptions::default(),
            SamplingConfig {
                max_new_tokens: 2,
                temperature: 0.0,
                ..SamplingConfig::default()
            },
            &CancellationToken::new(),
            |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
    assert!(!completion.generation.token_ids.is_empty());
    assert!(session.has_logits());
    assert_eq!(engine.session_position(&session), session.history().len());

    let snapshot = engine.export_session(&session, 4 * 1024 * 1024).unwrap();
    let mut restored = engine.restore_session(&snapshot, 4 * 1024 * 1024).unwrap();
    assert_eq!(restored.history(), session.history());
    assert_eq!(
        engine.session_position(&restored),
        engine.session_position(&session)
    );
    assert!(restored.has_logits());

    let continued_messages = [
        ChatMessage::user("a"),
        ChatMessage::assistant(completion.text),
        ChatMessage::user("a"),
    ];
    let continued_prompt = engine
        .tokenizer()
        .format_and_encode(&continued_messages, &ChatTemplateOptions::default())
        .unwrap();
    assert!(continued_prompt.starts_with(restored.history()));
    let continuation = engine
        .complete_in_session(
            &mut restored,
            &continued_messages,
            &ChatTemplateOptions::default(),
            SamplingConfig {
                max_new_tokens: 1,
                temperature: 0.0,
                ..SamplingConfig::default()
            },
            &CancellationToken::new(),
            |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
    assert!(continuation.cached_prompt_tokens > 0);
    assert_eq!(
        continuation.generation.stats.prompt_tokens,
        continuation.prompt_token_ids.len()
    );

    let mut corrupted = snapshot;
    corrupted[16] ^= 0x40;
    assert!(engine.restore_session(&corrupted, 4 * 1024 * 1024).is_err());
}
