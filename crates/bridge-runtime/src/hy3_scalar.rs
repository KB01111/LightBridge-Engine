use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use bridge_cache::{CacheConfig, CacheLease, CacheStats, CompressedCache, LoadError};
use bridge_core::ggml_type::GgmlType;
use bridge_core::sys::memory_status;
use bridge_format::{ExpertKey, Sidecar};
use bridge_gguf::Endianness;
use bridge_gguf_split::{open_set, GgufSet};
use bridge_io_windows::{
    file_storage, PositionedFile, ReadCancellation, ReadLimits, ReadSlotPool, SlotPoolError,
};
use bridge_kernels_cpu::{CpuBackend, CpuBackendConfig, CpuCapabilities};
use bridge_kernels_reference::{
    gemv_into, hy3_block_forward_token, hy3_moe_finish_token, hy3_moe_route_token, weighted_rms_norm_into,
    Hy3AttentionWeights, Hy3BlockExecution, Hy3BlockScratch, Hy3BlockWeights, Hy3FeedForwardWeights,
    Hy3RopeParams, Hy3StreamingMoeWeights, PackedMatrix, PayloadEndian, ReferenceExecutionMode,
    SelectedExpert, SwiGluExpert,
};
use bridge_kv_gqa::PagedKvCache;
use bridge_model_hy3::{
    validate_model_with_profile, validate_selected_model, ExpertSlab, Hy3Config, Hy3Profile, Hy3Tensor,
    Hy3TensorRole, ValidatedHy3Model,
};
use bridge_prepare::{tensor_directory_sha256, verify_source_bindings, DirectExpertStore};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::CausalModel;

pub const SELECTED_HY3_IQ2_M_BYTES: u64 = 96_019_311_104;
pub const SELECTED_HY3_IQ2_M_SHA256: &str =
    "1c02c57e4dc8b55a254a5329c6c248fa7bf741644b6936898793f272d3292ea7";
pub const DEFAULT_MEMORY_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

const HASH_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const MAX_STREAMING_EXPERTS: usize = 64;
const MAX_PREFILL_CHUNK: usize = 8;

#[derive(Debug, Clone)]
pub enum ExpertSourceOptions {
    Direct,
    Sidecar {
        data_path: PathBuf,
        manifest_path: PathBuf,
        verify_data_hash: bool,
        verify_source_bindings: bool,
    },
}

#[derive(Debug, Clone)]
pub struct Hy3ScalarOptions {
    pub context_capacity: usize,
    pub kv_page_tokens: usize,
    pub expert_cache_bytes: usize,
    pub cache_admit_after_requests: u64,
    pub execution_mode: ReferenceExecutionMode,
    pub cpu_threads: usize,
    pub cpu_set_ids: Vec<u32>,
    pub prefill_chunk: usize,
    pub speculative_ngram_t: Option<usize>,
    pub memory_headroom_bytes: usize,
    pub expert_source: ExpertSourceOptions,
}

impl Default for Hy3ScalarOptions {
    /// Creates the default scalar model runtime options.
    ///
    /// # Examples
    ///
    /// ```
    /// let options = Hy3ScalarOptions::default();
    /// assert_eq!(options.context_capacity, 2_048);
    /// assert_eq!(options.prefill_chunk, 1);
    /// assert!(options.speculative_ngram_t.is_none());
    /// ```
    fn default() -> Self {
        Self {
            context_capacity: 2_048,
            kv_page_tokens: 64,
            expert_cache_bytes: 2 * 1024 * 1024 * 1024,
            cache_admit_after_requests: 2,
            execution_mode: ReferenceExecutionMode::CpuParallelQ8K,
            cpu_threads: bridge_kernels_cpu::recommended_thread_count(),
            cpu_set_ids: Vec::new(),
            prefill_chunk: 1,
            speculative_ngram_t: None,
            memory_headroom_bytes: DEFAULT_MEMORY_HEADROOM_BYTES,
            expert_source: ExpertSourceOptions::Direct,
        }
    }
}

#[derive(Debug)]
pub struct Hy3ScalarModel {
    config: Hy3Config,
    context_capacity: usize,
    execution_mode: ReferenceExecutionMode,
    cuda_disabled: AtomicBool,
    cpu_backend: Option<CpuBackend>,
    prefill_chunk: usize,
    speculative_ngram_t: Option<usize>,
    kv_page_tokens: usize,
    source_paths: Vec<PathBuf>,
    source_sha256: Vec<String>,
    model_fingerprint: [u8; 32],
    resident_weight_bytes: usize,
    embeddings: OwnedMatrix,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
    output: OwnedMatrix,
    rope: Hy3RopeParams,
    expert_source: Option<ExpertSource>,
    expert_read_slots: Option<ReadSlotPool>,
    expert_cache: CompressedCache<ExpertKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectedPayloadReport {
    pub schema_valid: bool,
    pub files: Vec<SelectedPayloadFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectedPayloadFile {
    pub path: PathBuf,
    pub logical_bytes: u64,
    pub allocated_bytes: u64,
    pub sparse: bool,
    pub compressed: bool,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Hy3MemoryBudget {
    pub resident_weight_bytes: u64,
    pub expert_cache_bytes: u64,
    pub first_kv_page_bytes: u64,
    pub headroom_bytes: u64,
    pub required_available_bytes: u64,
}

impl Hy3MemoryBudget {
    pub fn for_validated(
        model: &ValidatedHy3Model,
        options: &Hy3ScalarOptions,
    ) -> Result<Self, Hy3ScalarError> {
        let resident_weight_bytes = model
            .tensors()
            .iter()
            .filter(|tensor| !tensor.role().is_routed_expert())
            .try_fold(0_u64, |total, tensor| {
                let range = tensor.location().absolute_range();
                total.checked_add(range.end - range.start)
            })
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        let page_tokens = options.kv_page_tokens.min(options.context_capacity) as u64;
        let config = model.config();
        let first_kv_page_bytes = u64::from(config.block_count)
            .checked_mul(u64::from(config.attention_kv_head_count))
            .and_then(|value| {
                value.checked_mul(u64::from(config.key_length) + u64::from(config.value_length))
            })
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| value.checked_mul(page_tokens))
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        let expert_cache_bytes =
            u64::try_from(options.expert_cache_bytes).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
        let headroom_bytes =
            u64::try_from(options.memory_headroom_bytes).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
        let required_available_bytes = resident_weight_bytes
            .checked_add(expert_cache_bytes)
            .and_then(|value| value.checked_add(first_kv_page_bytes))
            .and_then(|value| value.checked_add(headroom_bytes))
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        Ok(Self {
            resident_weight_bytes,
            expert_cache_bytes,
            first_kv_page_bytes,
            headroom_bytes,
            required_available_bytes,
        })
    }

    pub fn ensure_available(self, available: u64) -> Result<(), Hy3ScalarError> {
        if available == 0 || available >= self.required_available_bytes {
            return Ok(());
        }
        Err(Hy3ScalarError::InsufficientPhysicalMemory {
            required: self.required_available_bytes,
            available,
            resident_weights: self.resident_weight_bytes,
            expert_cache: self.expert_cache_bytes,
            first_kv_page: self.first_kv_page_bytes,
            headroom: self.headroom_bytes,
        })
    }
}

/// Validates the selected schema and authenticates every payload byte without
/// allocating executable model weights.
pub fn validate_selected_payload(entry: impl AsRef<Path>) -> Result<SelectedPayloadReport, Hy3ScalarError> {
    let set = open_set(entry)?;
    if set.files().len() != 1 {
        return Err(Hy3ScalarError::SelectedModelShardCount {
            actual: set.files().len(),
        });
    }
    validate_selected_model(&set)?;
    let file = &set.files()[0];
    if file.parsed().file_len != SELECTED_HY3_IQ2_M_BYTES {
        return Err(Hy3ScalarError::SelectedModelLength {
            expected: SELECTED_HY3_IQ2_M_BYTES,
            actual: file.parsed().file_len,
        });
    }
    let storage = file_storage(file.path()).map_err(|source| Hy3ScalarError::StorageInspection {
        path: file.path().to_owned(),
        source,
    })?;
    if storage.is_sparse && storage.allocated_bytes < storage.logical_bytes {
        return Err(Hy3ScalarError::SparseModelPayload {
            path: file.path().to_owned(),
            logical_bytes: storage.logical_bytes,
            allocated_bytes: storage.allocated_bytes,
        });
    }
    let hashes = verify_file_hashes(&set, &[SELECTED_HY3_IQ2_M_SHA256])?;
    Ok(SelectedPayloadReport {
        schema_valid: true,
        files: vec![SelectedPayloadFile {
            path: file.path().to_owned(),
            logical_bytes: storage.logical_bytes,
            allocated_bytes: storage.allocated_bytes,
            sparse: storage.is_sparse,
            compressed: storage.is_compressed,
            sha256: hashes[0].clone(),
        }],
    })
}

impl Hy3ScalarModel {
    /// Opens the exact selected Hy3 IQ2_M checkpoint and authenticates its
    /// complete payload before any tensor is made executable.
    pub fn open_selected(entry: impl AsRef<Path>, options: Hy3ScalarOptions) -> Result<Self, Hy3ScalarError> {
        let set = open_set(entry)?;
        if set.files().len() != 1 {
            return Err(Hy3ScalarError::SelectedModelShardCount {
                actual: set.files().len(),
            });
        }
        let file = &set.files()[0];
        if file.parsed().file_len != SELECTED_HY3_IQ2_M_BYTES {
            return Err(Hy3ScalarError::SelectedModelLength {
                expected: SELECTED_HY3_IQ2_M_BYTES,
                actual: file.parsed().file_len,
            });
        }
        let storage = file_storage(file.path()).map_err(|source| Hy3ScalarError::StorageInspection {
            path: file.path().to_owned(),
            source,
        })?;
        if storage.is_sparse && storage.allocated_bytes < storage.logical_bytes {
            return Err(Hy3ScalarError::SparseModelPayload {
                path: file.path().to_owned(),
                logical_bytes: storage.logical_bytes,
                allocated_bytes: storage.allocated_bytes,
            });
        }
        let source_sha256 = verify_file_hashes(&set, &[SELECTED_HY3_IQ2_M_SHA256])?;
        let validated = validate_selected_model(&set)?;
        Self::load_validated(set, validated, source_sha256, options, true)
    }

    /// Opens an explicitly authorized profile without a full-file hash.
    ///
    /// This entry point exists for deterministic fixtures and local oracle
    /// tests. Product commands use [`Self::open_selected`].
    pub fn open_profile_for_testing(
        entry: impl AsRef<Path>,
        profile: &Hy3Profile,
        options: Hy3ScalarOptions,
    ) -> Result<Self, Hy3ScalarError> {
        let set = open_set(entry)?;
        let validated = validate_model_with_profile(&set, profile)?;
        Self::load_validated(set, validated, Vec::new(), options, false)
    }

    pub const fn config(&self) -> &Hy3Config {
        &self.config
    }

    pub fn source_paths(&self) -> &[PathBuf] {
        &self.source_paths
    }

    pub fn source_sha256(&self) -> &[String] {
        &self.source_sha256
    }

    pub const fn model_fingerprint(&self) -> [u8; 32] {
        self.model_fingerprint
    }

    /// Returns the number of bytes occupied by resident model weights.
    ///
    /// # Examples
    ///
    /// ```
    /// let bytes = model.resident_weight_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    /// Identifies the currently active execution backend.
    ///
    /// # Examples
    ///
    /// ```
    /// let name = model.backend_name();
    /// assert!(!name.is_empty());
    /// ```
    pub fn backend_name(&self) -> &'static str {
        match self.active_execution_mode() {
            ReferenceExecutionMode::DequantF32 => "scalar_reference_dequant_f32",
            ReferenceExecutionMode::LlamaQ8K => "scalar_reference_q8_k",
            ReferenceExecutionMode::CpuParallelQ8K => self
                .cpu_backend
                .as_ref()
                .map(CpuBackend::backend_name)
                .unwrap_or("cpu_parallel_scalar_q8_k"),
            ReferenceExecutionMode::CpuParallelAvxVnni => "cpu_parallel_avx_vnni_q8_k",
            ReferenceExecutionMode::CpuParallelAvx512Vnni => "cpu_parallel_avx512_vnni_q8_k",
            ReferenceExecutionMode::CudaQ8K => "cuda_streaming_q8_k",
        }
    }

    pub fn cpu_threads(&self) -> Option<usize> {
        self.cpu_backend.as_ref().map(|backend| backend.config().threads)
    }

    /// Returns the CPU set identifiers configured for the CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// let set_ids = model.cpu_set_ids();
    /// ```
    pub fn cpu_set_ids...
    pub fn cpu_set_ids(&self) -> Option<&[u32]> {
        self.cpu_backend.as_ref().map(CpuBackend::cpu_set_ids)
    }

    /// Determines whether the active CPU execution path has SIMD acceleration available.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn example(model: &Hy3ScalarModel) {
    /// let simd_active = model.cpu_simd_active();
    /// assert!(simd_active || !simd_active);
    /// # }
    /// ```
    pub fn cpu_simd_active(&self) -> bool {
        match self.active_execution_mode() {
            ReferenceExecutionMode::CpuParallelAvxVnni => {
                CpuCapabilities::detect().avx_vnni_dot_kernel_available()
            }
            ReferenceExecutionMode::CpuParallelAvx512Vnni => {
                CpuCapabilities::detect().avx512_dot_kernel_available()
            }
            ReferenceExecutionMode::CudaQ8K => false,
            _ => self.cpu_backend.as_ref().is_some_and(CpuBackend::simd_active),
        }
    }

    /// Reports whether CUDA execution has fallen back to CPU execution.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// assert!(!model.cuda_fallback_active());
    /// ```
    ///
    /// # Returns
    ///
    /// `true` if CUDA execution is configured and disabled due to fallback, `false` otherwise.
    pub fn cuda_fallback_active(&self) -> bool {
        self.execution_mode == ReferenceExecutionMode::CudaQ8K && self.cuda_disabled.load(Ordering::Acquire)
    }

    /// Activates CPU fallback for a CUDA-configured model.
    ///
    /// Returns `true` only when this call changes the model from CUDA execution to CPU fallback.
    /// Repeated calls and calls on non-CUDA models return `false`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let model = load_cuda_model();
    /// assert!(model.fall_back_to_cpu());
    /// assert!(!model.fall_back_to_cpu());
    /// ```
    pub fn fall_back_to_cpu(&self) -> bool {
        self.execution_mode == ReferenceExecutionMode::CudaQ8K
            && !self.cuda_disabled.swap(true, Ordering::AcqRel)
    }

    /// Selects the execution mode currently used by the model, applying CPU fallback when CUDA execution has been disabled.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let model: Hy3ScalarModel = todo!();
    /// let mode = model.active_execution_mode();
    /// assert!(matches!(
    ///     mode,
    ///     ReferenceExecutionMode::CpuParallelQ8K | ReferenceExecutionMode::CudaQ8K
    /// ));
    /// ```
    fn active_execution_mode(&self) -> ReferenceExecutionMode {
        if self.cuda_fallback_active() {
            ReferenceExecutionMode::CpuParallelQ8K
        } else {
            self.execution_mode
        }
    }

    /// Indicates whether expert payload prefetching can run in parallel.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let model: Hy3ScalarModel = todo!();
    /// let parallel = model.parallel_expert_prefetch();
    /// ```
    pub fn parallel_expert_prefetch(&self) -> bool {
        self.cpu_backend.is_some()
    }

    pub fn cache_stats(&self) -> Result<CacheStats, Hy3ScalarError> {
        Ok(self.expert_cache.stats()?)
    }

    pub fn clear_unpinned_experts(&self) -> Result<usize, Hy3ScalarError> {
        Ok(self.expert_cache.clear_unpinned()?)
    }

    pub fn export_cache_heat(&self, maximum_entries: usize) -> Result<Vec<u8>, Hy3ScalarError> {
        Ok(self.expert_cache.export_heat_json(maximum_entries)?)
    }

    pub fn import_cache_heat(
        &self,
        bytes: &[u8],
        maximum_json_bytes: usize,
        maximum_entries: usize,
    ) -> Result<usize, Hy3ScalarError> {
        Ok(self
            .expert_cache
            .import_heat_json(bytes, maximum_json_bytes, maximum_entries)?)
    }

    pub fn export_kv_snapshot(
        &self,
        session: &Hy3ScalarSession,
        maximum_bytes: usize,
    ) -> Result<Vec<u8>, Hy3ScalarError> {
        for layer in 0..self.config.block_count as usize {
            let actual = session.cache.stored_tokens(layer)?;
            if actual != session.position {
                return Err(Hy3ScalarError::InconsistentKvLength {
                    layer,
                    expected: session.position,
                    actual,
                });
            }
        }
        Ok(session
            .cache
            .export_snapshot(self.model_fingerprint, maximum_bytes)?)
    }

    pub fn restore_kv_snapshot(
        &self,
        session: &mut Hy3ScalarSession,
        bytes: &[u8],
        maximum_bytes: usize,
    ) -> Result<(), Hy3ScalarError> {
        session
            .cache
            .restore_uniform_snapshot(self.model_fingerprint, bytes, maximum_bytes)?;
        let position = session.cache.stored_tokens(0)?;
        session.rollback_to(position)
    }

    /// Loads a validated Hy3 model and initializes its execution, tensor, KV, and expert-cache state.
    ///
    /// Performs option, payload endianness, backend, CUDA qualification, and optional memory checks
    /// before loading model weights and configuring expert payload access.
    ///
    /// # Parameters
    ///
    /// * `set` - GGUF model files containing the validated model payload.
    /// * `validated` - Validated model structure and configuration.
    /// * `source_sha256` - SHA-256 values associated with the model sources.
    /// * `options` - Runtime, memory, execution, and expert-source configuration.
    /// * `enforce_memory_preflight` - Whether to check the available physical memory before loading.
    ///
    /// # Errors
    ///
    /// Returns an error if validation, memory checks, backend qualification, tensor loading, expert
    /// source setup, or runtime resource allocation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let model = Hy3ScalarModel::load_validated(
    ///     set,
    ///     validated,
    ///     source_sha256,
    ///     options,
    ///     true,
    /// )?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    fn load_validated(
        set: GgufSet,
        validated: ValidatedHy3Model,
        source_sha256: Vec<String>,
        options: Hy3ScalarOptions,
        enforce_memory_preflight: bool,
    ) -> Result<Self, Hy3ScalarError> {
        validate_options(&validated, &options)?;
        if enforce_memory_preflight {
            let budget = Hy3MemoryBudget::for_validated(&validated, &options)?;
            budget.ensure_available(memory_status().available_physical)?;
        }
        for shard in set.files() {
            if shard.parsed().endianness != Endianness::Little {
                return Err(Hy3ScalarError::BigEndianPayload(shard.path().to_owned()));
            }
        }
        if options.execution_mode == ReferenceExecutionMode::CudaQ8K {
            let qualification = bridge_kernels_cuda::runtime_reusable_packed_q8k_canary()?;
            if !qualification.bit_exact || !qualification.deterministic {
                return Err(Hy3ScalarError::CudaQualification {
                    bit_exact: qualification.bit_exact,
                    deterministic: qualification.deterministic,
                });
            }
        }
        let cpu_backend = match options.execution_mode {
            ReferenceExecutionMode::CpuParallelQ8K
            | ReferenceExecutionMode::CpuParallelAvxVnni
            | ReferenceExecutionMode::CpuParallelAvx512Vnni
            | ReferenceExecutionMode::CudaQ8K => {
                let capabilities = CpuCapabilities::detect();
                match options.execution_mode {
                    ReferenceExecutionMode::CpuParallelAvxVnni
                        if !capabilities.avx_vnni_dot_kernel_available() =>
                    {
                        return Err(Hy3ScalarError::BackendUnavailable {
                            backend: "cpu_parallel_avx_vnni_q8_k",
                            reason: "AVX2 plus AVX-VNNI are required",
                        });
                    }
                    ReferenceExecutionMode::CpuParallelAvx512Vnni
                        if !capabilities.avx512_dot_kernel_available() =>
                    {
                        return Err(Hy3ScalarError::BackendUnavailable {
                            backend: "cpu_parallel_avx512_vnni_q8_k",
                            reason: "AVX-512F/BW/VL plus AVX-512 VNNI are required",
                        });
                    }
                    _ => {}
                }
                Some(CpuBackend::new_with_cpu_set(
                    CpuBackendConfig {
                        threads: options.cpu_threads,
                    },
                    &options.cpu_set_ids,
                )?)
            }
            ReferenceExecutionMode::DequantF32 | ReferenceExecutionMode::LlamaQ8K => None,
        };
        let model_fingerprint = build_model_fingerprint(&set, &source_sha256)?;

        let loader = TensorLoader::open(&set, &validated)?;
        let config = validated.config().clone();
        let embeddings = loader.matrix(&validated, Hy3TensorRole::TokenEmbedding)?;
        let output_norm = loader.f32_vector(&validated, Hy3TensorRole::OutputNorm)?;
        let output = loader.matrix(&validated, Hy3TensorRole::Output)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.block_count as usize)
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "Hy3 layer table",
                requested: config.block_count as usize,
            })?;
        for layer in 0..config.block_count {
            let attention = OwnedAttention {
                input_norm: loader.f32_vector(&validated, Hy3TensorRole::AttentionNorm { layer })?,
                query: loader.matrix(&validated, Hy3TensorRole::AttentionQ { layer })?,
                query_norm: loader.f32_vector(&validated, Hy3TensorRole::AttentionQNorm { layer })?,
                key: loader.matrix(&validated, Hy3TensorRole::AttentionK { layer })?,
                key_norm: loader.f32_vector(&validated, Hy3TensorRole::AttentionKNorm { layer })?,
                value: loader.matrix(&validated, Hy3TensorRole::AttentionV { layer })?,
                output: loader.matrix(&validated, Hy3TensorRole::AttentionOutput { layer })?,
            };
            let ffn_norm = loader.f32_vector(&validated, Hy3TensorRole::FfnNorm { layer })?;
            let feed_forward = if layer == 0 {
                FeedForwardWeights::Dense(OwnedExpert {
                    gate: loader.matrix(&validated, Hy3TensorRole::DenseGate { layer })?,
                    up: loader.matrix(&validated, Hy3TensorRole::DenseUp { layer })?,
                    down: loader.matrix(&validated, Hy3TensorRole::DenseDown { layer })?,
                })
            } else {
                FeedForwardWeights::Moe(OwnedMoe {
                    router: loader.matrix(&validated, Hy3TensorRole::RouterInput { layer })?,
                    selection_bias: loader
                        .f32_vector(&validated, Hy3TensorRole::RouterSelectionBias { layer })?,
                    shared: OwnedExpert {
                        gate: loader.matrix(&validated, Hy3TensorRole::SharedGate { layer })?,
                        up: loader.matrix(&validated, Hy3TensorRole::SharedUp { layer })?,
                        down: loader.matrix(&validated, Hy3TensorRole::SharedDown { layer })?,
                    },
                    expert_layout: ExpertPayloadLayout::from_model(&validated, layer)?,
                })
            };
            layers.push(LayerWeights {
                attention,
                ffn_norm,
                feed_forward,
            });
        }

        let resident_weight_bytes = embeddings
            .bytes_len()
            .checked_add(output.bytes_len())
            .and_then(|total| total.checked_add(output_norm.len().checked_mul(4)?))
            .and_then(|total| {
                layers
                    .iter()
                    .try_fold(total, |sum, layer| sum.checked_add(layer.resident_bytes()?))
            })
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        let expert_source = if config.block_count > 1 {
            Some(ExpertSource::open(&set, &validated, &options.expert_source)?)
        } else {
            None
        };
        let source_paths = set.files().iter().map(|shard| shard.path().to_owned()).collect();
        let rope = Hy3RopeParams::from_config(&config)?;
        let expert_read_slots = build_expert_read_slots(&layers, &config, options.expert_cache_bytes)?;
        let expert_cache = CompressedCache::new(CacheConfig {
            capacity_bytes: options.expert_cache_bytes,
            admit_after_requests: options.cache_admit_after_requests,
        })?;

        Ok(Self {
            config,
            context_capacity: options.context_capacity,
            execution_mode: options.execution_mode,
            cuda_disabled: AtomicBool::new(false),
            cpu_backend,
            prefill_chunk: options.prefill_chunk,
            speculative_ngram_t: options.speculative_ngram_t,
            kv_page_tokens: options.kv_page_tokens,
            source_paths,
            source_sha256,
            model_fingerprint,
            resident_weight_bytes,
            embeddings,
            layers,
            output_norm,
            output,
            rope,
            expert_source,
            expert_read_slots,
            expert_cache,
        })
    }

    /// Evaluates one token and optionally projects the resulting hidden state into logits.
    ///
    /// On evaluation failure, restores the session to its position before the call. CUDA
    /// kernel failures are retried using the CPU backend; if that retry fails, both errors
    /// are reported.
    ///
    /// # Parameters
    ///
    /// * `token_id` — Token identifier to evaluate.
    /// * `logits` — Buffer receiving vocabulary logits when projection is enabled.
    /// * `project_logits` — Whether to compute logits for the evaluated token.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate(&mut session, token_id, &mut logits, true)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails or the session cannot be restored after a
    /// failed evaluation.
    fn evaluate(
        &self,
        session: &mut Hy3ScalarSession,
        token_id: u32,
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        let committed_position = session.position;
        let error = match self.evaluate_once(session, token_id, logits, project_logits) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        rollback_after_error(session, committed_position, &error)?;
        if self.activate_cuda_fallback(&error) {
            let cuda = error.to_string();
            return match self.evaluate_once(session, token_id, logits, project_logits) {
                Ok(()) => Ok(()),
                Err(cpu) => {
                    rollback_after_error(session, committed_position, &cpu)?;
                    Err(Hy3ScalarError::CudaFallbackFailed {
                        cuda,
                        cpu: cpu.to_string(),
                    })
                }
            };
        }
        Err(error)
    }

    /// Evaluates one token and optionally computes its logits.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_once(&mut session, token_id, &mut logits, true)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if token evaluation or optional logit projection fails.
    fn evaluate_once(
        &self,
        session: &mut Hy3ScalarSession,
        token_id: u32,
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        match &self.cpu_backend {
            Some(backend) => {
                backend.install(|| self.evaluate_inner(session, token_id, logits, project_logits))
            }
            None => self.evaluate_inner(session, token_id, logits, project_logits),
        }
    }

    /// Loads the specified expert payload into the cache and returns a lease for it.
    ///
    /// # Parameters
    ///
    /// * `key` identifies the expert payload to load.
    /// * `layout` describes the payload's tensor segments and expected size.
    ///
    /// # Errors
    ///
    /// Returns an error if the expert source or read-slot pool is unavailable, the
    /// payload cannot be read, or cache insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let lease = model.load_expert_lease(key, &layout)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    fn load_expert_lease(
        &self,
        key: ExpertKey,
        layout: &ExpertPayloadLayout,
    ) -> Result<CacheLease<ExpertKey>, Hy3ScalarError> {
        let expected = layout.total_bytes()?;
        let source = self
            .expert_source
            .as_ref()
            .ok_or(Hy3ScalarError::MissingExpertSource)?;
        let slots = self
            .expert_read_slots
            .as_ref()
            .ok_or(Hy3ScalarError::MissingExpertReadSlots)?;
        self.expert_cache
            .get_or_try_insert_read_slot_charged(key, expected, slots.slot_bytes(), || {
                let cancellation = ReadCancellation::new();
                let mut slot = slots.acquire(&cancellation)?;
                let actual = slot.as_slice().len();
                let output = slot
                    .as_mut_slice()
                    .get_mut(..expected)
                    .ok_or(ExpertReadError::PayloadLength { expected, actual })?;
                source.read_tight_into(key, layout, output, &cancellation)?;
                Ok(slot)
            })
            .map_err(Hy3ScalarError::ExpertCache)
    }

    /// Evaluates a group of tokens and optionally projects their hidden states into logits.
    ///
    /// A single-token input uses the single-token evaluation path. On failure, the session
    /// is restored to its position before evaluation; CUDA kernel failures may be retried
    /// using the CPU backend.
    ///
    /// # Arguments
    ///
    /// * `session` - Session whose KV cache and execution state are updated.
    /// * `token_ids` - Tokens to evaluate as one grouped sequence.
    /// * `logits` - Output buffer for projected logits when `project_logits` is `true`.
    /// * `project_logits` - Whether to compute output logits for the evaluated tokens.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_grouped(&mut session, &[token_a, token_b], &mut logits, true)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation, validation, session rollback, or CUDA fallback fails.
    fn evaluate_grouped(
        &self,
        session: &mut Hy3ScalarSession,
        token_ids: &[u32],
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        if token_ids.len() == 1 {
            return self.evaluate(session, token_ids[0], logits, project_logits);
        }
        let committed_position = session.position;
        let error = match self.evaluate_grouped_once(session, token_ids, logits, project_logits) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        rollback_after_error(session, committed_position, &error)?;
        if self.activate_cuda_fallback(&error) {
            let cuda = error.to_string();
            return match self.evaluate_grouped_once(session, token_ids, logits, project_logits) {
                Ok(()) => Ok(()),
                Err(cpu) => {
                    rollback_after_error(session, committed_position, &cpu)?;
                    Err(Hy3ScalarError::CudaFallbackFailed {
                        cuda,
                        cpu: cpu.to_string(),
                    })
                }
            };
        }
        Err(error)
    }

    /// Evaluates a group of tokens and optionally computes logits.
    ///
    /// # Parameters
    ///
    /// * `session` - Session state to update.
    /// * `token_ids` - Tokens to evaluate in sequence.
    /// * `logits` - Buffer receiving projected logits when requested.
    /// * `project_logits` - Whether to compute logits for the evaluated group.
    ///
    /// # Returns
    ///
    /// `Ok(())` after successful evaluation, or a [`Hy3ScalarError`] describing the failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_grouped_once(&mut session, &token_ids, &mut logits, true)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    fn evaluate_grouped_once(
        &self,
        session: &mut Hy3ScalarSession,
        token_ids: &[u32],
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        match &self.cpu_backend {
            Some(backend) => {
                backend.install(|| self.evaluate_grouped_inner(session, token_ids, logits, project_logits))
            }
            None => self.evaluate_grouped_inner(session, token_ids, logits, project_logits),
        }
    }

    /// Evaluates exactly two tokens together for speculative decoding and writes logits for both tokens.
    ///
    /// The session is restored to its committed position if evaluation fails. CUDA kernel failures
    /// may trigger a CPU retry.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_ids` does not contain exactly two tokens, if `logits` does not
    /// have `vocabulary_size * 2` elements, or if evaluation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_speculative_grouped(&mut session, &[first_token, second_token], &mut logits)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// `token_ids` contains the two tokens to evaluate, and `logits` receives one vocabulary-sized
    /// row for each token.
    fn evaluate_speculative_grouped(
        &self,
        session: &mut Hy3ScalarSession,
        token_ids: &[u32],
        logits: &mut [f32],
    ) -> Result<(), Hy3ScalarError> {
        let expected = self
            .config
            .vocabulary_size
            .checked_mul(u32::try_from(token_ids.len()).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)? as usize;
        if token_ids.len() != 2 || logits.len() != expected {
            return Err(Hy3ScalarError::SpeculativeLogitShape {
                tokens: token_ids.len(),
                vocabulary_size: self.config.vocabulary_size as usize,
                actual: logits.len(),
            });
        }
        let committed_position = session.position;
        let error = match self.evaluate_speculative_grouped_once(session, token_ids, logits) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        rollback_after_error(session, committed_position, &error)?;
        if self.activate_cuda_fallback(&error) {
            let cuda = error.to_string();
            return match self.evaluate_speculative_grouped_once(session, token_ids, logits) {
                Ok(()) => Ok(()),
                Err(cpu) => {
                    rollback_after_error(session, committed_position, &cpu)?;
                    Err(Hy3ScalarError::CudaFallbackFailed {
                        cuda,
                        cpu: cpu.to_string(),
                    })
                }
            };
        }
        Err(error)
    }

    /// Evaluates a speculative token group and projects logits for each token.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_speculative_grouped_once(&mut session, &token_ids, &mut logits)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Arguments
    ///
    /// * `session` - Session state used for evaluation.
    /// * `token_ids` - Tokens to evaluate as a group.
    /// * `logits` - Output buffer receiving the projected logits for each token.
    ///
    /// # Returns
    ///
    /// `Ok(())` after evaluation and projection succeed; otherwise, the execution error.
    fn evaluate_speculative_grouped_once(
        &self,
        session: &mut Hy3ScalarSession,
        token_ids: &[u32],
        logits: &mut [f32],
    ) -> Result<(), Hy3ScalarError> {
        let mut execute = || -> Result<(), Hy3ScalarError> {
            self.evaluate_grouped_inner(session, token_ids, &mut [], false)?;
            self.project_grouped_logits_inner(session, token_ids.len(), logits)
        };
        match &self.cpu_backend {
            Some(backend) => backend.install(execute),
            None => execute(),
        }
    }

    /// Activates CPU fallback when a CUDA kernel error occurs in CUDA execution mode.
    ///
    /// Returns `true` when fallback is activated, or `false` when the error does not
    /// qualify for CUDA fallback.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let model = /* a Hy3ScalarModel configured for CUDA execution */ todo!();
    /// # let error = /* a CUDA kernel error */ todo!();
    /// assert!(model.activate_cuda_fallback(&error));
    /// ```
    fn activate_cuda_fallback(&self, error: &Hy3ScalarError) -> bool {
        if self.execution_mode == ReferenceExecutionMode::CudaQ8K
            && matches!(
                error,
                Hy3ScalarError::Kernel(bridge_kernels_reference::KernelError::Cuda { .. })
            )
        {
            self.fall_back_to_cpu();
            true
        } else {
            false
        }
    }

    /// Projects each grouped hidden state into a row of vocabulary logits.
    ///
    /// # Errors
    ///
    /// Returns an error if the logits buffer has an invalid shape, grouped hidden
    /// state is unavailable, or normalization or output projection fails.
    ///
    /// # Examples
    ///
    /// ```
    /// // Provide one vocabulary-sized logits row for each grouped token.
    /// let logits = vec![0.0_f32; vocabulary_size * token_count];
    /// assert_eq!(logits.len(), vocabulary_size * token_count);
    /// ```
    fn project_grouped_logits_inner(
        &self,
        session: &mut Hy3ScalarSession,
        token_count: usize,
        logits: &mut [f32],
    ) -> Result<(), Hy3ScalarError> {
        let execution_mode = self.active_execution_mode();
        let vocabulary_size = self.config.vocabulary_size as usize;
        let expected = vocabulary_size
            .checked_mul(token_count)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        if logits.len() != expected {
            return Err(Hy3ScalarError::SpeculativeLogitShape {
                tokens: token_count,
                vocabulary_size,
                actual: logits.len(),
            });
        }
        let hidden_width = self.config.embedding_length as usize;
        for (position, row) in logits.chunks_exact_mut(vocabulary_size).enumerate() {
            let hidden = grouped_position(&session.batch_hidden, position, hidden_width)?;
            weighted_rms_norm_into(
                hidden,
                &self.output_norm,
                self.config.rms_epsilon,
                &mut session.final_normalized,
            )?;
            gemv_into(
                execution_mode,
                self.output.view()?,
                &session.final_normalized,
                row,
                &mut session.decoded_block,
                &mut session.q8,
            )?;
        }
        Ok(())
    }

    /// Evaluates a group of tokens and advances the session by the group length.
    ///
    /// The final token's hidden state is retained in the session. When `project_logits`
    /// is enabled, the final hidden state is projected into `logits`.
    ///
    /// # Parameters
    ///
    /// * `token_ids` — Tokens to evaluate in sequence.
    /// * `logits` — Output buffer for the final token's vocabulary logits when projection
    ///   is enabled.
    /// * `project_logits` — Whether to compute the final token's logits.
    ///
    /// # Returns
    ///
    /// `Ok(())` after successful evaluation, or a `Hy3ScalarError` describing validation,
    /// execution, or resource-loading failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_grouped_inner(&mut session, &[first_token, second_token], &mut logits, true)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    fn evaluate_grouped_inner(
        &self,
        session: &mut Hy3ScalarSession,
        token_ids: &[u32],
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        let execution_mode = self.active_execution_mode();
        if token_ids.is_empty() || token_ids.len() > self.prefill_chunk || token_ids.len() > MAX_PREFILL_CHUNK
        {
            return Err(Hy3ScalarError::GroupedTokenCount {
                actual: token_ids.len(),
                maximum: self.prefill_chunk.min(MAX_PREFILL_CHUNK),
            });
        }
        if project_logits && logits.len() != self.config.vocabulary_size as usize {
            return Err(Hy3ScalarError::LogitLength {
                expected: self.config.vocabulary_size as usize,
                actual: logits.len(),
            });
        }
        for &token_id in token_ids {
            if token_id as usize >= self.config.vocabulary_size as usize {
                return Err(Hy3ScalarError::TokenOutOfRange {
                    token_id,
                    vocabulary_size: self.config.vocabulary_size as usize,
                });
            }
        }
        let end_position = session
            .position
            .checked_add(token_ids.len())
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        if end_position > self.context_capacity {
            return Err(Hy3ScalarError::ContextExhausted {
                position: session.position,
                capacity: self.context_capacity,
            });
        }

        let hidden_width = self.config.embedding_length as usize;
        let embeddings = self.embeddings.view()?;
        for (position, &token_id) in token_ids.iter().enumerate() {
            bridge_quant_layout::decode_row_into(
                self.embeddings.ty,
                embeddings.row(token_id as usize),
                self.embeddings.input_width,
                grouped_position_mut(&mut session.batch_hidden, position, hidden_width)?,
            )?;
        }

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_number = u32::try_from(layer_index).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
            match &layer.feed_forward {
                FeedForwardWeights::Dense(expert) => {
                    let weights = Hy3BlockWeights {
                        attention: layer.attention.view(&self.config)?,
                        ffn_norm: &layer.ffn_norm,
                        feed_forward: Hy3FeedForwardWeights::Dense(expert.view()?),
                    };
                    for position in 0..token_ids.len() {
                        let execution = Hy3BlockExecution {
                            mode: execution_mode,
                            layer: layer_index,
                            position: (session.position + position) as u64,
                            rope: self.rope,
                            rms_epsilon: self.config.rms_epsilon,
                        };
                        let hidden = grouped_position_mut(&mut session.batch_hidden, position, hidden_width)?;
                        let scratch = grouped_dense_scratch_mut(
                            &mut session.dense_scratch,
                            &mut session.batch_dense_scratch,
                            position,
                        )?;
                        hy3_block_forward_token(execution, weights, &mut session.cache, hidden, scratch)?;
                    }
                }
                FeedForwardWeights::Moe(moe) => {
                    let streaming = moe.streaming_weights(&layer.attention, &layer.ffn_norm, &self.config)?;
                    for position in 0..token_ids.len() {
                        let execution = Hy3BlockExecution {
                            mode: execution_mode,
                            layer: layer_index,
                            position: (session.position + position) as u64,
                            rope: self.rope,
                            rms_epsilon: self.config.rms_epsilon,
                        };
                        let hidden = grouped_position_mut(&mut session.batch_hidden, position, hidden_width)?;
                        let scratch = grouped_moe_scratch_mut(
                            &mut session.moe_scratch,
                            &mut session.batch_moe_scratch,
                            position,
                        )?;
                        hy3_moe_route_token(execution, streaming, &mut session.cache, hidden, scratch)?;
                        if scratch.routed().len() > MAX_STREAMING_EXPERTS {
                            return Err(Hy3ScalarError::SelectedExpertCount {
                                actual: scratch.routed().len(),
                                maximum: MAX_STREAMING_EXPERTS,
                            });
                        }
                        let routes = session.batch_routes.get_mut(position).ok_or(
                            Hy3ScalarError::GroupedScratchMissing {
                                kind: "route record",
                                position,
                            },
                        )?;
                        routes.clear();
                        routes.extend_from_slice(scratch.routed());
                    }

                    session.expert_needed.fill(false);
                    let route_expert_count = session.expert_needed.len();
                    for routes in session.batch_routes.iter().take(token_ids.len()) {
                        for route in routes {
                            let needed = session.expert_needed.get_mut(route.expert_id as usize).ok_or(
                                Hy3ScalarError::SelectedExpertId {
                                    expert_id: route.expert_id,
                                    expert_count: route_expert_count,
                                },
                            )?;
                            *needed = true;
                        }
                    }
                    let expert_count = self.config.expert_count as usize;
                    if session.expert_leases.len() != expert_count {
                        session.expert_leases.clear();
                        session.expert_leases.resize_with(expert_count, || None);
                    } else {
                        for lease in &mut session.expert_leases {
                            *lease = None;
                        }
                    }
                    let load_expert = |expert_id: usize| {
                        self.load_expert_lease(
                            ExpertKey {
                                layer: layer_number,
                                expert: expert_id as u32,
                            },
                            &moe.expert_layout,
                        )
                    };
                    if self.cpu_backend.is_some() {
                        session
                            .expert_leases
                            .par_iter_mut()
                            .zip(session.expert_needed.par_iter())
                            .enumerate()
                            .try_for_each(|(expert_id, (slot, needed))| {
                                if *needed {
                                    *slot = Some(load_expert(expert_id)?);
                                }
                                Ok::<(), Hy3ScalarError>(())
                            })?;
                    } else {
                        for (expert_id, (slot, needed)) in session
                            .expert_leases
                            .iter_mut()
                            .zip(&session.expert_needed)
                            .enumerate()
                        {
                            if *needed {
                                *slot = Some(load_expert(expert_id)?);
                            }
                        }
                    }

                    let shared = moe.shared.view()?;
                    for position in 0..token_ids.len() {
                        let routes = session.batch_routes.get(position).ok_or(
                            Hy3ScalarError::GroupedScratchMissing {
                                kind: "route record",
                                position,
                            },
                        )?;
                        let mut selected_storage =
                            [MaybeUninit::<SelectedExpert<'_>>::uninit(); MAX_STREAMING_EXPERTS];
                        for (index, route) in routes.iter().enumerate() {
                            let lease = session
                                .expert_leases
                                .get(route.expert_id as usize)
                                .and_then(Option::as_ref)
                                .ok_or(Hy3ScalarError::MissingExpertLease)?;
                            selected_storage[index].write(SelectedExpert {
                                expert_id: route.expert_id,
                                coefficient: route.coefficient,
                                expert: moe.expert_layout.view(lease.bytes())?,
                            });
                        }
                        // SAFETY: `routes.len()` contiguous entries were
                        // initialized and SelectedExpert has no drop glue.
                        let selected = unsafe {
                            std::slice::from_raw_parts(
                                selected_storage.as_ptr().cast::<SelectedExpert<'_>>(),
                                routes.len(),
                            )
                        };
                        let hidden = grouped_position_mut(&mut session.batch_hidden, position, hidden_width)?;
                        let scratch = grouped_moe_scratch_mut(
                            &mut session.moe_scratch,
                            &mut session.batch_moe_scratch,
                            position,
                        )?;
                        hy3_moe_finish_token(execution_mode, selected, shared, hidden, scratch)?;
                    }
                    for lease in &mut session.expert_leases {
                        *lease = None;
                    }
                }
            }
        }

        let final_hidden = grouped_position(&session.batch_hidden, token_ids.len() - 1, hidden_width)?;
        session.hidden.copy_from_slice(final_hidden);
        if project_logits {
            weighted_rms_norm_into(
                final_hidden,
                &self.output_norm,
                self.config.rms_epsilon,
                &mut session.final_normalized,
            )?;
            gemv_into(
                execution_mode,
                self.output.view()?,
                &session.final_normalized,
                logits,
                &mut session.decoded_block,
                &mut session.q8,
            )?;
        }
        session.position = end_position;
        Ok(())
    }

    /// Evaluates one token and advances the session, optionally projecting the final hidden state into vocabulary logits.
    ///
    /// Validates the token and context position, executes all model layers, and loads routed Mixture-of-Experts
    /// payloads when required. The `logits` buffer is used only when `project_logits` is `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if the token, context position, or projected-logit buffer length is invalid, or if
    /// model execution fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.evaluate_inner(&mut session, token_id, &mut logits, true)?;
    /// ```
    ///
    /// # Parameters
    ///
    /// * `token_id` — Token to evaluate.
    /// * `logits` — Output buffer for projected vocabulary logits.
    /// * `project_logits` — Whether to compute vocabulary logits.
    ///
    /// # Returns
    ///
    /// `Ok(())` after successful evaluation.
    fn evaluate_inner(
        &self,
        session: &mut Hy3ScalarSession,
        token_id: u32,
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Hy3ScalarError> {
        let execution_mode = self.active_execution_mode();
        if project_logits && logits.len() != self.config.vocabulary_size as usize {
            return Err(Hy3ScalarError::LogitLength {
                expected: self.config.vocabulary_size as usize,
                actual: logits.len(),
            });
        }
        let token = token_id as usize;
        if token >= self.config.vocabulary_size as usize {
            return Err(Hy3ScalarError::TokenOutOfRange {
                token_id,
                vocabulary_size: self.config.vocabulary_size as usize,
            });
        }
        if session.position >= self.context_capacity {
            return Err(Hy3ScalarError::ContextExhausted {
                position: session.position,
                capacity: self.context_capacity,
            });
        }

        bridge_quant_layout::decode_row_into(
            self.embeddings.ty,
            self.embeddings.view()?.row(token),
            self.embeddings.input_width,
            &mut session.hidden,
        )?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_number = u32::try_from(layer_index).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
            let execution = Hy3BlockExecution {
                mode: execution_mode,
                layer: layer_index,
                position: session.position as u64,
                rope: self.rope,
                rms_epsilon: self.config.rms_epsilon,
            };
            match &layer.feed_forward {
                FeedForwardWeights::Dense(expert) => {
                    let weights = Hy3BlockWeights {
                        attention: layer.attention.view(&self.config)?,
                        ffn_norm: &layer.ffn_norm,
                        feed_forward: Hy3FeedForwardWeights::Dense(expert.view()?),
                    };
                    hy3_block_forward_token(
                        execution,
                        weights,
                        &mut session.cache,
                        &mut session.hidden,
                        &mut session.dense_scratch,
                    )?;
                }
                FeedForwardWeights::Moe(moe) => {
                    let streaming = moe.streaming_weights(&layer.attention, &layer.ffn_norm, &self.config)?;
                    let scratch = session
                        .moe_scratch
                        .as_mut()
                        .ok_or(Hy3ScalarError::MissingMoeScratch)?;
                    hy3_moe_route_token(
                        execution,
                        streaming,
                        &mut session.cache,
                        &mut session.hidden,
                        scratch,
                    )?;
                    session.routes.clear();
                    session.routes.extend_from_slice(scratch.routed());
                    if session.routes.len() > MAX_STREAMING_EXPERTS {
                        return Err(Hy3ScalarError::SelectedExpertCount {
                            actual: session.routes.len(),
                            maximum: MAX_STREAMING_EXPERTS,
                        });
                    }
                    session.expert_leases.clear();
                    let load_route = |route: &bridge_model_hy3::RoutedExpert| {
                        let key = ExpertKey {
                            layer: layer_number,
                            expert: route.expert_id,
                        };
                        self.load_expert_lease(key, &moe.expert_layout)
                    };
                    session.expert_leases.resize_with(session.routes.len(), || None);
                    if self.cpu_backend.is_some() {
                        session
                            .expert_leases
                            .par_iter_mut()
                            .zip(session.routes.par_iter())
                            .try_for_each(|(slot, route)| {
                                *slot = Some(load_route(route)?);
                                Ok::<(), Hy3ScalarError>(())
                            })?;
                    } else {
                        for (slot, route) in session.expert_leases.iter_mut().zip(&session.routes) {
                            *slot = Some(load_route(route)?);
                        }
                    }

                    let mut selected_storage =
                        [MaybeUninit::<SelectedExpert<'_>>::uninit(); MAX_STREAMING_EXPERTS];
                    for (index, (route, lease)) in
                        session.routes.iter().zip(&session.expert_leases).enumerate()
                    {
                        let lease = lease.as_ref().ok_or(Hy3ScalarError::MissingExpertLease)?;
                        selected_storage[index].write(SelectedExpert {
                            expert_id: route.expert_id,
                            coefficient: route.coefficient,
                            expert: moe.expert_layout.view(lease.bytes())?,
                        });
                    }
                    // SAFETY: exactly `routes.len()` contiguous entries were
                    // initialized above and `SelectedExpert` has no drop glue.
                    let selected = unsafe {
                        std::slice::from_raw_parts(
                            selected_storage.as_ptr().cast::<SelectedExpert<'_>>(),
                            session.routes.len(),
                        )
                    };
                    hy3_moe_finish_token(
                        execution_mode,
                        selected,
                        moe.shared.view()?,
                        &mut session.hidden,
                        scratch,
                    )?;
                    session.expert_leases.clear();
                }
            }
        }

        if project_logits {
            weighted_rms_norm_into(
                &session.hidden,
                &self.output_norm,
                self.config.rms_epsilon,
                &mut session.final_normalized,
            )?;
            gemv_into(
                execution_mode,
                self.output.view()?,
                &session.final_normalized,
                logits,
                &mut session.decoded_block,
                &mut session.q8,
            )?;
        }
        session.position += 1;
        Ok(())
    }
}

impl CausalModel for Hy3ScalarModel {
    type Session = Hy3ScalarSession;
    type Error = Hy3ScalarError;

    fn vocabulary_size(&self) -> usize {
        self.config.vocabulary_size as usize
    }

    fn context_length(&self) -> usize {
        self.context_capacity
    }

    /// Reports the configured token count for grouped prefill execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # let model = /* a configured Hy3ScalarModel */;
    
    /// let chunk = model.preferred_prefill_chunk();
    /// assert!(chunk > 0);
    /// ```
    ///
    /// Returns the configured grouped prefill chunk size.
    fn preferred_prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    /// Reports the configured speculative n-gram width.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(model.speculative_ngram_t(), Some(2));
    /// ```
    ///
    /// # Returns
    ///
    /// The configured speculative n-gram width, or `None` when speculation is disabled.
    fn speculative_ngram_t(&self) -> Option<usize> {
        self.speculative_ngram_t
    }

    /// Creates a new session with an empty KV cache and execution workspaces.
    ///
    /// The session is initialized for token evaluation, grouped prefill, and any
    /// Mixture-of-Experts layers present in the model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let session = model.new_session()?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the model lacks required layers or if session resources
    /// cannot be initialized.
    fn new_session(&self) -> Result<Self::Session, Self::Error>
    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        let dense = self.layers.first().ok_or(Hy3ScalarError::MissingDenseLayer)?;
        let FeedForwardWeights::Dense(dense_expert) = &dense.feed_forward else {
            return Err(Hy3ScalarError::MissingDenseLayer);
        };
        let dense_weights = Hy3BlockWeights {
            attention: dense.attention.view(&self.config)?,
            ffn_norm: &dense.ffn_norm,
            feed_forward: Hy3FeedForwardWeights::Dense(dense_expert.view()?),
        };
        let streaming_moe = self
            .layers
            .iter()
            .find_map(|layer| match &layer.feed_forward {
                FeedForwardWeights::Moe(moe) => {
                    Some(moe.streaming_weights(&layer.attention, &layer.ffn_norm, &self.config))
                }
                FeedForwardWeights::Dense(_) => None,
            })
            .transpose()?;
        let moe_scratch = streaming_moe
            .map(|weights| Hy3BlockScratch::new_streaming_moe(weights, self.context_capacity))
            .transpose()?;
        let extra_positions = self.prefill_chunk.saturating_sub(1);
        let mut batch_dense_scratch = Vec::new();
        batch_dense_scratch
            .try_reserve_exact(extra_positions)
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "grouped dense scratch table",
                requested: extra_positions,
            })?;
        for _ in 0..extra_positions {
            batch_dense_scratch.push(Hy3BlockScratch::new(dense_weights, self.context_capacity)?);
        }
        let mut batch_moe_scratch = Vec::new();
        if let Some(weights) = streaming_moe {
            batch_moe_scratch
                .try_reserve_exact(extra_positions)
                .map_err(|_| Hy3ScalarError::AllocationFailed {
                    context: "grouped MoE scratch table",
                    requested: extra_positions,
                })?;
            for _ in 0..extra_positions {
                batch_moe_scratch.push(Hy3BlockScratch::new_streaming_moe(
                    weights,
                    self.context_capacity,
                )?);
            }
        }
        let hidden_width = self.config.embedding_length as usize;
        let batch_hidden_len = hidden_width
            .checked_mul(self.prefill_chunk)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        let mut batch_routes = Vec::new();
        batch_routes
            .try_reserve_exact(self.prefill_chunk)
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "grouped route table",
                requested: self.prefill_chunk,
            })?;
        for _ in 0..self.prefill_chunk {
            let mut routes = Vec::new();
            routes
                .try_reserve_exact(self.config.expert_used_count as usize)
                .map_err(|_| Hy3ScalarError::AllocationFailed {
                    context: "grouped position routes",
                    requested: self.config.expert_used_count as usize,
                })?;
            batch_routes.push(routes);
        }
        let expert_count = self.config.expert_count as usize;
        let mut expert_leases = Vec::new();
        expert_leases
            .try_reserve_exact(expert_count)
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "expert lease table",
                requested: expert_count,
            })?;
        expert_leases.resize_with(expert_count, || None);
        Ok(Hy3ScalarSession {
            cache: PagedKvCache::new_lazy(
                self.config.block_count as usize,
                self.config.attention_kv_head_count as usize,
                self.config.key_length as usize,
                self.config.value_length as usize,
                self.kv_page_tokens,
                self.context_capacity,
            )?,
            dense_scratch: Hy3BlockScratch::new(dense_weights, self.context_capacity)?,
            moe_scratch,
            batch_dense_scratch,
            batch_moe_scratch,
            hidden: fallible_zeroed(hidden_width, "hidden state")?,
            batch_hidden: fallible_zeroed(batch_hidden_len, "grouped hidden states")?,
            final_normalized: fallible_zeroed(hidden_width, "final normalized state")?,
            decoded_block: fallible_zeroed(256, "output decoded block")?,
            q8: fallible_zeroed(
                bridge_kernels_reference::required_q8_k_bytes(hidden_width)?,
                "output Q8_K row",
            )?,
            expert_leases,
            expert_needed: fallible_zeroed(expert_count, "grouped expert union")?,
            routes: Vec::with_capacity(self.config.expert_used_count as usize),
            batch_routes,
            position: 0,
        })
    }

    /// Resets the session to its initial state, clearing cached data, scratch buffers, expert selections, activations, and position.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// model.reset_session(&mut session);
    /// assert_eq!(session.position, 0);
    /// ```
    fn reset_session(&self, session: &mut Self::Session) {
        session.cache.reset();
        session.dense_scratch.reset();
        if let Some(scratch) = &mut session.moe_scratch {
            scratch.reset();
        }
        for scratch in &mut session.batch_dense_scratch {
            scratch.reset();
        }
        for scratch in &mut session.batch_moe_scratch {
            scratch.reset();
        }
        for lease in &mut session.expert_leases {
            *lease = None;
        }
        session.expert_needed.fill(false);
        session.routes.clear();
        for routes in &mut session.batch_routes {
            routes.clear();
        }
        session.hidden.fill(0.0);
        session.batch_hidden.fill(0.0);
        session.final_normalized.fill(0.0);
        session.position = 0;
    }

    fn position(&self, session: &Self::Session) -> usize {
        session.position
    }

    /// Evaluates one token and writes its vocabulary logits.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// model.evaluate_token(&mut session, token_id, &mut logits)?;
    /// # Ok::<(), _>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if token evaluation fails or `logits` has an invalid length.
    fn evaluate_token
    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error> {
        self.evaluate(session, token_id, logits, true)
    }

    /// Evaluates a token and optionally computes its vocabulary logits.
    ///
    /// # Parameters
    ///
    /// * `project_logits` — Whether to write the projected logits into `logits`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut logits = vec![0.0; model.vocabulary_size()];
    /// model.evaluate_token_with_projection(&mut session, token_id, &mut logits, true)?;
    /// # Ok::<(), _>(())
    /// ```
    ///
    /// # Returns
    ///
    /// `Ok(())` after the token is evaluated, or an error if evaluation fails.
    fn evaluate_token_with_projection(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Self::Error> {
        self.evaluate(session, token_id, logits, project_logits)
    }

    /// Evaluates multiple tokens as a grouped prefill and optionally computes their logits.
    ///
    /// When `project_logits` is `true`, `logits` must have room for one vocabulary-sized
    /// row per token.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let token_ids = [1, 2];
    /// let mut logits = vec![0.0; model.vocabulary_size() * token_ids.len()];
    /// model.evaluate_tokens_with_projection(
    ///     &mut session,
    ///     &token_ids,
    ///     &mut logits,
    ///     true,
    /// )?;
    /// # Ok::<(), ModelError>(())
    /// ```
    ///
    /// # Parameters
    ///
    /// * `project_logits` — Whether to compute output logits for the evaluated tokens.
    ///
    /// # Returns
    ///
    /// `Ok(())` after successful evaluation, or the runtime error that prevented it.
    fn evaluate_tokens_with_projection(
        &self,
        session: &mut Self::Session,
        token_ids: &[u32],
        logits: &mut [f32],
        project_logits: bool,
    ) -> Result<(), Self::Error> {
        self.evaluate_grouped(session, token_ids, logits, project_logits)
    }

    /// Evaluates a two-token speculative sequence and computes logits for both tokens.
    ///
    /// # Returns
    ///
    /// `Some(Ok(()))` when evaluation succeeds, or `Some(Err(error))` when it fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let result = model.evaluate_speculative_tokens(&mut session, &[first, second], &mut logits);
    /// result.unwrap()?;
    /// # Ok::<(), ModelError>(())
    /// ```
    fn evaluate_speculative_tokens(
        &self,
        session: &mut Self::Session,
        token_ids: &[u32],
        logits: &mut [f32],
    ) -> Option<Result<(), Self::Error>> {
        Some(self.evaluate_speculative_grouped(session, token_ids, logits))
    }

    /// Rewinds a speculative session to the specified position.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let result = model.rewind_speculative(&mut session, position);
    /// assert!(result.is_some());
    /// ```
    ///
    /// The result contains an error if the session cannot be rewound.
    fn rewind_speculative(
        &self,
        session: &mut Self::Session,
        position: usize,
    ) -> Option<Result<(), Self::Error>> {
        Some(session.rollback_to(position))
    }
}

#[derive(Debug)]
pub struct Hy3ScalarSession {
    cache: PagedKvCache,
    dense_scratch: Hy3BlockScratch,
    moe_scratch: Option<Hy3BlockScratch>,
    batch_dense_scratch: Vec<Hy3BlockScratch>,
    batch_moe_scratch: Vec<Hy3BlockScratch>,
    hidden: Vec<f32>,
    batch_hidden: Vec<f32>,
    final_normalized: Vec<f32>,
    decoded_block: Vec<f32>,
    q8: Vec<u8>,
    expert_leases: Vec<Option<CacheLease<ExpertKey>>>,
    expert_needed: Vec<bool>,
    routes: Vec<bridge_model_hy3::RoutedExpert>,
    batch_routes: Vec<Vec<bridge_model_hy3::RoutedExpert>>,
    position: usize,
}

impl Hy3ScalarSession {
    pub fn kv_stored_tokens(&self, layer: usize) -> Result<usize, Hy3ScalarError> {
        Ok(self.cache.stored_tokens(layer)?)
    }

    /// Restores the session to a previously committed position and clears its transient execution state.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// session.rollback_to(committed_position)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the KV cache cannot be rewound to `committed_position`.
    fn rollback_to(&mut self, committed_position: usize) -> Result<(), Hy3ScalarError> {
        self.cache.rewind_all(committed_position)?;
        self.dense_scratch.reset();
        if let Some(scratch) = &mut self.moe_scratch {
            scratch.reset();
        }
        for scratch in &mut self.batch_dense_scratch {
            scratch.reset();
        }
        for scratch in &mut self.batch_moe_scratch {
            scratch.reset();
        }
        for lease in &mut self.expert_leases {
            *lease = None;
        }
        self.expert_needed.fill(false);
        self.routes.clear();
        for routes in &mut self.batch_routes {
            routes.clear();
        }
        self.hidden.fill(0.0);
        self.batch_hidden.fill(0.0);
        self.final_normalized.fill(0.0);
        self.decoded_block.fill(0.0);
        self.q8.fill(0);
        self.position = committed_position;
        Ok(())
    }
}

/// Restores a session to its committed position after an evaluation error.
///
/// If restoring the session fails, returns a [`Hy3ScalarError::SessionRollback`]
/// containing both the original error and the rollback error.
///
/// # Examples
///
/// ```ignore
/// let result = rollback_after_error(&mut session, committed_position, &error);
/// assert!(result.is_ok());
/// ```
fn rollback_after_error(
    session: &mut Hy3ScalarSession,
    committed_position: usize,
    error: &Hy3ScalarError,
) -> Result<(), Hy3ScalarError> {
    session
        .rollback_to(committed_position)
        .map_err(|rollback| Hy3ScalarError::SessionRollback {
            original: error.to_string(),
            rollback: rollback.to_string(),
        })
}

/// Retrieves a hidden-state slice for a position in grouped storage.
///
/// # Examples
///
/// ```
/// let values = vec![1.0, 2.0, 3.0, 4.0];
/// assert_eq!(grouped_position(&values, 1, 2).unwrap(), &[3.0, 4.0]);
/// ```
///
/// # Errors
///
/// Returns an error if the requested range overflows or falls outside `values`.
fn grouped_position(values: &[f32], position: usize, width: usize) -> Result<&[f32], Hy3ScalarError> {
    let start = position
        .checked_mul(width)
        .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
    let end = start
        .checked_add(width)
        .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
    values
        .get(start..end)
        .ok_or(Hy3ScalarError::GroupedScratchMissing {
            kind: "hidden state",
            position,
        })
}

/// Gets the mutable hidden-state slice for a grouped position.
///
/// # Errors
///
/// Returns [`Hy3ScalarError::ArithmeticOverflow`] if the slice bounds overflow, or
/// [`Hy3ScalarError::GroupedScratchMissing`] if the requested position is unavailable.
///
/// # Examples
///
/// ```
/// let mut values = vec![0.0; 6];
/// let slice = grouped_position_mut(&mut values, 1, 3).unwrap();
/// slice[0] = 1.0;
/// assert_eq!(&values[3..6], &[1.0, 0.0, 0.0]);
/// ```
fn grouped_position_mut(
    values: &mut [f32],
    position: usize,
    width: usize,
) -> Result<&mut [f32], Hy3ScalarError> {
    let start = position
        .checked_mul(width)
        .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
    let end = start
        .checked_add(width)
        .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
    values
        .get_mut(start..end)
        .ok_or(Hy3ScalarError::GroupedScratchMissing {
            kind: "hidden state",
            position,
        })
}

/// Selects the dense scratch buffer for a grouped execution position.
///
/// Position zero uses the primary buffer; subsequent positions use the corresponding buffer in `extra`.
///
/// # Errors
///
/// Returns [`Hy3ScalarError::GroupedScratchMissing`] when `extra` has no buffer for the position.
///
/// # Examples
///
/// ```ignore
/// let scratch = grouped_dense_scratch_mut(&mut primary, &mut extra, 0)?;
/// assert!(std::ptr::eq(scratch, &primary));
/// # Ok::<(), Hy3ScalarError>(())
/// ```
fn grouped_dense_scratch_mut<'a>(
    primary: &'a mut Hy3BlockScratch,
    extra: &'a mut [Hy3BlockScratch],
    position: usize,
) -> Result<&'a mut Hy3BlockScratch, Hy3ScalarError> {
    if position == 0 {
        Ok(primary)
    } else {
        extra
            .get_mut(position - 1)
            .ok_or(Hy3ScalarError::GroupedScratchMissing {
                kind: "dense block",
                position,
            })
    }
}

/// Selects the MoE scratch buffer for a grouped position.
///
/// Position zero uses the primary scratch buffer; later positions use the corresponding buffer in `extra`.
///
/// # Errors
///
/// Returns [`Hy3ScalarError::MissingMoeScratch`] when position zero has no primary buffer, or
/// [`Hy3ScalarError::GroupedScratchMissing`] when the requested grouped buffer is unavailable.
///
/// # Examples
///
/// ```
/// let mut primary = None;
/// let mut extra = [];
///
/// assert!(grouped_moe_scratch_mut(&mut primary, &mut extra, 0).is_err());
/// ```
fn grouped_moe_scratch_mut<'a>(
    primary: &'a mut Option<Hy3BlockScratch>,
    extra: &'a mut [Hy3BlockScratch],
    position: usize,
) -> Result<&'a mut Hy3BlockScratch, Hy3ScalarError> {
    if position == 0 {
        primary.as_mut().ok_or(Hy3ScalarError::MissingMoeScratch)
    } else {
        extra
            .get_mut(position - 1)
            .ok_or(Hy3ScalarError::GroupedScratchMissing {
                kind: "MoE block",
                position,
            })
    }
}

#[derive(Debug)]
struct OwnedMatrix {
    ty: GgmlType,
    input_width: usize,
    output_width: usize,
    bytes: Vec<u8>,
}

impl OwnedMatrix {
    fn view(&self) -> Result<PackedMatrix<'_>, Hy3ScalarError> {
        Ok(PackedMatrix::from_parts(
            self.ty,
            PayloadEndian::Little,
            self.input_width,
            self.output_width,
            &self.bytes,
        )?)
    }

    fn bytes_len(&self) -> usize {
        self.bytes.len()
    }
}

#[derive(Debug)]
struct OwnedAttention {
    input_norm: Vec<f32>,
    query: OwnedMatrix,
    query_norm: Vec<f32>,
    key: OwnedMatrix,
    key_norm: Vec<f32>,
    value: OwnedMatrix,
    output: OwnedMatrix,
}

impl OwnedAttention {
    fn view(&self, config: &Hy3Config) -> Result<Hy3AttentionWeights<'_>, Hy3ScalarError> {
        Ok(Hy3AttentionWeights {
            input_norm: &self.input_norm,
            query: self.query.view()?,
            query_norm: &self.query_norm,
            key: self.key.view()?,
            key_norm: &self.key_norm,
            value: self.value.view()?,
            output: self.output.view()?,
            query_head_count: config.attention_head_count as usize,
            kv_head_count: config.attention_kv_head_count as usize,
            key_dimension: config.key_length as usize,
            value_dimension: config.value_length as usize,
        })
    }

    fn resident_bytes(&self) -> Option<usize> {
        self.input_norm
            .len()
            .checked_add(self.query_norm.len())?
            .checked_add(self.key_norm.len())?
            .checked_mul(4)?
            .checked_add(self.query.bytes_len())?
            .checked_add(self.key.bytes_len())?
            .checked_add(self.value.bytes_len())?
            .checked_add(self.output.bytes_len())
    }
}

#[derive(Debug)]
struct OwnedExpert {
    gate: OwnedMatrix,
    up: OwnedMatrix,
    down: OwnedMatrix,
}

impl OwnedExpert {
    fn view(&self) -> Result<SwiGluExpert<'_>, Hy3ScalarError> {
        Ok(SwiGluExpert::new(
            self.gate.view()?,
            self.up.view()?,
            self.down.view()?,
        )?)
    }

    fn resident_bytes(&self) -> Option<usize> {
        self.gate
            .bytes_len()
            .checked_add(self.up.bytes_len())?
            .checked_add(self.down.bytes_len())
    }
}

#[derive(Debug)]
struct OwnedMoe {
    router: OwnedMatrix,
    selection_bias: Vec<f32>,
    shared: OwnedExpert,
    expert_layout: ExpertPayloadLayout,
}

impl OwnedMoe {
    fn streaming_weights<'a>(
        &'a self,
        attention: &'a OwnedAttention,
        ffn_norm: &'a [f32],
        config: &Hy3Config,
    ) -> Result<Hy3StreamingMoeWeights<'a>, Hy3ScalarError> {
        Ok(Hy3StreamingMoeWeights {
            attention: attention.view(config)?,
            ffn_norm,
            router: self.router.view()?,
            selection_bias: &self.selection_bias,
            shared_expert: self.shared.view()?,
            expert_count: config.expert_count as usize,
            expert_used_count: config.expert_used_count as usize,
            weight_scale: config.expert_weights_scale,
        })
    }

    fn resident_bytes(&self) -> Option<usize> {
        self.router
            .bytes_len()
            .checked_add(self.selection_bias.len().checked_mul(4)?)?
            .checked_add(self.shared.resident_bytes()?)
    }
}

#[derive(Debug)]
enum FeedForwardWeights {
    Dense(OwnedExpert),
    Moe(OwnedMoe),
}

#[derive(Debug)]
struct LayerWeights {
    attention: OwnedAttention,
    ffn_norm: Vec<f32>,
    feed_forward: FeedForwardWeights,
}

impl LayerWeights {
    fn resident_bytes(&self) -> Option<usize> {
        let base = self
            .attention
            .resident_bytes()?
            .checked_add(self.ffn_norm.len().checked_mul(4)?)?;
        match &self.feed_forward {
            FeedForwardWeights::Dense(expert) => base.checked_add(expert.resident_bytes()?),
            FeedForwardWeights::Moe(moe) => base.checked_add(moe.resident_bytes()?),
        }
    }
}

#[derive(Debug, Clone)]
struct MatrixLayout {
    ty: GgmlType,
    input_width: usize,
    output_width: usize,
    bytes: usize,
}

#[derive(Debug, Clone)]
struct ExpertPayloadLayout {
    gate: MatrixLayout,
    up: MatrixLayout,
    down: MatrixLayout,
}

impl ExpertPayloadLayout {
    fn from_model(model: &ValidatedHy3Model, layer: u32) -> Result<Self, Hy3ScalarError> {
        let config = model.config();
        Ok(Self {
            gate: matrix_layout(
                required_tensor(model, Hy3TensorRole::RoutedGate { layer })?,
                config.expert_count,
            )?,
            up: matrix_layout(
                required_tensor(model, Hy3TensorRole::RoutedUp { layer })?,
                config.expert_count,
            )?,
            down: matrix_layout(
                required_tensor(model, Hy3TensorRole::RoutedDown { layer })?,
                config.expert_count,
            )?,
        })
    }

    fn total_bytes(&self) -> Result<usize, Hy3ScalarError> {
        self.gate
            .bytes
            .checked_add(self.up.bytes)
            .and_then(|total| total.checked_add(self.down.bytes))
            .ok_or(Hy3ScalarError::ArithmeticOverflow)
    }

    fn view<'a>(&self, bytes: &'a [u8]) -> Result<SwiGluExpert<'a>, Hy3ScalarError> {
        let total = self.total_bytes()?;
        if bytes.len() != total {
            return Err(Hy3ScalarError::ExpertPayloadLength {
                expected: total,
                actual: bytes.len(),
            });
        }
        let gate_end = self.gate.bytes;
        let up_end = gate_end
            .checked_add(self.up.bytes)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        Ok(SwiGluExpert::new(
            PackedMatrix::from_parts(
                self.gate.ty,
                PayloadEndian::Little,
                self.gate.input_width,
                self.gate.output_width,
                &bytes[..gate_end],
            )?,
            PackedMatrix::from_parts(
                self.up.ty,
                PayloadEndian::Little,
                self.up.input_width,
                self.up.output_width,
                &bytes[gate_end..up_end],
            )?,
            PackedMatrix::from_parts(
                self.down.ty,
                PayloadEndian::Little,
                self.down.input_width,
                self.down.output_width,
                &bytes[up_end..],
            )?,
        )?)
    }
}

#[derive(Debug)]
enum ExpertSource {
    Direct(DirectExpertStore),
    Sidecar(Sidecar),
}

impl ExpertSource {
    /// Opens an expert payload source for the validated model.
    ///
    /// Direct sources load payloads from the GGUF set. Sidecar sources verify the
    /// manifest's tensor-directory binding and optionally verify source bindings
    /// and the sidecar data hash.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let source = ExpertSource::open(&set, &model, &ExpertSourceOptions::Direct)?;
    /// # Ok::<(), Hy3ScalarError>(())
    /// ```
    fn open(
        set: &GgufSet,
        model: &ValidatedHy3Model,
        options: &ExpertSourceOptions,
    ) -> Result<Self, Hy3ScalarError> {
        match options {
            ExpertSourceOptions::Direct => Ok(Self::Direct(DirectExpertStore::open(set, model)?)),
            ExpertSourceOptions::Sidecar {
                data_path,
                manifest_path,
                verify_data_hash,
                verify_source_bindings: verify_bindings,
            } => {
                let sidecar = Sidecar::open(data_path, manifest_path)?;
                let actual_directory = tensor_directory_sha256(set)?;
                if sidecar.manifest().tensor_directory_sha256 != actual_directory {
                    return Err(Hy3ScalarError::SidecarDirectoryHash {
                        expected: sidecar.manifest().tensor_directory_sha256.clone(),
                        actual: actual_directory,
                    });
                }
                let cancellation = ReadCancellation::new();
                if *verify_bindings {
                    verify_source_bindings(set, sidecar.manifest(), HASH_CHUNK_BYTES, &cancellation)?;
                }
                if *verify_data_hash {
                    sidecar.verify_data_hash(&cancellation)?;
                }
                Ok(Self::Sidecar(sidecar))
            }
        }
    }

    /// Reads an expert payload into a tightly packed output buffer.
    ///
    /// The output buffer must have the size specified by `layout`, and each payload
    /// segment must match its corresponding layout segment.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload size or any segment length is invalid, the
    /// expert cannot be found, reading fails, or a size calculation overflows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// source.read_tight_into(key, &layout, &mut output, &cancellation)?;
    /// assert_eq!(output.len(), layout.total_bytes()?);
    /// # Ok::<(), ExpertReadError>(())
    /// ```
    fn read_tight_into(
        &self,
        key: ExpertKey,
        layout: &ExpertPayloadLayout,
        output: &mut [u8],
        cancellation: &ReadCancellation,
    ) -> Result<(), ExpertReadError> {
        let expected = layout
            .total_bytes()
            .map_err(|_| ExpertReadError::ArithmeticOverflow)?;
        if output.len() != expected {
            return Err(ExpertReadError::PayloadLength {
                expected,
                actual: output.len(),
            });
        }
        match self {
            Self::Direct(store) => {
                let record = store
                    .index()
                    .get(key)
                    .ok_or(bridge_prepare::PrepareError::MissingExpert(key))?;
                validate_segment_length(record.gate.length(), layout.gate.bytes, "gate")?;
                validate_segment_length(record.up.length(), layout.up.bytes, "up")?;
                validate_segment_length(record.down.length(), layout.down.bytes, "down")?;
                store.read_expert_into(key, output, cancellation)?;
            }
            Self::Sidecar(sidecar) => {
                let record = sidecar
                    .manifest()
                    .record(key)
                    .ok_or(bridge_format::SidecarError::MissingExpert(key))?;
                validate_segment_length(record.gate.length, layout.gate.bytes, "gate")?;
                validate_segment_length(record.up.length, layout.up.bytes, "up")?;
                validate_segment_length(record.down.length, layout.down.bytes, "down")?;
                sidecar.read_expert_into(key, output, cancellation)?;
            }
        }
        Ok(())
    }
}

/// Builds a reusable read-slot pool sized for the routed expert payloads in the model.
///
/// Returns `None` when the model has no MoE layers. Otherwise, the pool contains
/// enough aligned slots for the configured expert cache capacity.
///
/// # Examples
///
/// ```
/// # fn example() -> Result<(), Hy3ScalarError> {
/// let config = Hy3Config::default();
/// let slots = build_expert_read_slots(&[], &config, 0)?;
///
/// assert!(slots.is_none());
/// # Ok(())
/// # }
/// ```
fn build_expert_read_slots(
    layers: &[LayerWeights],
    config: &Hy3Config,
    cache_bytes: usize,
) -> Result<Option<ReadSlotPool>, Hy3ScalarError> {
    let mut expert_sizes = Vec::new();
    for layer in layers {
        let FeedForwardWeights::Moe(moe) = &layer.feed_forward else {
            continue;
        };
        expert_sizes.push(moe.expert_layout.total_bytes()?);
    }
    let Some((slot_count, slot_bytes)) =
        expert_slot_plan(&expert_sizes, config.expert_count as usize, cache_bytes)?
    else {
        return Ok(None);
    };
    Ok(Some(ReadSlotPool::new(slot_count, slot_bytes, 4_096)?))
}

/// Plans the number and size of reusable expert read slots for a cache.
///
/// Returns no plan when there are no expert layers. Otherwise, the slot size is
/// the largest expert payload size, and the slot count is bounded by the cache
/// capacity and total number of experts.
///
/// # Examples
///
/// ```
/// let plan = expert_slot_plan(&[100, 200], 4, 500).unwrap();
/// assert_eq!(plan, Some((2, 200)));
///
/// let empty = expert_slot_plan(&[], 4, 500).unwrap();
/// assert_eq!(empty, None);
/// ```
///
/// # Errors
///
/// Returns [`Hy3ScalarError::ArithmeticOverflow`] if the total expert count
/// cannot be represented by `usize`.
fn expert_slot_plan(
    expert_sizes: &[usize],
    experts_per_layer: usize,
    cache_bytes: usize,
) -> Result<Option<(usize, usize)>, Hy3ScalarError> {
    let Some(&slot_bytes) = expert_sizes.iter().max() else {
        return Ok(None);
    };
    let all_experts = expert_sizes
        .len()
        .checked_mul(experts_per_layer)
        .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
    let slot_count = (cache_bytes / slot_bytes).min(all_experts).max(1);
    Ok(Some((slot_count, slot_bytes)))
}

#[cfg(test)]
mod expert_slot_tests {
    use super::expert_slot_plan;

    #[test]
    fn heterogeneous_layer_records_share_slots_sized_for_the_largest_layout() {
        let smaller = 6_045_696;
        let larger = 6_733_824;
        let cache_bytes = 512 * 1024 * 1024;
        let (slot_count, slot_bytes) = expert_slot_plan(&[smaller, larger], 192, cache_bytes)
            .unwrap()
            .unwrap();

        assert_eq!(slot_bytes, larger);
        assert_eq!(slot_count, cache_bytes / larger);
        assert!(slot_count < 2 * 192);
    }

    #[test]
    fn model_without_moe_layers_needs_no_expert_slots() {
        assert_eq!(expert_slot_plan(&[], 192, 1024).unwrap(), None);
    }
}

struct TensorLoader {
    readers: Vec<PositionedFile>,
}

impl TensorLoader {
    fn open(set: &GgufSet, model: &ValidatedHy3Model) -> Result<Self, Hy3ScalarError> {
        let maximum = model
            .tensors()
            .iter()
            .filter(|tensor| !tensor.role().is_routed_expert())
            .map(|tensor| tensor.location().absolute_range().end - tensor.location().absolute_range().start)
            .max()
            .unwrap_or(1);
        let maximum = usize::try_from(maximum).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
        let mut readers = Vec::new();
        readers
            .try_reserve_exact(set.files().len())
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "static tensor readers",
                requested: set.files().len(),
            })?;
        for shard in set.files() {
            readers.push(PositionedFile::open(
                shard.path(),
                ReadLimits {
                    max_request_bytes: maximum.max(1),
                },
            )?);
        }
        Ok(Self { readers })
    }

    fn matrix(&self, model: &ValidatedHy3Model, role: Hy3TensorRole) -> Result<OwnedMatrix, Hy3ScalarError> {
        let tensor = required_tensor(model, role)?;
        let shape = tensor.location().descriptor().shape();
        if shape.len() != 2 {
            return Err(Hy3ScalarError::TensorRank {
                role,
                expected: 2,
                actual: shape.len(),
            });
        }
        let bytes = self.read(tensor)?;
        let matrix = OwnedMatrix {
            ty: tensor.location().descriptor().ty(),
            input_width: usize::try_from(shape[0]).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?,
            output_width: usize::try_from(shape[1]).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?,
            bytes,
        };
        matrix.view()?;
        Ok(matrix)
    }

    fn f32_vector(&self, model: &ValidatedHy3Model, role: Hy3TensorRole) -> Result<Vec<f32>, Hy3ScalarError> {
        let tensor = required_tensor(model, role)?;
        let descriptor = tensor.location().descriptor();
        if descriptor.shape().len() != 1 {
            return Err(Hy3ScalarError::TensorRank {
                role,
                expected: 1,
                actual: descriptor.shape().len(),
            });
        }
        if descriptor.ty() != GgmlType::F32 {
            return Err(Hy3ScalarError::TensorType {
                role,
                expected: GgmlType::F32,
                actual: descriptor.ty(),
            });
        }
        let bytes = self.read(tensor)?;
        let expected =
            usize::try_from(descriptor.shape()[0]).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
        if bytes.len()
            != expected
                .checked_mul(4)
                .ok_or(Hy3ScalarError::ArithmeticOverflow)?
        {
            return Err(Hy3ScalarError::F32PayloadLength {
                role,
                expected: expected * 4,
                actual: bytes.len(),
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(expected)
            .map_err(|_| Hy3ScalarError::AllocationFailed {
                context: "F32 tensor",
                requested: expected,
            })?;
        for (index, chunk) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_bits(u32::from_le_bytes(
                chunk.try_into().expect("chunks_exact yields four bytes"),
            ));
            if !value.is_finite() {
                return Err(Hy3ScalarError::NonFiniteF32 {
                    role,
                    index,
                    bits: value.to_bits(),
                });
            }
            values.push(value);
        }
        Ok(values)
    }

    fn read(&self, tensor: &Hy3Tensor) -> Result<Vec<u8>, Hy3ScalarError> {
        let reader =
            self.readers
                .get(tensor.location().shard_index())
                .ok_or(Hy3ScalarError::ShardIndexOutOfRange(
                    tensor.location().shard_index(),
                ))?;
        Ok(reader.read_exact_at(
            tensor.location().absolute_range().clone(),
            &ReadCancellation::new(),
        )?)
    }
}

/// Validates runtime options against the model configuration and cache requirements.
///
/// # Errors
///
/// Returns an error when context, KV paging, prefill, speculation, cache, or
/// grouped expert-cache settings are invalid or overflow during sizing.
///
/// # Examples
///
/// ```rust,ignore
/// assert!(validate_options(&model, &options).is_ok());
/// ```
fn validate_options(model: &ValidatedHy3Model, options: &Hy3ScalarOptions) -> Result<(), Hy3ScalarError> {
    if options.context_capacity == 0 || options.context_capacity > model.config().context_length as usize {
        return Err(Hy3ScalarError::InvalidContextCapacity {
            maximum: model.config().context_length,
            actual: options.context_capacity,
        });
    }
    if options.kv_page_tokens == 0 {
        return Err(Hy3ScalarError::ZeroKvPageTokens);
    }
    if !matches!(options.prefill_chunk, 1 | 2 | 4 | 8) {
        return Err(Hy3ScalarError::InvalidPrefillChunk(options.prefill_chunk));
    }
    if let Some(t) = options.speculative_ngram_t {
        if t != 2 {
            return Err(Hy3ScalarError::InvalidSpeculativeWidth(t));
        }
        if options.prefill_chunk < t {
            return Err(Hy3ScalarError::SpeculationRequiresGroupedPrefill {
                speculative_width: t,
                prefill_chunk: options.prefill_chunk,
            });
        }
    }
    CacheConfig {
        capacity_bytes: options.expert_cache_bytes,
        admit_after_requests: options.cache_admit_after_requests,
    }
    .validate()?;
    if model.config().block_count > 1 {
        let maximum_expert = (1..model.config().block_count).try_fold(0_usize, |maximum, layer| {
            Ok::<_, Hy3ScalarError>(
                maximum.max(ExpertPayloadLayout::from_model(model, layer)?.total_bytes()?),
            )
        })?;
        if options.expert_cache_bytes < maximum_expert {
            return Err(Hy3ScalarError::ExpertCacheTooSmall {
                minimum: maximum_expert,
                actual: options.expert_cache_bytes,
            });
        }
        let union_experts = (model.config().expert_used_count as usize)
            .checked_mul(options.prefill_chunk)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?
            .min(model.config().expert_count as usize);
        let grouped_minimum = maximum_expert
            .checked_mul(union_experts)
            .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        if options.expert_cache_bytes < grouped_minimum {
            return Err(Hy3ScalarError::GroupedExpertCacheTooSmall {
                chunk: options.prefill_chunk,
                minimum: grouped_minimum,
                actual: options.expert_cache_bytes,
            });
        }
    }
    Ok(())
}

fn verify_file_hashes(set: &GgufSet, expected: &[&str]) -> Result<Vec<String>, Hy3ScalarError> {
    if set.files().len() != expected.len() {
        return Err(Hy3ScalarError::IntegrityFileCount {
            expected: expected.len(),
            actual: set.files().len(),
        });
    }
    let cancellation = ReadCancellation::new();
    let mut hashes = Vec::new();
    hashes
        .try_reserve_exact(expected.len())
        .map_err(|_| Hy3ScalarError::AllocationFailed {
            context: "source hashes",
            requested: expected.len(),
        })?;
    let mut buffer = fallible_zeroed(HASH_CHUNK_BYTES, "integrity hash buffer")?;
    for (shard, &expected_hash) in set.files().iter().zip(expected) {
        let reader = PositionedFile::open(
            shard.path(),
            ReadLimits {
                max_request_bytes: HASH_CHUNK_BYTES,
            },
        )?;
        let mut hasher = Sha256::new();
        let mut offset = 0_u64;
        while offset < reader.length() {
            let remaining = reader.length() - offset;
            let length = usize::try_from(remaining.min(HASH_CHUNK_BYTES as u64))
                .map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
            reader.read_exact_at_into(offset, &mut buffer[..length], &cancellation)?;
            hasher.update(&buffer[..length]);
            offset = offset
                .checked_add(length as u64)
                .ok_or(Hy3ScalarError::ArithmeticOverflow)?;
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected_hash {
            return Err(Hy3ScalarError::IntegrityHash {
                path: shard.path().to_owned(),
                expected: expected_hash.to_owned(),
                actual,
            });
        }
        hashes.push(actual);
    }
    Ok(hashes)
}

fn build_model_fingerprint(set: &GgufSet, source_sha256: &[String]) -> Result<[u8; 32], Hy3ScalarError> {
    let mut hasher = Sha256::new();
    hasher.update(b"lightbridge-hy3-kv-v1\0");
    hasher.update(tensor_directory_sha256(set)?.as_bytes());
    for hash in source_sha256 {
        hasher.update([0]);
        hasher.update(hash.as_bytes());
    }
    Ok(hasher.finalize().into())
}

fn required_tensor(model: &ValidatedHy3Model, role: Hy3TensorRole) -> Result<&Hy3Tensor, Hy3ScalarError> {
    model
        .tensor_for_role(role)
        .ok_or(Hy3ScalarError::MissingTensorRole(role))
}

/// Derives the matrix layout for an expert tensor.
///
/// The tensor must have rank three and contain an expert slab for the specified number of experts.
///
/// # Examples
///
/// ```no_run
/// let layout = matrix_layout(&tensor, 8)?;
/// assert_eq!(layout.input_width, 4096);
/// # Ok::<(), Hy3ScalarError>(())
/// ```
fn matrix_layout(tensor: &Hy3Tensor, expert_count: u32) -> Result<MatrixLayout, Hy3ScalarError> {
    let descriptor = tensor.location().descriptor();
    let shape = descriptor.shape();
    if shape.len() != 3 {
        return Err(Hy3ScalarError::TensorRank {
            role: tensor.role(),
            expected: 3,
            actual: shape.len(),
        });
    }
    let ExpertSlab { relative_range, .. } = tensor.expert_slab(expert_count, 0)?;
    let bytes = usize::try_from(relative_range.end - relative_range.start)
        .map_err(|_| Hy3ScalarError::ArithmeticOverflow)?;
    Ok(MatrixLayout {
        ty: descriptor.ty(),
        input_width: usize::try_from(shape[0]).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?,
        output_width: usize::try_from(shape[1]).map_err(|_| Hy3ScalarError::ArithmeticOverflow)?,
        bytes,
    })
}

/// Validates that a segment has the expected byte length.
///
/// # Examples
///
/// ```
/// assert!(validate_segment_length(16, 16, "gate").is_ok());
/// assert!(validate_segment_length(8, 16, "gate").is_err());
/// ```
///
/// # Arguments
///
/// * `actual` - The observed segment length in bytes.
/// * `expected` - The required segment length in bytes.
/// * `segment` - The segment name used in a length-mismatch error.
///
/// # Errors
///
/// Returns an error if the observed length cannot be represented as `usize` or
/// differs from the expected length.
fn validate_segment_length(
    actual: u64,
    expected: usize,
    segment: &'static str,
) -> Result<(), ExpertReadError> {
    let actual = usize::try_from(actual).map_err(|_| ExpertReadError::ArithmeticOverflow)?;
    if actual != expected {
        return Err(ExpertReadError::SegmentLength {
            segment,
            expected,
            actual,
        });
    }
    Ok(())
}

fn fallible_zeroed<T>(length: usize, context: &'static str) -> Result<Vec<T>, Hy3ScalarError>
where
    T: Clone + Default,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Hy3ScalarError::AllocationFailed {
            context,
            requested: length,
        })?;
    values.resize(length, T::default());
    Ok(values)
}

#[derive(Debug, thiserror::Error)]
pub enum ExpertReadError {
    #[error(transparent)]
    Direct(#[from] bridge_prepare::PrepareError),
    #[error(transparent)]
    Sidecar(#[from] bridge_format::SidecarError),
    #[error("expert {segment} segment is {actual} bytes, expected {expected}")]
    SegmentLength {
        segment: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("checked arithmetic overflow while loading an expert")]
    ArithmeticOverflow,
    #[error("expert output buffer is {actual} bytes, expected {expected}")]
    PayloadLength { expected: usize, actual: usize },
    #[error(transparent)]
    ReadSlot(#[from] SlotPoolError),
}

#[derive(Debug, thiserror::Error)]
pub enum Hy3ScalarError {
    #[error(transparent)]
    Split(#[from] bridge_gguf_split::SplitError),
    #[error(transparent)]
    Model(#[from] bridge_model_hy3::Hy3Error),
    #[error(transparent)]
    Read(#[from] bridge_io_windows::ReadError),
    #[error(transparent)]
    Kernel(#[from] bridge_kernels_reference::KernelError),
    #[error(transparent)]
    Cpu(#[from] bridge_kernels_cpu::CpuBackendError),
    #[error(transparent)]
    Cuda(#[from] bridge_kernels_cuda::CudaRuntimeError),
    #[error(transparent)]
    Kv(#[from] bridge_kv_gqa::KvError),
    #[error(transparent)]
    Quant(#[from] bridge_quant_layout::QuantError),
    #[error(transparent)]
    Prepare(#[from] bridge_prepare::PrepareError),
    #[error(transparent)]
    Sidecar(#[from] bridge_format::SidecarError),
    #[error(transparent)]
    Cache(#[from] bridge_cache::CacheError),
    #[error(transparent)]
    CacheHeat(#[from] bridge_cache::HeatError),
    #[error(transparent)]
    ReadSlot(#[from] SlotPoolError),
    #[error("expert cache load failed: {0}")]
    ExpertCache(#[source] LoadError<ExpertReadError>),
    #[error("selected model has {actual} shards, expected exactly one")]
    SelectedModelShardCount { actual: usize },
    #[error("selected model length is {actual}, expected {expected}")]
    SelectedModelLength { expected: u64, actual: u64 },
    #[error("failed to inspect physical storage for {path:?}: {source}")]
    StorageInspection {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "selected model {path:?} is a sparse/incomplete payload: {allocated_bytes} allocated bytes for {logical_bytes} logical bytes"
    )]
    SparseModelPayload {
        path: PathBuf,
        logical_bytes: u64,
        allocated_bytes: u64,
    },
    #[error("integrity policy has {expected} hashes for {actual} source files")]
    IntegrityFileCount { expected: usize, actual: usize },
    #[error("model integrity hash for {path:?} is {actual}, expected {expected}")]
    IntegrityHash {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("big-endian tensor payloads are not executable: {0:?}")]
    BigEndianPayload(PathBuf),
    #[error("validated model is missing tensor role {0:?}")]
    MissingTensorRole(Hy3TensorRole),
    #[error("tensor role {role:?} has rank {actual}, expected {expected}")]
    TensorRank {
        role: Hy3TensorRole,
        expected: usize,
        actual: usize,
    },
    #[error("tensor role {role:?} has type {actual:?}, expected {expected:?}")]
    TensorType {
        role: Hy3TensorRole,
        expected: GgmlType,
        actual: GgmlType,
    },
    #[error("F32 tensor role {role:?} is {actual} bytes, expected {expected}")]
    F32PayloadLength {
        role: Hy3TensorRole,
        expected: usize,
        actual: usize,
    },
    #[error("F32 tensor role {role:?} index {index} is non-finite ({bits:#010x})")]
    NonFiniteF32 {
        role: Hy3TensorRole,
        index: usize,
        bits: u32,
    },
    #[error("tensor references missing shard index {0}")]
    ShardIndexOutOfRange(usize),
    #[error("context capacity is {actual}, expected 1..={maximum}")]
    InvalidContextCapacity { maximum: u64, actual: usize },
    #[error("KV page token count must be non-zero")]
    ZeroKvPageTokens,
    #[error("prefill chunk is {0}; expected 1, 2, 4, or 8")]
    InvalidPrefillChunk(usize),
    #[error("speculative width is {0}; only T=2 is supported")]
    InvalidSpeculativeWidth(usize),
    #[error(
        "T={speculative_width} speculation requires grouped prefill at least that wide, but \
         prefill chunk is {prefill_chunk}"
    )]
    SpeculationRequiresGroupedPrefill {
        speculative_width: usize,
        prefill_chunk: usize,
    },
    #[error("grouped token count is {actual}; expected 1..={maximum}")]
    GroupedTokenCount { actual: usize, maximum: usize },
    #[error(
        "speculative logits have {actual} values for {tokens} tokens and vocabulary \
         {vocabulary_size}"
    )]
    SpeculativeLogitShape {
        tokens: usize,
        vocabulary_size: usize,
        actual: usize,
    },
    #[error("grouped {kind} scratch is missing for position {position}")]
    GroupedScratchMissing { kind: &'static str, position: usize },
    #[error("expert cache is {actual} bytes, expected at least one complete expert ({minimum} bytes)")]
    ExpertCacheTooSmall { minimum: usize, actual: usize },
    #[error(
        "prefill chunk {chunk} needs at least {minimum} expert-cache bytes for its worst-case \
         route union, but only {actual} are configured"
    )]
    GroupedExpertCacheTooSmall {
        chunk: usize,
        minimum: usize,
        actual: usize,
    },
    #[error("model contains no dense layer zero")]
    MissingDenseLayer,
    #[error("MoE execution has no scratch workspace")]
    MissingMoeScratch,
    #[error("MoE execution has no expert source")]
    MissingExpertSource,
    #[error("MoE execution has no aligned expert read-slot pool")]
    MissingExpertReadSlots,
    #[error("selected expert lease was not populated")]
    MissingExpertLease,
    #[error("selected expert ID {expert_id} is outside expert count {expert_count}")]
    SelectedExpertId { expert_id: u32, expert_count: usize },
    #[error("selected expert count is {actual}, maximum supported without allocation is {maximum}")]
    SelectedExpertCount { actual: usize, maximum: usize },
    #[error("sidecar tensor-directory hash is {actual}, expected {expected}")]
    SidecarDirectoryHash { expected: String, actual: String },
    #[error("expert payload is {actual} bytes, expected {expected}")]
    ExpertPayloadLength { expected: usize, actual: usize },
    #[error("token ID {token_id} is outside vocabulary size {vocabulary_size}")]
    TokenOutOfRange { token_id: u32, vocabulary_size: usize },
    #[error("session position {position} exhausted context capacity {capacity}")]
    ContextExhausted { position: usize, capacity: usize },
    #[error("logit output has length {actual}, expected {expected}")]
    LogitLength { expected: usize, actual: usize },
    #[error(
        "insufficient available physical memory: need {required} bytes, have {available} \
         (resident weights {resident_weights}, expert cache {expert_cache}, first KV page \
         {first_kv_page}, safety headroom {headroom})"
    )]
    InsufficientPhysicalMemory {
        required: u64,
        available: u64,
        resident_weights: u64,
        expert_cache: u64,
        first_kv_page: u64,
        headroom: u64,
    },
    #[error("token evaluation failed ({original}) and session rollback also failed ({rollback})")]
    SessionRollback { original: String, rollback: String },
    #[error("CUDA token execution failed ({cuda}) and the authoritative CPU retry also failed ({cpu})")]
    CudaFallbackFailed { cuda: String, cpu: String },
    #[error(
        "CUDA reusable packed qualification did not pass: bit_exact={bit_exact}, \
         deterministic={deterministic}"
    )]
    CudaQualification { bit_exact: bool, deterministic: bool },
    #[error("KV layer {layer} stores {actual} tokens, expected committed position {expected}")]
    InconsistentKvLength {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error("checked arithmetic overflow in Hy3 scalar runtime")]
    ArithmeticOverflow,
    #[error("backend {backend} is unavailable: {reason}")]
    BackendUnavailable {
        backend: &'static str,
        reason: &'static str,
    },
    #[error("allocation failed while reserving {requested} entries for {context}")]
    AllocationFailed { context: &'static str, requested: usize },
}
