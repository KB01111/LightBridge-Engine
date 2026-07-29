use std::path::{Path, PathBuf};

use bridge_cache::{CacheConfig, CacheLease, CacheStats, CompressedCache, LoadError};
use bridge_core::ggml_type::GgmlType;
use bridge_core::sys::memory_status;
use bridge_format::{ExpertKey, Sidecar};
use bridge_gguf::Endianness;
use bridge_gguf_split::{open_set, GgufSet};
use bridge_io_windows::{file_storage, PositionedFile, ReadCancellation, ReadLimits};
use bridge_kernels_cpu::{CpuBackend, CpuBackendConfig};
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
    pub memory_headroom_bytes: usize,
    pub expert_source: ExpertSourceOptions,
}

impl Default for Hy3ScalarOptions {
    fn default() -> Self {
        Self {
            context_capacity: 2_048,
            kv_page_tokens: 64,
            expert_cache_bytes: 2 * 1024 * 1024 * 1024,
            cache_admit_after_requests: 2,
            execution_mode: ReferenceExecutionMode::CpuParallelQ8K,
            cpu_threads: bridge_kernels_cpu::recommended_thread_count(),
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
    cpu_backend: Option<CpuBackend>,
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

    pub const fn resident_weight_bytes(&self) -> usize {
        self.resident_weight_bytes
    }

    pub fn backend_name(&self) -> &'static str {
        match self.execution_mode {
            ReferenceExecutionMode::DequantF32 => "scalar_reference_dequant_f32",
            ReferenceExecutionMode::LlamaQ8K => "scalar_reference_q8_k",
            ReferenceExecutionMode::CpuParallelQ8K => self
                .cpu_backend
                .as_ref()
                .map(CpuBackend::backend_name)
                .unwrap_or("cpu_parallel_scalar_q8_k"),
        }
    }

    pub fn cpu_threads(&self) -> Option<usize> {
        self.cpu_backend.as_ref().map(|backend| backend.config().threads)
    }

    pub fn cpu_simd_active(&self) -> bool {
        self.cpu_backend.as_ref().is_some_and(CpuBackend::simd_active)
    }

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
        let cpu_backend = match options.execution_mode {
            ReferenceExecutionMode::CpuParallelQ8K => Some(CpuBackend::new(CpuBackendConfig {
                threads: options.cpu_threads,
            })?),
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
        let expert_cache = CompressedCache::new(CacheConfig {
            capacity_bytes: options.expert_cache_bytes,
            admit_after_requests: options.cache_admit_after_requests,
        })?;

        Ok(Self {
            config,
            context_capacity: options.context_capacity,
            execution_mode: options.execution_mode,
            cpu_backend,
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
            expert_cache,
        })
    }

    fn evaluate(
        &self,
        session: &mut Hy3ScalarSession,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Hy3ScalarError> {
        let committed_position = session.position;
        let result = match &self.cpu_backend {
            Some(backend) => backend.install(|| self.evaluate_inner(session, token_id, logits)),
            None => self.evaluate_inner(session, token_id, logits),
        };
        if let Err(error) = result {
            if let Err(rollback) = session.rollback_to(committed_position) {
                return Err(Hy3ScalarError::SessionRollback {
                    original: error.to_string(),
                    rollback: rollback.to_string(),
                });
            }
            return Err(error);
        }
        Ok(())
    }

    fn evaluate_inner(
        &self,
        session: &mut Hy3ScalarSession,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Hy3ScalarError> {
        if logits.len() != self.config.vocabulary_size as usize {
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
                mode: self.execution_mode,
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
                    let routes = scratch.routed().to_vec();
                    session.expert_leases.clear();
                    let load_route = |route: &bridge_model_hy3::RoutedExpert| {
                        let key = ExpertKey {
                            layer: layer_number,
                            expert: route.expert_id,
                        };
                        let expected = moe.expert_layout.total_bytes()?;
                        let source = self
                            .expert_source
                            .as_ref()
                            .ok_or(Hy3ScalarError::MissingExpertSource)?;
                        let layout = &moe.expert_layout;
                        self.expert_cache
                            .get_or_try_insert(key, expected, || source.read_tight(key, layout))
                            .map_err(Hy3ScalarError::ExpertCache)
                    };
                    session.expert_leases = if self.cpu_backend.is_some() {
                        routes.par_iter().map(load_route).collect::<Result<Vec<_>, _>>()?
                    } else {
                        let mut leases = Vec::new();
                        leases.try_reserve_exact(routes.len()).map_err(|_| {
                            Hy3ScalarError::AllocationFailed {
                                context: "selected expert leases",
                                requested: routes.len(),
                            }
                        })?;
                        for route in &routes {
                            leases.push(load_route(route)?);
                        }
                        leases
                    };

                    let mut selected = Vec::new();
                    selected.try_reserve_exact(routes.len()).map_err(|_| {
                        Hy3ScalarError::AllocationFailed {
                            context: "selected expert views",
                            requested: routes.len(),
                        }
                    })?;
                    for (route, lease) in routes.iter().zip(&session.expert_leases) {
                        selected.push(SelectedExpert {
                            expert_id: route.expert_id,
                            coefficient: route.coefficient,
                            expert: moe.expert_layout.view(lease.bytes())?,
                        });
                    }
                    hy3_moe_finish_token(
                        self.execution_mode,
                        &selected,
                        moe.shared.view()?,
                        &mut session.hidden,
                        scratch,
                    )?;
                    session.expert_leases.clear();
                }
            }
        }

        weighted_rms_norm_into(
            &session.hidden,
            &self.output_norm,
            self.config.rms_epsilon,
            &mut session.final_normalized,
        )?;
        gemv_into(
            self.execution_mode,
            self.output.view()?,
            &session.final_normalized,
            logits,
            &mut session.decoded_block,
            &mut session.q8,
        )?;
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
        let moe_scratch = self
            .layers
            .iter()
            .find_map(|layer| match &layer.feed_forward {
                FeedForwardWeights::Moe(moe) => {
                    Some(moe.streaming_weights(&layer.attention, &layer.ffn_norm, &self.config))
                }
                FeedForwardWeights::Dense(_) => None,
            })
            .transpose()?
            .map(|weights| Hy3BlockScratch::new_streaming_moe(weights, self.context_capacity))
            .transpose()?;
        let hidden_width = self.config.embedding_length as usize;
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
            hidden: fallible_zeroed(hidden_width, "hidden state")?,
            final_normalized: fallible_zeroed(hidden_width, "final normalized state")?,
            decoded_block: fallible_zeroed(256, "output decoded block")?,
            q8: fallible_zeroed(
                bridge_kernels_reference::required_q8_k_bytes(hidden_width)?,
                "output Q8_K row",
            )?,
            expert_leases: Vec::with_capacity(self.config.expert_used_count as usize),
            position: 0,
        })
    }

    fn reset_session(&self, session: &mut Self::Session) {
        session.cache.reset();
        session.dense_scratch.reset();
        if let Some(scratch) = &mut session.moe_scratch {
            scratch.reset();
        }
        session.expert_leases.clear();
        session.hidden.fill(0.0);
        session.final_normalized.fill(0.0);
        session.position = 0;
    }

    fn position(&self, session: &Self::Session) -> usize {
        session.position
    }

    fn evaluate_token(
        &self,
        session: &mut Self::Session,
        token_id: u32,
        logits: &mut [f32],
    ) -> Result<(), Self::Error> {
        self.evaluate(session, token_id, logits)
    }
}

#[derive(Debug)]
pub struct Hy3ScalarSession {
    cache: PagedKvCache,
    dense_scratch: Hy3BlockScratch,
    moe_scratch: Option<Hy3BlockScratch>,
    hidden: Vec<f32>,
    final_normalized: Vec<f32>,
    decoded_block: Vec<f32>,
    q8: Vec<u8>,
    expert_leases: Vec<CacheLease<ExpertKey>>,
    position: usize,
}

impl Hy3ScalarSession {
    pub fn kv_stored_tokens(&self, layer: usize) -> Result<usize, Hy3ScalarError> {
        Ok(self.cache.stored_tokens(layer)?)
    }

    fn rollback_to(&mut self, committed_position: usize) -> Result<(), Hy3ScalarError> {
        self.cache.rewind_all(committed_position)?;
        self.dense_scratch.reset();
        if let Some(scratch) = &mut self.moe_scratch {
            scratch.reset();
        }
        self.expert_leases.clear();
        self.hidden.fill(0.0);
        self.final_normalized.fill(0.0);
        self.decoded_block.fill(0.0);
        self.q8.fill(0);
        self.position = committed_position;
        Ok(())
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

    fn read_tight(&self, key: ExpertKey, layout: &ExpertPayloadLayout) -> Result<Vec<u8>, ExpertReadError> {
        let cancellation = ReadCancellation::new();
        let mut output = Vec::new();
        let expected = layout
            .total_bytes()
            .map_err(|_| ExpertReadError::ArithmeticOverflow)?;
        output
            .try_reserve_exact(expected)
            .map_err(|_| ExpertReadError::AllocationFailed { requested: expected })?;
        match self {
            Self::Direct(store) => {
                let bytes = store.read_expert(key, &cancellation)?;
                append_checked(&mut output, &bytes.gate, layout.gate.bytes, "gate")?;
                append_checked(&mut output, &bytes.up, layout.up.bytes, "up")?;
                append_checked(&mut output, &bytes.down, layout.down.bytes, "down")?;
            }
            Self::Sidecar(sidecar) => {
                let bytes = sidecar.read_expert(key, &cancellation)?;
                append_checked(&mut output, bytes.gate(), layout.gate.bytes, "gate")?;
                append_checked(&mut output, bytes.up(), layout.up.bytes, "up")?;
                append_checked(&mut output, bytes.down(), layout.down.bytes, "down")?;
            }
        }
        Ok(output)
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

fn append_checked(
    output: &mut Vec<u8>,
    bytes: &[u8],
    expected: usize,
    segment: &'static str,
) -> Result<(), ExpertReadError> {
    if bytes.len() != expected {
        return Err(ExpertReadError::SegmentLength {
            segment,
            expected,
            actual: bytes.len(),
        });
    }
    output.extend_from_slice(bytes);
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
    #[error("allocation failed while reserving {requested} expert bytes")]
    AllocationFailed { requested: usize },
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
    #[error("expert cache is {actual} bytes, expected at least one complete expert ({minimum} bytes)")]
    ExpertCacheTooSmall { minimum: usize, actual: usize },
    #[error("model contains no dense layer zero")]
    MissingDenseLayer,
    #[error("MoE execution has no scratch workspace")]
    MissingMoeScratch,
    #[error("MoE execution has no expert source")]
    MissingExpertSource,
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
    #[error("KV layer {layer} stores {actual} tokens, expected committed position {expected}")]
    InconsistentKvLength {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error("checked arithmetic overflow in Hy3 scalar runtime")]
    ArithmeticOverflow,
    #[error("allocation failed while reserving {requested} entries for {context}")]
    AllocationFailed { context: &'static str, requested: usize },
}
