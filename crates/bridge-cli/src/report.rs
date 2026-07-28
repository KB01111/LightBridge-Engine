use std::collections::BTreeMap;
use std::path::PathBuf;

use bridge_gguf::{Endianness, GgufFile, GgufValueType, MetadataError};
use bridge_gguf_split::GgufSet;
use bridge_model_hy3::{validate_selected_model, Hy3Error, Hy3TensorRole, ValidatedHy3Model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Aggregate {
    pub count: u64,
    pub logical_elements: u64,
    pub encoded_bytes: u64,
}

impl Aggregate {
    pub const fn zero() -> Self {
        Self {
            count: 0,
            logical_elements: 0,
            encoded_bytes: 0,
        }
    }

    pub fn checked_add(
        &mut self,
        count: u64,
        logical_elements: u64,
        encoded_bytes: u64,
    ) -> Result<(), ReportError> {
        let count = self
            .count
            .checked_add(count)
            .ok_or(ReportError::ArithmeticOverflow("aggregate tensor count"))?;
        let logical_elements = self
            .logical_elements
            .checked_add(logical_elements)
            .ok_or(ReportError::ArithmeticOverflow("aggregate logical element count"))?;
        let encoded_bytes = self
            .encoded_bytes
            .checked_add(encoded_bytes)
            .ok_or(ReportError::ArithmeticOverflow("aggregate encoded byte count"))?;
        self.count = count;
        self.logical_elements = logical_elements;
        self.encoded_bytes = encoded_bytes;
        Ok(())
    }
}

impl Default for Aggregate {
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileReport {
    pub path: String,
    pub ordinal: u32,
    pub count: u32,
    pub version: u32,
    pub endianness: String,
    pub metadata_count: u64,
    pub alignment: u64,
    pub logical_size: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufReport {
    pub version: u32,
    pub endianness: String,
    pub authoritative_metadata_count: u64,
    pub tensor_count: u64,
    pub alignment: u64,
    pub encoded_tensor_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneralReport {
    pub architecture: String,
    pub name: Option<String>,
    pub license: Option<String>,
    pub size_label: Option<String>,
    pub quantization_version: Option<u32>,
    pub file_type: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerReport {
    pub model: Option<String>,
    pub pretokenizer: Option<String>,
    pub token_count: u64,
    pub merge_count: Option<u64>,
    pub token_type_count: Option<u64>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub separator_token_id: Option<u32>,
    pub has_chat_template: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hy3Report {
    pub block_count: u32,
    pub context_length: u64,
    pub embedding_length: u32,
    pub dense_ffn_length: u32,
    pub expert_ffn_length: u32,
    pub shared_expert_ffn_length: u32,
    pub attention_head_count: u32,
    pub attention_kv_head_count: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub rms_epsilon: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    pub expert_weights_scale: f32,
    pub rope_base: f32,
    pub rope_scaling_type: String,
    pub yarn_factor: f32,
    pub yarn_original_context: u64,
    pub vocabulary_size: u32,
    pub has_mtp: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSummary {
    pub total: Aggregate,
    pub dense_layer_0_ffn: Aggregate,
    pub routed_experts: Aggregate,
    pub shared_experts: Aggregate,
    pub attention: Aggregate,
    pub routers: Aggregate,
    pub embeddings: Aggregate,
    pub norms: Aggregate,
    pub output: Aggregate,
}

impl TensorSummary {
    fn new() -> Self {
        Self {
            total: Aggregate::zero(),
            dense_layer_0_ffn: Aggregate::zero(),
            routed_experts: Aggregate::zero(),
            shared_experts: Aggregate::zero(),
            attention: Aggregate::zero(),
            routers: Aggregate::zero(),
            embeddings: Aggregate::zero(),
            norms: Aggregate::zero(),
            output: Aggregate::zero(),
        }
    }

    fn category_mut(&mut self, category: TensorCategory) -> &mut Aggregate {
        match category {
            TensorCategory::Dense => &mut self.dense_layer_0_ffn,
            TensorCategory::Routed => &mut self.routed_experts,
            TensorCategory::Shared => &mut self.shared_experts,
            TensorCategory::Attention => &mut self.attention,
            TensorCategory::Router => &mut self.routers,
            TensorCategory::Embedding => &mut self.embeddings,
            TensorCategory::Norm => &mut self.norms,
            TensorCategory::Output => &mut self.output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertProjectionReport {
    pub projection: String,
    pub physical_type: String,
    pub tensor_count: u64,
    pub expert_count: u32,
    pub slab_logical_elements: u64,
    pub slab_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertStorageReport {
    pub expert_count: u32,
    pub routed_banks: Aggregate,
    pub shared_experts: Aggregate,
    pub routed_projections: BTreeMap<String, ExpertProjectionReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InspectionReport {
    pub files: Vec<FileReport>,
    pub gguf: GgufReport,
    pub general: GeneralReport,
    pub tokenizer: TokenizerReport,
    pub hy3: Hy3Report,
    pub tensors: TensorSummary,
    pub types: BTreeMap<String, Aggregate>,
    pub roles: BTreeMap<String, Aggregate>,
    pub layers: BTreeMap<u32, Aggregate>,
    pub expert_storage: ExpertStorageReport,
    pub unsupported_execution_types: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error(transparent)]
    Hy3(#[from] Hy3Error),
    #[error(transparent)]
    Core(#[from] bridge_core::error::CoreError),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("allocation failed while reserving {0}")]
    AllocationFailed(&'static str),
    #[error("shard path is not valid Unicode and cannot be represented deterministically: {0:?}")]
    NonUnicodePath(PathBuf),
    #[error("metadata array {key:?} has element type {actual:?}, expected {expected:?}")]
    MetadataArrayElementType {
        key: &'static str,
        expected: GgufValueType,
        actual: GgufValueType,
    },
    #[error("routed expert tensor {name:?} logical element count is not divisible by {expert_count}")]
    ExpertElementCountNotDivisible { name: String, expert_count: u32 },
    #[error("routed expert projection group {key:?} has inconsistent slab dimensions")]
    InconsistentExpertProjection { key: String },
}

#[derive(Clone, Copy)]
enum TensorCategory {
    Dense,
    Routed,
    Shared,
    Attention,
    Router,
    Embedding,
    Norm,
    Output,
}

pub fn build_report(set: &GgufSet) -> Result<InspectionReport, ReportError> {
    let model = validate_selected_model(set)?;
    let primary = set
        .files()
        .first()
        .expect("selected Hy3 validation requires a non-empty set")
        .parsed();

    let files = build_files(set)?;
    let general = build_general(primary)?;
    let tokenizer = build_tokenizer(primary)?;
    let hy3 = build_hy3(&model);
    let mut tensors = TensorSummary::new();
    let mut types = BTreeMap::<String, Aggregate>::new();
    let mut roles = BTreeMap::<String, Aggregate>::new();
    let mut layers = BTreeMap::<u32, Aggregate>::new();
    let mut routed_projections = BTreeMap::<String, ExpertProjectionReport>::new();

    for tensor in model.tensors() {
        let descriptor = tensor.location().descriptor();
        let logical_elements = descriptor.element_count()?;
        let encoded_bytes = descriptor.encoded_bytes()?;
        let (role_key, layer, category) = role_details(tensor.role());

        tensors.total.checked_add(1, logical_elements, encoded_bytes)?;
        tensors
            .category_mut(category)
            .checked_add(1, logical_elements, encoded_bytes)?;
        types
            .entry(descriptor.ty().name().to_owned())
            .or_default()
            .checked_add(1, logical_elements, encoded_bytes)?;
        roles
            .entry(role_key.to_owned())
            .or_default()
            .checked_add(1, logical_elements, encoded_bytes)?;
        if let Some(layer) = layer {
            layers
                .entry(layer)
                .or_default()
                .checked_add(1, logical_elements, encoded_bytes)?;
        }
        if matches!(category, TensorCategory::Routed) {
            add_expert_projection(
                &mut routed_projections,
                role_key,
                tensor,
                logical_elements,
                model.config().expert_count,
            )?;
        }
    }

    let authoritative_metadata_count =
        checked_len(primary.metadata.len(), "authoritative GGUF metadata count")?;
    let unsupported_execution_types = types.keys().cloned().collect();
    let expert_storage = ExpertStorageReport {
        expert_count: model.config().expert_count,
        routed_banks: tensors.routed_experts.clone(),
        shared_experts: tensors.shared_experts.clone(),
        routed_projections,
    };
    let gguf = GgufReport {
        version: primary.version,
        endianness: endianness_name(primary.endianness).to_owned(),
        authoritative_metadata_count,
        tensor_count: tensors.total.count,
        alignment: primary.alignment,
        encoded_tensor_bytes: tensors.total.encoded_bytes,
    };

    Ok(InspectionReport {
        files,
        gguf,
        general,
        tokenizer,
        hy3,
        tensors,
        types,
        roles,
        layers,
        expert_storage,
        unsupported_execution_types,
        warnings: vec!["Tensor payload bytes were not read or verified.".to_owned()],
    })
}

fn build_files(set: &GgufSet) -> Result<Vec<FileReport>, ReportError> {
    let mut files = Vec::new();
    try_reserve_exact(&mut files, set.files().len(), "report files")?;
    for shard in set.files() {
        let parsed = shard.parsed();
        files.push(FileReport {
            path: shard
                .path()
                .to_str()
                .ok_or_else(|| ReportError::NonUnicodePath(shard.path().to_owned()))?
                .to_owned(),
            ordinal: shard.ordinal(),
            count: shard.count(),
            version: parsed.version,
            endianness: endianness_name(parsed.endianness).to_owned(),
            metadata_count: checked_len(parsed.metadata.len(), "per-file GGUF metadata count")?,
            alignment: parsed.alignment,
            logical_size: parsed.file_len,
            data_offset: parsed.data_offset,
        });
    }
    Ok(files)
}

const fn endianness_name(endianness: Endianness) -> &'static str {
    match endianness {
        Endianness::Little => "little",
        Endianness::Big => "big",
    }
}

fn build_general(metadata: &GgufFile) -> Result<GeneralReport, ReportError> {
    Ok(GeneralReport {
        architecture: metadata.get_string("general.architecture")?.to_owned(),
        name: optional_string(metadata, "general.name")?,
        license: optional_string(metadata, "general.license")?,
        size_label: optional_string(metadata, "general.size_label")?,
        quantization_version: optional_u32(metadata, "general.quantization_version")?,
        file_type: optional_u32(metadata, "general.file_type")?,
    })
}

fn build_tokenizer(metadata: &GgufFile) -> Result<TokenizerReport, ReportError> {
    Ok(TokenizerReport {
        model: optional_string(metadata, "tokenizer.ggml.model")?,
        pretokenizer: optional_string(metadata, "tokenizer.ggml.pre")?,
        token_count: array_count(metadata, "tokenizer.ggml.tokens", GgufValueType::String)?
            .expect("selected Hy3 validation requires tokenizer.ggml.tokens"),
        merge_count: array_count(metadata, "tokenizer.ggml.merges", GgufValueType::String)?,
        token_type_count: array_count(metadata, "tokenizer.ggml.token_type", GgufValueType::I32)?,
        bos_token_id: optional_u32(metadata, "tokenizer.ggml.bos_token_id")?,
        eos_token_id: optional_u32(metadata, "tokenizer.ggml.eos_token_id")?,
        padding_token_id: optional_u32(metadata, "tokenizer.ggml.padding_token_id")?,
        separator_token_id: optional_u32(metadata, "tokenizer.ggml.seperator_token_id")?,
        has_chat_template: optional_string(metadata, "tokenizer.chat_template")?.is_some(),
    })
}

fn build_hy3(model: &ValidatedHy3Model) -> Hy3Report {
    let config = model.config();
    Hy3Report {
        block_count: config.block_count,
        context_length: config.context_length,
        embedding_length: config.embedding_length,
        dense_ffn_length: config.dense_ffn_length,
        expert_ffn_length: config.expert_ffn_length,
        shared_expert_ffn_length: config.shared_expert_ffn_length,
        attention_head_count: config.attention_head_count,
        attention_kv_head_count: config.attention_kv_head_count,
        key_length: config.key_length,
        value_length: config.value_length,
        rms_epsilon: config.rms_epsilon,
        expert_count: config.expert_count,
        expert_used_count: config.expert_used_count,
        expert_weights_norm: config.expert_weights_norm,
        expert_gating_func: config.expert_gating_func,
        expert_weights_scale: config.expert_weights_scale,
        rope_base: config.rope_base,
        rope_scaling_type: config.rope_scaling_type.clone(),
        yarn_factor: config.yarn_factor,
        yarn_original_context: config.yarn_original_context,
        vocabulary_size: config.vocabulary_size,
        has_mtp: model.has_mtp(),
    }
}

fn optional_string(metadata: &GgufFile, key: &'static str) -> Result<Option<String>, MetadataError> {
    if has_metadata(metadata, key) {
        metadata.get_string(key).map(|value| Some(value.to_owned()))
    } else {
        Ok(None)
    }
}

fn optional_u32(metadata: &GgufFile, key: &'static str) -> Result<Option<u32>, MetadataError> {
    if has_metadata(metadata, key) {
        metadata.get_u32(key).map(Some)
    } else {
        Ok(None)
    }
}

fn array_count(
    metadata: &GgufFile,
    key: &'static str,
    expected: GgufValueType,
) -> Result<Option<u64>, ReportError> {
    if !has_metadata(metadata, key) {
        return Ok(None);
    }
    let array = metadata.get_array(key)?;
    if array.element_type != expected {
        return Err(ReportError::MetadataArrayElementType {
            key,
            expected,
            actual: array.element_type,
        });
    }
    Ok(Some(checked_len(
        array.values.len(),
        "metadata array element count",
    )?))
}

fn has_metadata(metadata: &GgufFile, key: &str) -> bool {
    metadata.metadata.iter().any(|(candidate, _)| candidate == key)
}

fn add_expert_projection(
    projections: &mut BTreeMap<String, ExpertProjectionReport>,
    role_key: &'static str,
    tensor: &bridge_model_hy3::Hy3Tensor,
    logical_elements: u64,
    expert_count: u32,
) -> Result<(), ReportError> {
    let descriptor = tensor.location().descriptor();
    let expert_count_u64 = u64::from(expert_count);
    if logical_elements % expert_count_u64 != 0 {
        return Err(ReportError::ExpertElementCountNotDivisible {
            name: descriptor.name().to_owned(),
            expert_count,
        });
    }
    let slab_logical_elements = logical_elements / expert_count_u64;
    let slab = tensor.expert_slab(expert_count, 0)?;
    let slab_bytes = slab
        .relative_range
        .end
        .checked_sub(slab.relative_range.start)
        .ok_or(ReportError::ArithmeticOverflow("expert slab byte count"))?;
    let physical_type = descriptor.ty().name();
    let key = format!("{role_key}/{physical_type}");

    match projections.get_mut(&key) {
        Some(existing)
            if existing.expert_count == expert_count
                && existing.slab_logical_elements == slab_logical_elements
                && existing.slab_bytes == slab_bytes =>
        {
            existing.tensor_count = existing
                .tensor_count
                .checked_add(1)
                .ok_or(ReportError::ArithmeticOverflow("expert projection tensor count"))?;
        }
        Some(_) => return Err(ReportError::InconsistentExpertProjection { key }),
        None => {
            projections.insert(
                key,
                ExpertProjectionReport {
                    projection: role_key.to_owned(),
                    physical_type: physical_type.to_owned(),
                    tensor_count: 1,
                    expert_count,
                    slab_logical_elements,
                    slab_bytes,
                },
            );
        }
    }
    Ok(())
}

fn role_details(role: Hy3TensorRole) -> (&'static str, Option<u32>, TensorCategory) {
    use Hy3TensorRole::{
        AttentionK, AttentionKNorm, AttentionNorm, AttentionOutput, AttentionQ, AttentionQNorm, AttentionV,
        DenseDown, DenseGate, DenseUp, FfnNorm, Output, OutputNorm, RoutedDown, RoutedGate, RoutedUp,
        RouterInput, RouterSelectionBias, SharedDown, SharedGate, SharedUp, TokenEmbedding,
    };
    match role {
        TokenEmbedding => ("token_embedding", None, TensorCategory::Embedding),
        OutputNorm => ("output_norm", None, TensorCategory::Norm),
        Output => ("output", None, TensorCategory::Output),
        AttentionNorm { layer } => ("attention_norm", Some(layer), TensorCategory::Norm),
        AttentionQ { layer } => ("attention_q", Some(layer), TensorCategory::Attention),
        AttentionQNorm { layer } => ("attention_q_norm", Some(layer), TensorCategory::Norm),
        AttentionK { layer } => ("attention_k", Some(layer), TensorCategory::Attention),
        AttentionKNorm { layer } => ("attention_k_norm", Some(layer), TensorCategory::Norm),
        AttentionV { layer } => ("attention_v", Some(layer), TensorCategory::Attention),
        AttentionOutput { layer } => ("attention_output", Some(layer), TensorCategory::Attention),
        FfnNorm { layer } => ("ffn_norm", Some(layer), TensorCategory::Norm),
        DenseGate { layer } => ("dense_gate", Some(layer), TensorCategory::Dense),
        DenseUp { layer } => ("dense_up", Some(layer), TensorCategory::Dense),
        DenseDown { layer } => ("dense_down", Some(layer), TensorCategory::Dense),
        RouterInput { layer } => ("router_input", Some(layer), TensorCategory::Router),
        RouterSelectionBias { layer } => ("router_selection_bias", Some(layer), TensorCategory::Router),
        RoutedGate { layer } => ("routed_gate", Some(layer), TensorCategory::Routed),
        RoutedUp { layer } => ("routed_up", Some(layer), TensorCategory::Routed),
        RoutedDown { layer } => ("routed_down", Some(layer), TensorCategory::Routed),
        SharedGate { layer } => ("shared_gate", Some(layer), TensorCategory::Shared),
        SharedUp { layer } => ("shared_up", Some(layer), TensorCategory::Shared),
        SharedDown { layer } => ("shared_down", Some(layer), TensorCategory::Shared),
    }
}

fn checked_len(len: usize, context: &'static str) -> Result<u64, ReportError> {
    u64::try_from(len).map_err(|_| ReportError::ArithmeticOverflow(context))
}

fn try_reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), ReportError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ReportError::AllocationFailed(context))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_categories_cover_the_total_without_overlap() {
        let mut summary = TensorSummary::new();
        for category in [
            TensorCategory::Dense,
            TensorCategory::Routed,
            TensorCategory::Shared,
            TensorCategory::Attention,
            TensorCategory::Router,
            TensorCategory::Embedding,
            TensorCategory::Norm,
            TensorCategory::Output,
        ] {
            summary.category_mut(category).checked_add(1, 2, 3).unwrap();
        }
        let mut combined = Aggregate::zero();
        for category in [
            &summary.dense_layer_0_ffn,
            &summary.routed_experts,
            &summary.shared_experts,
            &summary.attention,
            &summary.routers,
            &summary.embeddings,
            &summary.norms,
            &summary.output,
        ] {
            combined
                .checked_add(category.count, category.logical_elements, category.encoded_bytes)
                .unwrap();
        }
        assert_eq!(
            combined,
            Aggregate {
                count: 8,
                logical_elements: 16,
                encoded_bytes: 24,
            }
        );
    }

    #[test]
    fn reserve_failure_is_reported_as_allocation_not_arithmetic() {
        let mut values = Vec::<u8>::new();

        let error = try_reserve_exact(&mut values, usize::MAX, "test report entries").unwrap_err();

        assert!(matches!(
            error,
            ReportError::AllocationFailed("test report entries")
        ));
    }
}
