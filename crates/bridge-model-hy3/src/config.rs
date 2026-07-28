use bridge_gguf::{GgufArray, GgufFile, GgufValueType, MetadataError};

use crate::Hy3Error;

const ARCHITECTURE: &str = "general.architecture";
const TOKENS: &str = "tokenizer.ggml.tokens";

pub(crate) const SELECTED_METADATA_KEYS: &[&str] = &[
    ARCHITECTURE,
    "hy_v3.block_count",
    "hy_v3.context_length",
    "hy_v3.embedding_length",
    "hy_v3.feed_forward_length",
    "hy_v3.expert_feed_forward_length",
    "hy_v3.expert_shared_feed_forward_length",
    "hy_v3.attention.head_count",
    "hy_v3.attention.head_count_kv",
    "hy_v3.attention.key_length",
    "hy_v3.attention.value_length",
    "hy_v3.attention.layer_norm_rms_epsilon",
    "hy_v3.expert_count",
    "hy_v3.expert_used_count",
    "hy_v3.expert_weights_norm",
    "hy_v3.expert_gating_func",
    "hy_v3.expert_weights_scale",
    "hy_v3.rope.freq_base",
    "hy_v3.rope.scaling.type",
    "hy_v3.rope.scaling.factor",
    "hy_v3.rope.scaling.original_context_length",
    TOKENS,
];

#[derive(Debug, Clone, PartialEq)]
pub struct Hy3Config {
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
}

pub fn resolve_config(metadata: &GgufFile) -> Result<Hy3Config, Hy3Error> {
    let architecture = get_string(metadata, ARCHITECTURE)?;
    if architecture != "hy_v3" {
        return Err(Hy3Error::MetadataValue {
            key: ARCHITECTURE,
            expected: quoted("hy_v3"),
            actual: quoted(architecture),
        });
    }

    let rms_epsilon = finite_f32(metadata, "hy_v3.attention.layer_norm_rms_epsilon", 1.0e-5)?;
    let expert_weights_scale = finite_f32(metadata, "hy_v3.expert_weights_scale", 2.826)?;
    let rope_base = finite_f32(metadata, "hy_v3.rope.freq_base", 11_158_840.0)?;
    let yarn_factor = finite_f32(metadata, "hy_v3.rope.scaling.factor", 4.0)?;

    let tokens = get_array(metadata, TOKENS)?;
    if tokens.element_type != GgufValueType::String {
        return Err(Hy3Error::MetadataArrayElementType {
            key: TOKENS,
            expected: GgufValueType::String,
            actual: tokens.element_type,
        });
    }
    let vocabulary_size = u32::try_from(tokens.values.len()).map_err(|_| Hy3Error::MetadataValue {
        key: TOKENS,
        expected: "an ARRAY[STRING] count that fits u32".into(),
        actual: format!("ARRAY[STRING] count {}", tokens.values.len()),
    })?;

    Ok(Hy3Config {
        block_count: get_u32(metadata, "hy_v3.block_count")?,
        context_length: u64::from(get_u32(metadata, "hy_v3.context_length")?),
        embedding_length: get_u32(metadata, "hy_v3.embedding_length")?,
        dense_ffn_length: get_u32(metadata, "hy_v3.feed_forward_length")?,
        expert_ffn_length: get_u32(metadata, "hy_v3.expert_feed_forward_length")?,
        shared_expert_ffn_length: get_u32(metadata, "hy_v3.expert_shared_feed_forward_length")?,
        attention_head_count: get_u32(metadata, "hy_v3.attention.head_count")?,
        attention_kv_head_count: get_u32(metadata, "hy_v3.attention.head_count_kv")?,
        key_length: get_u32(metadata, "hy_v3.attention.key_length")?,
        value_length: get_u32(metadata, "hy_v3.attention.value_length")?,
        rms_epsilon,
        expert_count: get_u32(metadata, "hy_v3.expert_count")?,
        expert_used_count: get_u32(metadata, "hy_v3.expert_used_count")?,
        expert_weights_norm: get_bool(metadata, "hy_v3.expert_weights_norm")?,
        expert_gating_func: get_u32(metadata, "hy_v3.expert_gating_func")?,
        expert_weights_scale,
        rope_base,
        rope_scaling_type: get_string(metadata, "hy_v3.rope.scaling.type")?.to_owned(),
        yarn_factor,
        yarn_original_context: u64::from(get_u32(metadata, "hy_v3.rope.scaling.original_context_length")?),
        vocabulary_size,
    })
}

fn finite_f32(metadata: &GgufFile, key: &'static str, selected_expected: f32) -> Result<f32, Hy3Error> {
    let actual = get_f32(metadata, key)?;
    if actual.is_finite() {
        Ok(actual)
    } else {
        Err(Hy3Error::NonFiniteMetadata {
            key,
            expected: selected_expected,
            actual,
        })
    }
}

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

fn get_u32(metadata: &GgufFile, key: &'static str) -> Result<u32, Hy3Error> {
    map_metadata(metadata.get_u32(key), GgufValueType::U32)
}

fn get_f32(metadata: &GgufFile, key: &'static str) -> Result<f32, Hy3Error> {
    map_metadata(metadata.get_f32(key), GgufValueType::F32)
}

fn get_bool(metadata: &GgufFile, key: &'static str) -> Result<bool, Hy3Error> {
    map_metadata(metadata.get_bool(key), GgufValueType::Bool)
}

fn get_string<'a>(metadata: &'a GgufFile, key: &'static str) -> Result<&'a str, Hy3Error> {
    map_metadata(metadata.get_string(key), GgufValueType::String)
}

fn get_array<'a>(metadata: &'a GgufFile, key: &'static str) -> Result<&'a GgufArray, Hy3Error> {
    map_metadata(metadata.get_array(key), GgufValueType::Array)
}

fn map_metadata<T>(result: Result<T, MetadataError>, expected: GgufValueType) -> Result<T, Hy3Error> {
    result.map_err(|error| match error {
        MetadataError::Missing { key } => Hy3Error::MissingMetadataType { key, expected },
        MetadataError::WrongType { key, actual, .. } => Hy3Error::MetadataStoredType {
            key,
            expected,
            actual,
        },
    })
}

pub(crate) fn validate_exact_replica(authoritative: &Hy3Config, replica: &Hy3Config) -> Result<(), Hy3Error> {
    macro_rules! compare {
        ($field:ident, $key:literal) => {
            if authoritative.$field != replica.$field {
                return Err(Hy3Error::MetadataValue {
                    key: $key,
                    expected: authoritative.$field.to_string(),
                    actual: replica.$field.to_string(),
                });
            }
        };
    }

    compare!(block_count, "hy_v3.block_count");
    compare!(context_length, "hy_v3.context_length");
    compare!(embedding_length, "hy_v3.embedding_length");
    compare!(dense_ffn_length, "hy_v3.feed_forward_length");
    compare!(expert_ffn_length, "hy_v3.expert_feed_forward_length");
    compare!(
        shared_expert_ffn_length,
        "hy_v3.expert_shared_feed_forward_length"
    );
    compare!(attention_head_count, "hy_v3.attention.head_count");
    compare!(attention_kv_head_count, "hy_v3.attention.head_count_kv");
    compare!(key_length, "hy_v3.attention.key_length");
    compare!(value_length, "hy_v3.attention.value_length");
    compare!(rms_epsilon, "hy_v3.attention.layer_norm_rms_epsilon");
    compare!(expert_count, "hy_v3.expert_count");
    compare!(expert_used_count, "hy_v3.expert_used_count");
    compare!(expert_weights_norm, "hy_v3.expert_weights_norm");
    compare!(expert_gating_func, "hy_v3.expert_gating_func");
    compare!(expert_weights_scale, "hy_v3.expert_weights_scale");
    compare!(rope_base, "hy_v3.rope.freq_base");
    if authoritative.rope_scaling_type != replica.rope_scaling_type {
        return Err(Hy3Error::MetadataValue {
            key: "hy_v3.rope.scaling.type",
            expected: format!("{:?}", authoritative.rope_scaling_type),
            actual: format!("{:?}", replica.rope_scaling_type),
        });
    }
    compare!(yarn_factor, "hy_v3.rope.scaling.factor");
    compare!(
        yarn_original_context,
        "hy_v3.rope.scaling.original_context_length"
    );
    compare!(vocabulary_size, "tokenizer.ggml.tokens");
    Ok(())
}
