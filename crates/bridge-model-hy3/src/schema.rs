use std::collections::HashMap;

use bridge_core::ggml_type::GgmlType;
use bridge_core::tensor::TensorDesc;
use bridge_gguf_split::TensorLocation;

use crate::{ExpertSlab, Hy3Config, Hy3Error, Hy3Profile, Hy3TensorRole};

const MAX_BLOCK_COUNT: u32 = 1_024;
const MAX_MODEL_DIMENSION: u32 = 16_777_216;
const MAX_HEAD_COUNT: u32 = 4_096;
const MAX_HEAD_DIMENSION: u32 = 65_536;
const MAX_EXPERT_COUNT: u32 = 65_536;
const MAX_SCHEMA_TENSOR_COUNT: usize = 16_382;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    name: String,
    role: Hy3TensorRole,
    shape: Vec<u64>,
    ty: GgmlType,
}

impl TensorSpec {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn role(&self) -> Hy3TensorRole {
        self.role
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub const fn ty(&self) -> GgmlType {
        self.ty
    }
}

pub(crate) fn generate_schema_for_validated_selected_iq2_m(
    config: &Hy3Config,
) -> Result<Vec<TensorSpec>, Hy3Error> {
    validate_schema_config(config)?;
    let hidden = u64::from(config.embedding_length);
    let vocabulary = u64::from(config.vocabulary_size);
    let q_width = checked_mul(
        "blk.*.attn_q.weight",
        config.attention_head_count,
        config.key_length,
    )?;
    let k_width = checked_mul(
        "blk.*.attn_k.weight",
        config.attention_kv_head_count,
        config.key_length,
    )?;
    let v_width = checked_mul(
        "blk.*.attn_v.weight",
        config.attention_kv_head_count,
        config.value_length,
    )?;
    let attention_output_width = checked_mul(
        "blk.*.attn_output.weight",
        config.attention_head_count,
        config.value_length,
    )?;

    let block_count = usize::try_from(config.block_count).map_err(|_| Hy3Error::Arithmetic {
        name: "generated schema".into(),
        operation: "block count conversion",
    })?;
    let capacity = block_count
        .checked_sub(1)
        .and_then(|moe_blocks| moe_blocks.checked_mul(16))
        .and_then(|moe_entries| moe_entries.checked_add(14))
        .ok_or_else(|| Hy3Error::Arithmetic {
            name: "generated schema".into(),
            operation: "schema entry capacity",
        })?;
    let mut schema = Vec::new();
    schema
        .try_reserve_exact(capacity)
        .map_err(|_| Hy3Error::AllocationFailed {
            context: "generated tensor schema",
            requested: capacity,
        })?;
    push(
        &mut schema,
        "token_embd.weight".into(),
        Hy3TensorRole::TokenEmbedding,
        &[hidden, vocabulary],
        GgmlType::IQ3_S,
    );
    push(
        &mut schema,
        "output_norm.weight".into(),
        Hy3TensorRole::OutputNorm,
        &[hidden],
        GgmlType::F32,
    );
    push(
        &mut schema,
        "output.weight".into(),
        Hy3TensorRole::Output,
        &[hidden, vocabulary],
        GgmlType::Q5_K,
    );

    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        push(
            &mut schema,
            format!("{prefix}.attn_k.weight"),
            Hy3TensorRole::AttentionK { layer },
            &[hidden, k_width],
            GgmlType::IQ2_S,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_k_norm.weight"),
            Hy3TensorRole::AttentionKNorm { layer },
            &[u64::from(config.key_length)],
            GgmlType::F32,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_norm.weight"),
            Hy3TensorRole::AttentionNorm { layer },
            &[hidden],
            GgmlType::F32,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_output.weight"),
            Hy3TensorRole::AttentionOutput { layer },
            &[attention_output_width, hidden],
            GgmlType::IQ3_S,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_q.weight"),
            Hy3TensorRole::AttentionQ { layer },
            &[hidden, q_width],
            GgmlType::IQ2_S,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_q_norm.weight"),
            Hy3TensorRole::AttentionQNorm { layer },
            &[u64::from(config.key_length)],
            GgmlType::F32,
        );
        push(
            &mut schema,
            format!("{prefix}.attn_v.weight"),
            Hy3TensorRole::AttentionV { layer },
            &[hidden, v_width],
            GgmlType::Q4_K,
        );
        push(
            &mut schema,
            format!("{prefix}.ffn_norm.weight"),
            Hy3TensorRole::FfnNorm { layer },
            &[hidden],
            GgmlType::F32,
        );

        if layer == 0 {
            let dense = u64::from(config.dense_ffn_length);
            push(
                &mut schema,
                format!("{prefix}.ffn_down.weight"),
                Hy3TensorRole::DenseDown { layer },
                &[dense, hidden],
                GgmlType::IQ3_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_gate.weight"),
                Hy3TensorRole::DenseGate { layer },
                &[hidden, dense],
                GgmlType::IQ2_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_up.weight"),
                Hy3TensorRole::DenseUp { layer },
                &[hidden, dense],
                GgmlType::IQ2_S,
            );
        } else {
            let experts = u64::from(config.expert_count);
            let expert_ffn = u64::from(config.expert_ffn_length);
            let shared_ffn = u64::from(config.shared_expert_ffn_length);
            push(
                &mut schema,
                format!("{prefix}.exp_probs_b"),
                Hy3TensorRole::RouterSelectionBias { layer },
                &[experts],
                GgmlType::F32,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_gate_inp.weight"),
                Hy3TensorRole::RouterInput { layer },
                &[hidden, experts],
                GgmlType::F32,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_gate_exps.weight"),
                Hy3TensorRole::RoutedGate { layer },
                &[hidden, expert_ffn, experts],
                GgmlType::IQ2_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_up_exps.weight"),
                Hy3TensorRole::RoutedUp { layer },
                &[hidden, expert_ffn, experts],
                GgmlType::IQ2_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_down_exps.weight"),
                Hy3TensorRole::RoutedDown { layer },
                &[expert_ffn, hidden, experts],
                if layer <= 5 {
                    GgmlType::IQ3_S
                } else {
                    GgmlType::IQ2_S
                },
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_gate_shexp.weight"),
                Hy3TensorRole::SharedGate { layer },
                &[hidden, shared_ffn],
                GgmlType::IQ2_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_up_shexp.weight"),
                Hy3TensorRole::SharedUp { layer },
                &[hidden, shared_ffn],
                GgmlType::IQ2_S,
            );
            push(
                &mut schema,
                format!("{prefix}.ffn_down_shexp.weight"),
                Hy3TensorRole::SharedDown { layer },
                &[shared_ffn, hidden],
                if layer <= 4 {
                    GgmlType::IQ3_S
                } else {
                    GgmlType::IQ2_S
                },
            );
        }
    }

    Ok(schema)
}

/// Generates the exact tensor schema for the selected non-MTP IQ2_M checkpoint.
///
/// The supplied configuration must first match [`Hy3Profile::selected_iq2_m`];
/// self-consistent configurations for other Hy3 checkpoints are rejected.
pub fn generate_selected_iq2_m_schema(config: &Hy3Config) -> Result<Vec<TensorSpec>, Hy3Error> {
    Hy3Profile::selected_iq2_m().validate(config)?;
    generate_schema_for_validated_selected_iq2_m(config)
}

/// Validates descriptors against the exact selected non-MTP IQ2_M schema.
///
/// This boundary authorizes the selected profile before applying its
/// checkpoint-specific physical-type and layer-transition rules.
pub fn validate_selected_iq2_m_tensor_descriptors(
    config: &Hy3Config,
    tensors: &[TensorDesc],
) -> Result<(), Hy3Error> {
    Hy3Profile::selected_iq2_m().validate(config)?;
    validate_descriptor_iter(config, tensors.len(), tensors.iter())
}

pub(crate) fn validate_tensor_locations(
    config: &Hy3Config,
    tensors: &[TensorLocation],
) -> Result<(), Hy3Error> {
    validate_descriptor_iter(
        config,
        tensors.len(),
        tensors.iter().map(TensorLocation::descriptor),
    )
}

fn validate_descriptor_iter<'a>(
    config: &Hy3Config,
    tensor_count: usize,
    tensors: impl IntoIterator<Item = &'a TensorDesc>,
) -> Result<(), Hy3Error> {
    let expected_schema = generate_schema_for_validated_selected_iq2_m(config)?;
    if tensor_count > MAX_SCHEMA_TENSOR_COUNT {
        return Err(Hy3Error::TensorDirectoryCount {
            expected: expected_schema.len(),
            actual: tensor_count,
        });
    }
    let mut actual = HashMap::new();
    actual
        .try_reserve(tensor_count)
        .map_err(|_| Hy3Error::AllocationFailed {
            context: "tensor validation index",
            requested: tensor_count,
        })?;
    for tensor in tensors {
        if actual.insert(tensor.name(), tensor).is_some() {
            return Err(Hy3Error::DuplicateTensor {
                name: tensor.name().to_owned(),
            });
        }
    }

    for expected in expected_schema {
        let Some(actual_tensor) = actual.remove(expected.name()) else {
            return Err(Hy3Error::MissingTensor {
                name: expected.name,
                expected_shape: expected.shape,
                expected_type: expected.ty.name(),
            });
        };
        if expected.role.is_routed_expert() {
            validate_routed_shape(actual_tensor, config.expert_count)?;
        }
        if actual_tensor.shape() != expected.shape {
            return Err(Hy3Error::TensorShape {
                name: expected.name,
                expected: expected.shape,
                actual: actual_tensor.shape().to_vec(),
            });
        }
        if actual_tensor.ty() != expected.ty {
            return Err(Hy3Error::TensorType {
                name: expected.name,
                expected: expected.ty.name(),
                actual: actual_tensor.ty().name(),
            });
        }
    }

    if let Some((name, _)) = actual.into_iter().next() {
        return Err(Hy3Error::UnexpectedTensor {
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn validate_schema_config(config: &Hy3Config) -> Result<(), Hy3Error> {
    check_range(
        "block_count",
        u64::from(config.block_count),
        1,
        u64::from(MAX_BLOCK_COUNT),
    )?;
    check_range(
        "embedding_length",
        u64::from(config.embedding_length),
        1,
        u64::from(MAX_MODEL_DIMENSION),
    )?;
    check_range(
        "dense_ffn_length",
        u64::from(config.dense_ffn_length),
        1,
        u64::from(MAX_MODEL_DIMENSION),
    )?;
    check_range(
        "expert_ffn_length",
        u64::from(config.expert_ffn_length),
        1,
        u64::from(MAX_MODEL_DIMENSION),
    )?;
    check_range(
        "shared_expert_ffn_length",
        u64::from(config.shared_expert_ffn_length),
        1,
        u64::from(MAX_MODEL_DIMENSION),
    )?;
    check_range(
        "attention_head_count",
        u64::from(config.attention_head_count),
        1,
        u64::from(MAX_HEAD_COUNT),
    )?;
    check_range(
        "attention_kv_head_count",
        u64::from(config.attention_kv_head_count),
        1,
        u64::from(MAX_HEAD_COUNT),
    )?;
    check_range(
        "key_length",
        u64::from(config.key_length),
        1,
        u64::from(MAX_HEAD_DIMENSION),
    )?;
    check_range(
        "value_length",
        u64::from(config.value_length),
        1,
        u64::from(MAX_HEAD_DIMENSION),
    )?;
    check_range(
        "expert_count",
        u64::from(config.expert_count),
        1,
        u64::from(MAX_EXPERT_COUNT),
    )?;
    check_range(
        "expert_used_count",
        u64::from(config.expert_used_count),
        1,
        u64::from(config.expert_count),
    )?;
    check_range(
        "vocabulary_size",
        u64::from(config.vocabulary_size),
        1,
        u64::from(MAX_MODEL_DIMENSION),
    )?;
    if config.attention_kv_head_count > config.attention_head_count {
        return Err(Hy3Error::ConfigRelation {
            field: "attention_kv_head_count",
            expected: format!("at most attention_head_count {}", config.attention_head_count),
            actual: config.attention_kv_head_count.to_string(),
        });
    }
    Ok(())
}

fn check_range(field: &'static str, actual: u64, minimum: u64, maximum: u64) -> Result<(), Hy3Error> {
    if (minimum..=maximum).contains(&actual) {
        Ok(())
    } else {
        Err(Hy3Error::ConfigOutOfRange {
            field,
            minimum,
            maximum,
            actual,
        })
    }
}

pub fn checked_expert_slab(
    name: &str,
    shape: &[u64],
    payload_bytes: u64,
    expert_count: u32,
    expert: u32,
) -> Result<ExpertSlab, Hy3Error> {
    if shape.len() != 3 {
        return Err(Hy3Error::TensorRank {
            name: name.to_owned(),
            expected: 3,
            actual: shape.len(),
        });
    }
    if shape[2] != u64::from(expert_count) {
        return Err(Hy3Error::ExpertDimension {
            name: name.to_owned(),
            expected: expert_count,
            actual: shape[2],
        });
    }
    if expert >= expert_count {
        return Err(Hy3Error::ExpertIndexOutOfRange {
            name: name.to_owned(),
            expert,
            expert_count,
        });
    }
    if expert_count == 0 || payload_bytes % u64::from(expert_count) != 0 {
        return Err(Hy3Error::ExpertPayloadNotDivisible {
            name: name.to_owned(),
            payload_bytes,
            expert_count,
        });
    }

    let slab_bytes = payload_bytes / u64::from(expert_count);
    let start = u64::from(expert)
        .checked_mul(slab_bytes)
        .ok_or_else(|| Hy3Error::Arithmetic {
            name: name.to_owned(),
            operation: "expert index times slab bytes",
        })?;
    let end = start
        .checked_add(slab_bytes)
        .ok_or_else(|| Hy3Error::Arithmetic {
            name: name.to_owned(),
            operation: "expert slab end",
        })?;
    if end > payload_bytes {
        return Err(Hy3Error::Arithmetic {
            name: name.to_owned(),
            operation: "expert slab within payload",
        });
    }

    Ok(ExpertSlab {
        expert,
        relative_range: start..end,
    })
}

fn validate_routed_shape(tensor: &TensorDesc, expert_count: u32) -> Result<(), Hy3Error> {
    if tensor.shape().len() != 3 {
        return Err(Hy3Error::TensorRank {
            name: tensor.name().to_owned(),
            expected: 3,
            actual: tensor.shape().len(),
        });
    }
    if tensor.shape()[2] != u64::from(expert_count) {
        return Err(Hy3Error::ExpertDimension {
            name: tensor.name().to_owned(),
            expected: expert_count,
            actual: tensor.shape()[2],
        });
    }
    Ok(())
}

fn checked_mul(name: &str, left: u32, right: u32) -> Result<u64, Hy3Error> {
    u64::from(left)
        .checked_mul(u64::from(right))
        .ok_or_else(|| Hy3Error::Arithmetic {
            name: name.to_owned(),
            operation: "derived tensor dimension",
        })
}

fn push(schema: &mut Vec<TensorSpec>, name: String, role: Hy3TensorRole, shape: &[u64], ty: GgmlType) {
    schema.push(TensorSpec {
        name,
        role,
        shape: shape.to_vec(),
        ty,
    });
}
