use crate::{Hy3Config, Hy3Error};

#[derive(Debug, Clone)]
pub struct Hy3Profile {
    expected: Hy3Config,
}

impl Hy3Profile {
    pub fn selected_iq2_m() -> Self {
        Self {
            expected: Hy3Config {
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
            },
        }
    }

    pub const fn config(&self) -> &Hy3Config {
        &self.expected
    }

    pub fn validate(&self, actual: &Hy3Config) -> Result<(), Hy3Error> {
        compare_u32("hy_v3.block_count", self.expected.block_count, actual.block_count)?;
        compare_u64(
            "hy_v3.context_length",
            self.expected.context_length,
            actual.context_length,
        )?;
        compare_u32(
            "hy_v3.embedding_length",
            self.expected.embedding_length,
            actual.embedding_length,
        )?;
        compare_u32(
            "hy_v3.feed_forward_length",
            self.expected.dense_ffn_length,
            actual.dense_ffn_length,
        )?;
        compare_u32(
            "hy_v3.expert_feed_forward_length",
            self.expected.expert_ffn_length,
            actual.expert_ffn_length,
        )?;
        compare_u32(
            "hy_v3.expert_shared_feed_forward_length",
            self.expected.shared_expert_ffn_length,
            actual.shared_expert_ffn_length,
        )?;
        compare_u32(
            "hy_v3.attention.head_count",
            self.expected.attention_head_count,
            actual.attention_head_count,
        )?;
        compare_u32(
            "hy_v3.attention.head_count_kv",
            self.expected.attention_kv_head_count,
            actual.attention_kv_head_count,
        )?;
        compare_u32(
            "hy_v3.attention.key_length",
            self.expected.key_length,
            actual.key_length,
        )?;
        compare_u32(
            "hy_v3.attention.value_length",
            self.expected.value_length,
            actual.value_length,
        )?;
        compare_float(
            "hy_v3.attention.layer_norm_rms_epsilon",
            self.expected.rms_epsilon,
            actual.rms_epsilon,
        )?;
        compare_u32(
            "hy_v3.expert_count",
            self.expected.expert_count,
            actual.expert_count,
        )?;
        compare_u32(
            "hy_v3.expert_used_count",
            self.expected.expert_used_count,
            actual.expert_used_count,
        )?;
        compare_bool(
            "hy_v3.expert_weights_norm",
            self.expected.expert_weights_norm,
            actual.expert_weights_norm,
        )?;
        compare_u32(
            "hy_v3.expert_gating_func",
            self.expected.expert_gating_func,
            actual.expert_gating_func,
        )?;
        compare_float(
            "hy_v3.expert_weights_scale",
            self.expected.expert_weights_scale,
            actual.expert_weights_scale,
        )?;
        compare_float("hy_v3.rope.freq_base", self.expected.rope_base, actual.rope_base)?;
        compare_string(
            "hy_v3.rope.scaling.type",
            &self.expected.rope_scaling_type,
            &actual.rope_scaling_type,
        )?;
        compare_float(
            "hy_v3.rope.scaling.factor",
            self.expected.yarn_factor,
            actual.yarn_factor,
        )?;
        compare_u64(
            "hy_v3.rope.scaling.original_context_length",
            self.expected.yarn_original_context,
            actual.yarn_original_context,
        )?;
        compare_u32(
            "tokenizer.ggml.tokens",
            self.expected.vocabulary_size,
            actual.vocabulary_size,
        )
    }
}

fn compare_u32(key: &'static str, expected: u32, actual: u32) -> Result<(), Hy3Error> {
    compare_display(key, expected, actual)
}

fn compare_u64(key: &'static str, expected: u64, actual: u64) -> Result<(), Hy3Error> {
    compare_display(key, expected, actual)
}

fn compare_bool(key: &'static str, expected: bool, actual: bool) -> Result<(), Hy3Error> {
    compare_display(key, expected, actual)
}

fn compare_string(key: &'static str, expected: &str, actual: &str) -> Result<(), Hy3Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(Hy3Error::MetadataValue {
            key,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        })
    }
}

fn compare_display<T>(key: &'static str, expected: T, actual: T) -> Result<(), Hy3Error>
where
    T: PartialEq + std::fmt::Display,
{
    if expected == actual {
        Ok(())
    } else {
        Err(Hy3Error::MetadataValue {
            key,
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

fn compare_float(key: &'static str, expected: f32, actual: f32) -> Result<(), Hy3Error> {
    if !actual.is_finite() {
        return Err(Hy3Error::NonFiniteMetadata {
            key,
            expected,
            actual,
        });
    }
    let tolerance = (expected.abs() * 1.0e-6).clamp(1.0e-12, 1.0e-4);
    if (actual - expected).abs() <= tolerance {
        Ok(())
    } else {
        Err(Hy3Error::MetadataValue {
            key,
            expected: format!("{expected} (within {tolerance})"),
            actual: actual.to_string(),
        })
    }
}
