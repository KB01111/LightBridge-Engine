use bridge_model_hy3::Hy3Config;

use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::KernelError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hy3RopeParams {
    pub head_dimension: usize,
    pub context_length: u64,
    pub original_context_length: u64,
    pub frequency_base: f32,
    pub frequency_scale: f32,
    pub extension_factor: f32,
    pub attention_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Hy3RopeParams {
    pub fn from_config(config: &Hy3Config) -> Result<Self> {
        if config.key_length != config.value_length {
            return Err(KernelError::DimensionMismatch {
                field: "Hy3 key/value head dimension",
                expected: config.key_length as usize,
                actual: config.value_length as usize,
            });
        }
        let head_dimension =
            usize::try_from(config.key_length).map_err(|_| KernelError::ArithmeticOverflow {
                operation: "Hy3 RoPE head dimension conversion",
            })?;
        if head_dimension == 0 || head_dimension % 2 != 0 {
            return Err(KernelError::InvalidParameter {
                field: "Hy3 RoPE head dimension",
                reason: "must be positive and even",
            });
        }
        validate_finite_value("Hy3 RoPE frequency base", 0, config.rope_base)?;
        validate_finite_value("Hy3 YaRN factor", 0, config.yarn_factor)?;
        if config.rope_base <= 0.0 || config.yarn_factor <= 0.0 {
            return Err(KernelError::InvalidParameter {
                field: "Hy3 RoPE scale",
                reason: "frequency base and YaRN factor must be positive",
            });
        }
        if config.yarn_original_context == 0 || config.context_length == 0 {
            return Err(KernelError::InvalidParameter {
                field: "Hy3 RoPE context",
                reason: "context lengths must be positive",
            });
        }
        let expected_context = config
            .yarn_original_context
            .checked_mul(config.yarn_factor as u64)
            .ok_or(KernelError::ArithmeticOverflow {
                operation: "Hy3 scaled context length",
            })?;
        if expected_context != config.context_length {
            return Err(KernelError::DimensionMismatch {
                field: "Hy3 scaled context length",
                expected: usize::try_from(expected_context).unwrap_or(usize::MAX),
                actual: usize::try_from(config.context_length).unwrap_or(usize::MAX),
            });
        }

        let params = Self {
            head_dimension,
            context_length: config.context_length,
            original_context_length: config.yarn_original_context,
            frequency_base: config.rope_base,
            frequency_scale: 1.0_f32 / config.yarn_factor,
            extension_factor: 1.0,
            attention_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn validate(self) -> Result<()> {
        if self.head_dimension == 0 || self.head_dimension % 2 != 0 {
            return Err(KernelError::InvalidParameter {
                field: "RoPE head dimension",
                reason: "must be positive and even",
            });
        }
        if self.context_length == 0 || self.original_context_length == 0 {
            return Err(KernelError::InvalidParameter {
                field: "RoPE context",
                reason: "must be positive",
            });
        }
        for (field, value) in [
            ("RoPE frequency base", self.frequency_base),
            ("RoPE frequency scale", self.frequency_scale),
            ("RoPE extension factor", self.extension_factor),
            ("RoPE attention factor", self.attention_factor),
            ("RoPE beta fast", self.beta_fast),
            ("RoPE beta slow", self.beta_slow),
        ] {
            validate_finite_value(field, 0, value)?;
        }
        if self.frequency_base <= 0.0
            || self.frequency_scale <= 0.0
            || self.extension_factor < 0.0
            || self.attention_factor <= 0.0
            || self.beta_fast <= 0.0
            || self.beta_slow <= 0.0
        {
            return Err(KernelError::InvalidParameter {
                field: "RoPE parameters",
                reason: "base, scale, attention, and betas must be positive; extension must be non-negative",
            });
        }
        Ok(())
    }
}

/// Applies pinned llama.cpp NeoX/non-interleaved YaRN RoPE independently to
/// each head in place.
pub fn apply_neox_yarn_rope_in_place(
    values: &mut [f32],
    head_count: usize,
    position: u64,
    params: Hy3RopeParams,
) -> Result<()> {
    params.validate()?;
    if position >= params.context_length {
        return Err(KernelError::PositionOutOfRange {
            position,
            context_length: params.context_length,
        });
    }
    let expected = head_count
        .checked_mul(params.head_dimension)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "RoPE head value count",
        })?;
    if values.len() != expected {
        return Err(KernelError::DimensionMismatch {
            field: "RoPE values",
            expected,
            actual: values.len(),
        });
    }
    validate_finite_slice("RoPE input", values)?;

    let correction = correction_dimensions(params);
    let theta_scale = params
        .frequency_base
        .powf(-2.0_f32 / params.head_dimension as f32);
    let magnitude_scale = if params.extension_factor == 0.0 {
        params.attention_factor
    } else {
        params.attention_factor * (1.0_f32 + 0.1_f32 * (1.0_f32 / params.frequency_scale).ln())
    };
    let half = params.head_dimension / 2;

    for head in values.chunks(params.head_dimension) {
        let mut theta_extrapolated = position as f32;
        for pair in 0..half {
            let i0 = pair * 2;
            let theta_interpolated = params.frequency_scale * theta_extrapolated;
            let ramp = yarn_ramp(correction[0], correction[1], i0);
            let mix = ramp * params.extension_factor;
            let theta = theta_interpolated * (1.0_f32 - mix) + theta_extrapolated * mix;
            let cosine = theta.cos() * magnitude_scale;
            let sine = theta.sin() * magnitude_scale;
            let first = head[pair];
            let second = head[pair + half];
            validate_finite_value("RoPE output", pair, first * cosine - second * sine)?;
            validate_finite_value("RoPE output", pair + half, first * sine + second * cosine)?;
            theta_extrapolated *= theta_scale;
        }
    }
    for head in values.chunks_mut(params.head_dimension) {
        let mut theta_extrapolated = position as f32;
        for pair in 0..half {
            let i0 = pair * 2;
            let theta_interpolated = params.frequency_scale * theta_extrapolated;
            let ramp = yarn_ramp(correction[0], correction[1], i0);
            let mix = ramp * params.extension_factor;
            let theta = theta_interpolated * (1.0_f32 - mix) + theta_extrapolated * mix;
            let cosine = theta.cos() * magnitude_scale;
            let sine = theta.sin() * magnitude_scale;
            let first = head[pair];
            let second = head[pair + half];
            head[pair] = first * cosine - second * sine;
            head[pair + half] = first * sine + second * cosine;
            theta_extrapolated *= theta_scale;
        }
    }
    Ok(())
}

fn correction_dimensions(params: Hy3RopeParams) -> [f32; 2] {
    let correction = |rotations: f32| {
        params.head_dimension as f32
            * (params.original_context_length as f32 / (rotations * 2.0_f32 * std::f32::consts::PI)).ln()
            / (2.0_f32 * params.frequency_base.ln())
    };
    let start = correction(params.beta_fast).floor().max(0.0);
    let end = correction(params.beta_slow)
        .ceil()
        .min(params.head_dimension as f32 - 1.0);
    [start, end]
}

fn yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let interpolation = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0_f32 - interpolation.clamp(0.0, 1.0)
}
