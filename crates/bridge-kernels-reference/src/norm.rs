use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::KernelError;

pub fn weighted_rms_norm_into(input: &[f32], weight: &[f32], epsilon: f32, output: &mut [f32]) -> Result<()> {
    validate_norm_call(input, weight, epsilon, output)?;
    let scale = rms_scale(input, epsilon)?;
    for (index, (&value, &weight)) in input.iter().zip(weight).enumerate() {
        let normalized = value * scale * weight;
        validate_finite_value("RMSNorm output", index, normalized)?;
    }
    for ((destination, &value), &weight) in output.iter_mut().zip(input).zip(weight) {
        let normalized = value * scale * weight;
        *destination = normalized;
    }
    Ok(())
}

pub fn weighted_rms_norm_in_place(values: &mut [f32], weight: &[f32], epsilon: f32) -> Result<()> {
    validate_norm_lengths(values, weight)?;
    validate_epsilon(epsilon)?;
    validate_finite_slice("RMSNorm input", values)?;
    validate_finite_slice("RMSNorm weight", weight)?;
    let scale = rms_scale(values, epsilon)?;
    for (index, (&value, &weight)) in values.iter().zip(weight).enumerate() {
        let normalized = value * scale * weight;
        validate_finite_value("RMSNorm output", index, normalized)?;
    }
    for (value, &weight) in values.iter_mut().zip(weight) {
        let normalized = *value * scale * weight;
        *value = normalized;
    }
    Ok(())
}

pub fn weighted_head_rms_norm_in_place(
    values: &mut [f32],
    weight: &[f32],
    head_dimension: usize,
    epsilon: f32,
) -> Result<()> {
    if head_dimension == 0 {
        return Err(KernelError::InvalidParameter {
            field: "head_dimension",
            reason: "must be greater than zero",
        });
    }
    if weight.len() != head_dimension {
        return Err(KernelError::DimensionMismatch {
            field: "per-head RMSNorm weight",
            expected: head_dimension,
            actual: weight.len(),
        });
    }
    if values.len() % head_dimension != 0 {
        return Err(KernelError::DimensionMismatch {
            field: "per-head RMSNorm values",
            expected: head_dimension,
            actual: values.len(),
        });
    }
    validate_epsilon(epsilon)?;
    validate_finite_slice("per-head RMSNorm input", values)?;
    validate_finite_slice("per-head RMSNorm weight", weight)?;

    for head in values.chunks(head_dimension) {
        let scale = rms_scale(head, epsilon)?;
        for (index, (&value, &weight)) in head.iter().zip(weight).enumerate() {
            let normalized = value * scale * weight;
            validate_finite_value("per-head RMSNorm output", index, normalized)?;
        }
    }
    for head in values.chunks_mut(head_dimension) {
        let scale = rms_scale(head, epsilon)?;
        for (value, &weight) in head.iter_mut().zip(weight) {
            let normalized = *value * scale * weight;
            *value = normalized;
        }
    }
    Ok(())
}

pub fn residual_add_in_place(destination: &mut [f32], residual: &[f32]) -> Result<()> {
    if destination.len() != residual.len() {
        return Err(KernelError::DimensionMismatch {
            field: "residual",
            expected: destination.len(),
            actual: residual.len(),
        });
    }
    validate_finite_slice("residual destination", destination)?;
    validate_finite_slice("residual source", residual)?;
    for (index, (&left, &right)) in destination.iter().zip(residual).enumerate() {
        validate_finite_value("residual output", index, left + right)?;
    }
    for (destination, &residual) in destination.iter_mut().zip(residual) {
        *destination += residual;
    }
    Ok(())
}

fn validate_norm_call(input: &[f32], weight: &[f32], epsilon: f32, output: &[f32]) -> Result<()> {
    validate_norm_lengths(input, weight)?;
    if output.len() != input.len() {
        return Err(KernelError::DimensionMismatch {
            field: "RMSNorm output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    validate_epsilon(epsilon)?;
    validate_finite_slice("RMSNorm input", input)?;
    validate_finite_slice("RMSNorm weight", weight)
}

fn validate_norm_lengths(input: &[f32], weight: &[f32]) -> Result<()> {
    if input.is_empty() {
        return Err(KernelError::InvalidParameter {
            field: "RMSNorm input",
            reason: "must not be empty",
        });
    }
    if weight.len() != input.len() {
        return Err(KernelError::DimensionMismatch {
            field: "RMSNorm weight",
            expected: input.len(),
            actual: weight.len(),
        });
    }
    Ok(())
}

fn validate_epsilon(epsilon: f32) -> Result<()> {
    validate_finite_value("RMSNorm epsilon", 0, epsilon)?;
    if epsilon <= 0.0 {
        return Err(KernelError::InvalidParameter {
            field: "RMSNorm epsilon",
            reason: "must be greater than zero",
        });
    }
    Ok(())
}

fn rms_scale(values: &[f32], epsilon: f32) -> Result<f32> {
    let mut sum = 0.0_f32;
    for &value in values {
        sum += value * value;
    }
    let mean = sum / values.len() as f32;
    let scale = 1.0_f32 / (mean + epsilon).sqrt();
    validate_finite_value("RMSNorm scale", 0, scale)?;
    Ok(scale)
}
