use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::KernelError;

pub fn softmax_into(logits: &[f32], output: &mut [f32]) -> Result<()> {
    causal_softmax_into(logits, logits.len(), output)
}

/// Stable softmax over the unmasked prefix. Masked lanes are written as zero;
/// an all-masked row is the all-zero distribution.
pub fn causal_softmax_into(logits: &[f32], unmasked: usize, output: &mut [f32]) -> Result<()> {
    if output.len() != logits.len() {
        return Err(KernelError::DimensionMismatch {
            field: "softmax output",
            expected: logits.len(),
            actual: output.len(),
        });
    }
    if unmasked > logits.len() {
        return Err(KernelError::DimensionMismatch {
            field: "softmax unmasked prefix",
            expected: logits.len(),
            actual: unmasked,
        });
    }
    validate_finite_slice("softmax logits", &logits[..unmasked])?;
    if unmasked == 0 {
        output.fill(0.0);
        return Ok(());
    }

    let maximum = logits[..unmasked]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for &logit in &logits[..unmasked] {
        sum += (logit - maximum).exp();
    }
    validate_finite_value("softmax denominator", 0, sum)?;
    if sum == 0.0 {
        return Err(KernelError::InvalidParameter {
            field: "softmax denominator",
            reason: "must be greater than zero",
        });
    }

    for (destination, &logit) in output[..unmasked].iter_mut().zip(&logits[..unmasked]) {
        *destination = (logit - maximum).exp() / sum;
    }
    output[unmasked..].fill(0.0);
    Ok(())
}
