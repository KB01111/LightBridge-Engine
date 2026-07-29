use std::cmp::Ordering;

use crate::Hy3Error;

const ROUTING_SUM_FLOOR: f32 = 0.000_061_035_156; // 2^-14

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteCandidate {
    pub expert_id: u32,
    pub selection_score: f32,
    pub unbiased_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedExpert {
    pub expert_id: u32,
    pub coefficient: f32,
}

/// Selects Hy3 routed experts deterministically into caller-owned scratch.
///
/// The learned bias affects selection only. Equal selection scores use
/// ascending expert IDs as a BRIDGE policy; the selected output is also sorted
/// by expert ID so later accumulation has a stable order.
pub fn route_experts_into(
    logits: &[f32],
    selection_bias: &[f32],
    expert_used_count: usize,
    weight_scale: f32,
    candidates: &mut [RouteCandidate],
    selected: &mut [RoutedExpert],
) -> Result<(), Hy3Error> {
    let expert_count = logits.len();
    if expert_count > u32::MAX as usize {
        return Err(Hy3Error::RoutingLength {
            field: "logits representable expert IDs",
            expected: u32::MAX as usize,
            actual: expert_count,
        });
    }
    if selection_bias.len() != expert_count {
        return Err(Hy3Error::RoutingLength {
            field: "selection_bias",
            expected: expert_count,
            actual: selection_bias.len(),
        });
    }
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(Hy3Error::RoutingTopK {
            expert_count,
            expert_used_count,
        });
    }
    if candidates.len() != expert_count {
        return Err(Hy3Error::RoutingLength {
            field: "candidates",
            expected: expert_count,
            actual: candidates.len(),
        });
    }
    if selected.len() != expert_used_count {
        return Err(Hy3Error::RoutingLength {
            field: "selected",
            expected: expert_used_count,
            actual: selected.len(),
        });
    }

    validate_finite("weight_scale", 0, weight_scale)?;
    for (index, (&logit, &bias)) in logits.iter().zip(selection_bias).enumerate() {
        validate_finite("logits", index, logit)?;
        validate_finite("selection_bias", index, bias)?;
        let score = sigmoid(logit) + bias;
        validate_finite("selection_score", index, score)?;
    }

    for (index, (&logit, &bias)) in logits.iter().zip(selection_bias).enumerate() {
        let unbiased_weight = sigmoid(logit);
        candidates[index] = RouteCandidate {
            expert_id: u32::try_from(index).expect("expert count was bounded to u32"),
            selection_score: unbiased_weight + bias,
            unbiased_weight,
        };
    }
    candidates.sort_unstable_by(|left, right| {
        right
            .selection_score
            .partial_cmp(&left.selection_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.expert_id.cmp(&right.expert_id))
    });

    let mut weight_sum = 0.0_f32;
    for candidate in &candidates[..expert_used_count] {
        weight_sum += candidate.unbiased_weight;
    }
    weight_sum = weight_sum.max(ROUTING_SUM_FLOOR);

    for (output, candidate) in selected.iter_mut().zip(&candidates[..expert_used_count]) {
        *output = RoutedExpert {
            expert_id: candidate.expert_id,
            coefficient: candidate.unbiased_weight / weight_sum * weight_scale,
        };
    }
    selected.sort_unstable_by_key(|expert| expert.expert_id);
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-value).exp())
}

fn validate_finite(field: &'static str, index: usize, value: f32) -> Result<(), Hy3Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Hy3Error::NonFiniteRoutingValue {
            field,
            index,
            bits: value.to_bits(),
        })
    }
}
