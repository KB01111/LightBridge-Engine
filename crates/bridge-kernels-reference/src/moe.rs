use std::mem::MaybeUninit;

use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::{
    expert_swiglu_accumulate_into, KernelError, ReferenceExecutionMode, SwiGluExpert, SwiGluScratch,
};
use bridge_kernels_cuda::{packed_q8k_gemv_batch_into, CudaPackedQ8KBatchItem};
use bridge_model_hy3::RoutedExpert;
use bridge_quant_layout::quantize_row_q8_k_into;

use crate::activation::apply_swiglu;

const MAX_CUDA_BATCH_EXPERTS: usize = 65;
const MAX_CUDA_BATCH_ITEMS: usize = MAX_CUDA_BATCH_EXPERTS * 2;

#[derive(Debug, Clone, Copy)]
pub struct SelectedExpert<'a> {
    pub expert_id: u32,
    pub coefficient: f32,
    pub expert: SwiGluExpert<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct RoutedMoeSelection<'a> {
    pub routed: &'a [RoutedExpert],
    pub experts: &'a [SwiGluExpert<'a>],
    pub shared: SwiGluExpert<'a>,
}

/// Executes routed experts in ascending ID order, followed by the always-on
/// shared expert at coefficient 1.0.
///
/// Every expert is executed exactly once into an uncommitted candidate. The
/// candidate is validated and copied to `output` only after all experts
/// succeed, so malformed payloads cannot partially publish an MoE result.
pub fn moe_selected_into(
    mode: ReferenceExecutionMode,
    routed: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    input: &[f32],
    output: &mut [f32],
    candidate_output_scratch: &mut [f32],
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    validate_structure(routed, shared, input, output, candidate_output_scratch)?;
    let candidate = &mut candidate_output_scratch[..output.len()];
    candidate.fill(0.0);
    if mode == ReferenceExecutionMode::CudaQ8K {
        cuda_moe_selected_into(routed, shared, input, candidate, scratch)?;
    } else {
        for selected in routed {
            expert_swiglu_accumulate_into(
                mode,
                selected.expert,
                input,
                candidate,
                selected.coefficient,
                scratch,
            )?;
        }
        expert_swiglu_accumulate_into(mode, shared, input, candidate, 1.0, scratch)?;
    }
    validate_finite_slice("MoE candidate output", candidate)?;
    output.copy_from_slice(candidate);
    Ok(())
}

fn cuda_moe_selected_into(
    routed: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    input: &[f32],
    candidate: &mut [f32],
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    let expert_count = routed
        .len()
        .checked_add(1)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "CUDA MoE batch expert count",
        })?;
    if expert_count > MAX_CUDA_BATCH_EXPERTS {
        return Err(KernelError::InvalidParameter {
            field: "CUDA MoE batch expert count",
            reason: "must not exceed 65 total routed and shared experts",
        });
    }
    let expert_hidden = shared.hidden_width();
    let output_width = shared.output_width();
    let gate_up_values = expert_count
        .checked_mul(2)
        .and_then(|value| value.checked_mul(expert_hidden))
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "CUDA MoE gate/up batch output length",
        })?;
    let down_values = expert_count
        .checked_mul(output_width)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "CUDA MoE down batch output length",
        })?;
    let required_activation = gate_up_values.max(down_values);
    if scratch.activation.len() < required_activation {
        return Err(KernelError::ScratchTooSmall {
            field: "CUDA MoE batched activation/output",
            required: required_activation,
            actual: scratch.activation.len(),
        });
    }

    let input_q8_bytes = crate::required_q8_k_bytes(input.len())?;
    if scratch.q8.len() < input_q8_bytes {
        return Err(KernelError::ScratchTooSmall {
            field: "CUDA MoE input Q8_K",
            required: input_q8_bytes,
            actual: scratch.q8.len(),
        });
    }
    quantize_row_q8_k_into(input, &mut scratch.q8[..input_q8_bytes])?;
    {
        let input_q8 = &scratch.q8[..input_q8_bytes];
        let mut item_storage: [MaybeUninit<CudaPackedQ8KBatchItem<'_>>; MAX_CUDA_BATCH_ITEMS] =
            [MaybeUninit::uninit(); MAX_CUDA_BATCH_ITEMS];
        let mut item_count = 0;
        for selected in routed {
            for matrix in [selected.expert.gate(), selected.expert.up()] {
                item_storage[item_count].write(CudaPackedQ8KBatchItem {
                    weight_type: matrix.ty(),
                    weights: matrix.bytes(),
                    q8: input_q8,
                    logical_elements: matrix.input_width(),
                    rows: matrix.output_width(),
                });
                item_count += 1;
            }
        }
        for matrix in [shared.gate(), shared.up()] {
            item_storage[item_count].write(CudaPackedQ8KBatchItem {
                weight_type: matrix.ty(),
                weights: matrix.bytes(),
                q8: input_q8,
                logical_elements: matrix.input_width(),
                rows: matrix.output_width(),
            });
            item_count += 1;
        }
        // SAFETY: `item_count` contiguous entries were initialized above and
        // the batch item has no drop glue.
        let items = unsafe {
            std::slice::from_raw_parts(
                item_storage.as_ptr().cast::<CudaPackedQ8KBatchItem<'_>>(),
                item_count,
            )
        };
        packed_q8k_gemv_batch_into(items, &mut scratch.activation[..gate_up_values]).map_err(|error| {
            KernelError::Cuda {
                message: error.to_string(),
            }
        })?;
    }
    validate_finite_slice(
        "CUDA MoE gate/up batch output",
        &scratch.activation[..gate_up_values],
    )?;
    for expert_index in 0..expert_count {
        let start = expert_index * 2 * expert_hidden;
        let (gate, up) = scratch.activation[start..start + 2 * expert_hidden].split_at_mut(expert_hidden);
        apply_swiglu(gate, up)?;
    }

    let down_q8_bytes = crate::required_q8_k_bytes(expert_hidden)?;
    let total_down_q8_bytes =
        down_q8_bytes
            .checked_mul(expert_count)
            .ok_or(KernelError::ArithmeticOverflow {
                operation: "CUDA MoE down Q8_K batch length",
            })?;
    if scratch.q8.len() < total_down_q8_bytes {
        return Err(KernelError::ScratchTooSmall {
            field: "CUDA MoE down Q8_K batch",
            required: total_down_q8_bytes,
            actual: scratch.q8.len(),
        });
    }
    for expert_index in 0..expert_count {
        let gate_start = expert_index * 2 * expert_hidden;
        let q8_start = expert_index * down_q8_bytes;
        quantize_row_q8_k_into(
            &scratch.activation[gate_start..gate_start + expert_hidden],
            &mut scratch.q8[q8_start..q8_start + down_q8_bytes],
        )?;
    }
    {
        let mut item_storage: [MaybeUninit<CudaPackedQ8KBatchItem<'_>>; MAX_CUDA_BATCH_EXPERTS] =
            [MaybeUninit::uninit(); MAX_CUDA_BATCH_EXPERTS];
        for (expert_index, expert) in routed
            .iter()
            .map(|selected| selected.expert)
            .chain(std::iter::once(shared))
            .enumerate()
        {
            let matrix = expert.down();
            let q8_start = expert_index * down_q8_bytes;
            item_storage[expert_index].write(CudaPackedQ8KBatchItem {
                weight_type: matrix.ty(),
                weights: matrix.bytes(),
                q8: &scratch.q8[q8_start..q8_start + down_q8_bytes],
                logical_elements: matrix.input_width(),
                rows: matrix.output_width(),
            });
        }
        // SAFETY: exactly `expert_count` contiguous entries were initialized
        // above and the batch item has no drop glue.
        let items = unsafe {
            std::slice::from_raw_parts(
                item_storage.as_ptr().cast::<CudaPackedQ8KBatchItem<'_>>(),
                expert_count,
            )
        };
        packed_q8k_gemv_batch_into(items, &mut scratch.activation[..down_values]).map_err(|error| {
            KernelError::Cuda {
                message: error.to_string(),
            }
        })?;
    }
    validate_finite_slice("CUDA MoE down batch output", &scratch.activation[..down_values])?;
    for (expert_index, coefficient) in routed
        .iter()
        .map(|selected| selected.coefficient)
        .chain(std::iter::once(1.0))
        .enumerate()
    {
        let start = expert_index * output_width;
        for (destination, &value) in candidate
            .iter_mut()
            .zip(&scratch.activation[start..start + output_width])
        {
            *destination += coefficient * value;
        }
    }
    Ok(())
}

/// Executes routed experts selected by ID without constructing a temporary
/// borrowed expert list. This is the hot-path form used by the complete layer.
pub fn moe_routed_by_id_into(
    mode: ReferenceExecutionMode,
    selection: RoutedMoeSelection<'_>,
    input: &[f32],
    output: &mut [f32],
    candidate_output_scratch: &mut [f32],
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    let routed = selection.routed;
    let experts = selection.experts;
    let shared = selection.shared;
    if routed.is_empty() {
        return Err(KernelError::DimensionMismatch {
            field: "selected routed expert count",
            expected: 1,
            actual: 0,
        });
    }
    require_equal("MoE input", shared.input_width(), input.len())?;
    require_equal("MoE output", shared.output_width(), output.len())?;
    if candidate_output_scratch.len() < output.len() {
        return Err(KernelError::ScratchTooSmall {
            field: "candidate_output_scratch",
            required: output.len(),
            actual: candidate_output_scratch.len(),
        });
    }

    let mut previous = None;
    for selected in routed {
        validate_finite_value("routed expert coefficient", 0, selected.coefficient)?;
        if let Some(previous_id) = previous {
            if selected.expert_id == previous_id {
                return Err(KernelError::DuplicateRoutedExpert {
                    expert_id: selected.expert_id,
                });
            }
            if selected.expert_id < previous_id {
                return Err(KernelError::RoutedExpertOrder {
                    previous: previous_id,
                    current: selected.expert_id,
                });
            }
        }
        previous = Some(selected.expert_id);
        let expert = experts
            .get(selected.expert_id as usize)
            .ok_or(KernelError::InvalidParameter {
                field: "routed expert ID",
                reason: "must index an available expert",
            })?;
        require_equal(
            "routed expert input width",
            shared.input_width(),
            expert.input_width(),
        )?;
        require_equal(
            "routed expert output width",
            shared.output_width(),
            expert.output_width(),
        )?;
    }
    let candidate = &mut candidate_output_scratch[..output.len()];
    candidate.fill(0.0);
    for selected in routed {
        let expert = experts[selected.expert_id as usize];
        expert_swiglu_accumulate_into(mode, expert, input, candidate, selected.coefficient, scratch)?;
    }
    expert_swiglu_accumulate_into(mode, shared, input, candidate, 1.0, scratch)?;
    validate_finite_slice("MoE candidate output", candidate)?;
    output.copy_from_slice(candidate);
    Ok(())
}

fn validate_structure(
    routed: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    input: &[f32],
    output: &[f32],
    candidate_output_scratch: &[f32],
) -> Result<()> {
    if routed.is_empty() {
        return Err(KernelError::DimensionMismatch {
            field: "selected routed expert count",
            expected: 1,
            actual: 0,
        });
    }
    require_equal("MoE input", shared.input_width(), input.len())?;
    require_equal("MoE output", shared.output_width(), output.len())?;
    if candidate_output_scratch.len() < output.len() {
        return Err(KernelError::ScratchTooSmall {
            field: "candidate_output_scratch",
            required: output.len(),
            actual: candidate_output_scratch.len(),
        });
    }

    let mut previous = None;
    for selected in routed {
        validate_finite_value("routed expert coefficient", 0, selected.coefficient)?;
        if let Some(previous_id) = previous {
            if selected.expert_id == previous_id {
                return Err(KernelError::DuplicateRoutedExpert {
                    expert_id: selected.expert_id,
                });
            }
            if selected.expert_id < previous_id {
                return Err(KernelError::RoutedExpertOrder {
                    previous: previous_id,
                    current: selected.expert_id,
                });
            }
        }
        previous = Some(selected.expert_id);
        require_equal(
            "routed expert input width",
            shared.input_width(),
            selected.expert.input_width(),
        )?;
        require_equal(
            "routed expert hidden width",
            shared.hidden_width(),
            selected.expert.hidden_width(),
        )?;
        require_equal(
            "routed expert output width",
            shared.output_width(),
            selected.expert.output_width(),
        )?;
    }
    Ok(())
}

fn require_equal(field: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(KernelError::DimensionMismatch {
            field,
            expected,
            actual,
        })
    }
}
