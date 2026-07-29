use crate::error::Result;
use crate::gemv::{validate_finite_slice, validate_finite_value};
use crate::{
    expert_swiglu_accumulate_into, swiglu_project_into, KernelError, ReferenceExecutionMode, SwiGluExpert,
    SwiGluScratch,
};
use bridge_model_hy3::RoutedExpert;

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
/// The preflight output scratch lets every expert be executed and validated
/// before the final output is mutated. The accepted pass then accumulates each
/// down projection directly into `output`.
pub fn moe_selected_into(
    mode: ReferenceExecutionMode,
    routed: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    input: &[f32],
    output: &mut [f32],
    preflight_output_scratch: &mut [f32],
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    validate_structure(routed, shared, input, output, preflight_output_scratch)?;
    let preflight_output_scratch = &mut preflight_output_scratch[..output.len()];

    for selected in routed {
        swiglu_project_into(mode, selected.expert, input, preflight_output_scratch, scratch)?;
        validate_finite_slice("routed expert output", preflight_output_scratch)?;
    }
    swiglu_project_into(mode, shared, input, preflight_output_scratch, scratch)?;
    validate_finite_slice("shared expert output", preflight_output_scratch)?;

    output.fill(0.0);
    for selected in routed {
        expert_swiglu_accumulate_into(
            mode,
            selected.expert,
            input,
            output,
            selected.coefficient,
            scratch,
        )?;
    }
    expert_swiglu_accumulate_into(mode, shared, input, output, 1.0, scratch)
}

/// Executes routed experts selected by ID without constructing a temporary
/// borrowed expert list. This is the hot-path form used by the complete layer.
pub fn moe_routed_by_id_into(
    mode: ReferenceExecutionMode,
    selection: RoutedMoeSelection<'_>,
    input: &[f32],
    output: &mut [f32],
    preflight_output_scratch: &mut [f32],
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
    if preflight_output_scratch.len() < output.len() {
        return Err(KernelError::ScratchTooSmall {
            field: "MoE preflight output",
            required: output.len(),
            actual: preflight_output_scratch.len(),
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
        swiglu_project_into(
            mode,
            *expert,
            input,
            &mut preflight_output_scratch[..output.len()],
            scratch,
        )?;
    }
    swiglu_project_into(
        mode,
        shared,
        input,
        &mut preflight_output_scratch[..output.len()],
        scratch,
    )?;

    output.fill(0.0);
    for selected in routed {
        let expert = experts[selected.expert_id as usize];
        expert_swiglu_accumulate_into(mode, expert, input, output, selected.coefficient, scratch)?;
    }
    expert_swiglu_accumulate_into(mode, shared, input, output, 1.0, scratch)
}

fn validate_structure(
    routed: &[SelectedExpert<'_>],
    shared: SwiGluExpert<'_>,
    input: &[f32],
    output: &[f32],
    preflight_output_scratch: &[f32],
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
    if preflight_output_scratch.len() < output.len() {
        return Err(KernelError::ScratchTooSmall {
            field: "MoE preflight output",
            required: output.len(),
            actual: preflight_output_scratch.len(),
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
