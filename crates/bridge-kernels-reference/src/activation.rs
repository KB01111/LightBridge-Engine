use crate::error::Result;
use crate::gemv::{gemv_accumulate_scaled_into, gemv_into, validate_finite_value};
use crate::{KernelError, PackedMatrix, ReferenceExecutionMode};

#[derive(Debug, Clone, Copy)]
pub struct SwiGluExpert<'a> {
    gate: PackedMatrix<'a>,
    up: PackedMatrix<'a>,
    down: PackedMatrix<'a>,
}

impl<'a> SwiGluExpert<'a> {
    pub fn new(gate: PackedMatrix<'a>, up: PackedMatrix<'a>, down: PackedMatrix<'a>) -> Result<Self> {
        require_equal("SwiGLU gate/up input width", gate.input_width(), up.input_width())?;
        require_equal(
            "SwiGLU gate/up hidden width",
            gate.output_width(),
            up.output_width(),
        )?;
        require_equal("SwiGLU down input width", gate.output_width(), down.input_width())?;
        Ok(Self { gate, up, down })
    }

    pub const fn gate(self) -> PackedMatrix<'a> {
        self.gate
    }

    pub const fn up(self) -> PackedMatrix<'a> {
        self.up
    }

    pub const fn down(self) -> PackedMatrix<'a> {
        self.down
    }

    pub const fn input_width(self) -> usize {
        self.gate.input_width()
    }

    pub const fn hidden_width(self) -> usize {
        self.gate.output_width()
    }

    pub const fn output_width(self) -> usize {
        self.down.output_width()
    }
}

pub struct SwiGluScratch<'a> {
    pub activation: &'a mut [f32],
    pub decoded_block: &'a mut [f32],
    pub q8: &'a mut [u8],
}

impl<'a> SwiGluScratch<'a> {
    pub fn new(activation: &'a mut [f32], decoded_block: &'a mut [f32], q8: &'a mut [u8]) -> Self {
        Self {
            activation,
            decoded_block,
            q8,
        }
    }
}

pub fn swiglu_project_into(
    mode: ReferenceExecutionMode,
    expert: SwiGluExpert<'_>,
    input: &[f32],
    output: &mut [f32],
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    validate_expert_call(expert, input, output, scratch.activation)?;
    let hidden = expert.hidden_width();
    let (gate_values, remainder) = scratch.activation.split_at_mut(hidden);
    let up_values = &mut remainder[..hidden];

    gemv_into(
        mode,
        expert.gate(),
        input,
        gate_values,
        scratch.decoded_block,
        scratch.q8,
    )?;
    gemv_into(
        mode,
        expert.up(),
        input,
        up_values,
        scratch.decoded_block,
        scratch.q8,
    )?;
    apply_swiglu(gate_values, up_values)?;
    gemv_into(
        mode,
        expert.down(),
        gate_values,
        output,
        scratch.decoded_block,
        scratch.q8,
    )
}

pub fn expert_swiglu_accumulate_into(
    mode: ReferenceExecutionMode,
    expert: SwiGluExpert<'_>,
    input: &[f32],
    destination: &mut [f32],
    coefficient: f32,
    scratch: &mut SwiGluScratch<'_>,
) -> Result<()> {
    validate_finite_value("expert coefficient", 0, coefficient)?;
    validate_expert_call(expert, input, destination, scratch.activation)?;
    let hidden = expert.hidden_width();
    let (gate_values, remainder) = scratch.activation.split_at_mut(hidden);
    let up_values = &mut remainder[..hidden];

    gemv_into(
        mode,
        expert.gate(),
        input,
        gate_values,
        scratch.decoded_block,
        scratch.q8,
    )?;
    gemv_into(
        mode,
        expert.up(),
        input,
        up_values,
        scratch.decoded_block,
        scratch.q8,
    )?;
    apply_swiglu(gate_values, up_values)?;
    gemv_accumulate_scaled_into(
        mode,
        expert.down(),
        gate_values,
        destination,
        coefficient,
        scratch.decoded_block,
        scratch.q8,
    )
}

fn validate_expert_call(
    expert: SwiGluExpert<'_>,
    input: &[f32],
    output: &[f32],
    activation_scratch: &[f32],
) -> Result<()> {
    require_equal("SwiGLU input", expert.input_width(), input.len())?;
    require_equal("SwiGLU output", expert.output_width(), output.len())?;
    let required = expert
        .hidden_width()
        .checked_mul(2)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "SwiGLU gate/up scratch length",
        })?;
    if activation_scratch.len() < required {
        return Err(KernelError::ScratchTooSmall {
            field: "SwiGLU gate/up",
            required,
            actual: activation_scratch.len(),
        });
    }
    Ok(())
}

fn apply_swiglu(gate: &mut [f32], up: &[f32]) -> Result<()> {
    for (index, (gate_value, &up_value)) in gate.iter_mut().zip(up).enumerate() {
        let activated = *gate_value / (1.0_f32 + (-*gate_value).exp());
        let value = activated * up_value;
        validate_finite_value("SwiGLU activation", index, value)?;
        *gate_value = value;
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
