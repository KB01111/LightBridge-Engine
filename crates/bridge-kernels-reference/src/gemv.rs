use bridge_core::ggml_type::GgmlType;
use bridge_quant_layout::{
    decode_block_into, layout, quantize_row_q8_k_into, validate_vec_dot_q8_k, vec_dot_q8_k, vec_dot_q8_k_cpu,
    Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS,
};
use rayon::prelude::*;

use crate::error::Result;
use crate::{KernelError, PackedMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceExecutionMode {
    DequantF32,
    LlamaQ8K,
    CpuParallelQ8K,
}

pub fn required_q8_k_bytes(input_width: usize) -> Result<usize> {
    if input_width == 0 || input_width % Q8_K_BLOCK_ELEMENTS != 0 {
        return Err(KernelError::DimensionMismatch {
            field: "Q8_K input width",
            expected: Q8_K_BLOCK_ELEMENTS,
            actual: input_width,
        });
    }
    (input_width / Q8_K_BLOCK_ELEMENTS)
        .checked_mul(Q8_K_BLOCK_BYTES)
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "Q8_K scratch byte length",
        })
}

pub fn gemv_dequant_f32_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    decoded_block_scratch: &mut [f32],
) -> Result<()> {
    prepare_dequant(matrix, input, output, decoded_block_scratch)?;
    compute_dequant_into(matrix, input, output, decoded_block_scratch)
}

pub fn gemv_llama_q8k_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let q8 = prepare_llama_q8k(matrix, input, output, q8_scratch)?;
    for (row, destination) in output.iter_mut().enumerate() {
        *destination = vec_dot_q8_k(matrix.ty(), matrix.row(row), q8, matrix.input_width())?;
    }
    Ok(())
}

pub fn gemv_cpu_parallel_q8k_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let q8 = prepare_llama_q8k(matrix, input, output, q8_scratch)?;
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, destination)| {
            *destination = vec_dot_q8_k_cpu(matrix.ty(), matrix.row(row), q8, matrix.input_width())?;
            Ok::<(), KernelError>(())
        })
}

fn gemv_cpu_parallel_f32_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    decoded_block_scratch: &mut [f32],
) -> Result<()> {
    prepare_dequant(matrix, input, output, decoded_block_scratch)?;
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, destination)| {
            *destination = dot_f32_row(matrix, row, input)?;
            Ok::<(), KernelError>(())
        })
}

pub fn gemv_into(
    mode: ReferenceExecutionMode,
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    decoded_block_scratch: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    if matrix.ty() == GgmlType::F32 {
        return match mode {
            ReferenceExecutionMode::CpuParallelQ8K => {
                gemv_cpu_parallel_f32_into(matrix, input, output, decoded_block_scratch)
            }
            ReferenceExecutionMode::DequantF32 | ReferenceExecutionMode::LlamaQ8K => {
                gemv_dequant_f32_into(matrix, input, output, decoded_block_scratch)
            }
        };
    }
    match mode {
        ReferenceExecutionMode::DequantF32 => {
            gemv_dequant_f32_into(matrix, input, output, decoded_block_scratch)
        }
        ReferenceExecutionMode::LlamaQ8K => gemv_llama_q8k_into(matrix, input, output, q8_scratch),
        ReferenceExecutionMode::CpuParallelQ8K => {
            gemv_cpu_parallel_q8k_into(matrix, input, output, q8_scratch)
        }
    }
}

pub fn gemv_accumulate_scaled_into(
    mode: ReferenceExecutionMode,
    matrix: PackedMatrix<'_>,
    input: &[f32],
    destination: &mut [f32],
    scale: f32,
    decoded_block_scratch: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    validate_finite_value("accumulation scale", 0, scale)?;
    validate_finite_slice("destination", destination)?;
    if matrix.ty() == GgmlType::F32 {
        prepare_dequant(matrix, input, destination, decoded_block_scratch)?;
        match mode {
            ReferenceExecutionMode::CpuParallelQ8K => {
                destination
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(row, value)| {
                        *value += scale * dot_f32_row(matrix, row, input)?;
                        Ok::<(), KernelError>(())
                    })?;
            }
            ReferenceExecutionMode::DequantF32 | ReferenceExecutionMode::LlamaQ8K => {
                for (row, value) in destination.iter_mut().enumerate() {
                    *value += scale * dot_f32_row(matrix, row, input)?;
                }
            }
        }
        return Ok(());
    }
    match mode {
        ReferenceExecutionMode::DequantF32 => {
            prepare_dequant(matrix, input, destination, decoded_block_scratch)?;
            for (row, value) in destination.iter_mut().enumerate() {
                *value += scale * dot_dequant_row(matrix, row, input, decoded_block_scratch)?;
            }
        }
        ReferenceExecutionMode::LlamaQ8K => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            for (row, value) in destination.iter_mut().enumerate() {
                *value += scale * vec_dot_q8_k(matrix.ty(), matrix.row(row), q8, matrix.input_width())?;
            }
        }
        ReferenceExecutionMode::CpuParallelQ8K => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            destination
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, value)| {
                    *value +=
                        scale * vec_dot_q8_k_cpu(matrix.ty(), matrix.row(row), q8, matrix.input_width())?;
                    Ok::<(), KernelError>(())
                })?;
        }
    }
    Ok(())
}

pub(crate) fn validate_finite_slice(field: &'static str, values: &[f32]) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        validate_finite_value(field, index, value)?;
    }
    Ok(())
}

pub(crate) fn validate_finite_value(field: &'static str, index: usize, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(KernelError::NonFiniteValue {
            field,
            index,
            bits: value.to_bits(),
        })
    }
}

fn validate_dimensions(matrix: PackedMatrix<'_>, input: &[f32], output: &[f32]) -> Result<()> {
    if input.len() != matrix.input_width() {
        return Err(KernelError::DimensionMismatch {
            field: "GEMV input",
            expected: matrix.input_width(),
            actual: input.len(),
        });
    }
    if output.len() != matrix.output_width() {
        return Err(KernelError::DimensionMismatch {
            field: "GEMV output",
            expected: matrix.output_width(),
            actual: output.len(),
        });
    }
    validate_finite_slice("GEMV input", input)
}

fn prepare_dequant(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &[f32],
    decoded_block_scratch: &mut [f32],
) -> Result<()> {
    validate_dimensions(matrix, input, output)?;
    let block_layout = layout(matrix.ty())?;
    if decoded_block_scratch.len() < block_layout.block_elements {
        return Err(KernelError::ScratchTooSmall {
            field: "decoded block",
            required: block_layout.block_elements,
            actual: decoded_block_scratch.len(),
        });
    }
    let decoded = &mut decoded_block_scratch[..block_layout.block_elements];
    let blocks_per_row = matrix.input_width() / block_layout.block_elements;
    for row in 0..matrix.output_width() {
        let encoded = matrix.row(row);
        for block in 0..blocks_per_row {
            let start = block * block_layout.block_bytes;
            decode_block_into(
                matrix.ty(),
                &encoded[start..start + block_layout.block_bytes],
                decoded,
            )?;
            validate_finite_slice("GEMV weights", decoded)?;
        }
    }
    Ok(())
}

fn prepare_llama_q8k<'a>(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &[f32],
    q8_scratch: &'a mut [u8],
) -> Result<&'a [u8]> {
    validate_dimensions(matrix, input, output)?;
    if !matches!(
        matrix.ty(),
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S
    ) {
        return Err(KernelError::UnsupportedType { ty: matrix.ty() });
    }
    let required = required_q8_k_bytes(matrix.input_width())?;
    if q8_scratch.len() < required {
        return Err(KernelError::ScratchTooSmall {
            field: "Q8_K row",
            required,
            actual: q8_scratch.len(),
        });
    }
    let q8 = &mut q8_scratch[..required];
    quantize_row_q8_k_into(input, q8)?;
    for row in 0..matrix.output_width() {
        validate_vec_dot_q8_k(matrix.ty(), matrix.row(row), q8, matrix.input_width())?;
    }
    Ok(q8)
}

fn compute_dequant_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    decoded_block_scratch: &mut [f32],
) -> Result<()> {
    for (row, destination) in output.iter_mut().enumerate() {
        *destination = dot_dequant_row(matrix, row, input, decoded_block_scratch)?;
    }
    Ok(())
}

fn dot_f32_row(matrix: PackedMatrix<'_>, row: usize, input: &[f32]) -> Result<f32> {
    if matrix.ty() != GgmlType::F32 {
        return Err(KernelError::UnsupportedType { ty: matrix.ty() });
    }
    let encoded = matrix.row(row);
    let mut sum = 0.0_f32;
    for (lane, &input_value) in input.iter().enumerate() {
        let start = lane * 4;
        let weight = f32::from_bits(u32::from_le_bytes([
            encoded[start],
            encoded[start + 1],
            encoded[start + 2],
            encoded[start + 3],
        ]));
        sum += weight * input_value;
    }
    Ok(sum)
}

fn dot_dequant_row(
    matrix: PackedMatrix<'_>,
    row: usize,
    input: &[f32],
    decoded_block_scratch: &mut [f32],
) -> Result<f32> {
    let block_layout = layout(matrix.ty())?;
    let decoded = &mut decoded_block_scratch[..block_layout.block_elements];
    let encoded = matrix.row(row);
    let mut sum = 0.0_f32;
    for block in 0..matrix.input_width() / block_layout.block_elements {
        let encoded_start = block * block_layout.block_bytes;
        decode_block_into(
            matrix.ty(),
            &encoded[encoded_start..encoded_start + block_layout.block_bytes],
            decoded,
        )?;
        let input_start = block * block_layout.block_elements;
        for lane in 0..block_layout.block_elements {
            sum += decoded[lane] * input[input_start + lane];
        }
    }
    Ok(sum)
}
