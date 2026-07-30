use bridge_core::ggml_type::GgmlType;
use bridge_quant_layout::{
    decode_block_into, layout, quantize_row_q8_k_into, CpuDotBackend, ValidatedQ8KMatrix, Q8_K_BLOCK_BYTES,
    Q8_K_BLOCK_ELEMENTS,
};
use rayon::prelude::*;

use crate::error::Result;
use crate::{KernelError, PackedMatrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceExecutionMode {
    DequantF32,
    LlamaQ8K,
    CpuParallelQ8K,
    CpuParallelAvxVnni,
    CpuParallelAvx512Vnni,
    CudaQ8K,
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
    let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::Scalar)?;
    for (row, destination) in output.iter_mut().enumerate() {
        *destination = prepared.dot_row(row)?;
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
    let prepared = validate_prepared_q8k(matrix, q8, default_cpu_dot_backend())?;
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, destination)| {
            *destination = prepared.dot_row(row)?;
            Ok::<(), KernelError>(())
        })
}

pub fn gemv_cpu_parallel_avx512_vnni_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let q8 = prepare_llama_q8k(matrix, input, output, q8_scratch)?;
    let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::Avx512Vnni)?;
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, destination)| {
            *destination = prepared.dot_row(row)?;
            Ok::<(), KernelError>(())
        })
}

pub fn gemv_cpu_parallel_avx_vnni_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let q8 = prepare_llama_q8k(matrix, input, output, q8_scratch)?;
    let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::AvxVnni)?;
    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, destination)| {
            *destination = prepared.dot_row(row)?;
            Ok::<(), KernelError>(())
        })
}

pub fn gemv_cuda_q8k_into(
    matrix: PackedMatrix<'_>,
    input: &[f32],
    output: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let q8 = prepare_llama_q8k(matrix, input, output, q8_scratch)?;
    bridge_kernels_cuda::packed_q8k_gemv_into(matrix.ty(), matrix.bytes(), q8, matrix.input_width(), output)
        .map_err(cuda_error)?;
    validate_finite_slice("CUDA GEMV output", output)
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
            ReferenceExecutionMode::CpuParallelQ8K
            | ReferenceExecutionMode::CpuParallelAvxVnni
            | ReferenceExecutionMode::CpuParallelAvx512Vnni
            | ReferenceExecutionMode::CudaQ8K => {
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
        ReferenceExecutionMode::CpuParallelAvxVnni => {
            gemv_cpu_parallel_avx_vnni_into(matrix, input, output, q8_scratch)
        }
        ReferenceExecutionMode::CpuParallelAvx512Vnni => {
            gemv_cpu_parallel_avx512_vnni_into(matrix, input, output, q8_scratch)
        }
        ReferenceExecutionMode::CudaQ8K => gemv_cuda_q8k_into(matrix, input, output, q8_scratch),
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
            ReferenceExecutionMode::CpuParallelQ8K
            | ReferenceExecutionMode::CpuParallelAvxVnni
            | ReferenceExecutionMode::CpuParallelAvx512Vnni
            | ReferenceExecutionMode::CudaQ8K => {
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
            let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::Scalar)?;
            for (row, value) in destination.iter_mut().enumerate() {
                *value += scale * prepared.dot_row(row)?;
            }
        }
        ReferenceExecutionMode::CpuParallelQ8K => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            let prepared = validate_prepared_q8k(matrix, q8, default_cpu_dot_backend())?;
            destination
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, value)| {
                    *value += scale * prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CpuParallelAvxVnni => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::AvxVnni)?;
            destination
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, value)| {
                    *value += scale * prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CpuParallelAvx512Vnni => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::Avx512Vnni)?;
            destination
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, value)| {
                    *value += scale * prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CudaQ8K => {
            let q8 = prepare_llama_q8k(matrix, input, destination, q8_scratch)?;
            if decoded_block_scratch.len() < destination.len() {
                return Err(KernelError::ScratchTooSmall {
                    field: "CUDA accumulation output",
                    required: destination.len(),
                    actual: decoded_block_scratch.len(),
                });
            }
            let candidate = &mut decoded_block_scratch[..destination.len()];
            bridge_kernels_cuda::packed_q8k_gemv_into(
                matrix.ty(),
                matrix.bytes(),
                q8,
                matrix.input_width(),
                candidate,
            )
            .map_err(cuda_error)?;
            validate_finite_slice("CUDA GEMV output", candidate)?;
            for (value, &addition) in destination.iter_mut().zip(candidate.iter()) {
                *value += scale * addition;
            }
        }
    }
    Ok(())
}

/// Evaluates two projections of the same input while quantizing that input
/// only once on the packed Q8_K paths. This is the gate/up hot path for
/// SwiGLU experts.
pub fn gemv_pair_into(
    mode: ReferenceExecutionMode,
    matrices: [PackedMatrix<'_>; 2],
    input: &[f32],
    outputs: [&mut [f32]; 2],
    decoded_block_scratch: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let [first, second] = matrices;
    let [first_output, second_output] = outputs;
    let packed_pair = mode != ReferenceExecutionMode::DequantF32
        && first.ty() != GgmlType::F32
        && second.ty() != GgmlType::F32;
    if !packed_pair {
        gemv_into(
            mode,
            first,
            input,
            first_output,
            decoded_block_scratch,
            q8_scratch,
        )?;
        return gemv_into(
            mode,
            second,
            input,
            second_output,
            decoded_block_scratch,
            q8_scratch,
        );
    }

    if mode == ReferenceExecutionMode::CudaQ8K {
        let q8 = prepare_llama_q8k(first, input, first_output, q8_scratch)?;
        validate_dimensions(second, input, second_output)?;
        let required =
            first_output
                .len()
                .checked_add(second_output.len())
                .ok_or(KernelError::ArithmeticOverflow {
                    operation: "CUDA paired GEMV output scratch",
                })?;
        if decoded_block_scratch.len() < required {
            return Err(KernelError::ScratchTooSmall {
                field: "CUDA paired GEMV output",
                required,
                actual: decoded_block_scratch.len(),
            });
        }
        let (first_candidate, remainder) = decoded_block_scratch.split_at_mut(first_output.len());
        let second_candidate = &mut remainder[..second_output.len()];
        bridge_kernels_cuda::packed_q8k_gemv_pair_into(
            [first.ty(), second.ty()],
            [first.bytes(), second.bytes()],
            q8,
            first.input_width(),
            [first_candidate, second_candidate],
        )
        .map_err(cuda_error)?;
        validate_finite_slice("CUDA paired GEMV first output", first_candidate)?;
        validate_finite_slice("CUDA paired GEMV second output", second_candidate)?;
        first_output.copy_from_slice(first_candidate);
        second_output.copy_from_slice(second_candidate);
        return Ok(());
    }

    let q8 = prepare_llama_q8k(first, input, first_output, q8_scratch)?;
    validate_dimensions(second, input, second_output)?;
    let backend = dot_backend_for_mode(mode)?;
    let first_prepared = validate_prepared_q8k(first, q8, backend)?;
    let second_prepared = validate_prepared_q8k(second, q8, backend)?;
    compute_prepared_q8k_into(mode, first_prepared, first_output)?;
    compute_prepared_q8k_into(mode, second_prepared, second_output)
}

/// Evaluates three same-input projections. CUDA submits all three matrices
/// under one validation, transfer, and synchronization boundary.
pub fn gemv_triplet_into(
    mode: ReferenceExecutionMode,
    matrices: [PackedMatrix<'_>; 3],
    input: &[f32],
    outputs: [&mut [f32]; 3],
    decoded_block_scratch: &mut [f32],
    q8_scratch: &mut [u8],
) -> Result<()> {
    let [first, second, third] = matrices;
    let [first_output, second_output, third_output] = outputs;
    let packed_triplet = mode == ReferenceExecutionMode::CudaQ8K
        && first.ty() != GgmlType::F32
        && second.ty() != GgmlType::F32
        && third.ty() != GgmlType::F32;
    if !packed_triplet {
        gemv_pair_into(
            mode,
            [first, second],
            input,
            [first_output, second_output],
            decoded_block_scratch,
            q8_scratch,
        )?;
        return gemv_into(
            mode,
            third,
            input,
            third_output,
            decoded_block_scratch,
            q8_scratch,
        );
    }

    let q8 = prepare_llama_q8k(first, input, first_output, q8_scratch)?;
    validate_dimensions(second, input, second_output)?;
    validate_dimensions(third, input, third_output)?;
    let required = first_output
        .len()
        .checked_add(second_output.len())
        .and_then(|value| value.checked_add(third_output.len()))
        .ok_or(KernelError::ArithmeticOverflow {
            operation: "CUDA triplet GEMV output scratch",
        })?;
    if decoded_block_scratch.len() < required {
        return Err(KernelError::ScratchTooSmall {
            field: "CUDA triplet GEMV output",
            required,
            actual: decoded_block_scratch.len(),
        });
    }
    let items = [
        bridge_kernels_cuda::CudaPackedQ8KBatchItem {
            weight_type: first.ty(),
            weights: first.bytes(),
            q8,
            logical_elements: first.input_width(),
            rows: first.output_width(),
        },
        bridge_kernels_cuda::CudaPackedQ8KBatchItem {
            weight_type: second.ty(),
            weights: second.bytes(),
            q8,
            logical_elements: second.input_width(),
            rows: second.output_width(),
        },
        bridge_kernels_cuda::CudaPackedQ8KBatchItem {
            weight_type: third.ty(),
            weights: third.bytes(),
            q8,
            logical_elements: third.input_width(),
            rows: third.output_width(),
        },
    ];
    bridge_kernels_cuda::packed_q8k_gemv_batch_into(&items, &mut decoded_block_scratch[..required])
        .map_err(cuda_error)?;
    let (first_candidate, remainder) = decoded_block_scratch.split_at_mut(first_output.len());
    let (second_candidate, remainder) = remainder.split_at_mut(second_output.len());
    let third_candidate = &mut remainder[..third_output.len()];
    validate_finite_slice("CUDA triplet GEMV first output", first_candidate)?;
    validate_finite_slice("CUDA triplet GEMV second output", second_candidate)?;
    validate_finite_slice("CUDA triplet GEMV third output", third_candidate)?;
    first_output.copy_from_slice(first_candidate);
    second_output.copy_from_slice(second_candidate);
    third_output.copy_from_slice(third_candidate);
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
    Ok(q8)
}

fn default_cpu_dot_backend() -> CpuDotBackend {
    if CpuDotBackend::Avx2.available() {
        CpuDotBackend::Avx2
    } else {
        CpuDotBackend::Scalar
    }
}

fn dot_backend_for_mode(mode: ReferenceExecutionMode) -> Result<CpuDotBackend> {
    match mode {
        ReferenceExecutionMode::LlamaQ8K => Ok(CpuDotBackend::Scalar),
        ReferenceExecutionMode::CpuParallelQ8K => Ok(default_cpu_dot_backend()),
        ReferenceExecutionMode::CpuParallelAvxVnni => Ok(CpuDotBackend::AvxVnni),
        ReferenceExecutionMode::CpuParallelAvx512Vnni => Ok(CpuDotBackend::Avx512Vnni),
        ReferenceExecutionMode::CudaQ8K => Err(KernelError::InvalidParameter {
            field: "prepared Q8_K execution mode",
            reason: "CUDA uses its own validated packed executor",
        }),
        ReferenceExecutionMode::DequantF32 => Err(KernelError::InvalidParameter {
            field: "prepared Q8_K execution mode",
            reason: "must be a packed Q8_K mode",
        }),
    }
}

fn validate_prepared_q8k<'a>(
    matrix: PackedMatrix<'a>,
    q8: &'a [u8],
    backend: CpuDotBackend,
) -> Result<ValidatedQ8KMatrix<'a>> {
    Ok(ValidatedQ8KMatrix::new(
        matrix.ty(),
        matrix.bytes(),
        q8,
        matrix.input_width(),
        matrix.output_width(),
        backend,
    )?)
}

fn compute_prepared_q8k_into(
    mode: ReferenceExecutionMode,
    prepared: ValidatedQ8KMatrix<'_>,
    output: &mut [f32],
) -> Result<()> {
    match mode {
        ReferenceExecutionMode::LlamaQ8K => {
            for (row, destination) in output.iter_mut().enumerate() {
                *destination = prepared.dot_row(row)?;
            }
        }
        ReferenceExecutionMode::CpuParallelQ8K => {
            output
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, destination)| {
                    *destination = prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CpuParallelAvxVnni => {
            output
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, destination)| {
                    *destination = prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CpuParallelAvx512Vnni => {
            output
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, destination)| {
                    *destination = prepared.dot_row(row)?;
                    Ok::<(), KernelError>(())
                })?;
        }
        ReferenceExecutionMode::CudaQ8K => {
            return Err(KernelError::InvalidParameter {
                field: "prepared Q8_K execution mode",
                reason: "CUDA uses its own validated packed executor",
            });
        }
        ReferenceExecutionMode::DequantF32 => {
            return Err(KernelError::InvalidParameter {
                field: "prepared Q8_K execution mode",
                reason: "must be a packed Q8_K mode",
            });
        }
    }
    Ok(())
}

fn cuda_error(error: bridge_kernels_cuda::CudaRuntimeError) -> KernelError {
    KernelError::Cuda {
        message: error.to_string(),
    }
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
