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

/// Calculates the scratch-buffer size required to quantize an input row into Q8_K format.
///
/// # Errors
///
/// Returns an error if `input_width` is zero, is not a multiple of the Q8_K block size,
/// or if the required byte count overflows `usize`.
///
/// # Examples
///
/// ```
/// let bytes = required_q8_k_bytes(Q8_K_BLOCK_ELEMENTS).unwrap();
/// assert_eq!(bytes, Q8_K_BLOCK_BYTES);
/// ```
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

/// Computes a matrix–vector product using scalar Q8_K quantization.

///

/// `q8_scratch` must provide enough storage for the quantized input row.

/// The output slice must have one element per matrix row.

///

/// # Examples

///

/// ```no_run

/// # let matrix: PackedMatrix<'_> = unimplemented!();

/// # let input = vec![1.0_f32; 32];

/// # let mut output = vec![0.0_f32; 4];

/// # let mut q8_scratch = vec![0_u8; 256];

/// gemv_llama_q8k_into(matrix, &input, &mut output, &mut q8_scratch)?;

/// # Ok::<(), KernelError>(())

/// ```

///

/// # Errors

///

/// Returns an error if the dimensions, scratch buffer, matrix type, or

/// prepared Q8_K representation is invalid.
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

/// Computes a packed Q8_K matrix-vector product using the selected CPU backend in parallel.
///
/// # Examples
///
/// ```no_run
/// # let matrix = todo!();
/// # let input = todo!();
/// # let mut output = todo!();
/// # let mut q8_scratch = todo!();
/// gemv_cpu_parallel_q8k_into(matrix, input, &mut output, &mut q8_scratch)?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// `q8_scratch` must provide sufficient space for the input's Q8_K representation.
pub fn gemv_cpu_parallel_q8k_into(
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

/// Computes matrix-vector products in parallel using AVX512 VNNI Q8_K execution.

///

/// # Examples

///

/// ```

/// # let matrix: PackedMatrix<'_> = todo!();

/// # let input = vec![0.0; matrix.input_width()];

/// # let mut output = vec![0.0; matrix.output_width()];

/// # let mut q8_scratch = vec![0; required_q8_k_bytes(matrix.input_width()).unwrap()];

/// gemv_cpu_parallel_avx512_vnni_into(matrix, &input, &mut output, &mut q8_scratch)?;

/// # Ok::<(), KernelError>(())

/// ```

///

/// # Errors

///

/// Returns an error if the matrix, dimensions, scratch space, or AVX512 VNNI

/// execution parameters are invalid.
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

/// Computes a packed Q8_K matrix-vector product using parallel AVX VNNI execution.
///
/// # Parameters
///
/// * `q8_scratch` — Scratch storage used to hold the quantized input row.
///
/// # Examples
///
/// ```
/// # fn example(
/// #     matrix: PackedMatrix<'_>,
/// #     input: &[f32],
/// #     output: &mut [f32],
/// #     q8_scratch: &mut [u8],
/// # ) -> Result<()> {
/// gemv_cpu_parallel_avx_vnni_into(matrix, input, output, q8_scratch)?;
/// # Ok(())
/// # }
/// ```
pub fn gemv_cpu_parallel_avx_vnni_into(
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

/// Computes a matrix-vector product using the CUDA Q8_K execution path.
///
/// The input is quantized into `q8_scratch`, and `output` is populated with the
/// resulting values. The matrix must use a supported Q8_K-compatible packed
/// type, and the scratch buffer must have sufficient capacity.
///
/// # Examples
///
/// ```ignore
/// let mut output = vec![0.0; matrix.output_width()];
/// let mut q8_scratch = vec![0; required_q8_k_bytes(matrix.input_width())?];
///
/// gemv_cuda_q8k_into(matrix, &input, &mut output, &mut q8_scratch)?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// # Errors
///
/// Returns an error if dimensions or scratch capacity are invalid, the matrix
/// type is unsupported, CUDA execution fails, or the output contains a
/// non-finite value.
pub fn gemv_cuda_q8k_into(
matrix: PackedMatrix<'_>,
input: &[f32],
output: &mut [f32],
q8_scratch: &mut [u8],
) -> Result<()> {
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

/// Computes F32 matrix-vector products in parallel across output rows.
///
/// # Examples
///
/// ```ignore
/// gemv_cpu_parallel_f32_into(matrix, input, &mut output, &mut scratch)?;
/// # Ok::<(), KernelError>(())
/// ```
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

/// Computes a matrix-vector product using the selected reference execution mode.
///
/// # Examples
///
/// ```
/// # let mode = ReferenceExecutionMode::DequantF32;
/// # let matrix = matrix;
/// # let input = input;
/// # let mut output = output;
/// # let mut decoded_block_scratch = decoded_block_scratch;
/// # let mut q8_scratch = q8_scratch;
/// gemv_into(
///     mode,
///     matrix,
///     input,
///     &mut output,
///     &mut decoded_block_scratch,
///     &mut q8_scratch,
/// )?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// `decoded_block_scratch` provides workspace for dequantization, while
/// `q8_scratch` provides workspace for Q8_K input quantization.
pub fn gemv_into(
mode: ReferenceExecutionMode,
matrix: PackedMatrix<'_>,
input: &[f32],
output: &mut [f32],
decoded_block_scratch: &mut [f32],
q8_scratch: &mut [u8],
) -> Result<()> {
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

/// Adds the matrix–input product, multiplied by `scale`, to `destination`.
///
/// The destination and scale must contain finite values. The selected execution
/// mode determines the computation backend.
///
/// # Examples
///
/// ```no_run
/// # let mode = todo!();
/// # let matrix = todo!();
/// # let input = todo!();
/// # let mut destination = todo!();
/// # let mut decoded_block_scratch = todo!();
/// # let mut q8_scratch = todo!();
/// gemv_accumulate_scaled_into(
///     mode,
///     matrix,
///     input,
///     &mut destination,
///     1.0,
///     &mut decoded_block_scratch,
///     &mut q8_scratch,
/// )?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// # Parameters
///
/// `scale` is multiplied by each computed matrix–input product before it is
/// added to the corresponding destination element.
///
/// # Errors
///
/// Returns an error when dimensions, scratch capacity, values, or the selected
/// execution backend are invalid.
pub fn gemv_accumulate_scaled_into(
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

/// Evaluates two projections of the same input, sharing Q8_K quantization on packed execution paths.
///
/// This is the paired projection path used for gate and up projections in SwiGLU experts.
///
/// # Examples
///
/// ```no_run
/// # // Create `matrices`, `input`, `outputs`, and scratch buffers for the model.
/// gemv_pair_into(
///     mode,
///     matrices,
///     input,
///     outputs,
///     &mut decoded_block_scratch,
///     &mut q8_scratch,
/// )?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// # Errors
///
/// Returns an error when dimensions or scratch capacity are invalid, when the
/// selected execution mode is unsupported, or when execution produces invalid
/// values.
pub fn gemv_pair_into(
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

/// Evaluates three projections that share the same input.
///
/// Uses a CUDA batch execution path for three packed matrices when available;
/// otherwise, evaluates the projections sequentially using the selected mode.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let matrices = [todo!(), todo!(), todo!()];
/// let mut first = Vec::new();
/// let mut second = Vec::new();
/// let mut third = Vec::new();
///
/// gemv_triplet_into(
///     ReferenceExecutionMode::DequantF32,
///     matrices,
///     &[],
///     [&mut first, &mut second, &mut third],
///     &mut [],
///     &mut [],
/// )?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if dimensions, scratch capacity, values, or the selected
/// execution mode are invalid, or if CUDA execution fails.
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

/// Validates that every value in a slice is finite.
///
/// # Examples
///
/// ```
/// let values = [1.0, -2.5, 0.0];
/// assert!(validate_finite_slice("values", &values).is_ok());
/// ```
///
/// # Returns
///
/// `Ok(())` when all values are finite; otherwise, an error identifying the first non-finite value.
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

/// Prepares a quantized Q8_K representation of the input for a supported packed matrix.
///
/// # Errors
///
/// Returns an error if the dimensions are invalid, the matrix type is unsupported,
/// the scratch buffer is too small, or quantization fails.
///
/// # Examples
///
/// ```ignore
/// let q8 = prepare_llama_q8k(matrix, input, output, &mut q8_scratch)?;
/// assert!(!q8.is_empty());
/// # Ok::<(), KernelError>(())
/// ```
///
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

/// Selects the fastest available CPU backend supported for dot-product execution.
///
/// # Examples
///
/// ```
/// let backend = default_cpu_dot_backend();
/// assert!(matches!(backend, CpuDotBackend::Avx2 | CpuDotBackend::Scalar));
/// ```
fn default_cpu_dot_backend() -> CpuDotBackend {
    if CpuDotBackend::Avx2.available() {
        CpuDotBackend::Avx2
    } else {
        CpuDotBackend::Scalar
    }
}

/// Selects the CPU dot-product backend for a packed Q8_K execution mode.
///
/// # Errors
///
/// Returns an error for execution modes that use dequantized or CUDA execution.
///
/// # Examples
///
/// ```
/// let backend = dot_backend_for_mode(ReferenceExecutionMode::LlamaQ8K)?;
/// assert_eq!(backend, CpuDotBackend::Scalar);
/// # Ok::<(), KernelError>(())
/// ```
fn dot_backend_for_mode(mode: ReferenceExecutionMode) -> Result<CpuDotBackend> {
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

/// Validates a packed Q8_K matrix and prepares it for the selected CPU dot-product backend.
///
/// # Examples
///
/// ```
/// let prepared = validate_prepared_q8k(matrix, q8, CpuDotBackend::Scalar)?;
/// # Ok::<(), KernelError>(())
/// ```
///
/// # Errors
///
/// Returns an error if the matrix metadata, packed weights, Q8_K input, or selected
/// backend is invalid.
///
/// `backend` selects the CPU implementation used for dot products.
fn validate_prepared_q8k...
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

/// Computes output rows from a validated packed Q8_K matrix using the selected CPU execution mode.
///
/// # Errors
///
/// Returns an error if the selected mode is not a supported packed Q8_K CPU mode or if
/// computing any output row fails.
///
/// # Examples
///
/// ```no_run
/// # let prepared: ValidatedQ8KMatrix<'_> = todo!();
/// let mut output = vec![0.0; 4];
///
/// compute_prepared_q8k_into(
///     ReferenceExecutionMode::CpuParallelQ8K,
///     prepared,
///     &mut output,
/// )?;
/// # Ok::<(), KernelError>(())
/// ```
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

/// Converts a CUDA runtime error into a [`KernelError::Cuda`].
///
/// # Examples
///
/// ```ignore
/// let kernel_error = cuda_error(cuda_runtime_error);
/// assert!(matches!(kernel_error, KernelError::Cuda { .. }));
/// ```
///
/// The original error message is preserved in the returned kernel error.
fn cuda_error(error: bridge_kernels_cuda::CudaRuntimeError) -> KernelError {
    KernelError::Cuda {
        message: error.to_string(),
    }
}

/// Computes each output row by dequantizing the corresponding packed matrix weights.
///
/// The output slice is overwritten with the resulting dot products. The scratch
/// buffer is reused while processing individual weight blocks.
///
/// # Examples
///
/// ```no_run
/// # let matrix: PackedMatrix<'_> = todo!();
/// # let input: &[f32] = &[];
/// # let mut output = vec![0.0; matrix.output_width()];
/// # let mut decoded_block_scratch = vec![0.0; 1];
/// compute_dequant_into(matrix, input, &mut output, &mut decoded_block_scratch)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Returns
///
/// `Ok(())` after populating the output, or an error if dequantization fails.
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
