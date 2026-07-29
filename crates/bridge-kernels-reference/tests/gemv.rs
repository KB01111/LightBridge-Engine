use bridge_core::ggml_type::GgmlType;
use bridge_kernels_reference::{
    gemv_accumulate_scaled_into, gemv_dequant_f32_into, gemv_into, gemv_llama_q8k_into, required_q8_k_bytes,
    KernelError, PackedMatrix, PayloadEndian, ReferenceExecutionMode,
};
use bridge_quant_layout::{decode_row_into, vec_dot_q8_k};

const Q4: &[u8; 432] = include_bytes!("../../bridge-quant-layout/tests/fixtures/decode-q4-k.input.bin");
const Q5: &[u8; 528] = include_bytes!("../../bridge-quant-layout/tests/fixtures/decode-q5-k.input.bin");
const IQ2: &[u8; 246] = include_bytes!("../../bridge-quant-layout/tests/fixtures/decode-iq2-s.input.bin");
const IQ3: &[u8; 330] = include_bytes!("../../bridge-quant-layout/tests/fixtures/decode-iq3-s.input.bin");
const ACTIVATIONS: &[u8; 3072] =
    include_bytes!("../../bridge-quant-layout/tests/fixtures/q8-k-activations.input-f32le.bin");
const Q8: &[u8; 876] =
    include_bytes!("../../bridge-quant-layout/tests/fixtures/q8-k-activations.output-q8-k.bin");

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn activations() -> Vec<f32> {
    ACTIVATIONS
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn matrix<'a>(ty: GgmlType, input_width: usize, output_width: usize, bytes: &'a [u8]) -> PackedMatrix<'a> {
    PackedMatrix::from_parts(ty, PayloadEndian::Little, input_width, output_width, bytes).unwrap()
}

#[test]
fn rectangular_f32_gemv_uses_ggml_input_output_orientation() {
    let weights = f32_bytes(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]);
    let matrix = matrix(GgmlType::F32, 3, 2, &weights);
    let mut output = [f32::NAN; 2];
    let mut decoded = [0.0; 256];

    gemv_dequant_f32_into(matrix, &[2.0, -1.0, 0.5], &mut output, &mut decoded).unwrap();
    assert_eq!(output, [1.5, -0.5]);
}

#[test]
fn q8_k_modes_execute_f32_router_matrices_with_bit_exact_parallel_rows() {
    let weights = f32_bytes(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0, 0.25, -0.75, 2.5, 3.0, -2.0, 1.0]);
    let matrix = matrix(GgmlType::F32, 3, 4, &weights);
    let input = [2.0, -1.0, 0.5];
    let mut scalar = [f32::NAN; 4];
    let mut parallel = [f32::NAN; 4];
    let mut decoded = [0.0; 256];

    gemv_into(
        ReferenceExecutionMode::LlamaQ8K,
        matrix,
        &input,
        &mut scalar,
        &mut decoded,
        &mut [],
    )
    .unwrap();
    gemv_into(
        ReferenceExecutionMode::CpuParallelQ8K,
        matrix,
        &input,
        &mut parallel,
        &mut decoded,
        &mut [],
    )
    .unwrap();

    assert_eq!(scalar, [1.5, -0.5, 2.5, 8.5]);
    assert_eq!(parallel.map(f32::to_bits), scalar.map(f32::to_bits));
}

#[test]
fn dequant_mode_executes_all_five_selected_physical_types() {
    let input = activations();
    let cases: [(GgmlType, &[u8]); 4] = [
        (GgmlType::Q4_K, Q4),
        (GgmlType::Q5_K, Q5),
        (GgmlType::IQ2_S, IQ2),
        (GgmlType::IQ3_S, IQ3),
    ];

    for (ty, weights) in cases {
        let matrix = matrix(ty, 768, 1, weights);
        let mut decoded_row = vec![0.0; 768];
        decode_row_into(ty, weights, 768, &mut decoded_row).unwrap();
        let mut expected = 0.0_f32;
        for (&weight, &value) in decoded_row.iter().zip(&input) {
            expected += weight * value;
        }

        let mut output = [f32::NAN];
        let mut decoded = [0.0; 256];
        gemv_into(
            ReferenceExecutionMode::DequantF32,
            matrix,
            &input,
            &mut output,
            &mut decoded,
            &mut [],
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), expected.to_bits(), "{ty:?}");
    }
}

#[test]
fn llama_q8_k_mode_executes_every_selected_quantized_type() {
    let input = activations();
    let cases: [(GgmlType, &[u8]); 4] = [
        (GgmlType::Q4_K, Q4),
        (GgmlType::Q5_K, Q5),
        (GgmlType::IQ2_S, IQ2),
        (GgmlType::IQ3_S, IQ3),
    ];

    for (ty, weights) in cases {
        let matrix = matrix(ty, 768, 1, weights);
        let expected = vec_dot_q8_k(ty, weights, Q8, 768).unwrap();
        let mut output = [f32::NAN];
        let mut q8 = vec![0_u8; required_q8_k_bytes(768).unwrap()];
        gemv_llama_q8k_into(matrix, &input, &mut output, &mut q8).unwrap();
        assert_eq!(output[0].to_bits(), expected.to_bits(), "{ty:?}");
        assert_eq!(q8, Q8);
    }
}

#[test]
fn cpu_parallel_q8_k_is_bit_exact_for_every_selected_quantized_type() {
    let input = activations();
    let cases: [(GgmlType, &[u8]); 4] = [
        (GgmlType::Q4_K, Q4),
        (GgmlType::Q5_K, Q5),
        (GgmlType::IQ2_S, IQ2),
        (GgmlType::IQ3_S, IQ3),
    ];

    for (ty, row) in cases {
        let weights = row.repeat(8);
        let matrix = matrix(ty, 768, 8, &weights);
        let mut scalar = [f32::NAN; 8];
        let mut parallel = [f32::NAN; 8];
        let mut scalar_q8 = vec![0_u8; required_q8_k_bytes(768).unwrap()];
        let mut parallel_q8 = vec![0_u8; required_q8_k_bytes(768).unwrap()];
        gemv_into(
            ReferenceExecutionMode::LlamaQ8K,
            matrix,
            &input,
            &mut scalar,
            &mut [],
            &mut scalar_q8,
        )
        .unwrap();
        gemv_into(
            ReferenceExecutionMode::CpuParallelQ8K,
            matrix,
            &input,
            &mut parallel,
            &mut [],
            &mut parallel_q8,
        )
        .unwrap();
        assert_eq!(parallel.map(f32::to_bits), scalar.map(f32::to_bits), "{ty:?}");
        assert_eq!(parallel_q8, scalar_q8);
    }
}

#[test]
fn scaled_accumulation_preserves_initialized_destinations() {
    let weights = f32_bytes(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]);
    let matrix = matrix(GgmlType::F32, 3, 2, &weights);
    let mut destination = [10.0, -4.0];
    let mut decoded = [0.0; 256];
    gemv_accumulate_scaled_into(
        ReferenceExecutionMode::DequantF32,
        matrix,
        &[2.0, -1.0, 0.5],
        &mut destination,
        0.5,
        &mut decoded,
        &mut [],
    )
    .unwrap();
    assert_eq!(destination, [10.75, -4.25]);
}

#[test]
fn all_validation_failures_leave_output_unchanged() {
    let weights = f32_bytes(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]);
    let f32_matrix = matrix(GgmlType::F32, 3, 2, &weights);
    let sentinel = [f32::from_bits(0x7fc0_00a5), f32::from_bits(0x8000_0000)];

    let mut output = sentinel;
    assert!(matches!(
        gemv_dequant_f32_into(f32_matrix, &[1.0, 2.0], &mut output, &mut [0.0; 256]),
        Err(KernelError::DimensionMismatch {
            field: "GEMV input",
            expected: 3,
            actual: 2,
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut non_finite_weights = weights.clone();
    non_finite_weights[4..8].copy_from_slice(&f32::INFINITY.to_le_bytes());
    let non_finite_matrix = matrix(GgmlType::F32, 3, 2, &non_finite_weights);
    let mut output = sentinel;
    assert!(matches!(
        gemv_into(
            ReferenceExecutionMode::CpuParallelQ8K,
            non_finite_matrix,
            &[1.0, 2.0, 3.0],
            &mut output,
            &mut [0.0; 256],
            &mut [],
        ),
        Err(KernelError::NonFiniteValue {
            field: "GEMV weights",
            ..
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut output = sentinel;
    assert!(matches!(
        gemv_dequant_f32_into(f32_matrix, &[1.0, 2.0, f32::NAN], &mut output, &mut [0.0; 256]),
        Err(KernelError::NonFiniteValue {
            field: "GEMV input",
            index: 2,
            ..
        })
    ));
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    let mut bad_q4 = vec![0_u8; 288];
    bad_q4[144..146].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let bad_matrix = matrix(GgmlType::Q4_K, 256, 2, &bad_q4);
    let mut output = sentinel;
    assert!(gemv_dequant_f32_into(bad_matrix, &[0.0; 256], &mut output, &mut [0.0; 256]).is_err());
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));

    let q4_matrix = matrix(GgmlType::Q4_K, 256, 1, &Q4[..144]);
    let mut one_output = [f32::from_bits(0x7fc0_00a5)];
    assert!(matches!(
        gemv_llama_q8k_into(q4_matrix, &[0.0; 256], &mut one_output, &mut [0_u8; 291]),
        Err(KernelError::ScratchTooSmall {
            field: "Q8_K row",
            required: 292,
            actual: 291,
        })
    ));
    assert_eq!(one_output[0].to_bits(), 0x7fc0_00a5);

    let mut output = sentinel;
    assert_eq!(
        gemv_llama_q8k_into(f32_matrix, &[1.0, 2.0, 3.0], &mut output, &mut []),
        Err(KernelError::UnsupportedType { ty: GgmlType::F32 })
    );
    assert_eq!(output.map(f32::to_bits), sentinel.map(f32::to_bits));
}
