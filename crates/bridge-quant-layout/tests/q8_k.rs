use bridge_quant_layout::{
    quantize_row_q8_k_into, vec_dot_q8_k, vec_dot_q8_k_cpu_backend, CpuDotBackend, GgmlType, QuantError,
    ValidatedQ8KMatrix, Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS,
};

const Q4: &[u8; 432] = include_bytes!("fixtures/decode-q4-k.input.bin");
const Q5: &[u8; 528] = include_bytes!("fixtures/decode-q5-k.input.bin");
const IQ2: &[u8; 246] = include_bytes!("fixtures/decode-iq2-s.input.bin");
const IQ3: &[u8; 330] = include_bytes!("fixtures/decode-iq3-s.input.bin");
const Q8: &[u8; 876] = include_bytes!("fixtures/q8-k-activations.output-q8-k.bin");

fn sentinel_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}

#[test]
fn q8_k_abi_and_shape_validation_are_exact_and_atomic() {
    assert_eq!(Q8_K_BLOCK_ELEMENTS, 256);
    assert_eq!(Q8_K_BLOCK_BYTES, 292);

    let mut output = sentinel_bytes(Q8_K_BLOCK_BYTES);
    let before = output.clone();
    assert_eq!(
        quantize_row_q8_k_into(&[], &mut output),
        Err(QuantError::ZeroLogicalElements)
    );
    assert_eq!(output, before);

    assert_eq!(
        quantize_row_q8_k_into(&[0.0; 255], &mut output),
        Err(QuantError::LogicalElementsNotDivisible {
            ty: GgmlType::Q8_K,
            logical_elements: 255,
            block_elements: 256,
        })
    );
    assert_eq!(output, before);

    for actual in 0..Q8_K_BLOCK_BYTES {
        let mut truncated = sentinel_bytes(actual);
        let before = truncated.clone();
        assert_eq!(
            quantize_row_q8_k_into(&[0.0; Q8_K_BLOCK_ELEMENTS], &mut truncated),
            Err(QuantError::EncodedLengthMismatch {
                ty: GgmlType::Q8_K,
                expected: Q8_K_BLOCK_BYTES,
                actual,
            })
        );
        assert_eq!(truncated, before);
    }

    let actual = Q8_K_BLOCK_BYTES + 1;
    let mut overlong = sentinel_bytes(actual);
    let before = overlong.clone();
    assert_eq!(
        quantize_row_q8_k_into(&[0.0; Q8_K_BLOCK_ELEMENTS], &mut overlong),
        Err(QuantError::EncodedLengthMismatch {
            ty: GgmlType::Q8_K,
            expected: Q8_K_BLOCK_BYTES,
            actual,
        })
    );
    assert_eq!(overlong, before);
}

#[test]
fn q8_k_rejects_every_nonfinite_activation_before_mutation() {
    let nonfinite = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffa1_2345),
    ];

    for index in [0, 127, 255, 256, 511] {
        for value in nonfinite {
            let mut input = vec![0.0; 512];
            input[index] = value;
            let mut output = sentinel_bytes(Q8_K_BLOCK_BYTES * 2);
            let before = output.clone();
            assert_eq!(
                quantize_row_q8_k_into(&input, &mut output),
                Err(QuantError::NonFiniteActivation {
                    index,
                    bits: value.to_bits(),
                })
            );
            assert_eq!(output, before);
        }
    }
}

#[test]
fn q8_k_zero_blocks_are_fully_deterministic() {
    let mut output = vec![0xa5; Q8_K_BLOCK_BYTES * 2];
    quantize_row_q8_k_into(&[0.0; Q8_K_BLOCK_ELEMENTS * 2], &mut output).unwrap();
    assert!(output.iter().all(|&byte| byte == 0));
}

#[test]
fn dot_validation_rejects_bad_types_lengths_and_scales() {
    let q4 = vec![0_u8; 144];
    let q8 = vec![0_u8; Q8_K_BLOCK_BYTES];

    assert_eq!(
        vec_dot_q8_k(GgmlType::F32, &[], &[], 256),
        Err(QuantError::UnsupportedType { ty: GgmlType::F32 })
    );
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &q4, &q8, 0),
        Err(QuantError::ZeroLogicalElements)
    );
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &q4, &q8, 1),
        Err(QuantError::LogicalElementsNotDivisible {
            ty: GgmlType::Q4_K,
            logical_elements: 1,
            block_elements: 256,
        })
    );
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &q4[..143], &q8, 256),
        Err(QuantError::EncodedLengthMismatch {
            ty: GgmlType::Q4_K,
            expected: 144,
            actual: 143,
        })
    );
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &q4, &q8[..291], 256),
        Err(QuantError::EncodedLengthMismatch {
            ty: GgmlType::Q8_K,
            expected: 292,
            actual: 291,
        })
    );

    let mut nonfinite_weight = q4.clone();
    nonfinite_weight[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &nonfinite_weight, &q8, 256),
        Err(QuantError::NonFiniteScale {
            ty: GgmlType::Q4_K,
            block_index: 0,
            field: "d",
            bits: 0x7c00,
        })
    );

    let mut nonfinite_q8 = q8;
    let bits = f32::NAN.to_bits();
    nonfinite_q8[0..4].copy_from_slice(&bits.to_le_bytes());
    assert_eq!(
        vec_dot_q8_k(GgmlType::Q4_K, &q4, &nonfinite_q8, 256),
        Err(QuantError::NonFiniteQ8Scale { block_index: 0, bits })
    );
}

#[test]
fn every_available_cpu_backend_is_bit_exact_to_the_scalar_oracle() {
    let cases: [(GgmlType, &[u8]); 4] = [
        (GgmlType::Q4_K, Q4),
        (GgmlType::Q5_K, Q5),
        (GgmlType::IQ2_S, IQ2),
        (GgmlType::IQ3_S, IQ3),
    ];
    for (ty, weights) in cases {
        let expected = vec_dot_q8_k(ty, weights, Q8, 768).unwrap();
        for backend in [
            CpuDotBackend::Scalar,
            CpuDotBackend::Avx2,
            CpuDotBackend::AvxVnni,
            CpuDotBackend::Avx512Vnni,
        ] {
            if backend.available() {
                let actual = vec_dot_q8_k_cpu_backend(ty, weights, Q8, 768, backend).unwrap();
                assert_eq!(actual.to_bits(), expected.to_bits(), "{ty:?} {backend:?}");
            }
        }
    }
}

#[test]
fn validated_matrix_reuses_one_complete_validation_across_rows() {
    let rows = 3;
    let mut weights = Vec::new();
    for row in 0..rows {
        let mut row_data = Q4.to_vec();
        for byte in &mut row_data[12..] {
            *byte = byte.wrapping_add((row as u8).wrapping_mul(17));
        }
        weights.extend_from_slice(&row_data);
    }
    let expected: Vec<f32> = (0..rows)
        .map(|row| {
            let row_start = row * Q4.len();
            vec_dot_q8_k(GgmlType::Q4_K, &weights[row_start..row_start + Q4.len()], Q8, 768).unwrap()
        })
        .collect();

    for backend in [
        CpuDotBackend::Scalar,
        CpuDotBackend::Avx2,
        CpuDotBackend::AvxVnni,
        CpuDotBackend::Avx512Vnni,
    ] {
        if !backend.available() {
            continue;
        }
        let matrix = ValidatedQ8KMatrix::new(GgmlType::Q4_K, &weights, Q8, 768, rows, backend).unwrap();
        assert_eq!(matrix.output_rows(), rows);
        assert_eq!(matrix.backend(), backend);
        for row in 0..rows {
            assert_eq!(matrix.dot_row(row).unwrap().to_bits(), expected[row].to_bits());
        }
        assert_eq!(
            matrix.dot_row(rows),
            Err(QuantError::MatrixRowOutOfRange { row: rows, rows })
        );
    }
}
