use bridge_quant_layout::{
    quantize_row_q8_k_into, vec_dot_q8_k, GgmlType, QuantError, Q8_K_BLOCK_BYTES, Q8_K_BLOCK_ELEMENTS,
};

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
