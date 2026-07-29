use bridge_quant_layout::{
    decode_block_into, decode_f32_block_into, decode_iq2_s_block_into, decode_iq3_s_block_into,
    decode_q4_k_block_into, decode_q5_k_block_into, decode_row_into, layout, GgmlType, QuantError,
    QuantLayout,
};

const SENTINEL_BITS: [u32; 4] = [0x7fc0_00a5, 0xff80_00b6, 0x8000_0000, 0x3f12_3456];
type DecodeBlock = fn(&[u8], &mut [f32]) -> Result<(), QuantError>;

fn sentinel(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| f32::from_bits(SENTINEL_BITS[index % SENTINEL_BITS.len()]))
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn valid_block(ty: GgmlType) -> Vec<u8> {
    let size = match ty {
        GgmlType::F32 => 4,
        GgmlType::Q4_K => 144,
        GgmlType::Q5_K => 176,
        GgmlType::IQ2_S => 82,
        GgmlType::IQ3_S => 110,
        _ => panic!("test helper only accepts reference decoder types"),
    };
    vec![0; size]
}

fn unit_scale_q4(d_bits: u16, dmin_bits: u16) -> Vec<u8> {
    let mut block = vec![0_u8; 144];
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
    block[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    block
}

fn unit_scale_q5() -> Vec<u8> {
    let mut block = vec![0_u8; 176];
    block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    block
}

fn assert_unchanged(before: &[u32], output: &[f32]) {
    assert_eq!(bits(output), before);
}

#[test]
fn supported_layouts_are_exact_and_match_bridge_core() {
    let cases = [
        (GgmlType::F32, 0_u32, 1_usize, 4_usize),
        (GgmlType::Q4_K, 12, 256, 144),
        (GgmlType::Q5_K, 13, 256, 176),
        (GgmlType::IQ3_S, 21, 256, 110),
        (GgmlType::IQ2_S, 22, 256, 82),
    ];

    for (ty, id, block_elements, block_bytes) in cases {
        let expected = QuantLayout {
            ty,
            block_elements,
            block_bytes,
        };
        assert_eq!(ty.discriminant(), id);
        assert_eq!(layout(ty), Ok(expected));
        assert_eq!(usize::try_from(ty.block_size()).unwrap(), block_elements);
        assert_eq!(usize::try_from(ty.type_size()).unwrap(), block_bytes);
        assert_eq!(
            usize::try_from(ty.row_size(ty.block_size()).unwrap()).unwrap(),
            block_bytes
        );
    }
}

#[test]
fn every_non_reference_decoder_type_is_rejected_without_mutation() {
    let mut rejected = 0;
    for &ty in GgmlType::ALL {
        if matches!(
            ty,
            GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S
        ) {
            continue;
        }
        rejected += 1;
        assert_eq!(layout(ty), Err(QuantError::UnsupportedType { ty }));

        let mut block_output = sentinel(7);
        let block_before = bits(&block_output);
        assert_eq!(
            decode_block_into(ty, &[0; 9], &mut block_output),
            Err(QuantError::UnsupportedType { ty })
        );
        assert_unchanged(&block_before, &block_output);

        let mut row_output = sentinel(7);
        let row_before = bits(&row_output);
        assert_eq!(
            decode_row_into(ty, &[0; 9], 0, &mut row_output),
            Err(QuantError::UnsupportedType { ty })
        );
        assert_unchanged(&row_before, &row_output);
    }
    assert_eq!(rejected, GgmlType::ALL.len() - 5);
}

#[test]
fn every_truncated_and_overlong_encoded_block_is_rejected_atomically() {
    let cases = [
        (GgmlType::F32, 4_usize, 1_usize),
        (GgmlType::Q4_K, 144, 256),
        (GgmlType::Q5_K, 176, 256),
        (GgmlType::IQ2_S, 82, 256),
        (GgmlType::IQ3_S, 110, 256),
    ];

    let mut truncation_cases = 0;
    for (ty, block_bytes, block_elements) in cases {
        for actual in 0..block_bytes {
            truncation_cases += 1;
            let mut output = sentinel(block_elements);
            let before = bits(&output);
            assert_eq!(
                decode_block_into(ty, &vec![0; actual], &mut output),
                Err(QuantError::EncodedLengthMismatch {
                    ty,
                    expected: block_bytes,
                    actual,
                }),
                "{ty:?} encoded truncation at {actual}"
            );
            assert_unchanged(&before, &output);
        }

        let actual = block_bytes + 1;
        let mut output = sentinel(block_elements);
        let before = bits(&output);
        assert_eq!(
            decode_block_into(ty, &vec![0; actual], &mut output),
            Err(QuantError::EncodedLengthMismatch {
                ty,
                expected: block_bytes,
                actual,
            })
        );
        assert_unchanged(&before, &output);
    }
    assert_eq!(truncation_cases, 4 + 144 + 176 + 82 + 110);
}

#[test]
fn every_truncated_and_overlong_output_block_is_rejected_atomically() {
    let cases = [
        (GgmlType::F32, 1_usize),
        (GgmlType::Q4_K, 256),
        (GgmlType::Q5_K, 256),
        (GgmlType::IQ2_S, 256),
        (GgmlType::IQ3_S, 256),
    ];

    let mut truncation_cases = 0;
    for (ty, block_elements) in cases {
        let encoded = valid_block(ty);
        for actual in 0..block_elements {
            truncation_cases += 1;
            let mut output = sentinel(actual);
            let before = bits(&output);
            assert_eq!(
                decode_block_into(ty, &encoded, &mut output),
                Err(QuantError::OutputLengthMismatch {
                    ty,
                    expected: block_elements,
                    actual,
                }),
                "{ty:?} output truncation at {actual}"
            );
            assert_unchanged(&before, &output);
        }

        let actual = block_elements + 1;
        let mut output = sentinel(actual);
        let before = bits(&output);
        assert_eq!(
            decode_block_into(ty, &encoded, &mut output),
            Err(QuantError::OutputLengthMismatch {
                ty,
                expected: block_elements,
                actual,
            })
        );
        assert_unchanged(&before, &output);
    }
    assert_eq!(truncation_cases, 1 + 256 * 4);
}

#[test]
fn focused_entry_points_enforce_exact_lengths_atomically() {
    let cases: &[(GgmlType, usize, usize, DecodeBlock)] = &[
        (GgmlType::F32, 4, 1, decode_f32_block_into),
        (GgmlType::Q4_K, 144, 256, decode_q4_k_block_into),
        (GgmlType::Q5_K, 176, 256, decode_q5_k_block_into),
        (GgmlType::IQ2_S, 82, 256, decode_iq2_s_block_into),
        (GgmlType::IQ3_S, 110, 256, decode_iq3_s_block_into),
    ];

    for &(ty, block_bytes, block_elements, decode) in cases {
        for encoded_len in [block_bytes - 1, block_bytes + 1] {
            let mut output = sentinel(block_elements);
            let before = bits(&output);
            assert_eq!(
                decode(&vec![0; encoded_len], &mut output),
                Err(QuantError::EncodedLengthMismatch {
                    ty,
                    expected: block_bytes,
                    actual: encoded_len,
                })
            );
            assert_unchanged(&before, &output);
        }
        for output_len in [block_elements - 1, block_elements + 1] {
            let mut output = sentinel(output_len);
            let before = bits(&output);
            assert_eq!(
                decode(&vec![0; block_bytes], &mut output),
                Err(QuantError::OutputLengthMismatch {
                    ty,
                    expected: block_elements,
                    actual: output_len,
                })
            );
            assert_unchanged(&before, &output);
        }
    }
}

#[test]
fn row_shape_and_length_errors_are_exact_and_atomic() {
    let mut zero_output = sentinel(3);
    let zero_before = bits(&zero_output);
    assert_eq!(
        decode_row_into(GgmlType::Q4_K, &[], 0, &mut zero_output),
        Err(QuantError::ZeroLogicalElements)
    );
    assert_unchanged(&zero_before, &zero_output);

    for logical_elements in [1, 255, 257, 511] {
        let mut output = sentinel(9);
        let before = bits(&output);
        assert_eq!(
            decode_row_into(GgmlType::Q4_K, &[], logical_elements, &mut output),
            Err(QuantError::LogicalElementsNotDivisible {
                ty: GgmlType::Q4_K,
                logical_elements,
                block_elements: 256,
            })
        );
        assert_unchanged(&before, &output);
    }

    let row_cases = [
        (GgmlType::F32, 2_usize, 8_usize),
        (GgmlType::Q4_K, 512, 288),
        (GgmlType::Q5_K, 512, 352),
    ];
    for (ty, logical_elements, expected_encoded) in row_cases {
        for actual in [expected_encoded - 1, expected_encoded + 1] {
            let mut output = sentinel(logical_elements);
            let before = bits(&output);
            assert_eq!(
                decode_row_into(ty, &vec![0; actual], logical_elements, &mut output),
                Err(QuantError::EncodedLengthMismatch {
                    ty,
                    expected: expected_encoded,
                    actual,
                })
            );
            assert_unchanged(&before, &output);
        }

        let encoded = vec![0; expected_encoded];
        for actual in [logical_elements - 1, logical_elements + 1] {
            let mut output = sentinel(actual);
            let before = bits(&output);
            assert_eq!(
                decode_row_into(ty, &encoded, logical_elements, &mut output),
                Err(QuantError::OutputLengthMismatch {
                    ty,
                    expected: logical_elements,
                    actual,
                })
            );
            assert_unchanged(&before, &output);
        }
    }
}

#[test]
fn row_length_overflow_is_reported_before_slice_length_errors() {
    let mut output = sentinel(2);
    let before = bits(&output);
    assert_eq!(
        decode_row_into(GgmlType::F32, &[], usize::MAX, &mut output),
        Err(QuantError::ArithmeticOverflow {
            operation: "encoded row length",
        })
    );
    assert_unchanged(&before, &output);
}

#[test]
fn validation_order_is_deterministic_and_atomic() {
    let mut output = sentinel(1);
    let before = bits(&output);
    assert_eq!(
        decode_row_into(GgmlType::F16, &[], 0, &mut output),
        Err(QuantError::UnsupportedType { ty: GgmlType::F16 })
    );
    assert_unchanged(&before, &output);

    assert_eq!(
        decode_row_into(GgmlType::Q4_K, &[0], 0, &mut output),
        Err(QuantError::ZeroLogicalElements)
    );
    assert_unchanged(&before, &output);

    assert_eq!(
        decode_row_into(GgmlType::Q4_K, &[0], 1, &mut output),
        Err(QuantError::LogicalElementsNotDivisible {
            ty: GgmlType::Q4_K,
            logical_elements: 1,
            block_elements: 256,
        })
    );
    assert_unchanged(&before, &output);

    let mut invalid_scale = valid_block(GgmlType::Q4_K);
    invalid_scale[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert_eq!(
        decode_block_into(GgmlType::Q4_K, &invalid_scale[..143], &mut output),
        Err(QuantError::EncodedLengthMismatch {
            ty: GgmlType::Q4_K,
            expected: 144,
            actual: 143,
        })
    );
    assert_unchanged(&before, &output);

    assert_eq!(
        decode_block_into(GgmlType::Q4_K, &invalid_scale, &mut output),
        Err(QuantError::OutputLengthMismatch {
            ty: GgmlType::Q4_K,
            expected: 256,
            actual: 1,
        })
    );
    assert_unchanged(&before, &output);
}

#[test]
fn f32_identity_preserves_every_ieee_754_payload_bit() {
    let cases: [u32; 16] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xc020_0000,
        0x0000_0001,
        0x0080_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7fc1_2345,
        0xffc5_4321,
        0x7f80_0001,
        0x7fa1_2345,
        0xff80_0001,
        0xffa5_4321,
    ];
    let encoded: Vec<u8> = cases.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut output = sentinel(cases.len());

    decode_row_into(GgmlType::F32, &encoded, cases.len(), &mut output).unwrap();
    assert_eq!(bits(&output), cases);

    for (index, expected) in cases.into_iter().enumerate() {
        let mut lane = sentinel(1);
        decode_f32_block_into(&encoded[index * 4..index * 4 + 4], &mut lane).unwrap();
        assert_eq!(lane[0].to_bits(), expected);
    }
}

#[test]
fn finite_binary16_scales_convert_to_exact_f32_bits() {
    let cases = [
        (0x0000_u16, 0x0000_0000_u32),
        (0x8000, 0x8000_0000),
        (0x0001, 0x3380_0000),
        (0x03ff, 0x387f_c000),
        (0x0400, 0x3880_0000),
        (0x3c00, 0x3f80_0000),
        (0xbc00, 0xbf80_0000),
        (0x7bff, 0x477f_e000),
    ];

    for (half_bits, expected_bits) in cases {
        let mut block = unit_scale_q4(half_bits, 0);
        block[16] = 1;
        let mut output = vec![0.0; 256];
        decode_q4_k_block_into(&block, &mut output).unwrap();
        assert_eq!(
            output[0].to_bits(),
            expected_bits,
            "binary16 bits {half_bits:#06x}"
        );
    }
}

#[test]
fn every_nonfinite_binary16_scale_is_rejected_atomically() {
    let nonfinite = [0x7c00_u16, 0xfc00, 0x7e00, 0xfe01, 0x7d00, 0xfd01, 0x7fff, 0xffff];

    for ty in [GgmlType::Q4_K, GgmlType::Q5_K] {
        let block_bytes = if ty == GgmlType::Q4_K { 144 } else { 176 };
        for field in ["d", "dmin"] {
            for half_bits in nonfinite {
                let mut encoded = vec![0_u8; block_bytes];
                let offset = if field == "d" { 0 } else { 2 };
                encoded[offset..offset + 2].copy_from_slice(&half_bits.to_le_bytes());
                let mut output = sentinel(256);
                let before = bits(&output);
                assert_eq!(
                    decode_block_into(ty, &encoded, &mut output),
                    Err(QuantError::NonFiniteScale {
                        ty,
                        block_index: 0,
                        field,
                        bits: half_bits,
                    })
                );
                assert_unchanged(&before, &output);
            }
        }
    }

    for (ty, block_bytes) in [(GgmlType::IQ2_S, 82), (GgmlType::IQ3_S, 110)] {
        for half_bits in nonfinite {
            let mut encoded = vec![0_u8; block_bytes];
            encoded[0..2].copy_from_slice(&half_bits.to_le_bytes());
            let mut output = sentinel(256);
            let before = bits(&output);
            assert_eq!(
                decode_block_into(ty, &encoded, &mut output),
                Err(QuantError::NonFiniteScale {
                    ty,
                    block_index: 0,
                    field: "d",
                    bits: half_bits,
                })
            );
            assert_unchanged(&before, &output);
        }
    }
}

#[test]
fn row_prescans_every_scale_before_writing_any_lane() {
    for (ty, block_bytes) in [(GgmlType::Q4_K, 144_usize), (GgmlType::Q5_K, 176)] {
        for block_index in 0..3 {
            for field in ["d", "dmin"] {
                let mut encoded = vec![0_u8; block_bytes * 3];
                let field_offset = if field == "d" { 0 } else { 2 };
                let bits_value = if field == "d" { 0x7c00_u16 } else { 0x7e01_u16 };
                let offset = block_index * block_bytes + field_offset;
                encoded[offset..offset + 2].copy_from_slice(&bits_value.to_le_bytes());

                let mut output = sentinel(768);
                let before = bits(&output);
                assert_eq!(
                    decode_row_into(ty, &encoded, 768, &mut output),
                    Err(QuantError::NonFiniteScale {
                        ty,
                        block_index,
                        field,
                        bits: bits_value,
                    })
                );
                assert_unchanged(&before, &output);
            }
        }
    }

    for (ty, block_bytes) in [(GgmlType::IQ2_S, 82_usize), (GgmlType::IQ3_S, 110)] {
        for block_index in 0..3 {
            let mut encoded = vec![0_u8; block_bytes * 3];
            let bits_value = 0x7e01_u16;
            let offset = block_index * block_bytes;
            encoded[offset..offset + 2].copy_from_slice(&bits_value.to_le_bytes());

            let mut output = sentinel(768);
            let before = bits(&output);
            assert_eq!(
                decode_row_into(ty, &encoded, 768, &mut output),
                Err(QuantError::NonFiniteScale {
                    ty,
                    block_index,
                    field: "d",
                    bits: bits_value,
                })
            );
            assert_unchanged(&before, &output);
        }
    }
}

#[test]
fn q5_high_bit_mapping_covers_every_qh_bit_and_output_lane() {
    for expected_lane in 0..256 {
        let mut block = unit_scale_q5();
        let group = expected_lane / 64;
        let within_group = expected_lane % 64;
        let qh_index = within_group % 32;
        let bit_index = group * 2 + usize::from(within_group >= 32);
        block[16 + qh_index] = 1_u8 << bit_index;

        let mut output = vec![0.0; 256];
        decode_q5_k_block_into(&block, &mut output).unwrap();
        for (actual_lane, value) in output.iter().enumerate() {
            let expected = if actual_lane == expected_lane {
                16.0_f32.to_bits()
            } else {
                0.0_f32.to_bits()
            };
            assert_eq!(
                value.to_bits(),
                expected,
                "Q5_K high bit for lane {expected_lane} affected lane {actual_lane}"
            );
        }
    }
}
