use bridge_quant_layout::{
    decode_block_into, decode_f32_block_into, decode_iq2_s_block_into, decode_iq3_s_block_into,
    decode_q4_k_block_into, decode_q5_k_block_into, decode_row_into, quantize_row_q8_k_into,
    vec_dot_iq2_s_q8_k, vec_dot_iq3_s_q8_k, vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, GgmlType,
};

const F32_INPUT: &[u8; 64] = include_bytes!("fixtures/decode-f32.input.bin");
const F32_OUTPUT: &[u8; 64] = include_bytes!("fixtures/decode-f32.output-f32le.bin");
const Q4_K_INPUT: &[u8; 432] = include_bytes!("fixtures/decode-q4-k.input.bin");
const Q4_K_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-q4-k.output-f32le.bin");
const Q5_K_INPUT: &[u8; 528] = include_bytes!("fixtures/decode-q5-k.input.bin");
const Q5_K_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-q5-k.output-f32le.bin");
const IQ2_S_INPUT: &[u8; 246] = include_bytes!("fixtures/decode-iq2-s.input.bin");
const IQ2_S_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-iq2-s.output-f32le.bin");
const IQ3_S_INPUT: &[u8; 330] = include_bytes!("fixtures/decode-iq3-s.input.bin");
const IQ3_S_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-iq3-s.output-f32le.bin");
const Q8_K_INPUT: &[u8; 3072] = include_bytes!("fixtures/q8-k-activations.input-f32le.bin");
const Q8_K_OUTPUT: &[u8; 876] = include_bytes!("fixtures/q8-k-activations.output-q8-k.bin");
const Q4_K_DOT: &[u8; 4] = include_bytes!("fixtures/dot-q4-k-q8-k.output-f32le.bin");
const Q5_K_DOT: &[u8; 4] = include_bytes!("fixtures/dot-q5-k-q8-k.output-f32le.bin");
const IQ2_S_DOT: &[u8; 4] = include_bytes!("fixtures/dot-iq2-s-q8-k.output-f32le.bin");
const IQ3_S_DOT: &[u8; 4] = include_bytes!("fixtures/dot-iq3-s-q8-k.output-f32le.bin");

fn expected_bits(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn assert_exact_bits(actual: &[f32], expected: &[u8], label: &str) {
    let expected = expected_bits(expected);
    assert_eq!(actual.len(), expected.len());
    for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected, "{label} differs at lane {lane}");
    }
}

fn assert_packed_fixture(
    ty: GgmlType,
    block_bytes: usize,
    input: &[u8],
    expected: &[u8],
    focused: fn(&[u8], &mut [f32]) -> Result<(), bridge_quant_layout::QuantError>,
) {
    let mut row = vec![0.0; 768];
    decode_row_into(ty, input, 768, &mut row).unwrap();
    assert_exact_bits(&row, expected, &format!("{ty:?} row"));

    for block_index in 0..3 {
        let input = &input[block_index * block_bytes..(block_index + 1) * block_bytes];
        let expected = &expected[block_index * 1024..(block_index + 1) * 1024];
        let mut block = vec![0.0; 256];
        focused(input, &mut block).unwrap();
        assert_exact_bits(&block, expected, &format!("{ty:?} block {block_index}"));

        let mut generic = vec![0.0; 256];
        decode_block_into(ty, input, &mut generic).unwrap();
        assert_exact_bits(&generic, expected, &format!("generic {ty:?} block {block_index}"));
    }
}

#[test]
fn frozen_f32_fixture_is_bit_identical_as_a_row_and_as_blocks() {
    let mut row = vec![0.0; 16];
    decode_row_into(GgmlType::F32, F32_INPUT, 16, &mut row).unwrap();
    assert_exact_bits(&row, F32_OUTPUT, "F32 row");

    for block_index in 0..16 {
        let input = &F32_INPUT[block_index * 4..block_index * 4 + 4];
        let expected = &F32_OUTPUT[block_index * 4..block_index * 4 + 4];
        let mut block = [0.0_f32; 1];
        decode_f32_block_into(input, &mut block).unwrap();
        assert_exact_bits(&block, expected, &format!("F32 block {block_index}"));

        let mut generic = [0.0_f32; 1];
        decode_block_into(GgmlType::F32, input, &mut generic).unwrap();
        assert_exact_bits(&generic, expected, &format!("generic F32 block {block_index}"));
    }
}

#[test]
fn frozen_q4_k_fixture_is_bit_exact_as_a_row_and_each_block() {
    let mut row = vec![0.0; 768];
    decode_row_into(GgmlType::Q4_K, Q4_K_INPUT, 768, &mut row).unwrap();
    assert_exact_bits(&row, Q4_K_OUTPUT, "Q4_K row");

    for block_index in 0..3 {
        let input = &Q4_K_INPUT[block_index * 144..block_index * 144 + 144];
        let expected = &Q4_K_OUTPUT[block_index * 1024..block_index * 1024 + 1024];
        let mut block = vec![0.0; 256];
        decode_q4_k_block_into(input, &mut block).unwrap();
        assert_exact_bits(&block, expected, &format!("Q4_K block {block_index}"));

        let mut generic = vec![0.0; 256];
        decode_block_into(GgmlType::Q4_K, input, &mut generic).unwrap();
        assert_exact_bits(&generic, expected, &format!("generic Q4_K block {block_index}"));
    }
}

#[test]
fn frozen_q5_k_fixture_is_bit_exact_as_a_row_and_each_block() {
    let mut row = vec![0.0; 768];
    decode_row_into(GgmlType::Q5_K, Q5_K_INPUT, 768, &mut row).unwrap();
    assert_exact_bits(&row, Q5_K_OUTPUT, "Q5_K row");

    for block_index in 0..3 {
        let input = &Q5_K_INPUT[block_index * 176..block_index * 176 + 176];
        let expected = &Q5_K_OUTPUT[block_index * 1024..block_index * 1024 + 1024];
        let mut block = vec![0.0; 256];
        decode_q5_k_block_into(input, &mut block).unwrap();
        assert_exact_bits(&block, expected, &format!("Q5_K block {block_index}"));

        let mut generic = vec![0.0; 256];
        decode_block_into(GgmlType::Q5_K, input, &mut generic).unwrap();
        assert_exact_bits(&generic, expected, &format!("generic Q5_K block {block_index}"));
    }
}

#[test]
fn frozen_iq2_s_fixture_is_bit_exact_as_a_row_and_each_block() {
    assert_packed_fixture(
        GgmlType::IQ2_S,
        82,
        IQ2_S_INPUT,
        IQ2_S_OUTPUT,
        decode_iq2_s_block_into,
    );
}

#[test]
fn frozen_iq3_s_fixture_is_bit_exact_as_a_row_and_each_block() {
    assert_packed_fixture(
        GgmlType::IQ3_S,
        110,
        IQ3_S_INPUT,
        IQ3_S_OUTPUT,
        decode_iq3_s_block_into,
    );
}

#[test]
fn frozen_q8_k_activation_bytes_are_bit_exact() {
    let input: Vec<f32> = Q8_K_INPUT
        .chunks_exact(4)
        .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    let mut encoded = vec![0xa5; Q8_K_OUTPUT.len()];
    quantize_row_q8_k_into(&input, &mut encoded).unwrap();
    assert_eq!(encoded, Q8_K_OUTPUT);
}

#[test]
fn frozen_scalar_q8_k_dot_products_are_bit_exact() {
    let cases = [
        (
            "Q4_K x Q8_K",
            vec_dot_q4_k_q8_k(Q4_K_INPUT, Q8_K_OUTPUT, 768).unwrap(),
            Q4_K_DOT,
        ),
        (
            "Q5_K x Q8_K",
            vec_dot_q5_k_q8_k(Q5_K_INPUT, Q8_K_OUTPUT, 768).unwrap(),
            Q5_K_DOT,
        ),
        (
            "IQ2_S x Q8_K",
            vec_dot_iq2_s_q8_k(IQ2_S_INPUT, Q8_K_OUTPUT, 768).unwrap(),
            IQ2_S_DOT,
        ),
        (
            "IQ3_S x Q8_K",
            vec_dot_iq3_s_q8_k(IQ3_S_INPUT, Q8_K_OUTPUT, 768).unwrap(),
            IQ3_S_DOT,
        ),
    ];

    for (label, actual, expected) in cases {
        let expected = u32::from_le_bytes(*expected);
        assert_eq!(actual.to_bits(), expected, "{label}");
    }
}
