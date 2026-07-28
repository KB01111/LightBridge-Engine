use bridge_quant_layout::{
    decode_block_into, decode_f32_block_into, decode_q4_k_block_into, decode_q5_k_block_into,
    decode_row_into, GgmlType,
};

const F32_INPUT: &[u8; 64] = include_bytes!("fixtures/decode-f32.input.bin");
const F32_OUTPUT: &[u8; 64] = include_bytes!("fixtures/decode-f32.output-f32le.bin");
const Q4_K_INPUT: &[u8; 432] = include_bytes!("fixtures/decode-q4-k.input.bin");
const Q4_K_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-q4-k.output-f32le.bin");
const Q5_K_INPUT: &[u8; 528] = include_bytes!("fixtures/decode-q5-k.input.bin");
const Q5_K_OUTPUT: &[u8; 3072] = include_bytes!("fixtures/decode-q5-k.output-f32le.bin");

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
