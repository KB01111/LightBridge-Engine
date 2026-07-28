use crate::error::Result;
use crate::{GgmlType, QuantError};
use half::f16;

const F32_BLOCK_ELEMENTS: usize = 1;
const F32_BLOCK_BYTES: usize = 4;
const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;

const SCALE_BYTES: usize = 12;
const Q4_K_SCALES_OFFSET: usize = 4;
const Q4_K_QUANTS_OFFSET: usize = 16;
const Q5_K_SCALES_OFFSET: usize = 4;
const Q5_K_HIGH_BITS_OFFSET: usize = 16;
const Q5_K_QUANTS_OFFSET: usize = 48;

/// The exact packed layout of one supported GGML block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantLayout {
    /// GGML element type.
    pub ty: GgmlType,
    /// Logical scalar elements represented by one block.
    pub block_elements: usize,
    /// Encoded bytes occupied by one block.
    pub block_bytes: usize,
}

/// Returns the packed layout for a currently supported type.
pub fn layout(ty: GgmlType) -> Result<QuantLayout> {
    let (expected_elements, expected_bytes) = match ty {
        GgmlType::F32 => (F32_BLOCK_ELEMENTS, F32_BLOCK_BYTES),
        GgmlType::Q4_K => (K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES),
        GgmlType::Q5_K => (K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES),
        _ => return Err(QuantError::UnsupportedType { ty }),
    };

    let block_elements = checked_usize(ty.block_size(), "block elements")?;
    let block_bytes = checked_usize(ty.type_size(), "block bytes")?;
    debug_assert_eq!(block_elements, expected_elements);
    debug_assert_eq!(block_bytes, expected_bytes);

    Ok(QuantLayout {
        ty,
        block_elements,
        block_bytes,
    })
}

/// Decodes exactly one packed block into caller-owned output.
pub fn decode_block_into(ty: GgmlType, encoded: &[u8], output: &mut [f32]) -> Result<()> {
    let block_layout = layout(ty)?;
    validate_lengths(
        ty,
        encoded.len(),
        block_layout.block_bytes,
        output.len(),
        block_layout.block_elements,
    )?;
    validate_scales(ty, encoded, 1, block_layout.block_bytes)?;
    decode_validated_block(ty, encoded, output)
}

/// Decodes one exact packed row into caller-owned output.
pub fn decode_row_into(
    ty: GgmlType,
    encoded: &[u8],
    logical_elements: usize,
    output: &mut [f32],
) -> Result<()> {
    let row_layout = layout(ty)?;
    if logical_elements == 0 {
        return Err(QuantError::ZeroLogicalElements);
    }
    if logical_elements % row_layout.block_elements != 0 {
        return Err(QuantError::LogicalElementsNotDivisible {
            ty,
            logical_elements,
            block_elements: row_layout.block_elements,
        });
    }

    let block_count = logical_elements / row_layout.block_elements;
    let (expected_encoded, expected_output) =
        checked_row_lengths(block_count, row_layout.block_bytes, row_layout.block_elements)?;
    validate_lengths(ty, encoded.len(), expected_encoded, output.len(), expected_output)?;
    validate_scales(ty, encoded, block_count, row_layout.block_bytes)?;

    for block_index in 0..block_count {
        let encoded_start = block_index * row_layout.block_bytes;
        let output_start = block_index * row_layout.block_elements;
        decode_validated_block(
            ty,
            &encoded[encoded_start..encoded_start + row_layout.block_bytes],
            &mut output[output_start..output_start + row_layout.block_elements],
        )?;
    }
    Ok(())
}

/// Decodes exactly one raw little-endian F32 block.
pub fn decode_f32_block_into(encoded: &[u8], output: &mut [f32]) -> Result<()> {
    decode_block_into(GgmlType::F32, encoded, output)
}

/// Decodes exactly one packed Q4_K block.
pub fn decode_q4_k_block_into(encoded: &[u8], output: &mut [f32]) -> Result<()> {
    decode_block_into(GgmlType::Q4_K, encoded, output)
}

/// Decodes exactly one packed Q5_K block.
pub fn decode_q5_k_block_into(encoded: &[u8], output: &mut [f32]) -> Result<()> {
    decode_block_into(GgmlType::Q5_K, encoded, output)
}

fn checked_row_lengths(
    block_count: usize,
    block_bytes: usize,
    block_elements: usize,
) -> Result<(usize, usize)> {
    let encoded = block_count
        .checked_mul(block_bytes)
        .ok_or(QuantError::ArithmeticOverflow {
            operation: "encoded row length",
        })?;
    let output = block_count
        .checked_mul(block_elements)
        .ok_or(QuantError::ArithmeticOverflow {
            operation: "logical row length",
        })?;
    Ok((encoded, output))
}

fn checked_usize(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| QuantError::IntegerConversionOverflow { field, value })
}

fn validate_lengths(
    ty: GgmlType,
    encoded_actual: usize,
    encoded_expected: usize,
    output_actual: usize,
    output_expected: usize,
) -> Result<()> {
    if encoded_actual != encoded_expected {
        return Err(QuantError::EncodedLengthMismatch {
            ty,
            expected: encoded_expected,
            actual: encoded_actual,
        });
    }
    if output_actual != output_expected {
        return Err(QuantError::OutputLengthMismatch {
            ty,
            expected: output_expected,
            actual: output_actual,
        });
    }
    Ok(())
}

fn validate_scales(ty: GgmlType, encoded: &[u8], block_count: usize, block_bytes: usize) -> Result<()> {
    if ty == GgmlType::F32 {
        return Ok(());
    }

    for block_index in 0..block_count {
        let block_start = block_index * block_bytes;
        for (field, offset) in [("d", 0_usize), ("dmin", 2_usize)] {
            let bits = u16::from_le_bytes([encoded[block_start + offset], encoded[block_start + offset + 1]]);
            if !f16::from_bits(bits).is_finite() {
                return Err(QuantError::NonFiniteScale {
                    ty,
                    block_index,
                    field,
                    bits,
                });
            }
        }
    }
    Ok(())
}

fn decode_validated_block(ty: GgmlType, encoded: &[u8], output: &mut [f32]) -> Result<()> {
    match ty {
        GgmlType::F32 => decode_f32_validated(encoded, output),
        GgmlType::Q4_K => decode_q4_k_validated(encoded, output),
        GgmlType::Q5_K => decode_q5_k_validated(encoded, output),
        _ => return Err(QuantError::UnsupportedType { ty }),
    }
    Ok(())
}

fn decode_f32_validated(encoded: &[u8], output: &mut [f32]) {
    let bits = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
    output[0] = f32::from_bits(bits);
}

fn finite_scale(encoded: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([encoded[offset], encoded[offset + 1]])).to_f32()
}

fn scale_min(scales: &[u8], index: usize) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn decode_q4_k_validated(encoded: &[u8], output: &mut [f32]) {
    let d = finite_scale(encoded, 0);
    let dmin = finite_scale(encoded, 2);
    let scales = &encoded[Q4_K_SCALES_OFFSET..Q4_K_SCALES_OFFSET + SCALE_BYTES];
    let quants = &encoded[Q4_K_QUANTS_OFFSET..Q4_K_BLOCK_BYTES];

    for group in 0..4 {
        let (scale1, min1) = scale_min(scales, group * 2);
        let d1 = d * f32::from(scale1);
        let m1 = dmin * f32::from(min1);
        let (scale2, min2) = scale_min(scales, group * 2 + 1);
        let d2 = d * f32::from(scale2);
        let m2 = dmin * f32::from(min2);
        let quant_start = group * 32;
        let output_start = group * 64;

        for lane in 0..32 {
            output[output_start + lane] = d1 * f32::from(quants[quant_start + lane] & 0x0f) - m1;
        }
        for lane in 0..32 {
            output[output_start + 32 + lane] = d2 * f32::from(quants[quant_start + lane] >> 4) - m2;
        }
    }
}

fn decode_q5_k_validated(encoded: &[u8], output: &mut [f32]) {
    let d = finite_scale(encoded, 0);
    let dmin = finite_scale(encoded, 2);
    let scales = &encoded[Q5_K_SCALES_OFFSET..Q5_K_SCALES_OFFSET + SCALE_BYTES];
    let high_bits = &encoded[Q5_K_HIGH_BITS_OFFSET..Q5_K_QUANTS_OFFSET];
    let quants = &encoded[Q5_K_QUANTS_OFFSET..Q5_K_BLOCK_BYTES];
    let mut low_mask = 1_u8;
    let mut high_mask = 2_u8;

    for group in 0..4 {
        let (scale1, min1) = scale_min(scales, group * 2);
        let d1 = d * f32::from(scale1);
        let m1 = dmin * f32::from(min1);
        let (scale2, min2) = scale_min(scales, group * 2 + 1);
        let d2 = d * f32::from(scale2);
        let m2 = dmin * f32::from(min2);
        let quant_start = group * 32;
        let output_start = group * 64;

        for lane in 0..32 {
            let high = u8::from(high_bits[lane] & low_mask != 0) * 16;
            let quant = (quants[quant_start + lane] & 0x0f) + high;
            output[output_start + lane] = d1 * f32::from(quant) - m1;
        }
        for lane in 0..32 {
            let high = u8::from(high_bits[lane] & high_mask != 0) * 16;
            let quant = (quants[quant_start + lane] >> 4) + high;
            output[output_start + 32 + lane] = d2 * f32::from(quant) - m2;
        }

        low_mask <<= 2;
        high_mask <<= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_row_lengths_rejects_encoded_multiplication_overflow() {
        assert_eq!(
            checked_row_lengths(usize::MAX, 2, 1),
            Err(QuantError::ArithmeticOverflow {
                operation: "encoded row length",
            })
        );
    }

    #[test]
    fn checked_row_lengths_rejects_output_multiplication_overflow() {
        assert_eq!(
            checked_row_lengths(usize::MAX, 1, 2),
            Err(QuantError::ArithmeticOverflow {
                operation: "logical row length",
            })
        );
    }

    #[test]
    fn checked_usize_matches_the_target_pointer_width() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(checked_usize(u64::MAX, "test"), Ok(usize::MAX));

        #[cfg(not(target_pointer_width = "64"))]
        assert_eq!(
            checked_usize(u64::MAX, "test"),
            Err(QuantError::IntegerConversionOverflow {
                field: "test",
                value: u64::MAX,
            })
        );
    }
}
