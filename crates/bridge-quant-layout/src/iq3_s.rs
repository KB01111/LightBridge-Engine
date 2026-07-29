use crate::tables::{IQ3S_GRID, KMASK_IQ2XS};
use half::f16;

const GRID_INDEX_OFFSET: usize = 2;
const HIGH_INDEX_OFFSET: usize = 66;
const SIGN_OFFSET: usize = 74;
const SCALE_OFFSET: usize = 106;

pub(crate) fn decode_iq3_s_validated(encoded: &[u8], output: &mut [f32]) {
    let d = f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();

    for pair in 0..4 {
        let packed_scale = encoded[SCALE_OFFSET + pair];
        let scales = [
            d * f32::from(1 + 2 * (packed_scale & 0x0f)),
            d * f32::from(1 + 2 * (packed_scale >> 4)),
        ];

        for (half, &scale) in scales.iter().enumerate() {
            let group = pair * 2 + half;
            let high = usize::from(encoded[HIGH_INDEX_OFFSET + group]);
            let quant_start = GRID_INDEX_OFFSET + group * 8;
            let sign_start = SIGN_OFFSET + group * 4;
            let output_start = group * 32;

            for lane_group in 0..4 {
                let low1 = usize::from(encoded[quant_start + lane_group * 2]);
                let low2 = usize::from(encoded[quant_start + lane_group * 2 + 1]);
                let index1 = low1 | ((high << (8 - 2 * lane_group)) & 0x100);
                let index2 = low2 | ((high << (7 - 2 * lane_group)) & 0x100);
                let grid1 = IQ3S_GRID[index1].to_le_bytes();
                let grid2 = IQ3S_GRID[index2].to_le_bytes();
                let signs = encoded[sign_start + lane_group];
                let lane_start = output_start + lane_group * 8;

                for lane in 0..4 {
                    let sign1 = if signs & KMASK_IQ2XS[lane] == 0 {
                        1.0_f32
                    } else {
                        -1.0_f32
                    };
                    let sign2 = if signs & KMASK_IQ2XS[lane + 4] == 0 {
                        1.0_f32
                    } else {
                        -1.0_f32
                    };
                    output[lane_start + lane] = scale * f32::from(grid1[lane]) * sign1;
                    output[lane_start + lane + 4] = scale * f32::from(grid2[lane]) * sign2;
                }
            }
        }
    }
}
