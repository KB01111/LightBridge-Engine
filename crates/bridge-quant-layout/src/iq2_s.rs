use crate::tables::{IQ2S_GRID, KMASK_IQ2XS};
use half::f16;

const SCALE_OFFSET: usize = 74;
const GRID_INDEX_OFFSET: usize = 2;
const SIGN_OFFSET: usize = GRID_INDEX_OFFSET + 32;
const HIGH_INDEX_OFFSET: usize = 66;

pub(crate) fn decode_iq2_s_validated(encoded: &[u8], output: &mut [f32]) {
    let d = f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();

    for group in 0..8 {
        let scale = encoded[SCALE_OFFSET + group];
        let block_scales = [
            d * (0.5_f32 + f32::from(scale & 0x0f)) * 0.25_f32,
            d * (0.5_f32 + f32::from(scale >> 4)) * 0.25_f32,
        ];
        let high = usize::from(encoded[HIGH_INDEX_OFFSET + group]);

        for lane_group in 0..4 {
            let low = usize::from(encoded[GRID_INDEX_OFFSET + group * 4 + lane_group]);
            let shift = 8 - 2 * lane_group;
            let index = low | ((high << shift) & 0x300);
            let grid = IQ2S_GRID[index].to_le_bytes();
            let signs = encoded[SIGN_OFFSET + group * 4 + lane_group];
            let scale = block_scales[lane_group / 2];
            let output_start = group * 32 + lane_group * 8;

            for lane in 0..8 {
                let sign = if signs & KMASK_IQ2XS[lane] == 0 {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                output[output_start + lane] = scale * f32::from(grid[lane]) * sign;
            }
        }
    }
}
