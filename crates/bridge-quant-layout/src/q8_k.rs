use crate::error::Result;
use crate::k_quants::{layout, scale_min, validate_scales};
use crate::tables::{IQ2S_GRID, IQ3S_GRID, KMASK_IQ2XS};
use crate::{GgmlType, QuantError};
use half::f16;

pub const Q8_K_BLOCK_ELEMENTS: usize = 256;
pub const Q8_K_BLOCK_BYTES: usize = 292;

const Q8_D_OFFSET: usize = 0;
const Q8_QUANTS_OFFSET: usize = 4;
const Q8_BLOCK_SUMS_OFFSET: usize = 260;

/// Returns the authenticated IQ2_S codebook used by exact CPU and accelerator
/// oracles. The table is immutable and its digest is covered by this crate's
/// fixture tests.
pub fn iq2_s_grid_table() -> &'static [u64; 1024] {
    &IQ2S_GRID
}

/// Returns the authenticated IQ3_S codebook used by exact CPU and accelerator
/// oracles. The table is immutable and its digest is covered by this crate's
/// fixture tests.
pub fn iq3_s_grid_table() -> &'static [u32; 512] {
    &IQ3S_GRID
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuDotBackend {
    Scalar,
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

impl CpuDotBackend {
    pub fn available(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Self::AvxVnni => avx_vnni_available(),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Vnni => {
                std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                    && std::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            Self::Avx2 | Self::AvxVnni | Self::Avx512Vnni => false,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::AvxVnni => "avx_vnni",
            Self::Avx512Vnni => "avx512_vnni",
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx_vnni_available() -> bool {
    if !std::is_x86_feature_detected!("avx2") {
        return false;
    }
    // AVX-VNNI is enumerated by CPUID leaf 7, subleaf 1, EAX bit 4.
    // AVX2 detection above also proves that the OS saves the required YMM
    // state, so executing the VEX-encoded instruction is permitted.
    let features = std::arch::x86_64::__cpuid_count(7, 1);
    features.eax & (1 << 4) != 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VnniMode {
    None,
    Avx,
    Avx512,
}

/// A complete packed matrix and Q8_K activation whose shapes, scales, and CPU
/// dispatch capability have already been validated.
///
/// Constructing this value performs the complete failure-prone scan once.
/// Row dots then reuse that evidence without rescanning the common activation
/// or the same weight row immediately before computation. This lets callers
/// validate every row before publishing any output while avoiding duplicate
/// hot-path memory traffic.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedQ8KMatrix<'a> {
    weight_type: GgmlType,
    weights: &'a [u8],
    q8: &'a [u8],
    block_count: usize,
    row_bytes: usize,
    output_rows: usize,
    backend: CpuDotBackend,
}

impl<'a> ValidatedQ8KMatrix<'a> {
    pub fn new(
        weight_type: GgmlType,
        weights: &'a [u8],
        q8: &'a [u8],
        logical_elements: usize,
        output_rows: usize,
        backend: CpuDotBackend,
    ) -> Result<Self> {
        if !matches!(
            weight_type,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S
        ) {
            return Err(QuantError::UnsupportedType { ty: weight_type });
        }
        let block_count = validate_logical_elements(weight_type, logical_elements)?;
        let weight_layout = layout(weight_type)?;
        let row_bytes = checked_bytes(
            block_count,
            weight_layout.block_bytes,
            "packed dot weight row length",
        )?;
        let expected_weights = checked_bytes(row_bytes, output_rows, "packed dot matrix byte length")?;
        if weights.len() != expected_weights {
            return Err(QuantError::EncodedLengthMismatch {
                ty: weight_type,
                expected: expected_weights,
                actual: weights.len(),
            });
        }
        let expected_q8 = checked_bytes(block_count, Q8_K_BLOCK_BYTES, "packed dot Q8_K row length")?;
        if q8.len() != expected_q8 {
            return Err(QuantError::EncodedLengthMismatch {
                ty: GgmlType::Q8_K,
                expected: expected_q8,
                actual: q8.len(),
            });
        }
        if !backend.available() {
            return Err(QuantError::CpuBackendUnavailable {
                backend: backend.name(),
            });
        }

        let weight_blocks = block_count
            .checked_mul(output_rows)
            .ok_or(QuantError::ArithmeticOverflow {
                operation: "packed dot matrix block count",
            })?;
        validate_scales(weight_type, weights, weight_blocks, weight_layout.block_bytes)?;
        validate_q8_scales(q8, block_count)?;

        Ok(Self {
            weight_type,
            weights,
            q8,
            block_count,
            row_bytes,
            output_rows,
            backend,
        })
    }

    pub const fn backend(self) -> CpuDotBackend {
        self.backend
    }

    pub const fn output_rows(self) -> usize {
        self.output_rows
    }

    /// Computes one row after the constructor has completed every
    /// model-controlled validation step.
    pub fn dot_row(self, row: usize) -> Result<f32> {
        if row >= self.output_rows {
            return Err(QuantError::MatrixRowOutOfRange {
                row,
                rows: self.output_rows,
            });
        }
        // The constructor proved `row_bytes * output_rows` fits and exactly
        // matches `weights`, so this range is bounded for every accepted row.
        let start = row * self.row_bytes;
        let weights = &self.weights[start..start + self.row_bytes];
        match self.backend {
            CpuDotBackend::Scalar => Ok(dot_scalar(self.weight_type, weights, self.q8, self.block_count)),
            #[cfg(target_arch = "x86_64")]
            CpuDotBackend::Avx2 => {
                // SAFETY: construction proved AVX2 availability and all
                // packed block boundaries.
                Ok(unsafe {
                    dot_avx2(
                        self.weight_type,
                        weights,
                        self.q8,
                        self.block_count,
                        VnniMode::None,
                    )
                })
            }
            #[cfg(target_arch = "x86_64")]
            CpuDotBackend::AvxVnni => {
                // SAFETY: construction proved AVX2 plus AVX-VNNI availability
                // and all packed block boundaries.
                Ok(unsafe {
                    dot_avx2(
                        self.weight_type,
                        weights,
                        self.q8,
                        self.block_count,
                        VnniMode::Avx,
                    )
                })
            }
            #[cfg(target_arch = "x86_64")]
            CpuDotBackend::Avx512Vnni => {
                // SAFETY: construction proved the complete AVX-512/VNNI
                // feature set and every packed block boundary.
                Ok(unsafe {
                    dot_avx2(
                        self.weight_type,
                        weights,
                        self.q8,
                        self.block_count,
                        VnniMode::Avx512,
                    )
                })
            }
            #[cfg(not(target_arch = "x86_64"))]
            CpuDotBackend::Avx2 | CpuDotBackend::AvxVnni | CpuDotBackend::Avx512Vnni => {
                Err(QuantError::CpuBackendUnavailable {
                    backend: self.backend.name(),
                })
            }
        }
    }
}

/// Quantizes one or more exact 256-lane activation blocks with llama.cpp's
/// pinned scalar Q8_K reference semantics.
pub fn quantize_row_q8_k_into(input: &[f32], encoded: &mut [u8]) -> Result<()> {
    let block_count = validate_logical_elements(GgmlType::Q8_K, input.len())?;
    let expected = checked_bytes(block_count, Q8_K_BLOCK_BYTES, "Q8_K encoded row length")?;
    if encoded.len() != expected {
        return Err(QuantError::EncodedLengthMismatch {
            ty: GgmlType::Q8_K,
            expected,
            actual: encoded.len(),
        });
    }
    for (index, value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteActivation {
                index,
                bits: value.to_bits(),
            });
        }
    }

    for block_index in 0..block_count {
        let input_start = block_index * Q8_K_BLOCK_ELEMENTS;
        let output_start = block_index * Q8_K_BLOCK_BYTES;
        let values = &input[input_start..input_start + Q8_K_BLOCK_ELEMENTS];
        let output = &mut encoded[output_start..output_start + Q8_K_BLOCK_BYTES];

        let mut max = 0.0_f32;
        let mut absolute_max = 0.0_f32;
        for &value in values {
            let absolute = value.abs();
            if absolute > absolute_max {
                absolute_max = absolute;
                max = value;
            }
        }

        if absolute_max == 0.0 {
            output.fill(0);
            continue;
        }

        let inverse_scale = -127.0_f32 / max;
        for (lane, &value) in values.iter().enumerate() {
            let quant = nearest_int(inverse_scale * value).min(127) as i8;
            output[Q8_QUANTS_OFFSET + lane] = quant as u8;
        }
        for group in 0..16 {
            let mut sum = 0_i32;
            for lane in 0..16 {
                sum += i32::from(output[Q8_QUANTS_OFFSET + group * 16 + lane] as i8);
            }
            let encoded_sum = i16::try_from(sum)
                .expect("the sum of sixteen signed Q8 lanes always fits i16")
                .to_le_bytes();
            let sum_offset = Q8_BLOCK_SUMS_OFFSET + group * 2;
            output[sum_offset..sum_offset + 2].copy_from_slice(&encoded_sum);
        }
        output[Q8_D_OFFSET..Q8_D_OFFSET + 4].copy_from_slice(&(1.0_f32 / inverse_scale).to_le_bytes());
    }

    Ok(())
}

/// Computes the pinned scalar dot product for a selected packed weight row and
/// an already-quantized Q8_K activation row.
pub fn vec_dot_q8_k(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    logical_elements: usize,
) -> Result<f32> {
    ValidatedQ8KMatrix::new(
        weight_type,
        weights,
        q8,
        logical_elements,
        1,
        CpuDotBackend::Scalar,
    )?
    .dot_row(0)
}

/// Validates both packed rows without performing the dot product.
pub fn validate_vec_dot_q8_k(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    logical_elements: usize,
) -> Result<()> {
    ValidatedQ8KMatrix::new(
        weight_type,
        weights,
        q8,
        logical_elements,
        1,
        CpuDotBackend::Scalar,
    )
    .map(|_| ())
}

/// Runs the accepted exact dot product with AVX2 when available, falling back
/// to the scalar oracle. AVX-512 remains an explicit tuner-selected candidate.
pub fn vec_dot_q8_k_cpu(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    logical_elements: usize,
) -> Result<f32> {
    let backend = if CpuDotBackend::Avx2.available() {
        CpuDotBackend::Avx2
    } else {
        CpuDotBackend::Scalar
    };
    vec_dot_q8_k_cpu_backend(weight_type, weights, q8, logical_elements, backend)
}

/// Executes one explicitly selected CPU implementation without silently
/// substituting another backend.
pub fn vec_dot_q8_k_cpu_backend(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    logical_elements: usize,
    backend: CpuDotBackend,
) -> Result<f32> {
    ValidatedQ8KMatrix::new(weight_type, weights, q8, logical_elements, 1, backend)?.dot_row(0)
}

fn dot_scalar(weight_type: GgmlType, weights: &[u8], q8: &[u8], block_count: usize) -> f32 {
    match weight_type {
        GgmlType::Q4_K | GgmlType::Q5_K => dot_q4_or_q5_k(weight_type, weights, q8, block_count),
        GgmlType::IQ2_S => dot_iq2_s(weights, q8, block_count),
        GgmlType::IQ3_S => dot_iq3_s(weights, q8, block_count),
        _ => unreachable!("supported dot types were checked above"),
    }
}

pub fn vec_dot_q4_k_q8_k(weights: &[u8], q8: &[u8], logical_elements: usize) -> Result<f32> {
    vec_dot_q8_k(GgmlType::Q4_K, weights, q8, logical_elements)
}

pub fn vec_dot_q5_k_q8_k(weights: &[u8], q8: &[u8], logical_elements: usize) -> Result<f32> {
    vec_dot_q8_k(GgmlType::Q5_K, weights, q8, logical_elements)
}

pub fn vec_dot_iq2_s_q8_k(weights: &[u8], q8: &[u8], logical_elements: usize) -> Result<f32> {
    vec_dot_q8_k(GgmlType::IQ2_S, weights, q8, logical_elements)
}

pub fn vec_dot_iq3_s_q8_k(weights: &[u8], q8: &[u8], logical_elements: usize) -> Result<f32> {
    vec_dot_q8_k(GgmlType::IQ3_S, weights, q8, logical_elements)
}

fn validate_logical_elements(ty: GgmlType, logical_elements: usize) -> Result<usize> {
    if logical_elements == 0 {
        return Err(QuantError::ZeroLogicalElements);
    }
    if logical_elements % Q8_K_BLOCK_ELEMENTS != 0 {
        return Err(QuantError::LogicalElementsNotDivisible {
            ty,
            logical_elements,
            block_elements: Q8_K_BLOCK_ELEMENTS,
        });
    }
    Ok(logical_elements / Q8_K_BLOCK_ELEMENTS)
}

fn checked_bytes(block_count: usize, block_bytes: usize, operation: &'static str) -> Result<usize> {
    block_count
        .checked_mul(block_bytes)
        .ok_or(QuantError::ArithmeticOverflow { operation })
}

fn validate_q8_scales(encoded: &[u8], block_count: usize) -> Result<()> {
    for block_index in 0..block_count {
        let offset = block_index * Q8_K_BLOCK_BYTES + Q8_D_OFFSET;
        let bits = u32::from_le_bytes([
            encoded[offset],
            encoded[offset + 1],
            encoded[offset + 2],
            encoded[offset + 3],
        ]);
        if !f32::from_bits(bits).is_finite() {
            return Err(QuantError::NonFiniteQ8Scale { block_index, bits });
        }
    }
    Ok(())
}

fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0_f32);
    let bits = (value + 12_582_912.0_f32).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn dot_q4_or_q5_k(weight_type: GgmlType, weights: &[u8], q8: &[u8], block_count: usize) -> f32 {
    let block_bytes = layout(weight_type)
        .expect("validated weight type has a layout")
        .block_bytes;
    let mut sums = [0.0_f32; 8];
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * block_bytes..(block_index + 1) * block_bytes];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let mut quants = [0_i8; Q8_K_BLOCK_ELEMENTS];

        match weight_type {
            GgmlType::Q4_K => {
                for group in 0..4 {
                    for lane in 0..32 {
                        let packed = weight[16 + group * 32 + lane];
                        quants[group * 64 + lane] = (packed & 0x0f) as i8;
                        quants[group * 64 + 32 + lane] = (packed >> 4) as i8;
                    }
                }
            }
            GgmlType::Q5_K => {
                let mut low_mask = 1_u8;
                let mut high_mask = 2_u8;
                for group in 0..4 {
                    for lane in 0..32 {
                        let packed = weight[48 + group * 32 + lane];
                        let high_bits = weight[16 + lane];
                        quants[group * 64 + lane] =
                            ((packed & 0x0f) + u8::from(high_bits & low_mask != 0) * 16) as i8;
                        quants[group * 64 + 32 + lane] =
                            ((packed >> 4) + u8::from(high_bits & high_mask != 0) * 16) as i8;
                    }
                    low_mask <<= 2;
                    high_mask <<= 2;
                }
            }
            _ => unreachable!("caller restricts this helper to Q4_K and Q5_K"),
        }

        let packed_scales = &weight[4..16];
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..8 {
            (scales[index], mins[index]) = scale_min(packed_scales, index);
        }

        let mut minimum_sum = 0_i32;
        for group in 0..16 {
            minimum_sum += i32::from(read_i16(activation, Q8_BLOCK_SUMS_OFFSET + group * 2))
                * i32::from(mins[group / 2]);
        }

        let mut lane_sums = [0_i32; 8];
        let mut offset = 0;
        for &scale in &scales {
            for _ in 0..4 {
                for lane in 0..8 {
                    let activation_value = i32::from(activation[Q8_QUANTS_OFFSET + offset + lane] as i8);
                    let weight_value = i32::from(quants[offset + lane]);
                    lane_sums[lane] += i32::from(scale) * activation_value * weight_value;
                }
                offset += 8;
            }
        }

        let activation_scale = read_f32(activation, Q8_D_OFFSET);
        let scale = read_f16(weight, 0) * activation_scale;
        for lane in 0..8 {
            sums[lane] += scale * lane_sums[lane] as f32;
        }
        let minimum_scale = read_f16(weight, 2) * activation_scale;
        sum -= minimum_scale * minimum_sum as f32;
    }

    for lane_sum in sums {
        sum += lane_sum;
    }
    sum
}

fn dot_iq2_s(weights: &[u8], q8: &[u8], block_count: usize) -> f32 {
    const BLOCK_BYTES: usize = 82;
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let scale = read_f16(weight, 0) * read_f32(activation, Q8_D_OFFSET);
        let mut block_sum = 0_i32;

        for group in 0..8 {
            let packed_scale = weight[74 + group];
            let scale1 = i32::from(1 + 2 * (packed_scale & 0x0f));
            let scale2 = i32::from(1 + 2 * (packed_scale >> 4));
            let high = usize::from(weight[66 + group]);
            let mut sums = [0_i32; 2];

            for lane_group in 0..4 {
                let low = usize::from(weight[2 + group * 4 + lane_group]);
                let index = low | ((high << (8 - 2 * lane_group)) & 0x300);
                let grid = IQ2S_GRID[index].to_le_bytes();
                let signs = weight[34 + group * 4 + lane_group];
                let activation_start = Q8_QUANTS_OFFSET + group * 32 + lane_group * 8;

                for lane in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[lane] == 0 {
                        1_i32
                    } else {
                        -1_i32
                    };
                    sums[lane_group / 2] +=
                        i32::from(activation[activation_start + lane] as i8) * i32::from(grid[lane]) * sign;
                }
            }
            block_sum += scale1 * sums[0] + scale2 * sums[1];
        }

        sum += scale * block_sum as f32;
    }

    0.125_f32 * sum
}

fn dot_iq3_s(weights: &[u8], q8: &[u8], block_count: usize) -> f32 {
    const BLOCK_BYTES: usize = 110;
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let scale = read_f16(weight, 0) * read_f32(activation, Q8_D_OFFSET);
        let mut block_sum = 0_i32;

        for pair in 0..4 {
            let packed_scale = weight[106 + pair];
            let pair_scales = [
                i32::from(2 * (packed_scale & 0x0f) + 1),
                i32::from(2 * (packed_scale >> 4) + 1),
            ];

            for (half, &pair_scale) in pair_scales.iter().enumerate() {
                let group = pair * 2 + half;
                let high = usize::from(weight[66 + group]);
                let quant_start = 2 + group * 8;
                let sign_start = 74 + group * 4;
                let activation_group = Q8_QUANTS_OFFSET + group * 32;
                let mut group_sum = 0_i32;

                for lane_group in 0..4 {
                    let low1 = usize::from(weight[quant_start + lane_group * 2]);
                    let low2 = usize::from(weight[quant_start + lane_group * 2 + 1]);
                    let index1 = low1 | ((high << (8 - 2 * lane_group)) & 0x100);
                    let index2 = low2 | ((high << (7 - 2 * lane_group)) & 0x100);
                    let grid1 = IQ3S_GRID[index1].to_le_bytes();
                    let grid2 = IQ3S_GRID[index2].to_le_bytes();
                    let signs = weight[sign_start + lane_group];
                    let activation_start = activation_group + lane_group * 8;

                    for lane in 0..4 {
                        let sign1 = if signs & KMASK_IQ2XS[lane] == 0 {
                            1_i32
                        } else {
                            -1_i32
                        };
                        let sign2 = if signs & KMASK_IQ2XS[lane + 4] == 0 {
                            1_i32
                        } else {
                            -1_i32
                        };
                        group_sum += i32::from(activation[activation_start + lane] as i8)
                            * i32::from(grid1[lane])
                            * sign1;
                        group_sum += i32::from(activation[activation_start + lane + 4] as i8)
                            * i32::from(grid2[lane])
                            * sign2;
                    }
                }
                block_sum += group_sum * pair_scale;
            }
        }

        sum += scale * block_sum as f32;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    block_count: usize,
    vnni: VnniMode,
) -> f32 {
    match weight_type {
        GgmlType::Q4_K | GgmlType::Q5_K => {
            // SAFETY: this function is AVX2-enabled and the caller validated
            // every packed block boundary.
            unsafe { dot_q4_or_q5_k_avx2(weight_type, weights, q8, block_count, vnni) }
        }
        GgmlType::IQ2_S => {
            // SAFETY: same preconditions as above.
            unsafe { dot_iq2_s_avx2(weights, q8, block_count, vnni) }
        }
        GgmlType::IQ3_S => {
            // SAFETY: same preconditions as above.
            unsafe { dot_iq3_s_avx2(weights, q8, block_count, vnni) }
        }
        _ => unreachable!("supported dot types were checked before AVX2 dispatch"),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_or_q5_k_avx2(
    weight_type: GgmlType,
    weights: &[u8],
    q8: &[u8],
    block_count: usize,
    vnni: VnniMode,
) -> f32 {
    let block_bytes = layout(weight_type)
        .expect("validated weight type has a layout")
        .block_bytes;
    let mut sums = [0.0_f32; 8];
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * block_bytes..(block_index + 1) * block_bytes];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let mut quants = [0_i8; Q8_K_BLOCK_ELEMENTS];

        match weight_type {
            GgmlType::Q4_K => {
                for group in 0..4 {
                    for lane in 0..32 {
                        let packed = weight[16 + group * 32 + lane];
                        quants[group * 64 + lane] = (packed & 0x0f) as i8;
                        quants[group * 64 + 32 + lane] = (packed >> 4) as i8;
                    }
                }
            }
            GgmlType::Q5_K => {
                let mut low_mask = 1_u8;
                let mut high_mask = 2_u8;
                for group in 0..4 {
                    for lane in 0..32 {
                        let packed = weight[48 + group * 32 + lane];
                        let high_bits = weight[16 + lane];
                        quants[group * 64 + lane] =
                            ((packed & 0x0f) + u8::from(high_bits & low_mask != 0) * 16) as i8;
                        quants[group * 64 + 32 + lane] =
                            ((packed >> 4) + u8::from(high_bits & high_mask != 0) * 16) as i8;
                    }
                    low_mask <<= 2;
                    high_mask <<= 2;
                }
            }
            _ => unreachable!("caller restricts this helper to Q4_K and Q5_K"),
        }

        let packed_scales = &weight[4..16];
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..8 {
            (scales[index], mins[index]) = scale_min(packed_scales, index);
        }

        let mut minimum_sum = 0_i32;
        for group in 0..16 {
            minimum_sum += i32::from(read_i16(activation, Q8_BLOCK_SUMS_OFFSET + group * 2))
                * i32::from(mins[group / 2]);
        }

        let mut lane_sums = [0_i32; 8];
        for (scale_index, &scale) in scales.iter().enumerate() {
            let offset = scale_index * 32;
            let weights = &quants[offset..offset + 32];
            let activations = &activation[Q8_QUANTS_OFFSET + offset..Q8_QUANTS_OFFSET + offset + 32];
            let raw = if vnni == VnniMode::Avx512 {
                // SAFETY: explicit dispatch validated AVX-512/VNNI.
                unsafe { lane_sums_i8_32_avx512_vnni(weights, activations) }
            } else {
                // SAFETY: both slices contain the complete validated 32-lane
                // region, and this function is AVX2-enabled.
                unsafe { lane_sums_i8_32(weights, activations) }
            };
            for lane in 0..8 {
                lane_sums[lane] += i32::from(scale) * raw[lane];
            }
        }

        let activation_scale = read_f32(activation, Q8_D_OFFSET);
        let scale = read_f16(weight, 0) * activation_scale;
        for lane in 0..8 {
            sums[lane] += scale * lane_sums[lane] as f32;
        }
        let minimum_scale = read_f16(weight, 2) * activation_scale;
        sum -= minimum_scale * minimum_sum as f32;
    }

    for lane_sum in sums {
        sum += lane_sum;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_iq2_s_avx2(weights: &[u8], q8: &[u8], block_count: usize, vnni: VnniMode) -> f32 {
    const BLOCK_BYTES: usize = 82;
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let scale = read_f16(weight, 0) * read_f32(activation, Q8_D_OFFSET);
        let mut block_sum = 0_i32;

        for group in 0..8 {
            let packed_scale = weight[74 + group];
            let scale1 = i32::from(1 + 2 * (packed_scale & 0x0f));
            let scale2 = i32::from(1 + 2 * (packed_scale >> 4));
            let high = usize::from(weight[66 + group]);
            let mut signed_grid = [0_i8; 32];

            for lane_group in 0..4 {
                let low = usize::from(weight[2 + group * 4 + lane_group]);
                let index = low | ((high << (8 - 2 * lane_group)) & 0x300);
                let grid = IQ2S_GRID[index].to_le_bytes();
                let signs = weight[34 + group * 4 + lane_group];
                for lane in 0..8 {
                    let magnitude = grid[lane] as i8;
                    signed_grid[lane_group * 8 + lane] = if signs & KMASK_IQ2XS[lane] == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                }
            }

            let activation_start = Q8_QUANTS_OFFSET + group * 32;
            let sum1 = match vnni {
                VnniMode::Avx512 => {
                    // SAFETY: explicit dispatch validated AVX-512/VNNI.
                    unsafe {
                        dot_i8_avx512_vnni(
                            &signed_grid[..16],
                            &activation[activation_start..activation_start + 16],
                        )
                    }
                }
                VnniMode::Avx => {
                    // SAFETY: explicit dispatch validated AVX-VNNI and both
                    // slices contain exactly sixteen lanes.
                    unsafe {
                        dot_i8_avx_vnni_16(
                            &signed_grid[..16],
                            &activation[activation_start..activation_start + 16],
                        )
                    }
                }
                VnniMode::None => {
                    // SAFETY: each half is exactly 16 lanes and this function
                    // is AVX2-enabled.
                    unsafe {
                        dot_i8_16(
                            &signed_grid[..16],
                            &activation[activation_start..activation_start + 16],
                        )
                    }
                }
            };
            let sum2 = match vnni {
                VnniMode::Avx512 => {
                    // SAFETY: same VNNI conditions for the second half.
                    unsafe {
                        dot_i8_avx512_vnni(
                            &signed_grid[16..],
                            &activation[activation_start + 16..activation_start + 32],
                        )
                    }
                }
                VnniMode::Avx => {
                    // SAFETY: same AVX-VNNI conditions for the second half.
                    unsafe {
                        dot_i8_avx_vnni_16(
                            &signed_grid[16..],
                            &activation[activation_start + 16..activation_start + 32],
                        )
                    }
                }
                VnniMode::None => {
                    // SAFETY: same AVX2 conditions for the second half.
                    unsafe {
                        dot_i8_16(
                            &signed_grid[16..],
                            &activation[activation_start + 16..activation_start + 32],
                        )
                    }
                }
            };
            block_sum += scale1 * sum1 + scale2 * sum2;
        }

        sum += scale * block_sum as f32;
    }

    0.125_f32 * sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_iq3_s_avx2(weights: &[u8], q8: &[u8], block_count: usize, vnni: VnniMode) -> f32 {
    const BLOCK_BYTES: usize = 110;
    let mut sum = 0.0_f32;

    for block_index in 0..block_count {
        let weight = &weights[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
        let scale = read_f16(weight, 0) * read_f32(activation, Q8_D_OFFSET);
        let mut block_sum = 0_i32;

        for pair in 0..4 {
            let packed_scale = weight[106 + pair];
            let pair_scales = [
                i32::from(2 * (packed_scale & 0x0f) + 1),
                i32::from(2 * (packed_scale >> 4) + 1),
            ];

            for (half, &pair_scale) in pair_scales.iter().enumerate() {
                let group = pair * 2 + half;
                let high = usize::from(weight[66 + group]);
                let quant_start = 2 + group * 8;
                let sign_start = 74 + group * 4;
                let mut signed_grid = [0_i8; 32];

                for lane_group in 0..4 {
                    let low1 = usize::from(weight[quant_start + lane_group * 2]);
                    let low2 = usize::from(weight[quant_start + lane_group * 2 + 1]);
                    let index1 = low1 | ((high << (8 - 2 * lane_group)) & 0x100);
                    let index2 = low2 | ((high << (7 - 2 * lane_group)) & 0x100);
                    let grid1 = IQ3S_GRID[index1].to_le_bytes();
                    let grid2 = IQ3S_GRID[index2].to_le_bytes();
                    let signs = weight[sign_start + lane_group];
                    for lane in 0..4 {
                        let magnitude1 = grid1[lane] as i8;
                        let magnitude2 = grid2[lane] as i8;
                        signed_grid[lane_group * 8 + lane] = if signs & KMASK_IQ2XS[lane] == 0 {
                            magnitude1
                        } else {
                            -magnitude1
                        };
                        signed_grid[lane_group * 8 + lane + 4] = if signs & KMASK_IQ2XS[lane + 4] == 0 {
                            magnitude2
                        } else {
                            -magnitude2
                        };
                    }
                }

                let activation_start = Q8_QUANTS_OFFSET + group * 32;
                let group_sum = match vnni {
                    VnniMode::Avx512 => {
                        // SAFETY: explicit dispatch validated AVX-512/VNNI.
                        unsafe {
                            dot_i8_avx512_vnni(
                                &signed_grid,
                                &activation[activation_start..activation_start + 32],
                            )
                        }
                    }
                    VnniMode::Avx => {
                        // SAFETY: explicit dispatch validated AVX-VNNI and
                        // both slices contain exactly 32 lanes.
                        unsafe {
                            dot_i8_avx_vnni_32(
                                &signed_grid,
                                &activation[activation_start..activation_start + 32],
                            )
                        }
                    }
                    VnniMode::None => {
                        // SAFETY: both slices contain 32 lanes and this
                        // function is AVX2-enabled.
                        unsafe {
                            dot_i8_32(&signed_grid, &activation[activation_start..activation_start + 32])
                        }
                    }
                };
                block_sum += group_sum * pair_scale;
            }
        }

        sum += scale * block_sum as f32;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lane_sums_i8_32(weights: &[i8], activations: &[u8]) -> [i32; 8] {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
        _mm256_cvtepi8_epi16, _mm256_extracti128_si256, _mm256_mullo_epi16, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm_loadu_si128,
    };

    debug_assert!(weights.len() >= 32);
    debug_assert!(activations.len() >= 32);
    let mut accumulated = _mm256_setzero_si256();
    for offset in [0_usize, 16] {
        // SAFETY: the caller provides at least 32 bytes, so both unaligned
        // 16-byte loads are within their respective slices.
        let weight_i8 = unsafe { _mm_loadu_si128(weights.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: same bound as the weight load.
        let activation_i8 = unsafe { _mm_loadu_si128(activations.as_ptr().add(offset).cast::<__m128i>()) };
        let weight_i16 = _mm256_cvtepi8_epi16(weight_i8);
        let activation_i16 = _mm256_cvtepi8_epi16(activation_i8);
        let product_i16 = _mm256_mullo_epi16(weight_i16, activation_i16);
        let low_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(product_i16));
        let high_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_i16, 1));
        accumulated = _mm256_add_epi32(accumulated, low_i32);
        accumulated = _mm256_add_epi32(accumulated, high_i32);
    }
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` is exactly one 256-bit vector and unaligned stores are
    // permitted by this intrinsic.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), accumulated) };
    lanes
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx_vnni_16(weights: &[i8], activations: &[u8]) -> i32 {
    use std::arch::asm;

    debug_assert!(weights.len() >= 16);
    debug_assert!(activations.len() >= 16);
    let rebase = [0x80_u8; 16];
    let mut lanes = [0_i32; 4];
    // `VPDPBUSD` consumes unsigned activation bytes and signed weight bytes.
    // XOR with 0x80 maps each signed activation byte to `value + 128`;
    // subtracting 128 times the weight sum restores the exact signed dot.
    //
    // SAFETY: runtime dispatch proved AVX-VNNI support and every pointer
    // references a complete unaligned 16-byte region.
    unsafe {
        asm!(
            "vmovdqu xmm0, [{weights}]",
            "vmovdqu xmm1, [{activations}]",
            "vmovdqu xmm2, [{rebase}]",
            "vpxor xmm1, xmm1, xmm2",
            "vpxor xmm3, xmm3, xmm3",
            "vpdpbusd xmm3, xmm1, xmm0",
            "vmovdqu [{lanes}], xmm3",
            weights = in(reg) weights.as_ptr(),
            activations = in(reg) activations.as_ptr(),
            rebase = in(reg) rebase.as_ptr(),
            lanes = in(reg) lanes.as_mut_ptr(),
            lateout("xmm0") _,
            lateout("xmm1") _,
            lateout("xmm2") _,
            lateout("xmm3") _,
            options(nostack, preserves_flags),
        );
    }
    let shifted_sum = lanes.into_iter().sum::<i32>();
    let weight_sum = weights[..16].iter().map(|&weight| i32::from(weight)).sum::<i32>();
    shifted_sum - 128 * weight_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx_vnni_32(weights: &[i8], activations: &[u8]) -> i32 {
    use std::arch::asm;

    debug_assert!(weights.len() >= 32);
    debug_assert!(activations.len() >= 32);
    let rebase = [0x80_u8; 32];
    let mut lanes = [0_i32; 8];
    // SAFETY: runtime dispatch proved AVX-VNNI support and every pointer
    // references a complete unaligned 32-byte region.
    unsafe {
        asm!(
            "vmovdqu ymm0, [{weights}]",
            "vmovdqu ymm1, [{activations}]",
            "vmovdqu ymm2, [{rebase}]",
            "vpxor ymm1, ymm1, ymm2",
            "vpxor ymm3, ymm3, ymm3",
            "vpdpbusd ymm3, ymm1, ymm0",
            "vmovdqu [{lanes}], ymm3",
            weights = in(reg) weights.as_ptr(),
            activations = in(reg) activations.as_ptr(),
            rebase = in(reg) rebase.as_ptr(),
            lanes = in(reg) lanes.as_mut_ptr(),
            lateout("ymm0") _,
            lateout("ymm1") _,
            lateout("ymm2") _,
            lateout("ymm3") _,
            options(nostack, preserves_flags),
        );
    }
    let shifted_sum = lanes.into_iter().sum::<i32>();
    let weight_sum = weights[..32].iter().map(|&weight| i32::from(weight)).sum::<i32>();
    shifted_sum - 128 * weight_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn lane_sums_i8_32_avx512_vnni(weights: &[i8], activations: &[u8]) -> [i32; 8] {
    debug_assert!(weights.len() >= 32);
    debug_assert!(activations.len() >= 32);
    let mut reordered_weights = [0_i8; 64];
    let mut reordered_activations = [0_i8; 64];
    for lane in 0..8 {
        for group in 0..4 {
            let source = group * 8 + lane;
            let destination = lane * 4 + group;
            reordered_weights[destination] = weights[source];
            reordered_activations[destination] = (i16::from(activations[source] as i8) + 128) as i8;
        }
    }
    let mut lanes = [0_i32; 16];
    // SAFETY: runtime dispatch proved AVX-512 VNNI support and all three local
    // arrays contain a complete 64-byte unaligned vector.
    unsafe {
        dpbusd_64(
            reordered_weights.as_ptr(),
            reordered_activations.as_ptr(),
            lanes.as_mut_ptr(),
        );
    }
    let mut corrected = [0_i32; 8];
    for lane in 0..8 {
        let weight_sum = reordered_weights[lane * 4..lane * 4 + 4]
            .iter()
            .map(|&weight| i32::from(weight))
            .sum::<i32>();
        corrected[lane] = lanes[lane] - 128 * weight_sum;
    }
    corrected
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dot_i8_avx512_vnni(weights: &[i8], activations: &[u8]) -> i32 {
    debug_assert_eq!(weights.len(), activations.len());
    debug_assert!(weights.len() <= 64);
    let mut padded_weights = [0_i8; 64];
    let mut padded_activations = [0_i8; 64];
    padded_weights[..weights.len()].copy_from_slice(weights);
    for (destination, &activation) in padded_activations.iter_mut().zip(activations) {
        *destination = (i16::from(activation as i8) + 128) as i8;
    }
    let mut lanes = [0_i32; 16];
    // SAFETY: runtime dispatch proved AVX-512 VNNI support and all three local
    // arrays contain a complete 64-byte unaligned vector.
    unsafe {
        dpbusd_64(
            padded_weights.as_ptr(),
            padded_activations.as_ptr(),
            lanes.as_mut_ptr(),
        );
    }
    let shifted_sum = lanes.into_iter().sum::<i32>();
    let weight_sum = weights.iter().map(|&weight| i32::from(weight)).sum::<i32>();
    shifted_sum - 128 * weight_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dpbusd_64(weights: *const i8, activations: *const i8, output: *mut i32) {
    use std::arch::asm;

    // `VPDPBUSD` interprets its first byte source as unsigned and its second as
    // signed. Activations have already been rebased into the unsigned 0..=255
    // domain, while weights retain their signed byte representation.
    //
    // SAFETY: callers provide three valid 64-byte regions and runtime dispatch
    // has established every target feature used by these instructions.
    unsafe {
        asm!(
            "vmovdqu64 zmm0, [{weights}]",
            "vmovdqu64 zmm1, [{activations}]",
            "vpxord zmm2, zmm2, zmm2",
            "vpdpbusd zmm2, zmm1, zmm0",
            "vmovdqu64 [{output}], zmm2",
            weights = in(reg) weights,
            activations = in(reg) activations,
            output = in(reg) output,
            lateout("zmm0") _,
            lateout("zmm1") _,
            lateout("zmm2") _,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_32(weights: &[i8], activations: &[u8]) -> i32 {
    // SAFETY: the caller guarantees both 32-lane slices and AVX2 support.
    unsafe { lane_sums_i8_32(weights, activations) }.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_16(weights: &[i8], activations: &[u8]) -> i32 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_cvtepi8_epi16, _mm256_madd_epi16, _mm256_mullo_epi16, _mm256_set1_epi16,
        _mm256_storeu_si256, _mm_loadu_si128,
    };

    debug_assert!(weights.len() >= 16);
    debug_assert!(activations.len() >= 16);
    // SAFETY: the caller provides complete 16-byte slices.
    let weight_i8 = unsafe { _mm_loadu_si128(weights.as_ptr().cast::<__m128i>()) };
    // SAFETY: same bound as the weight load.
    let activation_i8 = unsafe { _mm_loadu_si128(activations.as_ptr().cast::<__m128i>()) };
    let product_i16 = _mm256_mullo_epi16(
        _mm256_cvtepi8_epi16(weight_i8),
        _mm256_cvtepi8_epi16(activation_i8),
    );
    let pair_sums = _mm256_madd_epi16(product_i16, _mm256_set1_epi16(1));
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` is exactly one vector wide.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), pair_sums) };
    lanes.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_int_matches_pinned_round_to_nearest_even_boundaries() {
        let cases = [
            (-2.5_f32, -2),
            (-1.5, -2),
            (-0.5, 0),
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (3.5, 4),
        ];
        for (input, expected) in cases {
            assert_eq!(nearest_int(input), expected);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_vnni_signed_dot_helpers_match_scalar() {
        if !avx_vnni_available() {
            return;
        }
        let weights = (0..32)
            .map(|index| (index as i8).wrapping_mul(17).wrapping_add(91))
            .collect::<Vec<_>>();
        let activations = (0..32)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(137))
            .collect::<Vec<_>>();
        let scalar = |length: usize| {
            weights[..length]
                .iter()
                .zip(&activations[..length])
                .map(|(&weight, &activation)| i32::from(weight) * i32::from(activation as i8))
                .sum::<i32>()
        };
        // SAFETY: the availability guard and complete slices satisfy both
        // helper contracts.
        assert_eq!(unsafe { dot_i8_avx_vnni_16(&weights, &activations) }, scalar(16));
        // SAFETY: same guard and complete 32-lane slices.
        assert_eq!(unsafe { dot_i8_avx_vnni_32(&weights, &activations) }, scalar(32));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_vnni_iq3_groups_match_avx2_integer_reduction() {
        if !avx_vnni_available() {
            return;
        }
        let weights = include_bytes!("../tests/fixtures/decode-iq3-s.input.bin");
        let q8 = include_bytes!("../tests/fixtures/q8-k-activations.output-q8-k.bin");
        for block_index in 0..3 {
            let weight = &weights[block_index * 110..(block_index + 1) * 110];
            let activation = &q8[block_index * Q8_K_BLOCK_BYTES..(block_index + 1) * Q8_K_BLOCK_BYTES];
            for pair in 0..4 {
                for half in 0..2 {
                    let group = pair * 2 + half;
                    let high = usize::from(weight[66 + group]);
                    let quant_start = 2 + group * 8;
                    let sign_start = 74 + group * 4;
                    let mut signed_grid = [0_i8; 32];
                    for lane_group in 0..4 {
                        let low1 = usize::from(weight[quant_start + lane_group * 2]);
                        let low2 = usize::from(weight[quant_start + lane_group * 2 + 1]);
                        let index1 = low1 | ((high << (8 - 2 * lane_group)) & 0x100);
                        let index2 = low2 | ((high << (7 - 2 * lane_group)) & 0x100);
                        let grid1 = IQ3S_GRID[index1].to_le_bytes();
                        let grid2 = IQ3S_GRID[index2].to_le_bytes();
                        let signs = weight[sign_start + lane_group];
                        for lane in 0..4 {
                            signed_grid[lane_group * 8 + lane] = if signs & KMASK_IQ2XS[lane] == 0 {
                                grid1[lane] as i8
                            } else {
                                -(grid1[lane] as i8)
                            };
                            signed_grid[lane_group * 8 + lane + 4] = if signs & KMASK_IQ2XS[lane + 4] == 0 {
                                grid2[lane] as i8
                            } else {
                                -(grid2[lane] as i8)
                            };
                        }
                    }
                    let start = Q8_QUANTS_OFFSET + group * 32;
                    let activations = &activation[start..start + 32];
                    // SAFETY: availability and complete slices satisfy both
                    // helper contracts.
                    let expected = unsafe { dot_i8_32(&signed_grid, activations) };
                    let actual = unsafe { dot_i8_avx_vnni_32(&signed_grid, activations) };
                    assert_eq!(actual, expected, "block {block_index}, pair {pair}, half {half}");
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_vnni_iq3_complete_dot_matches_avx2() {
        if !avx_vnni_available() {
            return;
        }
        let weights = include_bytes!("../tests/fixtures/decode-iq3-s.input.bin");
        let q8 = include_bytes!("../tests/fixtures/q8-k-activations.output-q8-k.bin");
        // SAFETY: availability and fixture lengths satisfy both helpers.
        let expected = unsafe { dot_iq3_s_avx2(weights, q8, 3, VnniMode::None) };
        let actual = unsafe { dot_iq3_s_avx2(weights, q8, 3, VnniMode::Avx) };
        assert_eq!(actual.to_bits(), expected.to_bits());
        let public = ValidatedQ8KMatrix::new(GgmlType::IQ3_S, weights, q8, 768, 1, CpuDotBackend::AvxVnni)
            .unwrap()
            .dot_row(0)
            .unwrap();
        assert_eq!(public.to_bits(), expected.to_bits());
    }
}
