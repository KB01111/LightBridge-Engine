typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef signed short i16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u16 read_u16(const u8 *bytes) {
    return (u16)bytes[0] | ((u16)bytes[1] << 8);
}

__device__ __forceinline__ i16 read_i16(const u8 *bytes) {
    return (i16)read_u16(bytes);
}

__device__ __forceinline__ u32 read_u32(const u8 *bytes) {
    return (u32)bytes[0]
        | ((u32)bytes[1] << 8)
        | ((u32)bytes[2] << 16)
        | ((u32)bytes[3] << 24);
}

__device__ __forceinline__ float read_f32(const u8 *bytes) {
    return __uint_as_float(read_u32(bytes));
}

__device__ __forceinline__ float half_to_float(u16 value) {
    u32 sign = ((u32)value & 0x8000u) << 16;
    u32 exponent = ((u32)value >> 10) & 0x1fu;
    u32 mantissa = (u32)value & 0x03ffu;
    u32 bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((u32)(unbiased + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float read_f16(const u8 *bytes) {
    return half_to_float(read_u16(bytes));
}

__device__ __forceinline__ void scale_min(
    const u8 *scales,
    int index,
    int *scale,
    int *minimum
) {
    if (index < 4) {
        *scale = (int)(scales[index] & 0x3fu);
        *minimum = (int)(scales[index + 4] & 0x3fu);
    } else {
        *scale = (int)((scales[index + 4] & 0x0fu)
            | ((scales[index - 4] >> 6) << 4));
        *minimum = (int)((scales[index + 4] >> 4)
            | ((scales[index] >> 6) << 4));
    }
}

__device__ __forceinline__ int q4_q5_quant(
    int kind,
    const u8 *weight,
    int offset
) {
    int group = offset / 64;
    int within = offset - group * 64;
    int lane = within & 31;
    int high_half = within >= 32;
    if (kind == 0) {
        u8 packed = weight[16 + group * 32 + lane];
        return high_half ? (int)(packed >> 4) : (int)(packed & 0x0f);
    }
    u8 packed = weight[48 + group * 32 + lane];
    u8 high_bits = weight[16 + lane];
    int mask = (high_half ? 2 : 1) << (2 * group);
    int low = high_half ? (int)(packed >> 4) : (int)(packed & 0x0f);
    return low + ((high_bits & mask) != 0 ? 16 : 0);
}

__device__ float dot_q4_q5(
    int kind,
    const u8 *weights,
    const u8 *q8,
    int block_count
) {
    int block_bytes = kind == 0 ? Q4_K_BLOCK_BYTES : Q5_K_BLOCK_BYTES;
    float lane_totals[8] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const u8 *weight = weights + block * block_bytes;
        const u8 *activation = q8 + block * Q8_K_BLOCK_BYTES;
        int scales[8];
        int minimums[8];
        for (int index = 0; index < 8; ++index) {
            scale_min(weight + 4, index, &scales[index], &minimums[index]);
        }

        int minimum_sum = 0;
        for (int group = 0; group < 16; ++group) {
            minimum_sum += (int)read_i16(activation + Q8_BLOCK_SUMS_OFFSET + group * 2)
                * minimums[group / 2];
        }

        int lane_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int offset = 0; offset < 256; ++offset) {
            int lane = offset & 7;
            int activation_value = (int)(i8)activation[Q8_QUANTS_OFFSET + offset];
            int weight_value = q4_q5_quant(kind, weight, offset);
            lane_sums[lane] += scales[offset / 32]
                * activation_value
                * weight_value;
        }

        float activation_scale = read_f32(activation);
        float scale = read_f16(weight) * activation_scale;
        for (int lane = 0; lane < 8; ++lane) {
            lane_totals[lane] += scale * (float)lane_sums[lane];
        }
        float minimum_scale = read_f16(weight + 2) * activation_scale;
        total -= minimum_scale * (float)minimum_sum;
    }
    for (int lane = 0; lane < 8; ++lane) {
        total += lane_totals[lane];
    }
    return total;
}

__device__ float dot_iq2(
    const u8 *weights,
    const u8 *q8,
    const u64 *grid,
    int block_count
) {
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const u8 *weight = weights + block * 82;
        const u8 *activation = q8 + block * 292;
        float scale = read_f16(weight) * read_f32(activation);
        int block_sum = 0;
        for (int group = 0; group < 8; ++group) {
            u8 packed_scale = weight[74 + group];
            int scale1 = 1 + 2 * (int)(packed_scale & 0x0f);
            int scale2 = 1 + 2 * (int)(packed_scale >> 4);
            int high = (int)weight[66 + group];
            int sums[2] = {0, 0};
            for (int lane_group = 0; lane_group < 4; ++lane_group) {
                int low = (int)weight[2 + group * 4 + lane_group];
                int index = low | ((high << (8 - 2 * lane_group)) & 0x300);
                u64 magnitudes = grid[index];
                u8 signs = weight[34 + group * 4 + lane_group];
                int activation_start = 4 + group * 32 + lane_group * 8;
                for (int lane = 0; lane < 8; ++lane) {
                    int magnitude = (int)((magnitudes >> (lane * 8)) & 0xffu);
                    int sign = (signs & (1u << lane)) == 0 ? 1 : -1;
                    sums[lane_group / 2] +=
                        (int)(i8)activation[activation_start + lane]
                        * magnitude
                        * sign;
                }
            }
            block_sum += scale1 * sums[0] + scale2 * sums[1];
        }
        total += scale * (float)block_sum;
    }
    return 0.125f * total;
}

__device__ float dot_iq3(
    const u8 *weights,
    const u8 *q8,
    const u32 *grid,
    int block_count
) {
    float total = 0.0f;
    for (int block = 0; block < block_count; ++block) {
        const u8 *weight = weights + block * 110;
        const u8 *activation = q8 + block * 292;
        float scale = read_f16(weight) * read_f32(activation);
        int block_sum = 0;
        for (int pair = 0; pair < 4; ++pair) {
            u8 packed_scale = weight[106 + pair];
            int pair_scales[2] = {
                2 * (int)(packed_scale & 0x0f) + 1,
                2 * (int)(packed_scale >> 4) + 1
            };
            for (int half = 0; half < 2; ++half) {
                int group = pair * 2 + half;
                int high = (int)weight[66 + group];
                int quant_start = 2 + group * 8;
                int sign_start = 74 + group * 4;
                int activation_group = 4 + group * 32;
                int group_sum = 0;
                for (int lane_group = 0; lane_group < 4; ++lane_group) {
                    int low1 = (int)weight[quant_start + lane_group * 2];
                    int low2 = (int)weight[quant_start + lane_group * 2 + 1];
                    int index1 = low1 | ((high << (8 - 2 * lane_group)) & 0x100);
                    int index2 = low2 | ((high << (7 - 2 * lane_group)) & 0x100);
                    u32 magnitudes1 = grid[index1];
                    u32 magnitudes2 = grid[index2];
                    u8 signs = weight[sign_start + lane_group];
                    int activation_start = activation_group + lane_group * 8;
                    for (int lane = 0; lane < 4; ++lane) {
                        int magnitude1 = (int)((magnitudes1 >> (lane * 8)) & 0xffu);
                        int magnitude2 = (int)((magnitudes2 >> (lane * 8)) & 0xffu);
                        int sign1 = (signs & (1u << lane)) == 0 ? 1 : -1;
                        int sign2 = (signs & (1u << (lane + 4))) == 0 ? 1 : -1;
                        group_sum +=
                            (int)(i8)activation[activation_start + lane]
                            * magnitude1
                            * sign1;
                        group_sum +=
                            (int)(i8)activation[activation_start + lane + 4]
                            * magnitude2
                            * sign2;
                    }
                }
                block_sum += group_sum * pair_scales[half];
            }
        }
        total += scale * (float)block_sum;
    }
    return total;
}

extern "C" __global__ void bridge_q8k_gemv_v1(
    int kind,
    const u8 *weights,
    const u8 *q8,
    const u64 *iq2_grid,
    const u32 *iq3_grid,
    int logical_elements,
    int rows,
    float *output
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    int block_count = logical_elements / 256;
    int block_bytes = kind == 0 ? 144 : (kind == 1 ? 176 : (kind == 2 ? 82 : 110));
    const u8 *row_weights = weights + row * block_count * block_bytes;
    if (kind == 0 || kind == 1) {
        output[row] = dot_q4_q5(kind, row_weights, q8, block_count);
    } else if (kind == 2) {
        output[row] = dot_iq2(row_weights, q8, iq2_grid, block_count);
    } else {
        output[row] = dot_iq3(row_weights, q8, iq3_grid, block_count);
    }
}
