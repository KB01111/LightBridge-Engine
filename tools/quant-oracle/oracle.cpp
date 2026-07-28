// Development-only scalar quantization oracle for LightBridge.
//
// This local harness is MIT licensed. It calls only authenticated llama.cpp
// b10153 scalar/reference functions and does not form part of the Rust runtime.

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "quants.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

extern "C" {
float ggml_table_f32_f16[1 << 16];
}

namespace {

constexpr std::size_t kBlockElements = 256;
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ5KBytes = 176;
constexpr std::size_t kIq2SBytes = 82;
constexpr std::size_t kIq3SBytes = 110;
constexpr std::size_t kQ8KBytes = 292;
constexpr std::size_t kQuantBlockCount = 3;
constexpr std::size_t kDotElements = kBlockElements * kQuantBlockCount;

static_assert(sizeof(float) == 4);
static_assert(std::numeric_limits<float>::is_iec559);
static_assert(sizeof(block_q4_K) == kQ4KBytes);
static_assert(sizeof(block_q5_K) == kQ5KBytes);
static_assert(sizeof(block_iq2_s) == kIq2SBytes);
static_assert(sizeof(block_iq3_s) == kIq3SBytes);
static_assert(sizeof(block_q8_K) == kQ8KBytes);
static_assert(alignof(block_q4_K) >= alignof(ggml_fp16_t));
static_assert(alignof(block_q5_K) >= alignof(ggml_fp16_t));
static_assert(alignof(block_iq2_s) >= alignof(ggml_fp16_t));
static_assert(alignof(block_iq3_s) >= alignof(ggml_fp16_t));
static_assert(alignof(block_q8_K) >= alignof(float));
static_assert(std::is_trivially_copyable_v<block_q4_K>);
static_assert(std::is_trivially_copyable_v<block_q5_K>);
static_assert(std::is_trivially_copyable_v<block_iq2_s>);
static_assert(std::is_trivially_copyable_v<block_iq3_s>);
static_assert(std::is_trivially_copyable_v<block_q8_K>);

using ByteVector = std::vector<std::uint8_t>;
using DotFunction = void (*)(
    int,
    float *,
    std::size_t,
    const void *,
    std::size_t,
    const void *,
    std::size_t,
    int);

struct Artifact {
    std::string name;
    ByteVector bytes;
};

enum class WeightType {
    F32,
    Q4K,
    Q5K,
    Iq2S,
    Iq3S,
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    std::uint8_t first = 0;
    std::memcpy(&first, &value, sizeof(first));
    return first == 1;
}

std::size_t checked_product(
    const std::size_t left,
    const std::size_t right,
    const char * context) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        fail(std::string(context) + " size overflow");
    }
    return left * right;
}

void validate_k_n(const std::size_t n, const char * context) {
    if (n == 0) {
        fail(std::string(context) + " requires n > 0");
    }
    if (n % kBlockElements != 0) {
        fail(std::string(context) + " requires n % 256 == 0");
    }
    if (n > static_cast<std::size_t>(INT_MAX)) {
        fail(std::string(context) + " requires n <= INT_MAX");
    }
}

void write_u16_le(ByteVector & bytes, const std::size_t offset, const std::uint16_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 2) {
        fail("u16 write is out of range");
    }
    bytes[offset] = static_cast<std::uint8_t>(value & 0xffu);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t read_u16_le(const std::uint8_t * bytes) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[0]) |
        (static_cast<std::uint16_t>(bytes[1]) << 8u));
}

std::int16_t read_i16_le(const std::uint8_t * bytes) {
    const std::uint16_t bits = read_u16_le(bytes);
    std::int16_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float read_f32_le(const std::uint8_t * bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

void write_binary(const std::filesystem::path & path, const void * data, const std::size_t bytes) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail("output is too large for std::streamsize");
    }
    const auto temporary =
        path.parent_path() / (path.filename().string() + ".bridge-quant-oracle.tmp");
    std::error_code error;
    if (std::filesystem::exists(temporary, error) || error) {
        fail("refusing to reuse oracle temporary path " + temporary.string());
    }
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            fail("cannot open temporary output " + temporary.string());
        }
        stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        if (!stream) {
            stream.close();
            std::filesystem::remove(temporary, error);
            fail("cannot write temporary output " + temporary.string());
        }
        stream.flush();
        if (!stream) {
            stream.close();
            std::filesystem::remove(temporary, error);
            fail("cannot flush temporary output " + temporary.string());
        }
        stream.close();
        if (!stream) {
            std::filesystem::remove(temporary, error);
            fail("cannot close temporary output " + temporary.string());
        }
    }
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD code = GetLastError();
        std::filesystem::remove(temporary, error);
        fail("cannot atomically promote output " + path.string() +
             " (Windows error " + std::to_string(code) + ")");
    }
#else
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        fail("cannot atomically promote output " + path.string());
    }
#endif
}

void write_binary(const std::filesystem::path & path, const ByteVector & bytes) {
    write_binary(path, bytes.data(), bytes.size());
}

ByteVector read_binary_exact(
    const std::filesystem::path & path,
    const std::size_t expected,
    const char * context) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status)) {
        fail(std::string(context) + " input must be a direct regular file");
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error || size != expected) {
        fail(std::string(context) + " input length mismatch");
    }
    ByteVector bytes(expected);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fail(std::string("cannot open ") + context + " input");
    }
    stream.read(
        reinterpret_cast<char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        fail(std::string("cannot read exact ") + context + " input");
    }
    return bytes;
}

ByteVector bytes_of_floats(const std::vector<float> & values) {
    const std::size_t byte_count = checked_product(values.size(), sizeof(float), "F32 output");
    ByteVector bytes(byte_count);
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), values.data(), byte_count);
    }
    return bytes;
}

std::uint32_t next_lcg(std::uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

ByteVector make_three_blocks(
    const std::size_t block_bytes,
    const std::array<std::uint16_t, 2> structural_scales,
    const std::array<std::uint16_t, 2> lcg_scales,
    const bool has_second_scale) {
    ByteVector result(checked_product(block_bytes, kQuantBlockCount, "quantized input"));

    for (std::size_t i = 0; i < block_bytes; ++i) {
        result[i] = static_cast<std::uint8_t>((i * 37u + (i >> 1u) * 11u + 0x53u) & 0xffu);
    }
    write_u16_le(result, 0, structural_scales[0]);
    if (has_second_scale) {
        write_u16_le(result, 2, structural_scales[1]);
    }

    std::uint32_t state = 0x6d2b79f5u;
    for (std::size_t i = 0; i < block_bytes; ++i) {
        result[block_bytes + i] = static_cast<std::uint8_t>(next_lcg(state) >> 24u);
    }
    write_u16_le(result, block_bytes, lcg_scales[0]);
    if (has_second_scale) {
        write_u16_le(result, block_bytes + 2, lcg_scales[1]);
    }

    for (std::size_t i = 0; i < block_bytes; ++i) {
        result[2 * block_bytes + i] =
            static_cast<std::uint8_t>((0xa5u ^ (i * 29u) ^ (i >> 2u)) & 0xffu);
    }
    write_u16_le(result, 2 * block_bytes, 0);
    if (has_second_scale) {
        write_u16_le(result, 2 * block_bytes + 2, 0);
    }

    return result;
}

template <typename Block>
std::vector<Block> copy_blocks(
    const ByteVector & encoded,
    const std::size_t n,
    const char * context) {
    validate_k_n(n, context);
    const std::size_t blocks = n / kBlockElements;
    const std::size_t expected = checked_product(blocks, sizeof(Block), context);
    if (encoded.size() != expected) {
        fail(std::string(context) + " input length mismatch");
    }
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t offset = checked_product(block, sizeof(Block), context);
        const auto * raw = encoded.data() + offset;
        const float d = ggml_fp16_to_fp32(
            static_cast<ggml_fp16_t>(read_u16_le(raw)));
        if (!std::isfinite(d)) {
            fail(std::string(context) + " rejects non-finite d scale");
        }
        if constexpr (
            std::is_same_v<Block, block_q4_K> ||
            std::is_same_v<Block, block_q5_K>) {
            const float dmin = ggml_fp16_to_fp32(
                static_cast<ggml_fp16_t>(read_u16_le(raw + 2)));
            if (!std::isfinite(dmin)) {
                fail(std::string(context) + " rejects non-finite dmin scale");
            }
        }
    }
    std::vector<Block> aligned(blocks);
    std::memcpy(aligned.data(), encoded.data(), expected);
    return aligned;
}

template <typename Block>
ByteVector decode_blocks(
    const ByteVector & encoded,
    const std::size_t n,
    void (*function)(const Block *, float *, std::int64_t),
    const char * context) {
    validate_k_n(n, context);
    const auto aligned = copy_blocks<Block>(encoded, n, context);
    const std::size_t output_bytes = checked_product(n, sizeof(float), context);
    std::vector<float> output(n);
    if (output_bytes != checked_product(output.size(), sizeof(float), context)) {
        fail(std::string(context) + " output length mismatch");
    }
    function(aligned.data(), output.data(), static_cast<std::int64_t>(n));
    if (!std::all_of(output.begin(), output.end(), [](const float value) {
            return std::isfinite(value);
        })) {
        fail(std::string(context) + " produced non-finite output");
    }
    return bytes_of_floats(output);
}

std::vector<float> make_q8_activations() {
    std::vector<float> values(kDotElements, 0.0f);

    for (std::size_t i = 0; i < kBlockElements; ++i) {
        const int centered = static_cast<int>(i % 31u) - 15;
        const float base = static_cast<float>(centered) * 0.125f;
        values[i] = (i % 13u == 0) ? -0.0f : ((i & 1u) == 0 ? base : -base);
    }

    std::uint32_t state = 0x12345678u;
    for (std::size_t i = 0; i < kBlockElements; ++i) {
        const std::int32_t centered =
            static_cast<std::int32_t>((next_lcg(state) >> 8u) & 0xffffu) - 32768;
        values[kBlockElements + i] = static_cast<float>(centered) / 4096.0f;
    }

    return values;
}

std::vector<block_q8_K> quantize_q8(const std::vector<float> & values) {
    const std::size_t n = values.size();
    validate_k_n(n, "Q8_K quantization");
    if (!std::all_of(values.begin(), values.end(), [](const float value) {
            return std::isfinite(value);
        })) {
        fail("Q8_K quantization rejects non-finite activation input");
    }
    const std::size_t blocks = n / kBlockElements;
    const std::size_t output_bytes = checked_product(blocks, sizeof(block_q8_K), "Q8_K output");
    std::vector<block_q8_K> output(blocks, {});
    if (output_bytes != checked_product(output.size(), sizeof(block_q8_K), "Q8_K output")) {
        fail("Q8_K output length mismatch");
    }
    quantize_row_q8_K_ref(values.data(), output.data(), static_cast<std::int64_t>(n));
    return output;
}

std::vector<block_q8_K> copy_q8_blocks(
    const ByteVector & encoded,
    const std::size_t n,
    const char * context) {
    validate_k_n(n, context);
    const std::size_t blocks = n / kBlockElements;
    const std::size_t expected = checked_product(blocks, sizeof(block_q8_K), context);
    if (encoded.size() != expected) {
        fail(std::string(context) + " Q8_K input length mismatch");
    }
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto * raw = encoded.data() + checked_product(block, sizeof(block_q8_K), context);
        const float d = read_f32_le(raw);
        if (!std::isfinite(d)) {
            fail(std::string(context) + " rejects non-finite Q8_K d");
        }
        for (std::size_t group = 0; group < 16; ++group) {
            std::int32_t sum = 0;
            for (std::size_t lane = 0; lane < 16; ++lane) {
                const std::uint8_t byte = raw[4 + group * 16 + lane];
                sum += byte <= 127u
                    ? static_cast<std::int32_t>(byte)
                    : static_cast<std::int32_t>(byte) - 256;
            }
            const std::int16_t stored = read_i16_le(raw + 260 + group * 2);
            if (sum != static_cast<std::int32_t>(stored)) {
                fail(std::string(context) + " rejects inconsistent Q8_K block sum");
            }
        }
    }
    std::vector<block_q8_K> aligned(blocks);
    std::memcpy(aligned.data(), encoded.data(), expected);
    return aligned;
}

template <typename WeightBlock>
float scalar_dot(
    const ByteVector & weights,
    const std::vector<block_q8_K> & activations,
    const std::size_t n,
    const DotFunction function,
    const char * context) {
    validate_k_n(n, context);
    const auto aligned_weights = copy_blocks<WeightBlock>(weights, n, context);
    const std::size_t blocks = n / kBlockElements;
    if (activations.size() != blocks) {
        fail(std::string(context) + " Q8_K block count mismatch");
    }
    const std::size_t activation_bytes = checked_product(blocks, sizeof(block_q8_K), context);
    if (activation_bytes != checked_product(activations.size(), sizeof(block_q8_K), context)) {
        fail(std::string(context) + " Q8_K byte length mismatch");
    }
    ByteVector activation_encoding(activation_bytes);
    std::memcpy(
        activation_encoding.data(),
        activations.data(),
        activation_encoding.size());
    const auto validated_activations =
        copy_q8_blocks(activation_encoding, n, context);

    float result = 0.0f;
    function(
        static_cast<int>(n),
        &result,
        0,
        aligned_weights.data(),
        0,
        validated_activations.data(),
        0,
        1);
    if (!std::isfinite(result)) {
        fail(std::string(context) + " produced a non-finite result");
    }
    return result;
}

void initialize_fp16_table() {
    for (std::size_t lane = 0; lane < (1u << 16u); ++lane) {
        ggml_table_f32_f16[lane] =
            ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(lane));
    }
}

WeightType parse_weight_type(const std::string & name) {
    if (name == "f32") {
        return WeightType::F32;
    }
    if (name == "q4-k") {
        return WeightType::Q4K;
    }
    if (name == "q5-k") {
        return WeightType::Q5K;
    }
    if (name == "iq2-s") {
        return WeightType::Iq2S;
    }
    if (name == "iq3-s") {
        return WeightType::Iq3S;
    }
    fail("unknown weight type: " + name);
}

std::size_t parse_n(const std::string & text) {
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](const char character) {
            return character >= '0' && character <= '9';
        })) {
        fail("n must be an unsigned base-10 integer");
    }
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(text, &consumed, 10);
    } catch (const std::exception &) {
        fail("n is outside the supported integer range");
    }
    if (consumed != text.size() || parsed == 0 || parsed > static_cast<unsigned long long>(INT_MAX)) {
        fail("n must satisfy 0 < n <= INT_MAX");
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t block_bytes(const WeightType type) {
    switch (type) {
        case WeightType::Q4K:
            return kQ4KBytes;
        case WeightType::Q5K:
            return kQ5KBytes;
        case WeightType::Iq2S:
            return kIq2SBytes;
        case WeightType::Iq3S:
            return kIq3SBytes;
        case WeightType::F32:
            fail("F32 does not use a 256-element packed block");
    }
    fail("unreachable weight type");
}

std::filesystem::path normalized_absolute(const std::filesystem::path & path) {
    std::error_code error;
    auto absolute = std::filesystem::absolute(path, error);
    if (error) {
        fail("cannot resolve path " + path.string());
    }
    return absolute.lexically_normal();
}

void validate_output_path(
    const std::filesystem::path & output,
    const std::vector<std::filesystem::path> & inputs) {
    const auto normalized_output = normalized_absolute(output);
    std::error_code error;
    const auto parent_status =
        std::filesystem::symlink_status(normalized_output.parent_path(), error);
    if (error || !std::filesystem::is_directory(parent_status)) {
        fail("output parent must be a direct existing directory");
    }
    const bool output_exists = std::filesystem::exists(normalized_output);
    for (const auto & input : inputs) {
        if (normalized_output == normalized_absolute(input)) {
            fail("output path must differ from every input path");
        }
        const auto input_status = std::filesystem::symlink_status(input, error);
        if (error || !std::filesystem::is_regular_file(input_status)) {
            fail("every input path must be a direct regular file");
        }
        if (output_exists) {
            const bool aliases = std::filesystem::equivalent(normalized_output, input, error);
            if (error) {
                fail("cannot compare output and input identities");
            }
            if (aliases) {
                fail("output path must not alias an input file");
            }
        }
    }
    if (output_exists) {
        const auto output_status = std::filesystem::symlink_status(normalized_output, error);
        if (error || !std::filesystem::is_regular_file(output_status)) {
            fail("output path must be absent or a direct regular file");
        }
    }
}

ByteVector decode_command(
    const WeightType type,
    const std::size_t n,
    const std::filesystem::path & input_path) {
    if (type == WeightType::F32) {
        const std::size_t bytes = checked_product(n, sizeof(float), "F32 identity");
        return read_binary_exact(input_path, bytes, "F32 identity");
    }
    validate_k_n(n, "packed decode");
    const std::size_t input_bytes =
        checked_product(n / kBlockElements, block_bytes(type), "packed decode");
    const ByteVector input = read_binary_exact(input_path, input_bytes, "packed decode");
    switch (type) {
        case WeightType::Q4K:
            return decode_blocks<block_q4_K>(
                input,
                n,
                dequantize_row_q4_K,
                "Q4_K dequantization");
        case WeightType::Q5K:
            return decode_blocks<block_q5_K>(
                input,
                n,
                dequantize_row_q5_K,
                "Q5_K dequantization");
        case WeightType::Iq2S:
            return decode_blocks<block_iq2_s>(
                input,
                n,
                dequantize_row_iq2_s,
                "IQ2_S dequantization");
        case WeightType::Iq3S:
            return decode_blocks<block_iq3_s>(
                input,
                n,
                dequantize_row_iq3_s,
                "IQ3_S dequantization");
        case WeightType::F32:
            break;
    }
    fail("unreachable decode type");
}

ByteVector q8_command(
    const std::size_t n,
    const std::filesystem::path & input_path) {
    validate_k_n(n, "Q8_K quantization");
    const std::size_t input_bytes = checked_product(n, sizeof(float), "Q8_K input");
    const ByteVector input = read_binary_exact(input_path, input_bytes, "Q8_K activation");
    std::vector<float> values(n);
    std::memcpy(values.data(), input.data(), input_bytes);
    const auto output = quantize_q8(values);
    ByteVector bytes(checked_product(output.size(), sizeof(block_q8_K), "Q8_K output"));
    std::memcpy(bytes.data(), output.data(), bytes.size());
    static_cast<void>(copy_q8_blocks(bytes, n, "Q8_K quantization output"));
    return bytes;
}

ByteVector dot_command(
    const WeightType type,
    const std::size_t n,
    const std::filesystem::path & weight_path,
    const std::filesystem::path & q8_path) {
    if (type == WeightType::F32) {
        fail("generic Q8_K dot does not support F32 weights");
    }
    validate_k_n(n, "generic dot");
    const std::size_t blocks = n / kBlockElements;
    const ByteVector weights = read_binary_exact(
        weight_path,
        checked_product(blocks, block_bytes(type), "dot weight input"),
        "dot weight");
    const ByteVector q8_bytes = read_binary_exact(
        q8_path,
        checked_product(blocks, sizeof(block_q8_K), "dot Q8_K input"),
        "dot Q8_K");
    const auto q8 = copy_q8_blocks(q8_bytes, n, "generic dot");
    initialize_fp16_table();
    float result = 0.0f;
    switch (type) {
        case WeightType::Q4K:
            result = scalar_dot<block_q4_K>(
                weights,
                q8,
                n,
                ggml_vec_dot_q4_K_q8_K_generic,
                "Q4_K x Q8_K dot");
            break;
        case WeightType::Q5K:
            result = scalar_dot<block_q5_K>(
                weights,
                q8,
                n,
                ggml_vec_dot_q5_K_q8_K_generic,
                "Q5_K x Q8_K dot");
            break;
        case WeightType::Iq2S:
            result = scalar_dot<block_iq2_s>(
                weights,
                q8,
                n,
                ggml_vec_dot_iq2_s_q8_K_generic,
                "IQ2_S x Q8_K dot");
            break;
        case WeightType::Iq3S:
            result = scalar_dot<block_iq3_s>(
                weights,
                q8,
                n,
                ggml_vec_dot_iq3_s_q8_K_generic,
                "IQ3_S x Q8_K dot");
            break;
        case WeightType::F32:
            break;
    }
    ByteVector output(sizeof(float));
    std::memcpy(output.data(), &result, sizeof(result));
    return output;
}

void generate(const std::filesystem::path & output_directory) {
    if (!host_is_little_endian()) {
        fail("the fixture oracle supports little-endian hosts only");
    }
    std::error_code error;
    const auto output_status = std::filesystem::symlink_status(output_directory, error);
    if (error || !std::filesystem::is_directory(output_status)) {
        fail("output directory must be a direct existing directory");
    }
    if (std::filesystem::directory_iterator(output_directory) !=
        std::filesystem::directory_iterator()) {
        fail("output staging directory must be empty");
    }

    std::vector<Artifact> artifacts;
    artifacts.reserve(16);
    const auto add_artifact = [&artifacts](std::string name, ByteVector bytes) {
        artifacts.push_back(Artifact{std::move(name), std::move(bytes)});
    };

    const std::array<std::uint32_t, 16> f32_bits = {
        0x00000000u,
        0x80000000u,
        0x3f800000u,
        0xbf800000u,
        0x3f000000u,
        0xc0200000u,
        0x00000001u,
        0x80000001u,
        0x00800000u,
        0x80800000u,
        0x7f7fffffu,
        0xff7fffffu,
        0x7f800000u,
        0xff800000u,
        0x7fc12345u,
        0xffc54321u,
    };
    ByteVector f32_bytes(checked_product(f32_bits.size(), sizeof(std::uint32_t), "F32 identity"));
    std::memcpy(f32_bytes.data(), f32_bits.data(), f32_bytes.size());
    add_artifact("decode-f32.input.bin", f32_bytes);
    add_artifact("decode-f32.output-f32le.bin", f32_bytes);

    const ByteVector q4 = make_three_blocks(
        kQ4KBytes,
        {0x3c00u, 0x3800u},
        {0x3400u, 0x3000u},
        true);
    const ByteVector q5 = make_three_blocks(
        kQ5KBytes,
        {0x3a00u, 0x3400u},
        {0x3800u, 0xb000u},
        true);
    const ByteVector iq2 = make_three_blocks(
        kIq2SBytes,
        {0x3800u, 0},
        {0x3400u, 0},
        false);
    const ByteVector iq3 = make_three_blocks(
        kIq3SBytes,
        {0x3c00u, 0},
        {0xb400u, 0},
        false);

    add_artifact("decode-q4-k.input.bin", q4);
    add_artifact(
        "decode-q4-k.output-f32le.bin",
        decode_blocks<block_q4_K>(
            q4,
            kDotElements,
            dequantize_row_q4_K,
            "Q4_K dequantization"));
    add_artifact("decode-q5-k.input.bin", q5);
    add_artifact(
        "decode-q5-k.output-f32le.bin",
        decode_blocks<block_q5_K>(
            q5,
            kDotElements,
            dequantize_row_q5_K,
            "Q5_K dequantization"));
    add_artifact("decode-iq2-s.input.bin", iq2);
    add_artifact(
        "decode-iq2-s.output-f32le.bin",
        decode_blocks<block_iq2_s>(
            iq2,
            kDotElements,
            dequantize_row_iq2_s,
            "IQ2_S dequantization"));
    add_artifact("decode-iq3-s.input.bin", iq3);
    add_artifact(
        "decode-iq3-s.output-f32le.bin",
        decode_blocks<block_iq3_s>(
            iq3,
            kDotElements,
            dequantize_row_iq3_s,
            "IQ3_S dequantization"));

    const std::vector<float> q8_input = make_q8_activations();
    const std::vector<block_q8_K> q8 = quantize_q8(q8_input);
    add_artifact("q8-k-activations.input-f32le.bin", bytes_of_floats(q8_input));
    ByteVector q8_encoded(checked_product(q8.size(), sizeof(block_q8_K), "Q8_K activation output"));
    std::memcpy(q8_encoded.data(), q8.data(), q8_encoded.size());
    add_artifact("q8-k-activations.output-q8-k.bin", std::move(q8_encoded));

    initialize_fp16_table();
    const float q4_dot = scalar_dot<block_q4_K>(
        q4,
        q8,
        kDotElements,
        ggml_vec_dot_q4_K_q8_K_generic,
        "Q4_K x Q8_K dot");
    const float q5_dot = scalar_dot<block_q5_K>(
        q5,
        q8,
        kDotElements,
        ggml_vec_dot_q5_K_q8_K_generic,
        "Q5_K x Q8_K dot");
    const float iq2_dot = scalar_dot<block_iq2_s>(
        iq2,
        q8,
        kDotElements,
        ggml_vec_dot_iq2_s_q8_K_generic,
        "IQ2_S x Q8_K dot");
    const float iq3_dot = scalar_dot<block_iq3_s>(
        iq3,
        q8,
        kDotElements,
        ggml_vec_dot_iq3_s_q8_K_generic,
        "IQ3_S x Q8_K dot");

    add_artifact("dot-q4-k-q8-k.output-f32le.bin", bytes_of_floats({q4_dot}));
    add_artifact("dot-q5-k-q8-k.output-f32le.bin", bytes_of_floats({q5_dot}));
    add_artifact("dot-iq2-s-q8-k.output-f32le.bin", bytes_of_floats({iq2_dot}));
    add_artifact("dot-iq3-s-q8-k.output-f32le.bin", bytes_of_floats({iq3_dot}));

    if (artifacts.size() != 16) {
        fail("internal fixture inventory mismatch");
    }
    for (const auto & artifact : artifacts) {
        write_binary(output_directory / artifact.name, artifact.bytes);
    }
}

} // namespace

int main(const int argc, const char * const argv[]) {
    try {
        if (!host_is_little_endian()) {
            fail("the fixture oracle supports little-endian hosts only");
        }
        if (argc == 3 && std::string(argv[1]) == "generate") {
            generate(std::filesystem::path(argv[2]));
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "decode") {
            const WeightType type = parse_weight_type(argv[2]);
            const std::size_t n = parse_n(argv[3]);
            const std::filesystem::path input(argv[4]);
            const std::filesystem::path output(argv[5]);
            validate_output_path(output, {input});
            const ByteVector staged = decode_command(type, n, input);
            write_binary(output, staged);
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "q8") {
            const std::size_t n = parse_n(argv[2]);
            const std::filesystem::path input(argv[3]);
            const std::filesystem::path output(argv[4]);
            validate_output_path(output, {input});
            const ByteVector staged = q8_command(n, input);
            write_binary(output, staged);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "dot") {
            const WeightType type = parse_weight_type(argv[2]);
            const std::size_t n = parse_n(argv[3]);
            const std::filesystem::path weights(argv[4]);
            const std::filesystem::path q8(argv[5]);
            const std::filesystem::path output(argv[6]);
            validate_output_path(output, {weights, q8});
            const ByteVector staged = dot_command(type, n, weights, q8);
            write_binary(output, staged);
            return 0;
        }
        std::cerr
            << "usage:\n"
            << "  bridge-quant-oracle generate <existing-output-directory>\n"
            << "  bridge-quant-oracle decode <f32|q4-k|q5-k|iq2-s|iq3-s> <n> <input> <output>\n"
            << "  bridge-quant-oracle q8 <n> <input-f32le> <output-q8-k>\n"
            << "  bridge-quant-oracle dot <q4-k|q5-k|iq2-s|iq3-s> <n> <weights> <q8-k> <output-f32le>\n";
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "bridge-quant-oracle: " << error.what() << '\n';
        return 1;
    }
}
