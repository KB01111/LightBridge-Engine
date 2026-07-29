#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char * kLlamaCommit = "b77d646751d01c0962bc203b6809e9d94f7d50b7";

std::int32_t parse_nonnegative_i32(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<std::int32_t>(value);
}

struct Step {
    std::int32_t position;
    std::int32_t input_id;
    std::int32_t greedy_id;
    float greedy_logit;
    std::int32_t runner_up_id;
    float runner_up_logit;
};

Step inspect_logits(
    const float * logits,
    std::int32_t vocabulary_size,
    std::int32_t position,
    std::int32_t input_id) {
    if (logits == nullptr) {
        throw std::runtime_error("llama.cpp returned no logits");
    }

    std::int32_t best_id = -1;
    std::int32_t second_id = -1;
    float best = -std::numeric_limits<float>::infinity();
    float second = -std::numeric_limits<float>::infinity();
    for (std::int32_t token_id = 0; token_id < vocabulary_size; ++token_id) {
        const float value = logits[token_id];
        if (!std::isfinite(value)) {
            throw std::runtime_error("llama.cpp returned a non-finite logit");
        }
        if (value > best) {
            second = best;
            second_id = best_id;
            best = value;
            best_id = token_id;
        } else if (value > second) {
            second = value;
            second_id = token_id;
        }
    }
    if (best_id < 0 || second_id < 0) {
        throw std::runtime_error("llama.cpp returned fewer than two logits");
    }

    return Step {
        position,
        input_id,
        best_id,
        best,
        second_id,
        second,
    };
}

void write_report(
    const std::string & output_path,
    std::int32_t vocabulary_size,
    std::int32_t context_length,
    std::int32_t threads,
    const std::vector<Step> & steps) {
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create oracle report: " + output_path);
    }

    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "{\n"
           << "  \"format\": \"lightbridge-llama-full-model-oracle-v1\",\n"
           << "  \"llama_commit\": \"" << kLlamaCommit << "\",\n"
           << "  \"vocabulary_size\": " << vocabulary_size << ",\n"
           << "  \"context_length\": " << context_length << ",\n"
           << "  \"threads\": " << threads << ",\n"
           << "  \"steps\": [\n";
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const Step & step = steps[index];
        output << "    {\n"
               << "      \"position\": " << step.position << ",\n"
               << "      \"input_id\": " << step.input_id << ",\n"
               << "      \"greedy_id\": " << step.greedy_id << ",\n"
               << "      \"greedy_logit\": " << step.greedy_logit << ",\n"
               << "      \"runner_up_id\": " << step.runner_up_id << ",\n"
               << "      \"runner_up_logit\": " << step.runner_up_logit << ",\n"
               << "      \"margin\": " << (step.greedy_logit - step.runner_up_logit) << "\n"
               << "    }";
        if (index + 1 != steps.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error("failed while writing oracle report: " + output_path);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 5) {
            throw std::runtime_error(
                "usage: bridge-hy3-full-model-oracle "
                "<model.gguf> <output.json> <threads> <token-id>...");
        }

        const std::string model_path = argv[1];
        const std::string output_path = argv[2];
        const std::int32_t threads = parse_nonnegative_i32(argv[3], "thread count");
        if (threads == 0) {
            throw std::runtime_error("thread count must be greater than zero");
        }

        std::vector<llama_token> token_ids;
        token_ids.reserve(static_cast<std::size_t>(argc - 4));
        for (int index = 4; index < argc; ++index) {
            token_ids.push_back(parse_nonnegative_i32(argv[index], "token ID"));
        }

        const std::size_t requested_context = std::max<std::size_t>(512, token_ids.size() + 8);
        if (requested_context > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("requested context is not representable");
        }
        const auto context_length = static_cast<std::int32_t>(requested_context);

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.check_tensors = false;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("llama.cpp failed to load the selected GGUF");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(context_length);
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("llama.cpp failed to create the selected-model context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        if (vocabulary_size < 2) {
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("llama.cpp reported an invalid vocabulary size");
        }

        std::vector<Step> steps;
        steps.reserve(token_ids.size());
        for (std::size_t position = 0; position < token_ids.size(); ++position) {
            const llama_token token_id = token_ids[position];
            if (token_id < 0 || token_id >= vocabulary_size) {
                throw std::runtime_error("token ID is outside the selected-model vocabulary");
            }
            llama_token input = token_id;
            llama_batch batch = llama_batch_get_one(&input, 1);
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("llama_decode failed");
            }
            steps.push_back(inspect_logits(
                llama_get_logits(context),
                vocabulary_size,
                static_cast<std::int32_t>(position),
                token_id));
        }

        write_report(output_path, vocabulary_size, context_length, threads, steps);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "bridge-hy3-full-model-oracle: " << error.what() << '\n';
        return 1;
    }
}
