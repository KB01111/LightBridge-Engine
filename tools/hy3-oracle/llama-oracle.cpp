#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
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
#include <vector>

namespace {

constexpr const char * kLlamaCommit = "b77d646751d01c0962bc203b6809e9d94f7d50b7";

struct Capture {
    std::array<std::int32_t, 32> selected_experts{};
    std::size_t selected_count = 0;
    bool saw_topk = false;
    bool invalid_topk = false;
};

bool capture_eval(ggml_tensor * tensor, bool ask, void * user_data) {
    const std::string name = tensor->name;
    const bool wanted = name.find("ffn_moe_topk-1") != std::string::npos;
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }
    if (tensor->type != GGML_TYPE_I32) {
        static_cast<Capture *>(user_data)->invalid_topk = true;
        return false;
    }

    auto * capture = static_cast<Capture *>(user_data);
    const std::size_t count = ggml_nelements(tensor);
    if (count == 0 || count > capture->selected_experts.size()) {
        capture->invalid_topk = true;
        return false;
    }
    ggml_backend_tensor_get(
        tensor,
        capture->selected_experts.data(),
        0,
        count * sizeof(std::int32_t));
    capture->selected_count = count;
    std::sort(
        capture->selected_experts.begin(),
        capture->selected_experts.begin() + static_cast<std::ptrdiff_t>(count));
    capture->saw_topk = true;
    return true;
}

std::int32_t parse_token(const char * text) {
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(std::string("invalid token ID: ") + text);
    }
    return static_cast<std::int32_t>(value);
}

struct Step {
    std::int32_t token_id;
    std::vector<std::int32_t> selected_experts;
    std::vector<float> logits;
    std::vector<float> probabilities;
    std::int32_t greedy_id;
};

std::vector<float> softmax(const std::vector<float> & logits) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    float sum = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float & value : probabilities) {
        value /= sum;
    }
    return probabilities;
}

void write_float_array(std::ostream & output, const std::vector<float> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

void write_int_array(std::ostream & output, const std::vector<std::int32_t> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

void write_report(
    const std::string & output_path,
    const std::vector<Step> & steps) {
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create oracle report: " + output_path);
    }
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "{\n"
           << "  \"format\": \"lightbridge-llama-hy3-oracle-v1\",\n"
           << "  \"llama_commit\": \"" << kLlamaCommit << "\",\n"
           << "  \"steps\": [\n";
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const Step & step = steps[index];
        output << "    {\n"
               << "      \"token_id\": " << step.token_id << ",\n"
               << "      \"selected_experts\": ";
        write_int_array(output, step.selected_experts);
        output << ",\n"
               << "      \"greedy_id\": " << step.greedy_id << ",\n"
               << "      \"logits\": ";
        write_float_array(output, step.logits);
        output << ",\n"
               << "      \"probabilities\": ";
        write_float_array(output, step.probabilities);
        output << "\n"
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
        if (argc < 4) {
            throw std::runtime_error(
                "usage: bridge-hy3-llama-oracle <model.gguf> <output.json> <token-id>...");
        }

        const std::string model_path = argv[1];
        const std::string output_path = argv[2];
        std::vector<std::int32_t> token_ids;
        token_ids.reserve(static_cast<std::size_t>(argc - 3));
        for (int index = 3; index < argc; ++index) {
            token_ids.push_back(parse_token(argv[index]));
        }

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.check_tensors = true;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("llama.cpp failed to load reduced GGUF");
        }

        Capture capture;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 1024;
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_threads = 1;
        context_params.n_threads_batch = 1;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.cb_eval = capture_eval;
        context_params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("llama.cpp failed to create reduced context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        if (vocabulary_size <= 0) {
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("llama.cpp reported an invalid vocabulary size");
        }

        std::vector<Step> steps;
        steps.reserve(token_ids.size());
        for (const std::int32_t token_id : token_ids) {
            if (token_id >= vocabulary_size) {
                throw std::runtime_error("token ID is outside the reduced vocabulary");
            }
            capture.selected_count = 0;
            capture.saw_topk = false;
            capture.invalid_topk = false;
            llama_token token = token_id;
            llama_batch batch = llama_batch_get_one(&token, 1);
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("llama_decode failed");
            }
            if (capture.invalid_topk) {
                throw std::runtime_error("llama.cpp exposed an invalid Hy3 top-k tensor");
            }
            float * raw_logits = llama_get_logits(context);
            if (raw_logits == nullptr) {
                throw std::runtime_error("llama.cpp returned no logits");
            }
            if (!capture.saw_topk) {
                throw std::runtime_error("llama.cpp did not expose the Hy3 layer-1 top-k tensor");
            }

            std::vector<float> logits(
                raw_logits,
                raw_logits + static_cast<std::size_t>(vocabulary_size));
            const auto greedy = static_cast<std::int32_t>(
                std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            steps.push_back(Step {
                token_id,
                std::vector<std::int32_t>(
                    capture.selected_experts.begin(),
                    capture.selected_experts.begin() +
                        static_cast<std::ptrdiff_t>(capture.selected_count)),
                logits,
                softmax(logits),
                greedy,
            });
        }

        write_report(output_path, steps);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "bridge-hy3-llama-oracle: " << error.what() << '\n';
        return 1;
    }
}
