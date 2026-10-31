#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"
#include "inference_engine/loader.hpp"
#include "inference_engine/sampler.hpp"
#include "inference_engine/tokenizer.hpp"
#include "inference_engine/transformer.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using inference_engine::RuntimeConfig;
using inference_engine::Status;

struct Arguments {
    RuntimeConfig runtime{};
    bool show_help = false;
    bool show_info = false;
    bool chat_mode = false;
    bool encode_only = false;
    bool decode_only = false;
    bool format_chat_only = false;
    bool add_bos = false;
    bool add_eos = false;
    std::string token_ids{};
    uint32_t bench_runs = 0; // number of benchmark runs (0 = single run)
};

void print_help() {
    std::cout
        << "inference-engine (TinyLlama-1.1B-Chat CPU inference runtime)\n"
        << "Usage: inference-engine --model <file.ie> --tokenizer <tokenizer.model> "
           "--prompt <text> [options]\n"
        << "Options:\n"
        << "  --model <path>       Path to FP32 .ie model file\n"
        << "  --tokenizer <path>   Path to TinyLlama SentencePiece tokenizer.model\n"
        << "  --prompt <text>      Input prompt text\n"
        << "  --chat               Format the prompt with TinyLlama's Zephyr template\n"
        << "  --threads <n>        Requested worker thread count (default: 1)\n"
        << "  --max-tokens <n>     Maximum new tokens (default: 64; 0 disables generation)\n"
        << "  --temp <value>       Sampling temperature (default: 0, greedy)\n"
        << "  --top-p <value>      Nucleus probability (0, 1], default: 1\n"
        << "  --seed <n>           Reproducible sampling seed (default: 42)\n"
        << "  --benchmark          Print measured load, TTFT, throughput, RSS, and CPU metrics\n"
        << "  --encode-only        Print SentencePiece token IDs and exit\n"
        << "  --decode-only        Decode IDs supplied with --token-ids and exit\n"
        << "  --token-ids <ids>    Space-separated IDs for --decode-only\n"
        << "  --add-bos            Add BOS in --encode-only mode\n"
        << "  --add-eos            Add EOS in --encode-only mode\n"
        << "  --format-chat        Print the formatted user chat prompt and exit\n"
        << "  --info               Print detected CPU backend and frozen model config\n"
        << "  --help, -h           Show this help\n";
}

bool parse_unsigned(const std::string& value, uint64_t* out) {
    if (out == nullptr || value.empty() || value[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_float(const std::string& value, float* out) {
    if (out == nullptr || value.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    *out = parsed;
    return true;
}

Status parse_arguments(int argc, char** argv, Arguments* args) {
    if (args == nullptr) {
        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "Argument output must not be null");
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args->show_help = true;
            continue;
        }
        if (arg == "--info") {
            args->show_info = true;
            continue;
        }
        if (arg == "--chat") {
            args->chat_mode = true;
            continue;
        }
        if (arg == "--benchmark") {
            args->runtime.benchmark_mode = true;
            continue;
        }
        if (arg == "--encode-only") {
            args->encode_only = true;
            continue;
        }
        if (arg == "--decode-only") {
            args->decode_only = true;
            continue;
        }
        if (arg == "--format-chat") {
            args->format_chat_only = true;
            continue;
        }
        if (arg == "--add-bos") {
            args->add_bos = true;
            continue;
        }
        if (arg == "--add-eos") {
            args->add_eos = true;
            continue;
        }

        if (arg == "--model" || arg == "--tokenizer" || arg == "--prompt" ||
            arg == "--threads" || arg == "--max-tokens" || arg == "--temp" ||
            arg == "--top-p" || arg == "--seed" || arg == "--token-ids" ||
            arg == "--bench-runs") {
            if (i + 1 >= argc) {
                return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                     "Missing value for " + arg);
            }
            const std::string value = argv[++i];
            uint64_t parsed_unsigned = 0;
            if (arg == "--model") {
                args->runtime.model_path = value;
            } else if (arg == "--tokenizer") {
                args->runtime.tokenizer_path = value;
            } else if (arg == "--prompt") {
                args->runtime.prompt = value;
            } else if (arg == "--threads") {
                if (!parse_unsigned(value, &parsed_unsigned) || parsed_unsigned == 0 ||
                    parsed_unsigned > std::numeric_limits<uint32_t>::max()) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--threads must be an integer in [1, UINT32_MAX]");
                }
                args->runtime.num_threads = static_cast<uint32_t>(parsed_unsigned);
            } else if (arg == "--max-tokens") {
                if (!parse_unsigned(value, &parsed_unsigned) ||
                    parsed_unsigned > std::numeric_limits<uint32_t>::max()) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--max-tokens must be an integer in [0, UINT32_MAX]");
                }
                args->runtime.max_new_tokens = static_cast<uint32_t>(parsed_unsigned);
            } else if (arg == "--seed") {
                if (!parse_unsigned(value, &parsed_unsigned)) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--seed must be an unsigned 64-bit integer");
                }
                args->runtime.seed = static_cast<uint64_t>(parsed_unsigned);
            } else if (arg == "--token-ids") {
                args->token_ids = value;
            } else if (arg == "--temp") {
                if (!parse_float(value, &args->runtime.temperature) ||
                    args->runtime.temperature < 0.0f) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--temp must be a finite non-negative number");
                }
            } else if (arg == "--top-p") {
                if (!parse_float(value, &args->runtime.top_p) ||
                    args->runtime.top_p <= 0.0f || args->runtime.top_p > 1.0f) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--top-p must be in (0, 1]");
                }
            } else if (arg == "--bench-runs") {
                if (!parse_unsigned(value, &parsed_unsigned) || parsed_unsigned == 0) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--bench-runs must be a positive integer");
                }
                args->bench_runs = static_cast<uint32_t>(parsed_unsigned);
            }
            continue;
        }

        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "Unknown option: " + arg);
    }

    const int modes = static_cast<int>(args->encode_only) +
                      static_cast<int>(args->decode_only) +
                      static_cast<int>(args->format_chat_only);
    if (modes > 1) {
        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "--encode-only, --decode-only, and --format-chat are exclusive");
    }
    if ((args->add_bos || args->add_eos) && !args->encode_only) {
        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "--add-bos and --add-eos require --encode-only");
    }
    if (args->decode_only && args->token_ids.empty()) {
        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "--decode-only requires --token-ids");
    }
    if (args->runtime.top_p <= 0.0f || args->runtime.top_p > 1.0f) {
        return Status::Error(inference_engine::StatusCode::InvalidArgument,
                             "--top-p must be in (0, 1]");
    }
    return Status::Ok();
}

bool parse_token_ids(const std::string& input, std::vector<int32_t>* ids) {
    if (ids == nullptr) {
        return false;
    }
    ids->clear();
    std::istringstream stream(input);
    int64_t value = 0;
    while (stream >> value) {
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        ids->push_back(static_cast<int32_t>(value));
    }
    return stream.eof() && !ids->empty();
}

double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME creation_time{};
    FILETIME exit_time{};
    FILETIME kernel_time{};
    FILETIME user_time{};
    if (GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time,
                        &kernel_time, &user_time) == 0) {
        return 0.0;
    }
    const uint64_t kernel =
        (static_cast<uint64_t>(kernel_time.dwHighDateTime) << 32) |
        kernel_time.dwLowDateTime;
    const uint64_t user =
        (static_cast<uint64_t>(user_time.dwHighDateTime) << 32) |
        user_time.dwLowDateTime;
    return static_cast<double>(kernel + user) / 10000000.0;
#elif defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    return static_cast<double>(usage.ru_utime.tv_sec) +
           static_cast<double>(usage.ru_utime.tv_usec) / 1000000.0 +
           static_cast<double>(usage.ru_stime.tv_sec) +
           static_cast<double>(usage.ru_stime.tv_usec) / 1000000.0;
#else
    return 0.0;
#endif
}

uint64_t peak_resident_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0) {
        return 0;
    }
    return static_cast<uint64_t>(counters.PeakWorkingSetSize);
#elif defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024u;
#endif
#else
    return 0;
#endif
}

void print_info() {
    const inference_engine::SimdBackend backend = inference_engine::detect_best_backend();
    std::cout << "Project: inference-engine\n"
              << "Target Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
              << "Detected SIMD Backend: " << inference_engine::backend_name(backend) << "\n"
              << "NEON Available: "
              << (inference_engine::is_backend_available(inference_engine::SimdBackend::Neon)
                      ? "yes"
                      : "no")
              << "\nAVX2 Available: "
              << (inference_engine::is_backend_available(inference_engine::SimdBackend::Avx2)
                      ? "yes"
                      : "no")
              << "\nLayers: " << inference_engine::kTinyLlamaNumLayers
              << "\nHidden dimension: " << inference_engine::kTinyLlamaHiddenDim
              << "\nVocabulary: " << inference_engine::kTinyLlamaVocabSize
              << "\nMaximum sequence length: " << inference_engine::kTinyLlamaMaxSeqLen << "\n";
}

int fail(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    return 1;
}

int fail(const Status& status) {
    return fail(std::string(inference_engine::status_code_name(status.code)) +
                ": " + status.message);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace inference_engine;



    Tokenizer tokenizer;
    Status status = tokenizer.load(args.runtime.tokenizer_path);
    if (!status.ok()) {
        return fail(status);
    }

    if (args.decode_only) {
        std::vector<int32_t> ids;
        if (!parse_token_ids(args.token_ids, &ids)) {
            return fail("--token-ids must contain one or more space-separated integers");
        }
        std::string decoded;
        status = tokenizer.decode(ids, &decoded);
        if (!status.ok()) {
            return fail(status);
        }
        std::cout << decoded;
        return 0;
    }

    std::string prompt = args.runtime.prompt;
    if (args.chat_mode || args.format_chat_only) {
        prompt = Tokenizer::format_chat_prompt({ChatMessage{"user", prompt}}, true);
    }
    if (args.format_chat_only) {
        std::cout << prompt;
        return 0;
    }

    if (args.encode_only) {
        std::vector<int32_t> ids;
        status = tokenizer.encode(prompt, args.add_bos, args.add_eos, &ids);
        if (!status.ok()) {
            return fail(status);
        }
        std::cout << "Encoded token IDs:";
        for (int32_t id : ids) {
            std::cout << ' ' << id;
        }
        std::cout << '\n';
        return 0;
    }

    ModelLoader loader;
    const auto model_load_start = Clock::now();
    status = loader.load_from_file(args.runtime.model_path);
    const auto model_load_end = Clock::now();
    if (!status.ok()) {
        return fail(status);
    }
    const double model_load_ms =
        std::chrono::duration<double, std::milli>(model_load_end - model_load_start).count();
    if (loader.config().vocab_size != kTinyLlamaVocabSize) {
        return fail("Loaded model vocabulary does not match the TinyLlama tokenizer");
    }

    std::vector<int32_t> prompt_ids;
    status = tokenizer.encode(prompt, true, false, &prompt_ids);
    if (!status.ok()) {
        return fail(status);
    }
    if (prompt_ids.empty()) {
        return fail("Prompt tokenization produced no tokens");
    }
    const uint64_t required_positions =
        static_cast<uint64_t>(prompt_ids.size()) +
        (args.runtime.max_new_tokens == 0 ? 0u : args.runtime.max_new_tokens - 1u);
    if (required_positions > loader.config().max_seq_len) {
        return fail("Prompt and requested generation exceed the model's maximum context length");
    }
    if (args.runtime.num_threads > 1) {
        std::cerr << "warning: this FP32 Transformer build does not yet use the requested "
                     "thread count\n";
    }

    TransformerEngine engine;
    status = engine.init(loader.config(), &loader, detect_best_backend(),
                         args.runtime.num_threads);
    if (!status.ok()) {
        return fail(status);
    }

    Sampler sampler(SamplerConfig{args.runtime.temperature, args.runtime.top_p,
                                  args.runtime.seed});
    std::vector<float> logits(loader.config().vocab_size, 0.0f);

    const double cpu_start = process_cpu_seconds();
    const auto prefill_start = Clock::now();
    for (size_t pos = 0; pos < prompt_ids.size(); ++pos) {
        status = engine.forward_token(prompt_ids[pos], static_cast<uint32_t>(pos), logits.data());
        if (!status.ok()) {
            return fail(status);
        }
    }
    const auto prefill_end = Clock::now();
    const double prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();

    std::vector<int32_t> generated_ids;
    generated_ids.reserve(args.runtime.max_new_tokens);
    uint32_t generated_token_count = 0;
    double ttft_ms = 0.0;
    double decode_seconds = 0.0;
    double cpu_seconds = 0.0;
    std::string previous_output;
    if (args.runtime.max_new_tokens > 0) {
        int32_t next_token = sampler.sample(logits.data(), logits.size());
        if (next_token < 0) {
            return fail("Sampler could not select a token from the model logits");
        }
        const auto first_token_time = Clock::now();
        ttft_ms = std::chrono::duration<double, std::milli>(
                      first_token_time - prefill_start).count();
        generated_token_count = 1;
        if (next_token != kEosTokenId) {
            generated_ids.push_back(next_token);
            // Stream incremental output for first token
            std::string current_output;
            status = tokenizer.decode(generated_ids, &current_output);
            if (!status.ok()) return fail(status);
            size_t common = 0;
            while (common < previous_output.size() && common < current_output.size() &&
                   previous_output[common] == current_output[common]) {
                ++common;
            }
            std::cout << current_output.substr(common);
            std::cout.flush();
            previous_output = std::move(current_output);
        }

        const auto decode_start = Clock::now();
        for (uint32_t generated_count = 1;
             generated_count < args.runtime.max_new_tokens && next_token != kEosTokenId;
             ++generated_count) {
            const uint32_t position =
                static_cast<uint32_t>(prompt_ids.size()) + generated_count - 1;
            status = engine.forward_token(next_token, position, logits.data());
            if (!status.ok()) {
                return fail(status);
            }
            next_token = sampler.sample(logits.data(), logits.size());
            if (next_token < 0) {
                return fail("Sampler could not select a token from the model logits");
            }
            ++generated_token_count;
            if (next_token != kEosTokenId) {
                generated_ids.push_back(next_token);
                // Stream incremental output for subsequent token
                std::string current_output;
                status = tokenizer.decode(generated_ids, &current_output);
                if (!status.ok()) return fail(status);
                size_t common = 0;
                while (common < previous_output.size() && common < current_output.size() &&
                       previous_output[common] == current_output[common]) {
                    ++common;
                }
                std::cout << current_output.substr(common);
                std::cout.flush();
                previous_output = std::move(current_output);
            }
        }
        const auto decode_end = Clock::now();
        decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
        cpu_seconds = process_cpu_seconds() - cpu_start;
    } else {
        cpu_seconds = process_cpu_seconds() - cpu_start;
    }

    std::string generated_text;
    status = tokenizer.decode(generated_ids, &generated_text);
    if (!status.ok()) {
        return fail(status);
    }
    // Ensure terminal ends with newline after streaming output
    if (!generated_text.empty() && generated_text.back() != '\n') {
        std::cout << '\n';
    }

    if (args.runtime.benchmark_mode) {
        const double generation_seconds = prefill_seconds + decode_seconds;
        const double prompt_tps =
            prefill_seconds > 0.0 ? static_cast<double>(prompt_ids.size()) / prefill_seconds : 0.0;
        const double decode_tps =
            decode_seconds > 0.0 && generated_token_count > 1
                ? static_cast<double>(generated_token_count - 1) / decode_seconds
                : 0.0;
        const uint64_t peak_bytes = peak_resident_bytes();
        std::cerr << std::fixed << std::setprecision(3)
                  << "Benchmark:\n"
                  << "  Model load: " << model_load_ms << " ms\n"
                  << "  TTFT: ";
        if (args.runtime.max_new_tokens == 0) {
            std::cerr << "n/a (generation disabled)\n";
        } else {
            std::cerr << ttft_ms << " ms\n";
        }
        std::cerr << "  Prompt tokens: " << prompt_ids.size() << "\n"
                  << "  Prompt throughput: " << prompt_tps << " tokens/s\n"
                  << "  Generated tokens: " << generated_token_count << "\n";
        if (generated_token_count > 1 && decode_seconds > 0.0) {
            std::cerr << "  Decode throughput: " << decode_tps << " tokens/s\n";
        } else {
            std::cerr << "  Decode throughput: n/a (fewer than two generated tokens)\n";
        }
        if (peak_bytes != 0) {
            std::cerr << "  Peak resident memory: "
                      << static_cast<double>(peak_bytes) / (1024.0 * 1024.0) << " MiB\n";
        } else {
            std::cerr << "  Peak resident memory: unavailable on this platform\n";
        }
        if (generation_seconds > 0.0 && cpu_seconds > 0.0) {
            std::cerr << "  CPU utilization: "
                      << (cpu_seconds / generation_seconds) * 100.0 << "%\n";
        } else {
            std::cerr << "  CPU utilization: unavailable\n";
        }
        std::cerr << "  Threads requested: " << args.runtime.num_threads << "\n";
    }
    return 0;
}
