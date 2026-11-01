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
#include <filesystem>
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
           "[--prompt <text> | --chat] [options]\n"
        << "Options:\n"
        << "  --model <path>       Path to FP32 or Q4_0 .ie model file\n"
        << "  --tokenizer <path>   Path to TinyLlama SentencePiece tokenizer.model\n"
        << "  --prompt <text>      Input prompt text (omit with --chat for interactive REPL)\n"
        << "  --chat               Format with Zephyr chat template (interactive if --prompt omitted)\n"
        << "  --threads <n>        Requested worker thread count (default: 1)\n"
        << "  --max-tokens <n>     Maximum new tokens (default: 64; 0 disables generation)\n"
        << "  --temp <value>       Sampling temperature (default: 0, greedy)\n"
        << "  --top-p <value>      Nucleus probability (0, 1], default: 1\n"
        << "  --top-k <n>          Top-k candidate filtering (default: 0, disabled)\n"
        << "  --seed <n>           Reproducible sampling seed (default: 42)\n"
        << "  --benchmark          Print measured load, TTFT, throughput, RSS, and CPU metrics\n"
        << "  --bench-runs <n>     Run benchmark across <n> iterations and report statistical summary\n"
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
            arg == "--threads" || arg == "--max-tokens" || arg == "--max-new-tokens" ||
            arg == "--temp" || arg == "--temperature" ||
            arg == "--top-p" || arg == "--top-k" || arg == "--seed" || arg == "--token-ids" ||
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
            } else if (arg == "--max-tokens" || arg == "--max-new-tokens") {
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
            } else if (arg == "--temp" || arg == "--temperature") {
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
            } else if (arg == "--top-k") {
                if (!parse_unsigned(value, &parsed_unsigned) ||
                    parsed_unsigned > std::numeric_limits<uint32_t>::max()) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--top-k must be an integer in [0, UINT32_MAX]");
                }
                args->runtime.top_k = static_cast<uint32_t>(parsed_unsigned);
            } else if (arg == "--bench-runs") {
                if (!parse_unsigned(value, &parsed_unsigned) || parsed_unsigned == 0) {
                    return Status::Error(inference_engine::StatusCode::InvalidArgument,
                                         "--bench-runs must be a positive integer");
                }
                args->bench_runs = static_cast<uint32_t>(parsed_unsigned);
                args->runtime.benchmark_mode = true;
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

struct SummaryStats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
};

SummaryStats compute_summary(std::vector<double> vals) {
    if (vals.empty()) {
        return SummaryStats{};
    }
    std::sort(vals.begin(), vals.end());
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    SummaryStats s;
    s.mean = sum / static_cast<double>(vals.size());
    s.min = vals.front();
    s.max = vals.back();
    const size_t n = vals.size();
    if (n % 2 == 1) {
        s.median = vals[n / 2];
    } else {
        s.median = 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    }
    return s;
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

    Arguments args;
    const Status parse_status = parse_arguments(argc, argv, &args);
    if (!parse_status.ok()) {
        print_help();
        return fail(parse_status);
    }
    if (args.show_help) {
        print_help();
        return 0;
    }
    if (args.show_info) {
        print_info();
        return 0;
    }

    // Auto-detect tokenizer if not explicitly specified
    if (args.runtime.tokenizer_path.empty()) {
        const std::vector<std::string> candidates = {
            "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model",
            "models/tokenizer.model",
            "tokenizer.model"
        };
        for (const auto& path : candidates) {
            std::error_code ec;
            if (std::filesystem::exists(path, ec)) {
                args.runtime.tokenizer_path = path;
                break;
            }
        }
    }

    // Auto-detect model if not explicitly specified
    if (args.runtime.model_path.empty() && !args.encode_only && !args.decode_only && !args.format_chat_only) {
        const std::vector<std::string> candidates = {
            "models/tinyllama-1.1b-q4_0.ie",
            "models/tinyllama-1.1b-fp32.ie"
        };
        for (const auto& path : candidates) {
            std::error_code ec;
            if (std::filesystem::exists(path, ec)) {
                args.runtime.model_path = path;
                break;
            }
        }
    }

    const bool is_interactive_chat =
        args.chat_mode && args.runtime.prompt.empty() &&
        !args.encode_only && !args.decode_only && !args.format_chat_only;

    if (args.runtime.tokenizer_path.empty()) {
        print_help();
        return fail("--tokenizer is required (or place tokenizer.model in models/)");
    }
    {
        std::error_code ec;
        if (!std::filesystem::exists(args.runtime.tokenizer_path, ec)) {
            return fail("Tokenizer file not found: " + args.runtime.tokenizer_path);
        }
    }

    if (!args.encode_only && !args.decode_only && !args.format_chat_only) {
        if (args.runtime.model_path.empty()) {
            print_help();
            return fail("--model is required for inference (or place .ie model in models/)");
        }
        std::error_code ec;
        if (!std::filesystem::exists(args.runtime.model_path, ec)) {
            return fail("Model file not found: " + args.runtime.model_path);
        }
    }

    if (!args.decode_only && !is_interactive_chat && args.runtime.prompt.empty() &&
        !args.encode_only && !args.format_chat_only) {
        return fail("--prompt must not be empty (or run --chat for interactive mode)");
    }

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

    if (args.format_chat_only) {
        std::string prompt = Tokenizer::format_chat_prompt({ChatMessage{"user", args.runtime.prompt}}, true);
        std::cout << prompt;
        return 0;
    }

    if (args.encode_only) {
        std::string prompt = args.runtime.prompt;
        if (args.chat_mode) {
            prompt = Tokenizer::format_chat_prompt({ChatMessage{"user", prompt}}, true);
        }
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

    TransformerEngine engine;
    status = engine.init(loader.config(), &loader, detect_best_backend(),
                         args.runtime.num_threads);
    if (!status.ok()) {
        return fail(status);
    }

    Sampler sampler(SamplerConfig{args.runtime.temperature, args.runtime.top_p,
                                  args.runtime.seed, args.runtime.top_k});

    // Handle interactive chat session
    if (is_interactive_chat) {
        const TensorView* probe = loader.find_tensor("model.layers.0.mlp.gate_proj.weight");
        const std::string format_str = (probe && probe->dtype() == DataType::Q4_0) ? "Q4_0" : "FP32";
        const SimdBackend backend = detect_best_backend();

        std::cout << "BIOS V2 \u2014 Own Inference Engine\n\n"
                  << "Model: TinyLlama 1.1B Chat\n"
                  << "Format: " << format_str << "\n"
                  << "Backend: " << backend_name(backend) << "\n"
                  << "Threads: " << args.runtime.num_threads << "\n\n"
                  << "Type /help for available commands, /clear to reset context, /exit to quit.\n\n";

        std::vector<ChatMessage> history;
        std::vector<int32_t> committed_tokens;
        std::vector<float> logits(loader.config().vocab_size, 0.0f);
        std::string user_line;

        while (true) {
            std::cout << "BIOS> ";
            std::cout.flush();

            if (!std::getline(std::cin, user_line)) {
                std::cout << "\n";
                break;
            }

            size_t first = user_line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                continue;
            }
            size_t last = user_line.find_last_not_of(" \t\r\n");
            std::string trimmed = user_line.substr(first, last - first + 1);

            if (trimmed == "/exit" || trimmed == "/quit") {
                break;
            }
            if (trimmed == "/help") {
                std::cout << "\nAvailable commands:\n"
                          << "  /clear  Reset conversation history and KV cache\n"
                          << "  /exit   Exit the chat session\n"
                          << "  /quit   Exit the chat session\n"
                          << "  /help   Display this help message\n\n";
                continue;
            }
            if (trimmed == "/clear") {
                engine.kv_cache().clear();
                history.clear();
                committed_tokens.clear();
                std::cout << "\n[Conversation history and KV cache cleared.]\n\n";
                continue;
            }

            history.push_back(ChatMessage{"user", trimmed});
            std::string prompt_str = Tokenizer::format_chat_prompt(history, true);
            std::vector<int32_t> full_tokens;
            status = tokenizer.encode(prompt_str, true, false, &full_tokens);
            if (!status.ok()) {
                std::cerr << "error: Failed to tokenize chat input: " << status.message << "\n\n";
                history.pop_back();
                continue;
            }

            if (full_tokens.size() >= loader.config().max_seq_len) {
                std::cout << "\n[Warning: Context limit reached (" << full_tokens.size()
                          << " >= " << loader.config().max_seq_len
                          << " tokens). Please use /clear to start a new session.]\n\n";
                history.pop_back();
                continue;
            }
            if (full_tokens.size() > loader.config().max_seq_len - 128) {
                std::cout << "[Warning: Approaching context limit: " << full_tokens.size()
                          << " / " << loader.config().max_seq_len << " tokens]\n";
            }

            // KV Cache Prefix Matching
            size_t match_len = 0;
            while (match_len < committed_tokens.size() &&
                   match_len < full_tokens.size() &&
                   committed_tokens[match_len] == full_tokens[match_len]) {
                ++match_len;
            }

            if (match_len < committed_tokens.size()) {
                status = engine.kv_cache().truncate(static_cast<uint32_t>(match_len));
                if (!status.ok()) {
                    std::cerr << "error: Failed to rewind KV cache: " << status.message << "\n\n";
                    history.pop_back();
                    continue;
                }
                committed_tokens.resize(match_len);
            }

            // Prefill remaining tokens
            bool prefill_failed = false;
            for (size_t i = match_len; i < full_tokens.size(); ++i) {
                const uint32_t pos = static_cast<uint32_t>(i);
                status = engine.forward_token(full_tokens[i], pos, logits.data());
                if (!status.ok()) {
                    std::cerr << "error: Inference forward pass failed: " << status.message << "\n\n";
                    prefill_failed = true;
                    break;
                }
                committed_tokens.push_back(full_tokens[i]);
            }
            if (prefill_failed) {
                history.pop_back();
                continue;
            }

            std::cout << "\nBIOS:\n";
            std::cout.flush();

            std::vector<int32_t> turn_generated_ids;
            turn_generated_ids.reserve(args.runtime.max_new_tokens);
            std::string previous_output;

            if (args.runtime.max_new_tokens > 0) {
                int32_t next_token = sampler.sample(logits.data(), logits.size());
                if (next_token < 0) {
                    std::cerr << "error: Sampler could not select a token\n\n";
                    history.pop_back();
                    continue;
                }

                if (next_token != kEosTokenId) {
                    turn_generated_ids.push_back(next_token);
                    std::string current_output;
                    status = tokenizer.decode(turn_generated_ids, &current_output);
                    if (status.ok()) {
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

                for (uint32_t gen_i = 1;
                     gen_i < args.runtime.max_new_tokens && next_token != kEosTokenId;
                     ++gen_i) {
                    if (engine.kv_cache().seq_len() >= loader.config().max_seq_len) {
                        std::cout << "\n[Notice: Max sequence length reached during generation.]";
                        break;
                    }
                    const uint32_t pos = engine.kv_cache().seq_len();
                    status = engine.forward_token(next_token, pos, logits.data());
                    if (!status.ok()) {
                        std::cerr << "\nerror: Inference error during decode: " << status.message << "\n";
                        break;
                    }
                    committed_tokens.push_back(next_token);

                    next_token = sampler.sample(logits.data(), logits.size());
                    if (next_token < 0 || next_token == kEosTokenId) {
                        break;
                    }
                    turn_generated_ids.push_back(next_token);

                    std::string current_output;
                    status = tokenizer.decode(turn_generated_ids, &current_output);
                    if (status.ok()) {
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
            }

            std::string turn_text;
            tokenizer.decode(turn_generated_ids, &turn_text);
            history.push_back(ChatMessage{"assistant", turn_text});
            std::cout << "\n\n";
        }

        return 0;
    }

    std::string prompt = args.runtime.prompt;
    if (args.chat_mode) {
        prompt = Tokenizer::format_chat_prompt({ChatMessage{"user", prompt}}, true);
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
        return fail("Prompt and requested generation exceed the model's maximum context length (" +
                    std::to_string(required_positions) + " > " +
                    std::to_string(loader.config().max_seq_len) + ")");
    }

    const uint32_t total_runs = std::max(1u, args.bench_runs);
    struct RunMetric {
        double ttft_ms = 0.0;
        double prompt_tps = 0.0;
        double decode_tps = 0.0;
        double prefill_sec = 0.0;
        double decode_sec = 0.0;
        double cpu_sec = 0.0;
        uint32_t gen_tokens = 0;
        uint64_t peak_bytes = 0;
    };
    std::vector<RunMetric> metrics;
    metrics.reserve(total_runs);

    std::vector<float> logits(loader.config().vocab_size, 0.0f);
    std::vector<int32_t> final_generated_ids;

    for (uint32_t run = 0; run < total_runs; ++run) {
        engine.kv_cache().clear();
        RunMetric metric{};
        const bool stream_tokens = (run == 0 && total_runs == 1);

        const double cpu_start = process_cpu_seconds();
        const auto prefill_start = Clock::now();
        for (size_t pos = 0; pos < prompt_ids.size(); ++pos) {
            status = engine.forward_token(prompt_ids[pos], static_cast<uint32_t>(pos), logits.data());
            if (!status.ok()) {
                return fail(status);
            }
        }
        const auto prefill_end = Clock::now();
        metric.prefill_sec = std::chrono::duration<double>(prefill_end - prefill_start).count();
        metric.prompt_tps = metric.prefill_sec > 0.0 ? static_cast<double>(prompt_ids.size()) / metric.prefill_sec : 0.0;

        std::vector<int32_t> generated_ids;
        generated_ids.reserve(args.runtime.max_new_tokens);
        std::string previous_output;

        if (args.runtime.max_new_tokens > 0) {
            int32_t next_token = sampler.sample(logits.data(), logits.size());
            if (next_token < 0) {
                return fail("Sampler could not select a token from the model logits");
            }
            const auto first_token_time = Clock::now();
            metric.ttft_ms = std::chrono::duration<double, std::milli>(first_token_time - prefill_start).count();
            metric.gen_tokens = 1;

            if (next_token != kEosTokenId) {
                generated_ids.push_back(next_token);
                if (stream_tokens) {
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
                ++metric.gen_tokens;
                if (next_token != kEosTokenId) {
                    generated_ids.push_back(next_token);
                    if (stream_tokens) {
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
            }
            const auto decode_end = Clock::now();
            metric.decode_sec = std::chrono::duration<double>(decode_end - decode_start).count();
            metric.decode_tps = (metric.decode_sec > 0.0 && metric.gen_tokens > 1)
                                    ? static_cast<double>(metric.gen_tokens - 1) / metric.decode_sec
                                    : 0.0;
            metric.cpu_sec = process_cpu_seconds() - cpu_start;
        } else {
            metric.cpu_sec = process_cpu_seconds() - cpu_start;
        }

        metric.peak_bytes = peak_resident_bytes();
        metrics.push_back(metric);
        if (run == 0) {
            final_generated_ids = std::move(generated_ids);
        }
    }

    if (total_runs == 1) {
        std::string generated_text;
        status = tokenizer.decode(final_generated_ids, &generated_text);
        if (!status.ok()) {
            return fail(status);
        }
        if (!generated_text.empty() && generated_text.back() != '\n') {
            std::cout << '\n';
        }
    }

    if (args.runtime.benchmark_mode) {
        if (total_runs > 1) {
            std::vector<double> ttfts, ptpss, dtpss;
            ttfts.reserve(total_runs);
            ptpss.reserve(total_runs);
            dtpss.reserve(total_runs);
            uint64_t max_peak = 0;
            for (const auto& m : metrics) {
                ttfts.push_back(m.ttft_ms);
                ptpss.push_back(m.prompt_tps);
                if (m.gen_tokens > 1) dtpss.push_back(m.decode_tps);
                if (m.peak_bytes > max_peak) max_peak = m.peak_bytes;
            }
            SummaryStats s_ttft = compute_summary(ttfts);
            SummaryStats s_ptps = compute_summary(ptpss);
            SummaryStats s_dtps = compute_summary(dtpss);

            std::cerr << std::fixed << std::setprecision(3)
                      << "=== Multi-Run Benchmark Summary (N = " << total_runs << " runs) ===\n"
                      << "  Model load: " << model_load_ms << " ms\n"
                      << "  Prompt tokens: " << prompt_ids.size() << "\n"
                      << "  Generated tokens (per run): " << metrics.front().gen_tokens << "\n"
                      << "  Prompt throughput: mean = " << s_ptps.mean << " tok/s, median = "
                      << s_ptps.median << " tok/s (min = " << s_ptps.min << ", max = " << s_ptps.max << ")\n"
                      << "  Decode throughput: mean = " << s_dtps.mean << " tok/s, median = "
                      << s_dtps.median << " tok/s (min = " << s_dtps.min << ", max = " << s_dtps.max << ")\n"
                      << "  TTFT: mean = " << s_ttft.mean << " ms, median = "
                      << s_ttft.median << " ms (min = " << s_ttft.min << ", max = " << s_ttft.max << ")\n"
                      << "  Peak resident memory: " << static_cast<double>(max_peak) / (1024.0 * 1024.0) << " MiB\n"
                      << "  Threads requested: " << args.runtime.num_threads << "\n";
        } else {
            const auto& m = metrics.front();
            const double gen_sec = m.prefill_sec + m.decode_sec;
            std::cerr << std::fixed << std::setprecision(3)
                      << "Benchmark:\n"
                      << "  Model load: " << model_load_ms << " ms\n"
                      << "  TTFT: ";
            if (args.runtime.max_new_tokens == 0) {
                std::cerr << "n/a (generation disabled)\n";
            } else {
                std::cerr << m.ttft_ms << " ms\n";
            }
            std::cerr << "  Prompt tokens: " << prompt_ids.size() << "\n"
                      << "  Prompt throughput: " << m.prompt_tps << " tokens/s\n"
                      << "  Generated tokens: " << m.gen_tokens << "\n";
            if (m.gen_tokens > 1 && m.decode_sec > 0.0) {
                std::cerr << "  Decode throughput: " << m.decode_tps << " tokens/s\n";
            } else {
                std::cerr << "  Decode throughput: n/a (fewer than two generated tokens)\n";
            }
            if (m.peak_bytes != 0) {
                std::cerr << "  Peak resident memory: "
                          << static_cast<double>(m.peak_bytes) / (1024.0 * 1024.0) << " MiB\n";
            } else {
                std::cerr << "  Peak resident memory: unavailable on this platform\n";
            }
            if (gen_sec > 0.0 && m.cpu_sec > 0.0) {
                std::cerr << "  CPU utilization: "
                          << (m.cpu_sec / gen_sec) * 100.0 << "%\n";
            } else {
                std::cerr << "  CPU utilization: unavailable\n";
            }
            std::cerr << "  Threads requested: " << args.runtime.num_threads << "\n";
        }
    }
    return 0;
}
