#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"
#include "inference_engine/loader.hpp"
#include "inference_engine/sampler.hpp"
#include "inference_engine/tokenizer.hpp"
#include "inference_engine/transformer.hpp"

namespace inference_engine::studio {

using StudioClock = std::chrono::steady_clock;

inline uint64_t current_resident_bytes() noexcept {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0) {
        return 0;
    }
    return static_cast<uint64_t>(counters.WorkingSetSize);
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t size_pages = 0;
    uint64_t resident_pages = 0;
    if (statm >> size_pages >> resident_pages) {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            return resident_pages * static_cast<uint64_t>(page_size);
        }
    }
    return 0;
#else
    return 0;
#endif
}

inline std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
                break;
        }
    }
    return out.str();
}

inline std::string fmt_double(double v, int prec = 3) {
    if (!std::isfinite(v)) {
        return "0.0";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(prec) << v;
    return out.str();
}

// Minimal, dependency-free JSON parser for studio worker commands
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val{};
    std::vector<JsonValue> arr_val{};
    std::unordered_map<std::string, JsonValue> obj_val{};

    std::string get_string(const std::string& key, const std::string& def = "") const {
        auto it = obj_val.find(key);
        if (it != obj_val.end() && it->second.type == Type::String) {
            return it->second.str_val;
        }
        return def;
    }

    double get_number(const std::string& key, double def = 0.0) const {
        auto it = obj_val.find(key);
        if (it != obj_val.end() && it->second.type == Type::Number) {
            return it->second.num_val;
        }
        return def;
    }

    bool get_bool(const std::string& key, bool def = false) const {
        auto it = obj_val.find(key);
        if (it != obj_val.end() && it->second.type == Type::Bool) {
            return it->second.bool_val;
        }
        return def;
    }
};

class MiniJsonParser {
public:
    explicit MiniJsonParser(const std::string& src) : src_(src), pos_(0) {}

    bool parse(JsonValue* out) {
        skip_ws();
        if (!parse_value(out)) {
            return false;
        }
        skip_ws();
        return pos_ == src_.size();
    }

private:
    const std::string& src_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < src_.size() &&
               (src_[pos_] == ' ' || src_[pos_] == '\t' || src_[pos_] == '\r' || src_[pos_] == '\n')) {
            ++pos_;
        }
    }

    bool parse_value(JsonValue* out) {
        skip_ws();
        if (pos_ >= src_.size()) return false;
        char c = src_[pos_];
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == '\"') {
            out->type = JsonValue::Type::String;
            return parse_string(&out->str_val);
        }
        if (c == 't' && src_.compare(pos_, 4, "true") == 0) {
            out->type = JsonValue::Type::Bool;
            out->bool_val = true;
            pos_ += 4;
            return true;
        }
        if (c == 'f' && src_.compare(pos_, 5, "false") == 0) {
            out->type = JsonValue::Type::Bool;
            out->bool_val = false;
            pos_ += 5;
            return true;
        }
        if (c == 'n' && src_.compare(pos_, 4, "null") == 0) {
            out->type = JsonValue::Type::Null;
            pos_ += 4;
            return true;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number(out);
        }
        return false;
    }

    bool parse_string(std::string* out) {
        if (pos_ >= src_.size() || src_[pos_] != '\"') return false;
        ++pos_;
        out->clear();
        while (pos_ < src_.size()) {
            char c = src_[pos_++];
            if (c == '\"') return true;
            if (c == '\\') {
                if (pos_ >= src_.size()) return false;
                char esc = src_[pos_++];
                switch (esc) {
                    case '\"': out->push_back('\"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > src_.size()) return false;
                        unsigned int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char hc = src_[pos_++];
                            code <<= 4;
                            if (hc >= '0' && hc <= '9') code |= static_cast<unsigned int>(hc - '0');
                            else if (hc >= 'a' && hc <= 'f') code |= static_cast<unsigned int>(hc - 'a' + 10);
                            else if (hc >= 'A' && hc <= 'F') code |= static_cast<unsigned int>(hc - 'A' + 10);
                            else return false;
                        }
                        if (code <= 0x7F) {
                            out->push_back(static_cast<char>(code));
                        } else if (code <= 0x7FF) {
                            out->push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out->push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                            out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        out->push_back(esc);
                        break;
                }
            } else {
                out->push_back(c);
            }
        }
        return false;
    }

    bool parse_number(JsonValue* out) {
        const char* start = src_.c_str() + pos_;
        char* end = nullptr;
        errno = 0;
        double val = std::strtod(start, &end);
        if (errno != 0 || end == start) return false;
        pos_ = static_cast<size_t>(end - src_.c_str());
        out->type = JsonValue::Type::Number;
        out->num_val = val;
        return true;
    }

    bool parse_array(JsonValue* out) {
        if (src_[pos_] != '[') return false;
        ++pos_;
        out->type = JsonValue::Type::Array;
        out->arr_val.clear();
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (pos_ < src_.size()) {
            JsonValue elem;
            if (!parse_value(&elem)) return false;
            out->arr_val.push_back(std::move(elem));
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < src_.size() && src_[pos_] == ']') {
                ++pos_;
                return true;
            }
            return false;
        }
        return false;
    }

    bool parse_object(JsonValue* out) {
        if (src_[pos_] != '{') return false;
        ++pos_;
        out->type = JsonValue::Type::Object;
        out->obj_val.clear();
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (pos_ < src_.size()) {
            skip_ws();
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_] != ':') return false;
            ++pos_;
            JsonValue val;
            if (!parse_value(&val)) return false;
            out->obj_val[std::move(key)] = std::move(val);
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < src_.size() && src_[pos_] == '}') {
                ++pos_;
                return true;
            }
            return false;
        }
        return false;
    }
};

struct ExtendedStats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

inline ExtendedStats compute_extended_stats(std::vector<double> vals) {
    if (vals.empty()) return ExtendedStats{};
    std::sort(vals.begin(), vals.end());
    double sum = 0.0;
    for (double v : vals) sum += v;
    ExtendedStats s;
    s.mean = sum / static_cast<double>(vals.size());
    s.min = vals.front();
    s.max = vals.back();
    const size_t n = vals.size();
    s.median = (n % 2 == 1) ? vals[n / 2] : 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
    double sq_sum = 0.0;
    for (double v : vals) {
        const double d = v - s.mean;
        sq_sum += d * d;
    }
    s.stddev = std::sqrt(sq_sum / static_cast<double>(n));
    return s;
}

inline std::string detect_os_arch() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
    return "macOS ARM64 (Apple Silicon)";
#elif defined(__APPLE__)
    return "macOS x86_64";
#elif defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
    return "Linux x86_64";
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
    return "Linux ARM64";
#elif defined(_WIN32)
    return "Windows x86_64";
#else
    return "POSIX CPU";
#endif
}

class StudioWorker {
public:
    using CpuSecondsFn = double (*)();
    using PeakRssFn = uint64_t (*)();

    StudioWorker(const std::string& initial_model_path,
                 const std::string& initial_tokenizer_path,
                 uint32_t initial_threads,
                 CpuSecondsFn cpu_fn,
                 PeakRssFn peak_rss_fn)
        : model_path_(initial_model_path),
          tokenizer_path_(initial_tokenizer_path),
          num_threads_(initial_threads > 0 ? initial_threads : 4),
          active_backend_(detect_best_backend()),
          cpu_seconds_fn_(cpu_fn),
          peak_rss_fn_(peak_rss_fn) {}

    int run() {
        emit_log("INFO", "Initializing BIOS V2 Studio Worker runtime...");
        if (!initialize_tokenizer()) {
            emit_log("ERROR", "Tokenizer initialization failed: " + last_error_);
        }
        if (!model_path_.empty()) {
            load_and_init_model(model_path_, num_threads_, active_backend_);
        } else {
            emit_log("WARN", "No default .ie model file found at startup.");
        }

        emit_status_report("engine_ready");

        std::thread reader(&StudioWorker::stdin_reader_loop, this);

        while (true) {
            JsonValue cmd;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() { return !cmd_queue_.empty() || stdin_closed_; });
                if (cmd_queue_.empty() && stdin_closed_) {
                    break;
                }
                cmd = std::move(cmd_queue_.front());
                cmd_queue_.pop();
            }

            const std::string action = cmd.get_string("cmd");
            if (action == "shutdown" || action == "exit") {
                emit_log("INFO", "Studio worker shutting down cleanly.");
                break;
            } else if (action == "status") {
                emit_status_report("status_report", cmd.get_string("request_id"));
            } else if (action == "clear") {
                handle_clear(cmd);
            } else if (action == "configure") {
                handle_configure(cmd);
            } else if (action == "chat") {
                handle_chat(cmd);
            } else if (action == "benchmark") {
                handle_benchmark(cmd);
            } else {
                emit_error(cmd.get_string("request_id"), "Unknown command: " + action);
            }
        }

        if (reader.joinable()) {
            reader.join();
        }
        return 0;
    }

private:
    std::string model_path_;
    std::string tokenizer_path_;
    uint32_t num_threads_ = 4;
    SimdBackend active_backend_ = SimdBackend::Scalar;
    CpuSecondsFn cpu_seconds_fn_ = nullptr;
    PeakRssFn peak_rss_fn_ = nullptr;

    Tokenizer tokenizer_{};
    ModelLoader loader_{};
    TransformerEngine engine_{};
    std::vector<int32_t> committed_tokens_{};
    double model_load_ms_ = 0.0;
    uint64_t model_file_bytes_ = 0;
    std::string model_format_ = "UNKNOWN";
    std::string last_error_{};
    std::string engine_state_ = "idle";

    std::mutex stdout_mutex_{};
    std::mutex queue_mutex_{};
    std::condition_variable queue_cv_{};
    std::queue<JsonValue> cmd_queue_{};
    bool stdin_closed_ = false;
    std::atomic<bool> stop_requested_{false};

    void stdin_reader_loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            JsonValue parsed;
            MiniJsonParser parser(line);
            if (!parser.parse(&parsed) || parsed.type != JsonValue::Type::Object) {
                emit_error("", "Malformed JSON command received on stdin");
                continue;
            }
            const std::string cmd_name = parsed.get_string("cmd");
            if (cmd_name == "stop") {
                stop_requested_.store(true, std::memory_order_relaxed);
                emit_log("WARN", "Stop generation signal received by C++ engine.");
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                cmd_queue_.push(std::move(parsed));
            }
            queue_cv_.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stdin_closed_ = true;
        }
        queue_cv_.notify_one();
    }

    void emit_raw_json(const std::string& json_line) {
        std::lock_guard<std::mutex> lock(stdout_mutex_);
        std::cout << json_line << '\n';
        std::cout.flush();
    }

    void emit_log(const std::string& level, const std::string& message) {
        std::ostringstream ss;
        ss << "{\"event\":\"log\",\"level\":\"" << json_escape(level)
           << "\",\"message\":\"" << json_escape(message) << "\"}";
        emit_raw_json(ss.str());
    }

    void emit_error(const std::string& request_id, const std::string& message) {
        emit_log("ERROR", message);
        std::ostringstream ss;
        ss << "{\"event\":\"generation_error\",\"request_id\":\"" << json_escape(request_id)
           << "\",\"error\":\"" << json_escape(message) << "\"}";
        emit_raw_json(ss.str());
    }

    bool initialize_tokenizer() {
        if (tokenizer_path_.empty()) {
            last_error_ = "Tokenizer path is empty";
            return false;
        }
        Status st = tokenizer_.load(tokenizer_path_);
        if (!st.ok()) {
            last_error_ = st.message;
            return false;
        }
        emit_log("INFO", "Loaded SentencePiece tokenizer (" + tokenizer_path_ + ", vocab=32000)");
        return true;
    }

    bool load_and_init_model(const std::string& path, uint32_t threads, SimdBackend backend) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            last_error_ = "Model file not found: " + path;
            engine_state_ = "error";
            emit_log("ERROR", last_error_);
            return false;
        }

        const bool need_reload = (!loader_.is_loaded() || path != model_path_);
        if (need_reload) {
            emit_log("INFO", "Memory-mapping .ie model file: " + path);
            const auto t0 = StudioClock::now();
            Status st = loader_.load_from_file(path);
            const auto t1 = StudioClock::now();
            if (!st.ok()) {
                last_error_ = st.message;
                engine_state_ = "error";
                emit_log("ERROR", "ModelLoader failed: " + st.message);
                return false;
            }
            model_load_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
            model_path_ = path;
            model_file_bytes_ = static_cast<uint64_t>(std::filesystem::file_size(path, ec));

            const TensorView* gate_probe = loader_.find_tensor("model.layers.0.mlp.gate_proj.weight");
            const TensorView* embed_probe = loader_.find_tensor("model.embed_tokens.weight");
            if (gate_probe && gate_probe->dtype() == DataType::Q4_0) {
                if (embed_probe && embed_probe->dtype() == DataType::FP32) {
                    model_format_ = "Q4_0_MIXED";
                } else {
                    model_format_ = "Q4_0";
                }
            } else {
                model_format_ = "FP32";
            }
        }

        if (!is_backend_available(backend)) {
            backend = detect_best_backend();
        }
        num_threads_ = std::max(1u, threads);
        active_backend_ = backend;

        Status st = engine_.init(loader_.config(), &loader_, active_backend_, num_threads_);
        if (!st.ok()) {
            last_error_ = st.message;
            engine_state_ = "error";
            emit_log("ERROR", "TransformerEngine init failed: " + st.message);
            return false;
        }

        committed_tokens_.clear();
        engine_state_ = "ready";
        last_error_.clear();

        std::ostringstream msg;
        msg << "Engine ready: model=" << model_path_ << " (" << model_format_
            << ", load=" << fmt_double(model_load_ms_, 2) << " ms), backend="
            << backend_name(active_backend_) << ", threads=" << num_threads_;
        emit_log("INFO", msg.str());
        return true;
    }

    uint64_t compute_total_parameters() const {
        if (!loader_.is_loaded()) return 0;
        uint64_t total = 0;
        for (const auto& desc : loader_.tensor_descriptors()) {
            uint64_t elems = 0;
            if (desc.compute_num_elements(&elems)) {
                total += elems;
            }
        }
        return total;
    }

    void emit_status_report(const std::string& event_name, const std::string& request_id = "") {
        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;
        const uint32_t hw_threads = std::max(1u, std::thread::hardware_concurrency());

        std::ostringstream ss;
        ss << "{\"event\":\"" << json_escape(event_name) << "\"";
        if (!request_id.empty()) {
            ss << ",\"request_id\":\"" << json_escape(request_id) << "\"";
        }
        ss << ",\"engine_state\":\"" << json_escape(engine_state_) << "\""
           << ",\"last_error\":\"" << json_escape(last_error_) << "\""
           << ",\"model\":{"
           << "\"loaded\":" << (loader_.is_loaded() && engine_.initialized() ? "true" : "false")
           << ",\"name\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\""
           << ",\"architecture\":\"LlamaForCausalLM (TinyLlama 1.1B)\""
           << ",\"path\":\"" << json_escape(model_path_) << "\""
           << ",\"tokenizer_path\":\"" << json_escape(tokenizer_path_) << "\""
           << ",\"format\":\"" << json_escape(model_format_) << "\""
           << ",\"file_size_bytes\":" << model_file_bytes_
           << ",\"load_time_ms\":" << fmt_double(model_load_ms_, 3)
           << ",\"is_mmap\":" << (loader_.is_mmap() ? "true" : "false")
           << ",\"weight_loading_strategy\":\"" << (loader_.is_mmap() ? "Zero-Copy POSIX mmap (64B aligned)" : "Heap Buffer") << "\""
           << ",\"num_tensors\":" << loader_.num_tensors()
           << ",\"parameter_count\":" << compute_total_parameters();

        const ModelConfig& cfg = loader_.is_loaded() ? loader_.config() : ModelConfig{};
        ss << ",\"config\":{"
           << "\"num_layers\":" << cfg.num_layers
           << ",\"hidden_dim\":" << cfg.hidden_dim
           << ",\"intermediate_dim\":" << cfg.intermediate_dim
           << ",\"num_query_heads\":" << cfg.num_query_heads
           << ",\"num_kv_heads\":" << cfg.num_kv_heads
           << ",\"head_dim\":" << cfg.head_dim
           << ",\"gqa_group_size\":" << cfg.gqa_group_size()
           << ",\"vocab_size\":" << cfg.vocab_size
           << ",\"max_seq_len\":" << cfg.max_seq_len
           << ",\"rms_norm_eps\":" << fmt_double(cfg.rms_norm_eps, 6)
           << ",\"rope_theta\":" << fmt_double(cfg.rope_theta, 1)
           << "}";

        // Include all tensor directory entries from the loaded .ie file
        ss << ",\"tensors\":[";
        if (loader_.is_loaded()) {
            const auto& descs = loader_.tensor_descriptors();
            for (size_t i = 0; i < descs.size(); ++i) {
                if (i > 0) ss << ",";
                const auto& d = descs[i];
                ss << "{\"name\":\"" << json_escape(d.name) << "\""
                   << ",\"dtype\":\"" << dtype_name(d.dtype) << "\""
                   << ",\"ndim\":" << d.ndim
                   << ",\"shape\":[";
                for (uint32_t dim_i = 0; dim_i < d.ndim; ++dim_i) {
                    if (dim_i > 0) ss << ",";
                    ss << d.shape[dim_i];
                }
                ss << "],\"payload_offset\":" << d.payload_offset
                   << ",\"payload_nbytes\":" << d.payload_nbytes << "}";
            }
        }
        ss << "]}";

        const double kv_mib =
            engine_.initialized()
                ? (2.0 * static_cast<double>(engine_.kv_cache().num_layers()) *
                   static_cast<double>(engine_.kv_cache().max_seq_len()) *
                   static_cast<double>(engine_.kv_cache().kv_dim()) * sizeof(float)) /
                      (1024.0 * 1024.0)
                : 0.0;

        ss << ",\"runtime\":{"
           << "\"active_backend\":\"" << backend_name(active_backend_) << "\""
           << ",\"best_backend\":\"" << backend_name(detect_best_backend()) << "\""
           << ",\"neon_available\":" << (is_backend_available(SimdBackend::Neon) ? "true" : "false")
           << ",\"avx2_available\":" << (is_backend_available(SimdBackend::Avx2) ? "true" : "false")
           << ",\"scalar_available\":true"
           << ",\"num_threads\":" << num_threads_
           << ",\"hardware_threads\":" << hw_threads
           << ",\"kv_seq_len\":" << (engine_.initialized() ? engine_.kv_cache().seq_len() : 0)
           << ",\"kv_max_seq_len\":" << (engine_.initialized() ? engine_.kv_cache().max_seq_len() : cfg.max_seq_len)
           << ",\"kv_cache_size_mib\":" << fmt_double(kv_mib, 2)
           << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
           << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2)
           << ",\"os_arch\":\"" << json_escape(detect_os_arch()) << "\""
#if defined(__VERSION__)
           << ",\"compiler\":\"" << json_escape(__VERSION__) << "\""
#else
           << ",\"compiler\":\"C++17\""
#endif
           << "}";

        // Discover available .ie models on disk
        ss << ",\"available_models\":[";
        const std::vector<std::pair<std::string, std::string>> candidate_models = {
            {"models/tinyllama-1.1b-q4_0.ie", "Q4_0"},
            {"models/tinyllama-1.1b-fp32.ie", "FP32"},
        };
        bool first_m = true;
        for (const auto& cm : candidate_models) {
            std::error_code ec;
            if (std::filesystem::exists(cm.first, ec)) {
                if (!first_m) ss << ",";
                first_m = false;
                const uint64_t fsize = static_cast<uint64_t>(std::filesystem::file_size(cm.first, ec));
                ss << "{\"path\":\"" << json_escape(cm.first) << "\""
                   << ",\"format\":\"" << json_escape(cm.second) << "\""
                   << ",\"size_bytes\":" << fsize << "}";
            }
        }
        ss << "]}";

        emit_raw_json(ss.str());
    }

    void handle_clear(const JsonValue& cmd) {
        const std::string req_id = cmd.get_string("request_id");
        if (engine_.initialized()) {
            engine_.kv_cache().clear();
        }
        committed_tokens_.clear();
        emit_log("INFO", "KV cache and conversation prefix cleared (seq_len reset to 0).");

        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;
        std::ostringstream ss;
        ss << "{\"event\":\"context_cleared\",\"request_id\":\"" << json_escape(req_id)
           << "\",\"kv_seq_len\":0"
           << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
           << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2) << "}";
        emit_raw_json(ss.str());
    }

    SimdBackend parse_backend_string(const std::string& b_str) const {
        if (b_str == "NEON" || b_str == "neon") return SimdBackend::Neon;
        if (b_str == "AVX2" || b_str == "avx2" || b_str == "AVX2/FMA") return SimdBackend::Avx2;
        if (b_str == "Scalar" || b_str == "scalar") return SimdBackend::Scalar;
        return active_backend_;
    }

    void handle_configure(const JsonValue& cmd) {
        const std::string req_id = cmd.get_string("request_id");
        std::string new_model = cmd.get_string("model_path", model_path_);
        uint32_t new_threads = static_cast<uint32_t>(cmd.get_number("threads", num_threads_));
        if (new_threads == 0) new_threads = num_threads_;
        SimdBackend new_backend = parse_backend_string(cmd.get_string("backend", backend_name(active_backend_)));

        load_and_init_model(new_model, new_threads, new_backend);
        emit_status_report("status_report", req_id);
    }

    void handle_chat(const JsonValue& cmd) {
        const std::string req_id = cmd.get_string("request_id");
        stop_requested_.store(false, std::memory_order_relaxed);

        // Check if thread count or backend or model changed
        const uint32_t req_threads = static_cast<uint32_t>(cmd.get_number("threads", num_threads_));
        const std::string req_backend_str = cmd.get_string("backend", backend_name(active_backend_));
        const SimdBackend req_backend = parse_backend_string(req_backend_str);
        const std::string req_model = cmd.get_string("model_path", model_path_);

        if (!loader_.is_loaded() || !engine_.initialized() ||
            req_model != model_path_ ||
            (req_threads > 0 && req_threads != num_threads_) ||
            (is_backend_available(req_backend) && req_backend != active_backend_)) {
            if (!load_and_init_model(req_model, req_threads > 0 ? req_threads : num_threads_, req_backend)) {
                emit_error(req_id, "Engine is not initialized: " + last_error_);
                return;
            }
        }

        if (!tokenizer_.is_loaded()) {
            emit_error(req_id, "Tokenizer is not loaded");
            return;
        }

        // Extract conversation messages or single prompt
        std::vector<ChatMessage> history;
        auto msg_it = cmd.obj_val.find("messages");
        if (msg_it != cmd.obj_val.end() && msg_it->second.type == JsonValue::Type::Array) {
            for (const auto& item : msg_it->second.arr_val) {
                if (item.type == JsonValue::Type::Object) {
                    std::string role = item.get_string("role", "user");
                    std::string content = item.get_string("content", "");
                    if (!content.empty()) {
                        history.push_back(ChatMessage{role, content});
                    }
                }
            }
        }
        if (history.empty()) {
            std::string single_prompt = cmd.get_string("prompt");
            if (!single_prompt.empty()) {
                history.push_back(ChatMessage{"user", single_prompt});
            }
        }
        if (history.empty()) {
            emit_error(req_id, "Prompt or messages must not be empty");
            return;
        }

        const float temperature = static_cast<float>(cmd.get_number("temperature", 0.0));
        const float top_p = static_cast<float>(cmd.get_number("top_p", 1.0));
        const uint32_t top_k = static_cast<uint32_t>( std::max(0.0, cmd.get_number("top_k", 0.0)));
        const uint32_t max_tokens = static_cast<uint32_t>(std::max(1.0, cmd.get_number("max_tokens", 128.0)));
        const uint64_t seed = static_cast<uint64_t>(std::max(0.0, cmd.get_number("seed", 42.0)));

        if (temperature < 0.0f) {
            emit_error(req_id, "Temperature must be non-negative");
            return;
        }
        if (top_p <= 0.0f || top_p > 1.0f) {
            emit_error(req_id, "Top-p must be in (0, 1]");
            return;
        }

        const std::string formatted_prompt = Tokenizer::format_chat_prompt(history, true);
        std::vector<int32_t> full_tokens;
        Status st = tokenizer_.encode(formatted_prompt, true, false, &full_tokens);
        if (!st.ok() || full_tokens.empty()) {
            emit_error(req_id, "Failed to tokenize chat prompt: " + st.message);
            return;
        }

        const uint32_t max_seq_len = loader_.config().max_seq_len;
        if (full_tokens.size() >= max_seq_len) {
            std::ostringstream err;
            err << "Context window limit reached (" << full_tokens.size() << " >= " << max_seq_len
                << " tokens). Clear conversation to reset the KV cache.";
            emit_error(req_id, err.str());
            return;
        }

        engine_state_ = "generating";

        // KV-cache prefix matching
        size_t match_len = 0;
        while (match_len < committed_tokens_.size() &&
               match_len < full_tokens.size() &&
               committed_tokens_[match_len] == full_tokens[match_len]) {
            ++match_len;
        }
        // Always evaluate at least the final prompt token so logits are freshly computed
        if (match_len >= full_tokens.size() && !full_tokens.empty()) {
            match_len = full_tokens.size() - 1;
        }
        if (match_len < committed_tokens_.size()) {
            Status trunc_st = engine_.kv_cache().truncate(static_cast<uint32_t>(match_len));
            if (!trunc_st.ok()) {
                engine_.kv_cache().clear();
                committed_tokens_.clear();
                match_len = 0;
            } else {
                committed_tokens_.resize(match_len);
            }
        }

        const size_t reused_kv_tokens = match_len;
        const size_t new_prompt_tokens = full_tokens.size() - match_len;

        {
            std::ostringstream log_ss;
            log_ss << "Generation started (prompt_tokens=" << full_tokens.size()
                   << ", reused_kv=" << reused_kv_tokens << ", new_prefill=" << new_prompt_tokens
                   << ", max_tokens=" << max_tokens << ", temp=" << fmt_double(temperature, 2)
                   << ", top_p=" << fmt_double(top_p, 2) << ", top_k=" << top_k << ")";
            emit_log("INFO", log_ss.str());
        }

        {
            std::ostringstream start_ev;
            start_ev << "{\"event\":\"generation_started\",\"request_id\":\"" << json_escape(req_id) << "\""
                     << ",\"prompt_tokens\":" << full_tokens.size()
                     << ",\"reused_kv_tokens\":" << reused_kv_tokens
                     << ",\"new_prompt_tokens\":" << new_prompt_tokens
                     << ",\"kv_seq_len\":" << engine_.kv_cache().seq_len()
                     << ",\"max_seq_len\":" << max_seq_len
                     << ",\"threads\":" << num_threads_
                     << ",\"backend\":\"" << backend_name(active_backend_) << "\"}";
            emit_raw_json(start_ev.str());
        }

        Sampler sampler(SamplerConfig{temperature, top_p, seed, top_k});
        std::vector<float> logits(loader_.config().vocab_size, 0.0f);

        const double cpu_start = cpu_seconds_fn_ ? cpu_seconds_fn_() : 0.0;
        const auto request_start = StudioClock::now();

        // Prefill phase for un-cached tokens
        for (size_t i = match_len; i < full_tokens.size(); ++i) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                engine_state_ = "ready";
                emit_cancelled(req_id, full_tokens.size(), reused_kv_tokens, i - match_len, 0,
                               0.0, 0.0, 0.0,
                               std::chrono::duration<double, std::milli>(StudioClock::now() - request_start).count(),
                               cpu_start, "");
                return;
            }
            const uint32_t pos = static_cast<uint32_t>(i);
            st = engine_.forward_token(full_tokens[i], pos, logits.data());
            if (!st.ok()) {
                engine_state_ = "ready";
                emit_error(req_id, "Prefill forward_token failed: " + st.message);
                return;
            }
            committed_tokens_.push_back(full_tokens[i]);
        }

        const auto prefill_end = StudioClock::now();
        const double prefill_sec = std::chrono::duration<double>(prefill_end - request_start).count();
        const double prefill_ms = prefill_sec * 1000.0;
        const double prompt_tps =
            prefill_sec > 0.0 ? static_cast<double>(new_prompt_tokens) / prefill_sec : 0.0;

        std::vector<int32_t> turn_generated_ids;
        turn_generated_ids.reserve(max_tokens);
        std::string previous_output;
        uint32_t generated_count = 0;
        std::string stop_reason = "max_tokens";

        // Sample first token
        int32_t next_token = sampler.sample(logits.data(), logits.size());
        if (next_token < 0) {
            engine_state_ = "ready";
            emit_error(req_id, "Sampler failed to select initial token");
            return;
        }
        const auto first_token_time = StudioClock::now();
        const double ttft_ms =
            std::chrono::duration<double, std::milli>(first_token_time - request_start).count();
        generated_count = 1;

        if (next_token == kEosTokenId) {
            stop_reason = "eos";
        } else {
            turn_generated_ids.push_back(next_token);
            std::string current_output;
            st = tokenizer_.decode(turn_generated_ids, &current_output);
            std::string delta;
            if (st.ok()) {
                size_t common = 0;
                while (common < previous_output.size() && common < current_output.size() &&
                       previous_output[common] == current_output[common]) {
                    ++common;
                }
                delta = current_output.substr(common);
                previous_output = std::move(current_output);
            }
            emit_token_and_metrics(req_id, next_token, delta, generated_count,
                                   full_tokens.size(), reused_kv_tokens, new_prompt_tokens,
                                   ttft_ms, prefill_ms, prompt_tps, 0.0, 0.0, ttft_ms, cpu_start);
        }

        const auto decode_start = StudioClock::now();
        auto prev_step_time = decode_start;

        for (uint32_t gen_i = 1; gen_i < max_tokens && next_token != kEosTokenId; ++gen_i) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                const auto now = StudioClock::now();
                const double dec_sec = std::chrono::duration<double>(now - decode_start).count();
                const double dec_tps = (dec_sec > 0.0 && generated_count > 1)
                                           ? static_cast<double>(generated_count - 1) / dec_sec
                                           : 0.0;
                const double tot_ms = std::chrono::duration<double, std::milli>(now - request_start).count();
                engine_state_ = "ready";
                emit_cancelled(req_id, full_tokens.size(), reused_kv_tokens, new_prompt_tokens,
                               generated_count, ttft_ms, prompt_tps, dec_tps, tot_ms, cpu_start, previous_output);
                return;
            }

            if (engine_.kv_cache().seq_len() >= max_seq_len) {
                stop_reason = "context_limit";
                emit_log("WARN", "Reached maximum sequence length (2048) during decode.");
                break;
            }

            const uint32_t pos = engine_.kv_cache().seq_len();
            st = engine_.forward_token(next_token, pos, logits.data());
            if (!st.ok()) {
                engine_state_ = "ready";
                emit_error(req_id, "Decode forward_token failed: " + st.message);
                return;
            }
            committed_tokens_.push_back(next_token);

            next_token = sampler.sample(logits.data(), logits.size());
            if (next_token < 0) {
                engine_state_ = "ready";
                emit_error(req_id, "Sampler failed during decode");
                return;
            }
            ++generated_count;

            const auto now = StudioClock::now();
            const double decode_sec = std::chrono::duration<double>(now - decode_start).count();
            const double step_sec = std::chrono::duration<double>(now - prev_step_time).count();
            prev_step_time = now;

            const double decode_tps =
                (decode_sec > 0.0 && generated_count > 1)
                    ? static_cast<double>(generated_count - 1) / decode_sec
                    : 0.0;
            const double instant_tps = step_sec > 0.0 ? (1.0 / step_sec) : decode_tps;
            const double total_ms = std::chrono::duration<double, std::milli>(now - request_start).count();

            if (next_token == kEosTokenId) {
                stop_reason = "eos";
                break;
            }

            turn_generated_ids.push_back(next_token);
            std::string current_output;
            st = tokenizer_.decode(turn_generated_ids, &current_output);
            std::string delta;
            if (st.ok()) {
                size_t common = 0;
                while (common < previous_output.size() && common < current_output.size() &&
                       previous_output[common] == current_output[common]) {
                    ++common;
                }
                delta = current_output.substr(common);
                previous_output = std::move(current_output);
            }

            emit_token_and_metrics(req_id, next_token, delta, generated_count,
                                   full_tokens.size(), reused_kv_tokens, new_prompt_tokens,
                                   ttft_ms, prefill_ms, prompt_tps, decode_tps, instant_tps,
                                   total_ms, cpu_start);
        }

        const auto gen_end = StudioClock::now();
        const double decode_sec = std::chrono::duration<double>(gen_end - decode_start).count();
        const double decode_ms = decode_sec * 1000.0;
        const double total_ms = std::chrono::duration<double, std::milli>(gen_end - request_start).count();
        const double decode_tps =
            (decode_sec > 0.0 && generated_count > 1)
                ? static_cast<double>(generated_count - 1) / decode_sec
                : 0.0;
        const double cpu_used = cpu_seconds_fn_ ? (cpu_seconds_fn_() - cpu_start) : 0.0;
        const double wall_sec = total_ms / 1000.0;
        const double cpu_pct = (wall_sec > 0.0 && cpu_used > 0.0) ? (cpu_used / wall_sec) * 100.0 : 0.0;
        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;

        engine_state_ = "ready";

        {
            std::ostringstream log_ss;
            log_ss << "Generation completed: " << generated_count << " tokens, TTFT="
                   << fmt_double(ttft_ms, 1) << " ms, prefill=" << fmt_double(prompt_tps, 1)
                   << " tok/s, decode=" << fmt_double(decode_tps, 2) << " tok/s, kv_seq_len="
                   << engine_.kv_cache().seq_len();
            emit_log("INFO", log_ss.str());
        }

        std::ostringstream done_ev;
        done_ev << "{\"event\":\"generation_completed\",\"request_id\":\"" << json_escape(req_id) << "\""
                << ",\"text\":\"" << json_escape(previous_output) << "\""
                << ",\"stop_reason\":\"" << json_escape(stop_reason) << "\""
                << ",\"metrics\":{"
                << "\"ttft_ms\":" << fmt_double(ttft_ms, 3)
                << ",\"prefill_ms\":" << fmt_double(prefill_ms, 3)
                << ",\"decode_ms\":" << fmt_double(decode_ms, 3)
                << ",\"total_ms\":" << fmt_double(total_ms, 3)
                << ",\"prompt_tokens\":" << full_tokens.size()
                << ",\"reused_kv_tokens\":" << reused_kv_tokens
                << ",\"new_prompt_tokens\":" << new_prompt_tokens
                << ",\"generated_tokens\":" << generated_count
                << ",\"prompt_tps\":" << fmt_double(prompt_tps, 3)
                << ",\"decode_tps\":" << fmt_double(decode_tps, 3)
                << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
                << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2)
                << ",\"cpu_utilization_pct\":" << fmt_double(cpu_pct, 1)
                << ",\"kv_seq_len\":" << engine_.kv_cache().seq_len()
                << ",\"max_seq_len\":" << max_seq_len
                << ",\"threads\":" << num_threads_
                << ",\"backend\":\"" << backend_name(active_backend_) << "\""
                << "}}";
        emit_raw_json(done_ev.str());
    }

    void emit_token_and_metrics(const std::string& req_id,
                                int32_t token_id,
                                const std::string& delta,
                                uint32_t generated_count,
                                size_t prompt_tokens,
                                size_t reused_kv_tokens,
                                size_t new_prompt_tokens,
                                double ttft_ms,
                                double prefill_ms,
                                double prompt_tps,
                                double decode_tps,
                                double instant_tps,
                                double total_ms,
                                double cpu_start) {
        const double cpu_used = cpu_seconds_fn_ ? (cpu_seconds_fn_() - cpu_start) : 0.0;
        const double wall_sec = total_ms / 1000.0;
        const double cpu_pct = (wall_sec > 0.0 && cpu_used > 0.0) ? (cpu_used / wall_sec) * 100.0 : 0.0;
        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;

        std::ostringstream ss;
        ss << "{\"event\":\"token\",\"request_id\":\"" << json_escape(req_id) << "\""
           << ",\"token_id\":" << token_id
           << ",\"delta\":\"" << json_escape(delta) << "\""
           << ",\"text\":\"" << json_escape(delta) << "\""
           << ",\"generated_tokens\":" << generated_count
           << ",\"metrics\":{"
           << "\"ttft_ms\":" << fmt_double(ttft_ms, 3)
           << ",\"prefill_ms\":" << fmt_double(prefill_ms, 3)
           << ",\"total_ms\":" << fmt_double(total_ms, 3)
           << ",\"prompt_tokens\":" << prompt_tokens
           << ",\"reused_kv_tokens\":" << reused_kv_tokens
           << ",\"new_prompt_tokens\":" << new_prompt_tokens
           << ",\"generated_tokens\":" << generated_count
           << ",\"prompt_tps\":" << fmt_double(prompt_tps, 3)
           << ",\"decode_tps\":" << fmt_double(decode_tps, 3)
           << ",\"instant_tps\":" << fmt_double(instant_tps, 3)
           << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
           << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2)
           << ",\"cpu_utilization_pct\":" << fmt_double(cpu_pct, 1)
           << ",\"kv_seq_len\":" << engine_.kv_cache().seq_len()
           << ",\"max_seq_len\":" << loader_.config().max_seq_len
           << ",\"threads\":" << num_threads_
           << ",\"backend\":\"" << backend_name(active_backend_) << "\""
           << "}}";
        emit_raw_json(ss.str());
    }

    void emit_cancelled(const std::string& req_id,
                        size_t prompt_tokens,
                        size_t reused_kv_tokens,
                        size_t new_prompt_tokens,
                        uint32_t generated_count,
                        double ttft_ms,
                        double prompt_tps,
                        double decode_tps,
                        double total_ms,
                        double cpu_start,
                        const std::string& partial_text) {
        const double cpu_used = cpu_seconds_fn_ ? (cpu_seconds_fn_() - cpu_start) : 0.0;
        const double wall_sec = total_ms / 1000.0;
        const double cpu_pct = (wall_sec > 0.0 && cpu_used > 0.0) ? (cpu_used / wall_sec) * 100.0 : 0.0;
        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;

        emit_log("WARN", "Generation cancelled at " + std::to_string(generated_count) + " generated tokens.");

        std::ostringstream ss;
        ss << "{\"event\":\"generation_cancelled\",\"request_id\":\"" << json_escape(req_id) << "\""
           << ",\"text\":\"" << json_escape(partial_text) << "\""
           << ",\"metrics\":{"
           << "\"ttft_ms\":" << fmt_double(ttft_ms, 3)
           << ",\"total_ms\":" << fmt_double(total_ms, 3)
           << ",\"prompt_tokens\":" << prompt_tokens
           << ",\"reused_kv_tokens\":" << reused_kv_tokens
           << ",\"new_prompt_tokens\":" << new_prompt_tokens
           << ",\"generated_tokens\":" << generated_count
           << ",\"prompt_tps\":" << fmt_double(prompt_tps, 3)
           << ",\"decode_tps\":" << fmt_double(decode_tps, 3)
           << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
           << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2)
           << ",\"cpu_utilization_pct\":" << fmt_double(cpu_pct, 1)
           << ",\"kv_seq_len\":" << engine_.kv_cache().seq_len()
           << ",\"max_seq_len\":" << (loader_.is_loaded() ? loader_.config().max_seq_len : 2048)
           << ",\"threads\":" << num_threads_
           << ",\"backend\":\"" << backend_name(active_backend_) << "\""
           << "}}";
        emit_raw_json(ss.str());
    }

    void handle_benchmark(const JsonValue& cmd) {
        const std::string req_id = cmd.get_string("request_id");
        stop_requested_.store(false, std::memory_order_relaxed);

        const std::string req_model = cmd.get_string("model_path", model_path_);
        const uint32_t req_threads = static_cast<uint32_t>(std::max(1.0, cmd.get_number("threads", num_threads_)));
        const SimdBackend req_backend = parse_backend_string(cmd.get_string("backend", backend_name(active_backend_)));
        const uint32_t runs = static_cast<uint32_t>(std::clamp(cmd.get_number("runs", 3.0), 1.0, 20.0));
        const uint32_t max_tokens = static_cast<uint32_t>(std::clamp(cmd.get_number("max_tokens", 32.0), 2.0, 512.0));
        std::string raw_prompt = cmd.get_string("prompt", "Explain entropy.");
        if (raw_prompt.empty()) raw_prompt = "Explain entropy.";

        if (!load_and_init_model(req_model, req_threads, req_backend)) {
            emit_error(req_id, "Benchmark model initialization failed: " + last_error_);
            return;
        }

        const std::string formatted = Tokenizer::format_chat_prompt({ChatMessage{"user", raw_prompt}}, true);
        std::vector<int32_t> prompt_ids;
        Status st = tokenizer_.encode(formatted, true, false, &prompt_ids);
        if (!st.ok() || prompt_ids.empty()) {
            emit_error(req_id, "Failed to tokenize benchmark prompt");
            return;
        }

        engine_state_ = "benchmarking";
        {
            std::ostringstream log_ss;
            log_ss << "Starting Benchmark Lab run: model=" << model_format_
                   << ", threads=" << num_threads_ << ", backend=" << backend_name(active_backend_)
                   << ", runs=" << runs << ", max_tokens=" << max_tokens;
            emit_log("INFO", log_ss.str());
        }

        {
            std::ostringstream start_ev;
            start_ev << "{\"event\":\"benchmark_started\",\"request_id\":\"" << json_escape(req_id) << "\""
                     << ",\"runs\":" << runs
                     << ",\"prompt_tokens\":" << prompt_ids.size()
                     << ",\"max_tokens\":" << max_tokens
                     << ",\"threads\":" << num_threads_
                     << ",\"format\":\"" << json_escape(model_format_) << "\""
                     << ",\"backend\":\"" << backend_name(active_backend_) << "\"}";
            emit_raw_json(start_ev.str());
        }

        struct SingleRun {
            uint32_t run_index = 0;
            double ttft_ms = 0.0;
            double prefill_ms = 0.0;
            double decode_ms = 0.0;
            double total_ms = 0.0;
            double prompt_tps = 0.0;
            double decode_tps = 0.0;
            double cpu_pct = 0.0;
            uint32_t gen_tokens = 0;
            double rss_mib = 0.0;
        };
        std::vector<SingleRun> results;
        results.reserve(runs);

        std::vector<float> logits(loader_.config().vocab_size, 0.0f);
        std::string sample_output;

        for (uint32_t r = 0; r < runs; ++r) {
            if (stop_requested_.load(std::memory_order_relaxed)) {
                engine_state_ = "ready";
                emit_log("WARN", "Benchmark cancelled by user.");
                emit_error(req_id, "Benchmark cancelled by user");
                return;
            }

            engine_.kv_cache().clear();
            committed_tokens_.clear();
            Sampler sampler(SamplerConfig{0.0f, 1.0f, 42, 0});

            const double cpu_t0 = cpu_seconds_fn_ ? cpu_seconds_fn_() : 0.0;
            const auto t_start = StudioClock::now();

            for (size_t pos = 0; pos < prompt_ids.size(); ++pos) {
                st = engine_.forward_token(prompt_ids[pos], static_cast<uint32_t>(pos), logits.data());
                if (!st.ok()) {
                    engine_state_ = "ready";
                    emit_error(req_id, "Benchmark prefill failed: " + st.message);
                    return;
                }
            }
            const auto t_prefill = StudioClock::now();
            const double prefill_sec = std::chrono::duration<double>(t_prefill - t_start).count();

            int32_t next_token = sampler.sample(logits.data(), logits.size());
            const auto t_first = StudioClock::now();
            const double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();

            uint32_t gen_count = 1;
            std::vector<int32_t> gen_ids;
            if (next_token != kEosTokenId) {
                gen_ids.push_back(next_token);
            }

            const auto t_dec_start = StudioClock::now();
            for (uint32_t g = 1; g < max_tokens && next_token != kEosTokenId; ++g) {
                if (stop_requested_.load(std::memory_order_relaxed)) break;
                const uint32_t pos = static_cast<uint32_t>(prompt_ids.size()) + g - 1;
                st = engine_.forward_token(next_token, pos, logits.data());
                if (!st.ok()) break;
                next_token = sampler.sample(logits.data(), logits.size());
                ++gen_count;
                if (next_token != kEosTokenId) {
                    gen_ids.push_back(next_token);
                }
            }
            const auto t_end = StudioClock::now();
            const double dec_sec = std::chrono::duration<double>(t_end - t_dec_start).count();
            const double tot_sec = std::chrono::duration<double>(t_end - t_start).count();
            const double cpu_used = cpu_seconds_fn_ ? (cpu_seconds_fn_() - cpu_t0) : 0.0;

            if (r == 0) {
                tokenizer_.decode(gen_ids, &sample_output);
            }

            SingleRun sr{};
            sr.run_index = r + 1;
            sr.ttft_ms = ttft_ms;
            sr.prefill_ms = prefill_sec * 1000.0;
            sr.decode_ms = dec_sec * 1000.0;
            sr.total_ms = tot_sec * 1000.0;
            sr.prompt_tps = prefill_sec > 0.0 ? static_cast<double>(prompt_ids.size()) / prefill_sec : 0.0;
            sr.decode_tps = (dec_sec > 0.0 && gen_count > 1) ? static_cast<double>(gen_count - 1) / dec_sec : 0.0;
            sr.cpu_pct = (tot_sec > 0.0 && cpu_used > 0.0) ? (cpu_used / tot_sec) * 100.0 : 0.0;
            sr.gen_tokens = gen_count;
            sr.rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
            results.push_back(sr);

            std::ostringstream step_ev;
            step_ev << "{\"event\":\"benchmark_progress\",\"request_id\":\"" << json_escape(req_id) << "\""
                    << ",\"completed_runs\":" << (r + 1)
                    << ",\"total_runs\":" << runs
                    << ",\"run\":{"
                    << "\"run_index\":" << sr.run_index
                    << ",\"ttft_ms\":" << fmt_double(sr.ttft_ms, 3)
                    << ",\"prompt_tps\":" << fmt_double(sr.prompt_tps, 3)
                    << ",\"decode_tps\":" << fmt_double(sr.decode_tps, 3)
                    << ",\"total_ms\":" << fmt_double(sr.total_ms, 3)
                    << ",\"cpu_utilization_pct\":" << fmt_double(sr.cpu_pct, 1)
                    << ",\"generated_tokens\":" << sr.gen_tokens
                    << "}}";
            emit_raw_json(step_ev.str());
        }

        engine_.kv_cache().clear();
        committed_tokens_.clear();
        engine_state_ = "ready";

        std::vector<double> v_ttft, v_ptps, v_dtps, v_cpu;
        for (const auto& r : results) {
            v_ttft.push_back(r.ttft_ms);
            v_ptps.push_back(r.prompt_tps);
            v_dtps.push_back(r.decode_tps);
            v_cpu.push_back(r.cpu_pct);
        }
        const ExtendedStats s_ttft = compute_extended_stats(v_ttft);
        const ExtendedStats s_ptps = compute_extended_stats(v_ptps);
        const ExtendedStats s_dtps = compute_extended_stats(v_dtps);
        const ExtendedStats s_cpu = compute_extended_stats(v_cpu);

        const double cur_rss_mib = static_cast<double>(current_resident_bytes()) / (1024.0 * 1024.0);
        const double peak_rss_mib =
            peak_rss_fn_ ? static_cast<double>(peak_rss_fn_()) / (1024.0 * 1024.0) : 0.0;

        {
            std::ostringstream log_ss;
            log_ss << "Benchmark finished (" << runs << " runs): decode mean="
                   << fmt_double(s_dtps.mean, 2) << " tok/s, median=" << fmt_double(s_dtps.median, 2)
                   << " tok/s, TTFT mean=" << fmt_double(s_ttft.mean, 1) << " ms";
            emit_log("INFO", log_ss.str());
        }

        std::ostringstream out;
        out << "{\"event\":\"benchmark_completed\",\"request_id\":\"" << json_escape(req_id) << "\""
            << ",\"result\":{"
            << "\"model_path\":\"" << json_escape(model_path_) << "\""
            << ",\"format\":\"" << json_escape(model_format_) << "\""
            << ",\"backend\":\"" << backend_name(active_backend_) << "\""
            << ",\"threads\":" << num_threads_
            << ",\"runs\":" << runs
            << ",\"model_load_ms\":" << fmt_double(model_load_ms_, 3)
            << ",\"prompt\":\"" << json_escape(raw_prompt) << "\""
            << ",\"sample_output\":\"" << json_escape(sample_output) << "\""
            << ",\"prompt_tokens\":" << prompt_ids.size()
            << ",\"generated_tokens\":" << (results.empty() ? 0 : results.front().gen_tokens)
            << ",\"current_rss_mib\":" << fmt_double(cur_rss_mib, 2)
            << ",\"peak_rss_mib\":" << fmt_double(peak_rss_mib, 2)
            << ",\"cpu_utilization_mean_pct\":" << fmt_double(s_cpu.mean, 1)
            << ",\"os_arch\":\"" << json_escape(detect_os_arch()) << "\""
#if defined(__VERSION__)
            << ",\"compiler\":\"" << json_escape(__VERSION__) << "\""
#else
            << ",\"compiler\":\"C++17\""
#endif
            << ",\"ttft_ms\":{\"mean\":" << fmt_double(s_ttft.mean, 3)
            << ",\"median\":" << fmt_double(s_ttft.median, 3)
            << ",\"min\":" << fmt_double(s_ttft.min, 3)
            << ",\"max\":" << fmt_double(s_ttft.max, 3)
            << ",\"stddev\":" << fmt_double(s_ttft.stddev, 3) << "}"
            << ",\"prompt_tps\":{\"mean\":" << fmt_double(s_ptps.mean, 3)
            << ",\"median\":" << fmt_double(s_ptps.median, 3)
            << ",\"min\":" << fmt_double(s_ptps.min, 3)
            << ",\"max\":" << fmt_double(s_ptps.max, 3)
            << ",\"stddev\":" << fmt_double(s_ptps.stddev, 3) << "}"
            << ",\"decode_tps\":{\"mean\":" << fmt_double(s_dtps.mean, 3)
            << ",\"median\":" << fmt_double(s_dtps.median, 3)
            << ",\"min\":" << fmt_double(s_dtps.min, 3)
            << ",\"max\":" << fmt_double(s_dtps.max, 3)
            << ",\"stddev\":" << fmt_double(s_dtps.stddev, 3) << "}"
            << ",\"individual_runs\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) out << ",";
            const auto& r = results[i];
            out << "{\"run_index\":" << r.run_index
                << ",\"ttft_ms\":" << fmt_double(r.ttft_ms, 3)
                << ",\"prefill_ms\":" << fmt_double(r.prefill_ms, 3)
                << ",\"decode_ms\":" << fmt_double(r.decode_ms, 3)
                << ",\"total_ms\":" << fmt_double(r.total_ms, 3)
                << ",\"prompt_tps\":" << fmt_double(r.prompt_tps, 3)
                << ",\"decode_tps\":" << fmt_double(r.decode_tps, 3)
                << ",\"cpu_utilization_pct\":" << fmt_double(r.cpu_pct, 1)
                << ",\"generated_tokens\":" << r.gen_tokens << "}";
        }
        out << "]}}";
        emit_raw_json(out.str());
    }
};

}  // namespace inference_engine::studio

