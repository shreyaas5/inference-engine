#include <iostream>
#include <string>

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"

int main(int argc, char** argv) {
    using namespace inference_engine;

    RuntimeConfig rt_cfg{};
    rt_cfg.backend = detect_best_backend();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "inference-engine (TinyLlama-1.1B CPU Inference Runtime)\n"
                      << "Usage: inference-engine [options]\n"
                      << "Options:\n"
                      << "  --model <path>       Path to .ie model file\n"
                      << "  --tokenizer <path>   Path to SentencePiece tokenizer.model\n"
                      << "  --prompt <text>      Input prompt string\n"
                      << "  --threads <n>        Worker thread count (default: 1)\n"
                      << "  --max-tokens <n>     Maximum tokens to generate (default: 64)\n"
                      << "  --benchmark          Run deterministic benchmark harness\n"
                      << "  --info               Print detected CPU backend and config\n";
            return 0;
        }
        if (arg == "--info") {
            std::cout << "Project: inference-engine\n"
                      << "Target Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
                      << "Detected SIMD Backend: " << backend_name(rt_cfg.backend) << "\n"
                      << "NEON Available: " << (is_backend_available(SimdBackend::Neon) ? "yes" : "no") << "\n"
                      << "AVX2 Available: " << (is_backend_available(SimdBackend::Avx2) ? "yes" : "no") << "\n";
            return 0;
        }
    }

    std::cout << "inference-engine M0 foundation binary initialized (detected backend: "
              << backend_name(rt_cfg.backend) << "). Run with --help for usage.\n";
    return 0;
}
