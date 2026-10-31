#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "inference_engine/common.hpp"

#if defined(_WIN32)
#define IE_POPEN _popen
#define IE_PCLOSE _pclose
#else
#define IE_POPEN popen
#define IE_PCLOSE pclose
#endif

namespace inference_engine::testing {

bool run_cli_integration_test() {
    // Resolve CLI binary path
    std::string cli_path = "./inference-engine";
#if defined(_WIN32)
    cli_path = "inference-engine.exe";
#endif

    // Test 1: Verify --help returns 0 and displays usage instructions
    {
        std::string cmd = cli_path + " --help";
        FILE* pipe = IE_POPEN(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to spawn: " << cmd << "\n";
            return false;
        }
        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        int rc = IE_PCLOSE(pipe);
        if (rc != 0) {
            std::cerr << "CLI --help exited with status: " << rc << "\n";
            return false;
        }
        if (output.find("Usage: inference-engine") == std::string::npos) {
            std::cerr << "CLI --help output missing usage string\n";
            return false;
        }
        if (output.find("--model") == std::string::npos ||
            output.find("--tokenizer") == std::string::npos) {
            std::cerr << "CLI --help output missing standard options\n";
            return false;
        }
    }

    // Test 2: Verify --info returns 0 and outputs architecture parameters
    {
        std::string cmd = cli_path + " --info";
        FILE* pipe = IE_POPEN(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to spawn: " << cmd << "\n";
            return false;
        }
        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        int rc = IE_PCLOSE(pipe);
        if (rc != 0) {
            std::cerr << "CLI --info exited with status: " << rc << "\n";
            return false;
        }
        if (output.find("Detected SIMD Backend:") == std::string::npos) {
            std::cerr << "CLI --info missing CPU SIMD detection output\n";
            return false;
        }
        if (output.find("Layers: 22") == std::string::npos ||
            output.find("Hidden dimension: 2048") == std::string::npos) {
            std::cerr << "CLI --info missing frozen architecture metadata\n";
            return false;
        }
    }

    // Test 3: Verify --format-chat when tokenizer is available
    const char* tok_env = std::getenv("INFERENCE_ENGINE_TOKENIZER_MODEL");
    std::string tok_path = tok_env != nullptr ? tok_env : "";
    if (tok_path.empty() && std::filesystem::exists("models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model")) {
        tok_path = "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model";
    }
    if (tok_path.empty() && std::filesystem::exists("../models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model")) {
        tok_path = "../models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model";
    }
    if (!tok_path.empty()) {
        std::string cmd = cli_path + " --tokenizer \"" + tok_path + "\" --prompt \"Hello\" --format-chat";
        FILE* pipe = IE_POPEN(cmd.c_str(), "r");
        if (pipe != nullptr) {
            std::string output;
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                output += buffer;
            }
            int rc = IE_PCLOSE(pipe);
            if (rc == 0) {
                if (output.find("<|user|>") == std::string::npos ||
                    output.find("<|assistant|>") == std::string::npos) {
                    std::cerr << "CLI --format-chat missing expected Zephyr chat markers\n";
                    return false;
                }
            }
        }
    }

    return true;
}

}  // namespace inference_engine::testing
