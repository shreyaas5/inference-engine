#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include "inference_engine/loader.hpp"
#include "inference_engine/common.hpp"

namespace inference_engine::testing {

bool run_cli_integration_test() {
    // Build a minimal synthetic model
    ModelConfig cfg{};
    cfg.num_layers = 1;
    cfg.hidden_dim = 32;
    cfg.intermediate_dim = 64;
    cfg.num_query_heads = 2;
    cfg.num_kv_heads = 1; // GQA ratio 2:1
    cfg.head_dim = 16;
    cfg.vocab_size = 32;
    cfg.max_seq_len = 16;
    cfg.weight_dtype = DataType::FP32;

    // Helper to create simple FP32 tensor payloads
    auto make_fp32_tensor = [&](const std::string &name, const std::array<uint64_t, 4> &shape, float base, float step) {
        TensorDesc d{};
        d.name = name;
        d.dtype = DataType::FP32;
        d.ndim = 0;
        for (size_t i = 0; i < 4; ++i) {
            if (shape[i] != 0) ++d.ndim;
        }
        d.shape = shape;
        uint64_t elems = 0;
        d.compute_num_elements(&elems);
        std::vector<uint8_t> payload(elems * sizeof(float));
        for (size_t i = 0; i < elems; ++i) {
            float val = base + static_cast<float>(i) * step;
            std::memcpy(payload.data() + i * sizeof(float), &val, sizeof(float));
        }
        return std::make_pair(d, payload);
    };

    std::vector<TensorDesc> descs;
    std::vector<std::vector<uint8_t>> payloads;

    // Embedding matrix
    auto [embed_desc, embed_payload] = make_fp32_tensor("model.embed_tokens.weight", {cfg.vocab_size, cfg.hidden_dim, 0, 0}, 0.1f, 0.001f);
    descs.push_back(embed_desc);
    payloads.push_back(std::move(embed_payload));

    // LM head weight
    auto [lm_desc, lm_payload] = make_fp32_tensor("lm_head.weight", {cfg.vocab_size, cfg.hidden_dim, 0, 0}, -0.2f, 0.002f);
    descs.push_back(lm_desc);
    payloads.push_back(std::move(lm_payload));

    // Layer norm weights
    auto [ln_desc, ln_payload] = make_fp32_tensor("model.layers.0.input_layernorm.weight", {cfg.hidden_dim, 0, 0, 0}, 1.0f, 0.0f);
    descs.push_back(ln_desc);
    payloads.push_back(std::move(ln_payload));
    auto [final_ln_desc, final_ln_payload] = make_fp32_tensor("model.norm.weight", {cfg.hidden_dim, 0, 0, 0}, 1.0f, 0.0f);
    descs.push_back(final_ln_desc);
    payloads.push_back(std::move(final_ln_payload));

    // Attention and MLP weights
    auto [q_desc, q_payload] = make_fp32_tensor("model.layers.0.self_attn.q_proj.weight", {cfg.q_dim(), cfg.hidden_dim, 0, 0}, 0.05f, 0.0005f);
    descs.push_back(q_desc);
    payloads.push_back(std::move(q_payload));
    auto [k_desc, k_payload] = make_fp32_tensor("model.layers.0.self_attn.k_proj.weight", {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, 0.06f, 0.0006f);
    descs.push_back(k_desc);
    payloads.push_back(std::move(k_payload));
    auto [v_desc, v_payload] = make_fp32_tensor("model.layers.0.self_attn.v_proj.weight", {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, 0.07f, 0.0007f);
    descs.push_back(v_desc);
    payloads.push_back(std::move(v_payload));
    auto [o_desc, o_payload] = make_fp32_tensor("model.layers.0.self_attn.o_proj.weight", {cfg.hidden_dim, cfg.q_dim(), 0, 0}, 0.08f, 0.0008f);
    descs.push_back(o_desc);
    payloads.push_back(std::move(o_payload));
    auto [gate_desc, gate_payload] = make_fp32_tensor("model.layers.0.mlp.gate_proj.weight", {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, 0.09f, 0.0009f);
    descs.push_back(gate_desc);
    payloads.push_back(std::move(gate_payload));
    auto [up_desc, up_payload] = make_fp32_tensor("model.layers.0.mlp.up_proj.weight", {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, 0.10f, 0.0010f);
    descs.push_back(up_desc);
    payloads.push_back(std::move(up_payload));
    auto [down_desc, down_payload] = make_fp32_tensor("model.layers.0.mlp.down_proj.weight", {cfg.hidden_dim, cfg.intermediate_dim, 0, 0}, 0.11f, 0.0011f);
    descs.push_back(down_desc);
    payloads.push_back(std::move(down_payload));

    // Build the .ie image
    std::vector<uint8_t> image;
    Status st = ModelLoader::build_ie_image(cfg, descs, payloads, &image);
    if (!st.ok()) {
        std::cerr << "Failed to build synthetic .ie model: " << st.message << "\n";
        return false;
    }

    // Write to a temporary file
    std::filesystem::path temp_model = std::filesystem::temp_directory_path() / "cli_synthetic_test.ie";
    {
        std::ofstream ofs(temp_model, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            std::cerr << "Unable to open temporary file for model output\n";
            return false;
        }
        ofs.write(reinterpret_cast<const char *>(image.data()), static_cast<std::streamsize>(image.size()));
    }

    // CLI executable path (assumes CTest runs with build dir as CWD)
    std::string cli_path = "./inference-engine";
    std::ostringstream cmd;
    cmd << cli_path
        << " --model " << temp_model.string()
        << " --prompt \"Hello\""
        << " --max-new-tokens 4"
        << " --temperature 0"
        << " --benchmark";

    // Execute and capture output (Windows _popen)
    FILE *pipe = _popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to execute CLI command\n";
        std::filesystem::remove(temp_model);
        return false;
    }
    std::string cli_output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        cli_output += buffer;
    }
    int rc = _pclose(pipe);
    std::filesystem::remove(temp_model);

    if (rc != 0) {
        std::cerr << "CLI exited with non-zero status: " << rc << "\n";
        return false;
    }
    if (cli_output.empty()) {
        std::cerr << "CLI produced no output\n";
        return false;
    }
    if (cli_output.find("benchmark") == std::string::npos) {
        std::cerr << "CLI output missing benchmark information\n";
        return false;
    }
    if (cli_output.find("Hello") == std::string::npos) {
        std::cerr << "CLI output missing original prompt\n";
        return false;
    }
    return true;
}

} // namespace inference_engine::testing
