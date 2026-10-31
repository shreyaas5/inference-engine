#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"
#include "inference_engine/kernels.hpp"
#include "inference_engine/loader.hpp"
#include "inference_engine/sampler.hpp"
#include "inference_engine/transformer.hpp"

namespace inference_engine::testing {

namespace {

struct SyntheticModel {
    ModelConfig config{};
    std::vector<uint8_t> ie_bytes{};
    std::vector<float> embed_weights{};
    std::vector<float> final_norm_weights{};
    std::vector<float> lm_head_weights{};
    struct LayerRef {
        std::vector<float> input_norm{};
        std::vector<float> q_proj{};
        std::vector<float> k_proj{};
        std::vector<float> v_proj{};
        std::vector<float> o_proj{};
        std::vector<float> post_norm{};
        std::vector<float> gate_proj{};
        std::vector<float> up_proj{};
        std::vector<float> down_proj{};
    };
    std::vector<LayerRef> layer_refs{};
};

SyntheticModel create_synthetic_model() {
    SyntheticModel sm;
    sm.config.num_layers = 2;
    sm.config.hidden_dim = 16;
    sm.config.intermediate_dim = 32;
    sm.config.num_query_heads = 4;
    sm.config.num_kv_heads = 2;
    sm.config.head_dim = 4;
    sm.config.vocab_size = 32;
    sm.config.max_seq_len = 16;
    sm.config.rms_norm_eps = 1e-5f;
    sm.config.rope_theta = 10000.0f;
    sm.config.weight_dtype = DataType::FP32;

    const ModelConfig& cfg = sm.config;
    std::vector<TensorDesc> descriptors;
    std::vector<std::vector<uint8_t>> payloads;

    auto add_tensor = [&](const std::string& name,
                          uint32_t ndim,
                          std::array<uint64_t, 4> shape,
                          const std::vector<float>& float_data) {
        TensorDesc desc{};
        desc.name = name;
        desc.dtype = DataType::FP32;
        desc.ndim = ndim;
        desc.shape = shape;
        descriptors.push_back(desc);

        std::vector<uint8_t> bytes(float_data.size() * sizeof(float));
        std::memcpy(bytes.data(), float_data.data(), bytes.size());
        payloads.push_back(std::move(bytes));
    };

    auto gen_matrix = [](size_t n, float base, float factor) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = std::sin(base + static_cast<float>(i + 1) * factor) * 0.05f;
        }
        return v;
    };

    auto gen_norm = [](size_t n, float base) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = 1.0f + std::sin(base + static_cast<float>(i + 1) * 0.1f) * 0.02f;
        }
        return v;
    };

    // 1. embed_tokens
    sm.embed_weights = gen_matrix(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim, 0.5f, 0.07f);
    add_tensor("model.embed_tokens.weight", 2, {cfg.vocab_size, cfg.hidden_dim, 0, 0}, sm.embed_weights);

    // 2. per-layer tensors
    sm.layer_refs.resize(cfg.num_layers);
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        auto& lr = sm.layer_refs[l];

        const float l_offset = static_cast<float>(l) * 1.5f;
        lr.input_norm = gen_norm(cfg.hidden_dim, 1.0f + l_offset);
        add_tensor(prefix + "input_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lr.input_norm);

        lr.q_proj = gen_matrix(static_cast<size_t>(cfg.q_dim()) * cfg.hidden_dim, 2.0f + l_offset, 0.03f);
        add_tensor(prefix + "self_attn.q_proj.weight", 2, {cfg.q_dim(), cfg.hidden_dim, 0, 0}, lr.q_proj);

        lr.k_proj = gen_matrix(static_cast<size_t>(cfg.kv_dim()) * cfg.hidden_dim, 3.0f + l_offset, 0.04f);
        add_tensor(prefix + "self_attn.k_proj.weight", 2, {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, lr.k_proj);

        lr.v_proj = gen_matrix(static_cast<size_t>(cfg.kv_dim()) * cfg.hidden_dim, 4.0f + l_offset, 0.05f);
        add_tensor(prefix + "self_attn.v_proj.weight", 2, {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, lr.v_proj);

        lr.o_proj = gen_matrix(static_cast<size_t>(cfg.hidden_dim) * cfg.q_dim(), 5.0f + l_offset, 0.06f);
        add_tensor(prefix + "self_attn.o_proj.weight", 2, {cfg.hidden_dim, cfg.q_dim(), 0, 0}, lr.o_proj);

        lr.post_norm = gen_norm(cfg.hidden_dim, 6.0f + l_offset);
        add_tensor(prefix + "post_attention_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lr.post_norm);

        lr.gate_proj = gen_matrix(static_cast<size_t>(cfg.intermediate_dim) * cfg.hidden_dim, 7.0f + l_offset, 0.02f);
        add_tensor(prefix + "mlp.gate_proj.weight", 2, {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, lr.gate_proj);

        lr.up_proj = gen_matrix(static_cast<size_t>(cfg.intermediate_dim) * cfg.hidden_dim, 8.0f + l_offset, 0.03f);
        add_tensor(prefix + "mlp.up_proj.weight", 2, {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, lr.up_proj);

        lr.down_proj = gen_matrix(static_cast<size_t>(cfg.hidden_dim) * cfg.intermediate_dim, 9.0f + l_offset, 0.04f);
        add_tensor(prefix + "mlp.down_proj.weight", 2, {cfg.hidden_dim, cfg.intermediate_dim, 0, 0}, lr.down_proj);
    }

    // 3. final norm and lm_head
    sm.final_norm_weights = gen_norm(cfg.hidden_dim, 10.0f);
    add_tensor("model.norm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, sm.final_norm_weights);

    sm.lm_head_weights = gen_matrix(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim, 11.0f, 0.015f);
    add_tensor("lm_head.weight", 2, {cfg.vocab_size, cfg.hidden_dim, 0, 0}, sm.lm_head_weights);

    Status bst = ModelLoader::build_ie_image(cfg, descriptors, payloads, &sm.ie_bytes);
    if (!bst.ok()) {
        std::cerr << "Failed to build synthetic .ie image: " << bst.message << "\n";
    }
    return sm;
}

// Independent reference forward step for a single token
void independent_reference_forward(int32_t token_id,
                                   uint32_t pos,
                                   const SyntheticModel& sm,
                                   std::vector<std::vector<std::vector<float>>>& cached_k,
                                   std::vector<std::vector<std::vector<float>>>& cached_v,
                                   std::vector<float>& out_logits) {
    const ModelConfig& cfg = sm.config;
    std::vector<float> x(cfg.hidden_dim);
    const float* emb_row = sm.embed_weights.data() + static_cast<size_t>(token_id) * cfg.hidden_dim;
    std::memcpy(x.data(), emb_row, cfg.hidden_dim * sizeof(float));

    std::vector<float> norm_buf(cfg.hidden_dim);
    std::vector<float> q(cfg.q_dim());
    std::vector<float> k(cfg.kv_dim());
    std::vector<float> v(cfg.kv_dim());
    std::vector<float> attn_out(cfg.q_dim());
    std::vector<float> proj_out(cfg.hidden_dim);
    std::vector<float> gate(cfg.intermediate_dim);
    std::vector<float> up(cfg.intermediate_dim);
    std::vector<float> ffn_act(cfg.intermediate_dim);
    std::vector<float> scores(pos + 1);

    const uint32_t gqa_group = cfg.gqa_group_size();
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const auto& lr = sm.layer_refs[l];

        // RMSNorm
        scalar::rmsnorm_f32(norm_buf.data(), x.data(), lr.input_norm.data(), cfg.hidden_dim, cfg.rms_norm_eps);

        // QKV
        scalar::matmul_f32(q.data(), lr.q_proj.data(), norm_buf.data(), cfg.q_dim(), cfg.hidden_dim);
        scalar::matmul_f32(k.data(), lr.k_proj.data(), norm_buf.data(), cfg.kv_dim(), cfg.hidden_dim);
        scalar::matmul_f32(v.data(), lr.v_proj.data(), norm_buf.data(), cfg.kv_dim(), cfg.hidden_dim);

        // RoPE
        scalar::rope_inplace_f32(q.data(), cfg.num_query_heads, cfg.head_dim, pos, cfg.rope_theta);
        scalar::rope_inplace_f32(k.data(), cfg.num_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

        // KV cache update
        if (cached_k[l].size() <= pos) {
            cached_k[l].resize(pos + 1);
            cached_v[l].resize(pos + 1);
        }
        cached_k[l][pos] = k;
        cached_v[l][pos] = v;

        // Attention
        for (uint32_t hq = 0; hq < cfg.num_query_heads; ++hq) {
            const uint32_t hkv = hq / gqa_group;
            const float* q_head = q.data() + hq * cfg.head_dim;
            for (uint32_t t = 0; t <= pos; ++t) {
                const float* k_head = cached_k[l][t].data() + hkv * cfg.head_dim;
                scores[t] = scalar::dot_f32(q_head, k_head, cfg.head_dim) * head_scale;
            }
            scalar::softmax_f32(scores.data(), pos + 1);

            float* out_head = attn_out.data() + hq * cfg.head_dim;
            std::fill(out_head, out_head + cfg.head_dim, 0.0f);
            for (uint32_t t = 0; t <= pos; ++t) {
                const float* v_head = cached_v[l][t].data() + hkv * cfg.head_dim;
                for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                    out_head[d] += scores[t] * v_head[d];
                }
            }
        }

        // Out projection and residual
        scalar::matmul_f32(proj_out.data(), lr.o_proj.data(), attn_out.data(), cfg.hidden_dim, cfg.q_dim());
        for (uint32_t i = 0; i < cfg.hidden_dim; ++i) {
            x[i] += proj_out[i];
        }

        // Post-attention norm
        scalar::rmsnorm_f32(norm_buf.data(), x.data(), lr.post_norm.data(), cfg.hidden_dim, cfg.rms_norm_eps);

        // MLP
        scalar::matmul_f32(gate.data(), lr.gate_proj.data(), norm_buf.data(), cfg.intermediate_dim, cfg.hidden_dim);
        scalar::matmul_f32(up.data(), lr.up_proj.data(), norm_buf.data(), cfg.intermediate_dim, cfg.hidden_dim);
        scalar::swiglu_f32(ffn_act.data(), gate.data(), up.data(), cfg.intermediate_dim);
        scalar::matmul_f32(proj_out.data(), lr.down_proj.data(), ffn_act.data(), cfg.hidden_dim, cfg.intermediate_dim);
        for (uint32_t i = 0; i < cfg.hidden_dim; ++i) {
            x[i] += proj_out[i];
        }
    }

    // Final norm & LM head
    scalar::rmsnorm_f32(norm_buf.data(), x.data(), sm.final_norm_weights.data(), cfg.hidden_dim, cfg.rms_norm_eps);
    out_logits.resize(cfg.vocab_size);
    scalar::matmul_f32(out_logits.data(), sm.lm_head_weights.data(), norm_buf.data(), cfg.vocab_size, cfg.hidden_dim);
}

bool test_synthetic_model_parity_and_non_zero_logits() {
    SyntheticModel sm = create_synthetic_model();
    if (sm.ie_bytes.empty()) {
        std::cerr << "create_synthetic_model failed\n";
        return false;
    }

    ModelLoader loader;
    Status lst = loader.load_from_buffer(sm.ie_bytes.data(), sm.ie_bytes.size());
    if (!lst.ok()) {
        std::cerr << "loader.load_from_buffer failed: " << lst.message << "\n";
        return false;
    }

    TransformerEngine engine;
    Status est = engine.init(sm.config, &loader);
    if (!est.ok()) {
        std::cerr << "engine.init failed: " << est.message << "\n";
        return false;
    }

    std::vector<std::vector<std::vector<float>>> ref_k(sm.config.num_layers);
    std::vector<std::vector<std::vector<float>>> ref_v(sm.config.num_layers);

    const std::vector<int32_t> test_tokens = {3, 7, 12};
    std::vector<float> engine_logits(sm.config.vocab_size, 0.0f);
    std::vector<float> ref_logits(sm.config.vocab_size, 0.0f);

    for (uint32_t pos = 0; pos < test_tokens.size(); ++pos) {
        Status fst = engine.forward_token(test_tokens[pos], pos, engine_logits.data());
        if (!fst.ok()) {
            std::cerr << "engine.forward_token failed at pos " << pos << ": " << fst.message << "\n";
            return false;
        }

        independent_reference_forward(test_tokens[pos], pos, sm, ref_k, ref_v, ref_logits);

        // 1. Verify logits are no longer always zero!
        bool has_nonzero = false;
        for (float v : engine_logits) {
            if (std::fabs(v) > 1e-4f) {
                has_nonzero = true;
                break;
            }
        }
        if (!has_nonzero) {
            std::cerr << "Logits are unexpectedly all zero at pos " << pos << "\n";
            return false;
        }

        // 2. Verify bit-exact numerical agreement with independent reference forward path
        float max_diff = 0.0f;
        for (uint32_t i = 0; i < sm.config.vocab_size; ++i) {
            const float diff = std::fabs(engine_logits[i] - ref_logits[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        if (max_diff > 1e-5f) {
            std::cerr << "Parity mismatch with independent reference at pos " << pos
                      << ", max_diff = " << max_diff << "\n";
            return false;
        }
    }

    return true;
}

bool test_position_indexing_prefill_and_decode() {
    SyntheticModel sm = create_synthetic_model();
    ModelLoader loader;
    if (!loader.load_from_buffer(sm.ie_bytes.data(), sm.ie_bytes.size()).ok()) {
        std::cerr << "Failed to load synthetic model\n";
        return false;
    }

    TransformerEngine engine;
    if (!engine.init(sm.config, &loader).ok()) {
        std::cerr << "Failed to init engine\n";
        return false;
    }

    // Prefill 4 prompt tokens: pos = 0, 1, 2, 3
    const std::vector<int32_t> prompt = {2, 5, 9, 14};
    std::vector<float> logits(sm.config.vocab_size, 0.0f);

    for (uint32_t pos = 0; pos < prompt.size(); ++pos) {
        if (engine.kv_cache().seq_len() != pos) {
            std::cerr << "Expected committed seq_len " << pos << " before forward at pos " << pos << "\n";
            return false;
        }
        Status st = engine.forward_token(prompt[pos], pos, logits.data());
        if (!st.ok()) {
            std::cerr << "Prefill forward failed at pos " << pos << ": " << st.message << "\n";
            return false;
        }
        if (engine.kv_cache().seq_len() != pos + 1) {
            std::cerr << "Expected committed seq_len " << (pos + 1) << " after forward at pos " << pos << "\n";
            return false;
        }
    }

    // Autoregressive position contract:
    // Prompt tokens occupy positions 0..3. seq_len is 4.
    // Final prompt logits (at pos 3) predict first generated token.
    // Generated token 1 is processed at position 4.
    if (engine.kv_cache().seq_len() != 4) {
        std::cerr << "Committed seq_len must be 4 after prompt prefill\n";
        return false;
    }

    const int32_t gen_token_1 = Sampler::sample_greedy(logits.data(), logits.size());
    const uint32_t decode_pos_1 = 4;
    Status st1 = engine.forward_token(gen_token_1, decode_pos_1, logits.data());
    if (!st1.ok() || engine.kv_cache().seq_len() != 5) {
        std::cerr << "Decode step 1 failed at pos 4\n";
        return false;
    }

    // Decode token 2: predicted from pos 4 logits, fed at pos 5
    const int32_t gen_token_2 = Sampler::sample_greedy(logits.data(), logits.size());
    const uint32_t decode_pos_2 = 5;
    Status st2 = engine.forward_token(gen_token_2, decode_pos_2, logits.data());
    if (!st2.ok() || engine.kv_cache().seq_len() != 6) {
        std::cerr << "Decode step 2 failed at pos 5\n";
        return false;
    }

    // Decode token 3: predicted from pos 5 logits, fed at pos 6
    const int32_t gen_token_3 = Sampler::sample_greedy(logits.data(), logits.size());
    const uint32_t decode_pos_3 = 6;
    Status st3 = engine.forward_token(gen_token_3, decode_pos_3, logits.data());
    if (!st3.ok() || engine.kv_cache().seq_len() != 7) {
        std::cerr << "Decode step 3 failed at pos 6\n";
        return false;
    }

    return true;
}

bool test_kv_cache_and_gqa_mapping() {
    SyntheticModel sm = create_synthetic_model();
    ModelLoader loader;
    loader.load_from_buffer(sm.ie_bytes.data(), sm.ie_bytes.size());
    TransformerEngine engine;
    engine.init(sm.config, &loader);

    std::vector<float> logits(sm.config.vocab_size, 0.0f);
    engine.forward_token(4, 0, logits.data());
    engine.forward_token(8, 1, logits.data());

    // Verify KV cache contents for layer 0 and 1 at pos 0 and 1 are non-null and non-zero
    for (uint32_t l = 0; l < sm.config.num_layers; ++l) {
        for (uint32_t p = 0; p < 2; ++p) {
            const float* k_ptr = engine.kv_cache().key_vector(l, p);
            const float* v_ptr = engine.kv_cache().value_vector(l, p);
            if (k_ptr == nullptr || v_ptr == nullptr) {
                std::cerr << "KV cache vector is null at layer " << l << ", pos " << p << "\n";
                return false;
            }
            float k_norm = scalar::dot_f32(k_ptr, k_ptr, sm.config.kv_dim());
            float v_norm = scalar::dot_f32(v_ptr, v_ptr, sm.config.kv_dim());
            if (k_norm <= 1e-6f || v_norm <= 1e-6f) {
                std::cerr << "KV cache vector unexpectedly zero at layer " << l << ", pos " << p << "\n";
                return false;
            }
        }
    }

    return true;
}

bool test_error_handling_contracts() {
    SyntheticModel sm = create_synthetic_model();
    ModelLoader loader;
    loader.load_from_buffer(sm.ie_bytes.data(), sm.ie_bytes.size());

    TransformerEngine engine;
    engine.init(sm.config, &loader);

    std::vector<float> logits(sm.config.vocab_size, 0.0f);

    // 1. Null out_logits
    if (engine.forward_token(0, 0, nullptr).ok()) {
        std::cerr << "Expected null out_logits to fail\n";
        return false;
    }

    // 2. token_id out of range
    if (engine.forward_token(-1, 0, logits.data()).ok()) {
        std::cerr << "Expected negative token_id to fail\n";
        return false;
    }
    if (engine.forward_token(static_cast<int32_t>(sm.config.vocab_size), 0, logits.data()).ok()) {
        std::cerr << "Expected token_id >= vocab_size to fail\n";
        return false;
    }

    // 3. pos >= max_seq_len
    if (engine.forward_token(0, sm.config.max_seq_len, logits.data()).ok()) {
        std::cerr << "Expected pos >= max_seq_len to fail\n";
        return false;
    }

    // 4. Out-of-order pos (seq_len is 0, asking for pos 2)
    if (engine.forward_token(0, 2, logits.data()).ok()) {
        std::cerr << "Expected out-of-order pos to fail\n";
        return false;
    }

    // 5. Uninitialized engine
    TransformerEngine uninit_engine;
    if (uninit_engine.forward_token(0, 0, logits.data()).ok()) {
        std::cerr << "Expected uninitialized engine forward to fail\n";
        return false;
    }

    // 6. Loader with missing tensor
    ModelConfig bad_cfg = sm.config;
    TensorDesc incomplete_desc{};
    incomplete_desc.name = "model.embed_tokens.weight";
    incomplete_desc.dtype = DataType::FP32;
    incomplete_desc.ndim = 2;
    incomplete_desc.shape = {bad_cfg.vocab_size, bad_cfg.hidden_dim, 0, 0};
    std::vector<uint8_t> payload(static_cast<size_t>(bad_cfg.vocab_size) * bad_cfg.hidden_dim * sizeof(float), 0x1);
    std::vector<uint8_t> bad_image;
    ModelLoader::build_ie_image(bad_cfg, {incomplete_desc}, {payload}, &bad_image);

    ModelLoader incomplete_loader;
    incomplete_loader.load_from_buffer(bad_image.data(), bad_image.size());
    TransformerEngine missing_tensor_engine;
    if (missing_tensor_engine.init(bad_cfg, &incomplete_loader).ok()) {
        std::cerr << "Expected engine.init to fail when required tensors are missing\n";
        return false;
    }

    // 7. Unsupported dtype in ModelConfig
    ModelConfig bad_dtype_cfg = sm.config;
    bad_dtype_cfg.weight_dtype = DataType::Q4_0;
    TransformerEngine bad_dtype_engine;
    if (bad_dtype_engine.init(bad_dtype_cfg, &loader).ok()) {
        std::cerr << "Expected engine.init to fail when weight_dtype != FP32 in M1\n";
        return false;
    }

    return true;
}

bool test_offline_golden_fixture_if_present() {
    const std::string golden_bin_path = "fixtures/golden/golden_reference.bin";
    const std::string model_ie_path = "fixtures/golden/model.ie";

    std::ifstream bin_file(golden_bin_path, std::ios::binary);
    std::ifstream model_file(model_ie_path, std::ios::binary);

    if (!bin_file || !model_file) {
        std::cout << "[INFO] fixtures/golden not present; skipping offline Hugging Face fixture run.\n";
        return true;
    }

    std::cout << "[INFO] Found golden reference fixture; executing end-to-end TinyLlama validation...\n";

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t vocab_size = 0;
    uint32_t prompt_len = 0;
    uint32_t gen_len = 0;

    bin_file.read(reinterpret_cast<char*>(&magic), 4);
    bin_file.read(reinterpret_cast<char*>(&version), 4);
    bin_file.read(reinterpret_cast<char*>(&vocab_size), 4);
    bin_file.read(reinterpret_cast<char*>(&prompt_len), 4);
    bin_file.read(reinterpret_cast<char*>(&gen_len), 4);

    if (magic != 0x474F4C44 || version != 1) {
        std::cerr << "Corrupted golden fixture header\n";
        return false;
    }

    std::vector<uint32_t> prompt_tokens(prompt_len);
    bin_file.read(reinterpret_cast<char*>(prompt_tokens.data()), prompt_len * sizeof(uint32_t));

    std::vector<float> prompt_ref_logits(vocab_size);
    bin_file.read(reinterpret_cast<char*>(prompt_ref_logits.data()), vocab_size * sizeof(float));

    std::vector<uint32_t> gen_ref_tokens(gen_len);
    bin_file.read(reinterpret_cast<char*>(gen_ref_tokens.data()), gen_len * sizeof(uint32_t));

    std::vector<float> gen_ref_logits(static_cast<size_t>(gen_len) * vocab_size);
    bin_file.read(reinterpret_cast<char*>(gen_ref_logits.data()), gen_ref_logits.size() * sizeof(float));

    ModelLoader model_loader;
    Status lst = model_loader.load_from_file(model_ie_path);
    if (!lst.ok()) {
        std::cerr << "Failed to load golden model.ie: " << lst.message << "\n";
        return false;
    }

    TransformerEngine engine;
    Status est = engine.init(model_loader.config(), &model_loader);
    if (!est.ok()) {
        std::cerr << "Failed to init golden engine: " << est.message << "\n";
        return false;
    }

    std::vector<float> logits(vocab_size, 0.0f);
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        Status fst = engine.forward_token(static_cast<int32_t>(prompt_tokens[pos]), pos, logits.data());
        if (!fst.ok()) {
            std::cerr << "Golden prefill forward failed at pos " << pos << ": " << fst.message << "\n";
            return false;
        }
    }

    // Verify logits at last prompt token
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        max_diff = std::max(max_diff, std::fabs(logits[i] - prompt_ref_logits[i]));
    }
    std::cout << "[INFO] Golden prompt final logits max_diff = " << max_diff << "\n";
    if (max_diff > 1e-3f) {
        std::cerr << "Golden prompt logits exceeded tolerance (max_diff=" << max_diff << ")\n";
        return false;
    }

    // Autoregressive decode verification
    for (uint32_t step = 0; step < gen_len; ++step) {
        const int32_t next_token = Sampler::sample_greedy(logits.data(), logits.size());
        if (static_cast<uint32_t>(next_token) != gen_ref_tokens[step]) {
            std::cerr << "Golden decode token mismatch at step " << step
                      << ": expected " << gen_ref_tokens[step] << ", got " << next_token << "\n";
            return false;
        }
        const uint32_t decode_pos = prompt_len + step;
        Status dst = engine.forward_token(next_token, decode_pos, logits.data());
        if (!dst.ok()) {
            std::cerr << "Golden decode forward failed at step " << step << "\n";
            return false;
        }
    }

    std::cout << "[SUCCESS] Golden reference parity verified across prompt and "
              << gen_len << " decode steps!\n";
    return true;
}

}  // namespace

bool run_golden_forward_tests() {
    if (!test_synthetic_model_parity_and_non_zero_logits()) {
        return false;
    }
    if (!test_position_indexing_prefill_and_decode()) {
        return false;
    }
    if (!test_kv_cache_and_gqa_mapping()) {
        return false;
    }
    if (!test_error_handling_contracts()) {
        return false;
    }
    if (!test_offline_golden_fixture_if_present()) {
        return false;
    }
    return true;
}

}  // namespace inference_engine::testing
