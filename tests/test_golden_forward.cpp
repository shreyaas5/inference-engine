#include <chrono>
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
#include "inference_engine/tokenizer.hpp"
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

SyntheticModel create_synthetic_model_64(DataType weight_dtype, bool keep_embeddings_fp32 = false) {
    SyntheticModel sm;
    sm.config.num_layers = 2;
    sm.config.hidden_dim = 64;
    sm.config.intermediate_dim = 128;
    sm.config.num_query_heads = 4;
    sm.config.num_kv_heads = 2;
    sm.config.head_dim = 16;
    sm.config.vocab_size = 64;
    sm.config.max_seq_len = 32;
    sm.config.rms_norm_eps = 1e-5f;
    sm.config.rope_theta = 10000.0f;
    sm.config.weight_dtype = weight_dtype;

    const ModelConfig& cfg = sm.config;
    std::vector<TensorDesc> descriptors;
    std::vector<std::vector<uint8_t>> payloads;

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

    auto add_f32 = [&](const std::string& name, uint32_t ndim, std::array<uint64_t, 4> shape, const std::vector<float>& data) {
        TensorDesc desc{};
        desc.name = name;
        desc.dtype = DataType::FP32;
        desc.ndim = ndim;
        desc.shape = shape;
        descriptors.push_back(desc);

        std::vector<uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        payloads.push_back(std::move(bytes));
    };

    auto add_2d = [&](const std::string& name, uint64_t rows, uint64_t cols, const std::vector<float>& data, bool force_fp32 = false) {
        if (weight_dtype == DataType::FP32 || force_fp32) {
            add_f32(name, 2, {rows, cols, 0, 0}, data);
        } else {
            TensorDesc desc{};
            desc.name = name;
            desc.dtype = DataType::Q4_0;
            desc.ndim = 2;
            desc.shape = {rows, cols, 0, 0};
            descriptors.push_back(desc);

            std::vector<BlockQ4_0> blocks(data.size() / kQ4_0BlockElements);
            quantize_row_q4_0(data.data(), blocks.data(), data.size());
            std::vector<uint8_t> bytes(blocks.size() * sizeof(BlockQ4_0));
            std::memcpy(bytes.data(), blocks.data(), bytes.size());
            payloads.push_back(std::move(bytes));
        }
    };

    sm.embed_weights = gen_matrix(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim, 0.5f, 0.07f);
    add_2d("model.embed_tokens.weight", cfg.vocab_size, cfg.hidden_dim, sm.embed_weights, keep_embeddings_fp32);

    sm.layer_refs.resize(cfg.num_layers);
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        auto& lr = sm.layer_refs[l];
        const float l_offset = static_cast<float>(l) * 1.5f;

        lr.input_norm = gen_norm(cfg.hidden_dim, 1.0f + l_offset);
        add_f32(prefix + "input_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lr.input_norm);

        lr.q_proj = gen_matrix(static_cast<size_t>(cfg.q_dim()) * cfg.hidden_dim, 2.0f + l_offset, 0.03f);
        add_2d(prefix + "self_attn.q_proj.weight", cfg.q_dim(), cfg.hidden_dim, lr.q_proj);

        lr.k_proj = gen_matrix(static_cast<size_t>(cfg.kv_dim()) * cfg.hidden_dim, 3.0f + l_offset, 0.04f);
        add_2d(prefix + "self_attn.k_proj.weight", cfg.kv_dim(), cfg.hidden_dim, lr.k_proj);

        lr.v_proj = gen_matrix(static_cast<size_t>(cfg.kv_dim()) * cfg.hidden_dim, 4.0f + l_offset, 0.05f);
        add_2d(prefix + "self_attn.v_proj.weight", cfg.kv_dim(), cfg.hidden_dim, lr.v_proj);

        lr.o_proj = gen_matrix(static_cast<size_t>(cfg.hidden_dim) * cfg.q_dim(), 5.0f + l_offset, 0.06f);
        add_2d(prefix + "self_attn.o_proj.weight", cfg.hidden_dim, cfg.q_dim(), lr.o_proj);

        lr.post_norm = gen_norm(cfg.hidden_dim, 6.0f + l_offset);
        add_f32(prefix + "post_attention_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lr.post_norm);

        lr.gate_proj = gen_matrix(static_cast<size_t>(cfg.intermediate_dim) * cfg.hidden_dim, 7.0f + l_offset, 0.02f);
        add_2d(prefix + "mlp.gate_proj.weight", cfg.intermediate_dim, cfg.hidden_dim, lr.gate_proj);

        lr.up_proj = gen_matrix(static_cast<size_t>(cfg.intermediate_dim) * cfg.hidden_dim, 8.0f + l_offset, 0.03f);
        add_2d(prefix + "mlp.up_proj.weight", cfg.intermediate_dim, cfg.hidden_dim, lr.up_proj);

        lr.down_proj = gen_matrix(static_cast<size_t>(cfg.hidden_dim) * cfg.intermediate_dim, 9.0f + l_offset, 0.04f);
        add_2d(prefix + "mlp.down_proj.weight", cfg.hidden_dim, cfg.intermediate_dim, lr.down_proj);
    }

    sm.final_norm_weights = gen_norm(cfg.hidden_dim, 10.0f);
    add_f32("model.norm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, sm.final_norm_weights);

    sm.lm_head_weights = gen_matrix(static_cast<size_t>(cfg.vocab_size) * cfg.hidden_dim, 11.0f, 0.015f);
    add_2d("lm_head.weight", cfg.vocab_size, cfg.hidden_dim, sm.lm_head_weights, keep_embeddings_fp32);

    Status bst = ModelLoader::build_ie_image(cfg, descriptors, payloads, &sm.ie_bytes);
    if (!bst.ok()) {
        std::cerr << "Failed to build synthetic .ie image: " << bst.message << "\n";
    }
    return sm;
}

bool test_q4_0_transformer_forward_and_numerical_metrics() {
    std::cout << "[INFO] Running comprehensive Q4_0 Transformer forward pass and numerical validation...\n";

    SyntheticModel sm_fp32 = create_synthetic_model_64(DataType::FP32);
    SyntheticModel sm_q4_pure = create_synthetic_model_64(DataType::Q4_0, false);
    SyntheticModel sm_q4_mixed = create_synthetic_model_64(DataType::Q4_0, true);

    if (sm_fp32.ie_bytes.empty() || sm_q4_pure.ie_bytes.empty() || sm_q4_mixed.ie_bytes.empty()) {
        std::cerr << "Failed to build synthetic models for Q4_0 testing\n";
        return false;
    }

    ModelLoader loader_fp32;
    if (!loader_fp32.load_from_buffer(sm_fp32.ie_bytes.data(), sm_fp32.ie_bytes.size()).ok()) return false;
    ModelLoader loader_q4_pure;
    if (!loader_q4_pure.load_from_buffer(sm_q4_pure.ie_bytes.data(), sm_q4_pure.ie_bytes.size()).ok()) return false;
    ModelLoader loader_q4_mixed;
    if (!loader_q4_mixed.load_from_buffer(sm_q4_mixed.ie_bytes.data(), sm_q4_mixed.ie_bytes.size()).ok()) return false;

    TransformerEngine engine_fp32;
    if (!engine_fp32.init(sm_fp32.config, &loader_fp32, SimdBackend::Scalar, 1).ok()) return false;

    TransformerEngine engine_q4_scalar;
    if (!engine_q4_scalar.init(sm_q4_pure.config, &loader_q4_pure, SimdBackend::Scalar, 1).ok()) return false;

    const SimdBackend best_simd = detect_best_backend();
    TransformerEngine engine_q4_simd;
    if (!engine_q4_simd.init(sm_q4_pure.config, &loader_q4_pure, best_simd, 1).ok()) return false;

    TransformerEngine engine_q4_mt;
    if (!engine_q4_mt.init(sm_q4_pure.config, &loader_q4_pure, best_simd, 4).ok()) return false;

    TransformerEngine engine_q4_mixed;
    if (!engine_q4_mixed.init(sm_q4_mixed.config, &loader_q4_mixed, best_simd, 1).ok()) return false;

    const std::vector<int32_t> prompt_tokens = {5, 11, 22, 33, 44};
    const uint32_t prompt_len = static_cast<uint32_t>(prompt_tokens.size());
    const uint32_t decode_steps = 4;
    const uint32_t vocab_size = sm_fp32.config.vocab_size;

    std::vector<float> logits_fp32(vocab_size, 0.0f);
    std::vector<float> logits_q4_scalar(vocab_size, 0.0f);
    std::vector<float> logits_q4_simd(vocab_size, 0.0f);
    std::vector<float> logits_q4_mt(vocab_size, 0.0f);
    std::vector<float> logits_q4_mixed(vocab_size, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        if (!engine_fp32.forward_token(prompt_tokens[pos], pos, logits_fp32.data()).ok()) return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        if (!engine_q4_scalar.forward_token(prompt_tokens[pos], pos, logits_q4_scalar.data()).ok()) return false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        if (!engine_q4_simd.forward_token(prompt_tokens[pos], pos, logits_q4_simd.data()).ok()) return false;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        if (!engine_q4_mt.forward_token(prompt_tokens[pos], pos, logits_q4_mt.data()).ok()) return false;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    for (uint32_t pos = 0; pos < prompt_len; ++pos) {
        if (!engine_q4_mixed.forward_token(prompt_tokens[pos], pos, logits_q4_mixed.data()).ok()) return false;
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    const double time_fp32_prefill_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double time_q4_scalar_prefill_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    const double time_q4_simd_prefill_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    const double time_q4_mt_prefill_us = std::chrono::duration<double, std::micro>(t4 - t3).count();
    const double time_q4_mixed_prefill_us = std::chrono::duration<double, std::micro>(t5 - t4).count();

    // Verify prompt prefill parity between Scalar and SIMD and MT Q4_0
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (std::fabs(logits_q4_scalar[i] - logits_q4_simd[i]) > 1e-4f) {
            std::cerr << "Prefill Q4_0 Scalar vs SIMD divergence at " << i << "\n";
            return false;
        }
        if (std::fabs(logits_q4_simd[i] - logits_q4_mt[i]) > 1e-4f) {
            std::cerr << "Prefill Q4_0 SIMD vs MultiThread divergence at " << i << "\n";
            return false;
        }
    }

    auto compute_metrics = [&](const std::vector<float>& ref, const std::vector<float>& act) {
        float max_err = 0.0f;
        float sum_err = 0.0f;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < vocab_size; ++i) {
            const float diff = std::fabs(ref[i] - act[i]);
            max_err = std::max(max_err, diff);
            sum_err += diff;
            sum_sq += diff * diff;
        }
        const float mae = sum_err / static_cast<float>(vocab_size);
        const float rmse = std::sqrt(sum_sq / static_cast<float>(vocab_size));
        const int32_t tok_ref = Sampler::sample_greedy(ref.data(), ref.size());
        const int32_t tok_act = Sampler::sample_greedy(act.data(), act.size());
        return std::make_tuple(max_err, mae, rmse, tok_ref == tok_act, tok_ref, tok_act);
    };

    auto [pre_max, pre_mae, pre_rmse, pre_match, pre_tok_fp, pre_tok_q4] =
        compute_metrics(logits_fp32, logits_q4_simd);

    std::cout << "[METRICS] Prompt Prefill (5 tokens) FP32 vs Q4_0 Pure: MaxErr=" << pre_max
              << ", MAE=" << pre_mae << ", RMSE=" << pre_rmse
              << ", GreedyAgreement=" << (pre_match ? "YES" : "NO")
              << " (FP32 tok=" << pre_tok_fp << ", Q4_0 tok=" << pre_tok_q4 << ")\n";

    if (pre_mae > 0.35f || pre_rmse > 0.45f) {
        std::cerr << "Q4_0 prefill quantization error exceeded acceptable bounds\n";
        return false;
    }

    int32_t curr_token_fp32 = pre_tok_fp;
    int32_t curr_token_q4 = pre_tok_q4;

    double decode_time_fp32 = 0.0;
    double decode_time_q4_simd = 0.0;
    double decode_time_q4_mt = 0.0;

    for (uint32_t step = 0; step < decode_steps; ++step) {
        const uint32_t decode_pos = prompt_len + step;

        auto dt0 = std::chrono::high_resolution_clock::now();
        if (!engine_fp32.forward_token(curr_token_fp32, decode_pos, logits_fp32.data()).ok()) return false;
        auto dt1 = std::chrono::high_resolution_clock::now();
        if (!engine_q4_scalar.forward_token(curr_token_q4, decode_pos, logits_q4_scalar.data()).ok()) return false;
        auto dt2 = std::chrono::high_resolution_clock::now();
        if (!engine_q4_simd.forward_token(curr_token_q4, decode_pos, logits_q4_simd.data()).ok()) return false;
        auto dt3 = std::chrono::high_resolution_clock::now();
        if (!engine_q4_mt.forward_token(curr_token_q4, decode_pos, logits_q4_mt.data()).ok()) return false;
        auto dt4 = std::chrono::high_resolution_clock::now();

        decode_time_fp32 += std::chrono::duration<double, std::micro>(dt1 - dt0).count();
        decode_time_q4_simd += std::chrono::duration<double, std::micro>(dt3 - dt2).count();
        decode_time_q4_mt += std::chrono::duration<double, std::micro>(dt4 - dt3).count();

        for (uint32_t i = 0; i < vocab_size; ++i) {
            if (std::fabs(logits_q4_scalar[i] - logits_q4_simd[i]) > 1e-4f) {
                std::cerr << "Decode step " << step << " Q4_0 Scalar vs SIMD divergence\n";
                return false;
            }
            if (std::fabs(logits_q4_simd[i] - logits_q4_mt[i]) > 1e-4f) {
                std::cerr << "Decode step " << step << " Q4_0 SIMD vs MT divergence\n";
                return false;
            }
        }

        auto [d_max, d_mae, d_rmse, d_match, d_tok_fp, d_tok_q4] =
            compute_metrics(logits_fp32, logits_q4_simd);

        std::cout << "[METRICS] Decode Step " << step << " (pos " << decode_pos << "): MaxErr=" << d_max
                  << ", MAE=" << d_mae << ", RMSE=" << d_rmse
                  << ", GreedyAgreement=" << (d_match ? "YES" : "NO")
                  << " (FP32=" << d_tok_fp << ", Q4=" << d_tok_q4 << ")\n";

        curr_token_fp32 = d_tok_fp;
        curr_token_q4 = d_tok_q4;
    }

    std::cout << "[BENCHMARK] Timing (Prefill 5 tokens / Decode 4 tokens):\n"
              << "  - FP32 Scalar:        Prefill=" << time_fp32_prefill_us << " us, Decode=" << decode_time_fp32 << " us\n"
              << "  - Q4_0 Scalar (1T):   Prefill=" << time_q4_scalar_prefill_us << " us\n"
              << "  - Q4_0 SIMD (1T):     Prefill=" << time_q4_simd_prefill_us << " us, Decode=" << decode_time_q4_simd << " us\n"
              << "  - Q4_0 SIMD (4T):     Prefill=" << time_q4_mt_prefill_us << " us, Decode=" << decode_time_q4_mt << " us\n"
              << "  - Q4_0 Mixed (1T):    Prefill=" << time_q4_mixed_prefill_us << " us\n";

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
    bad_dtype_cfg.weight_dtype = DataType::I32;
    TransformerEngine bad_dtype_engine;
    if (bad_dtype_engine.init(bad_dtype_cfg, &loader).ok()) {
        std::cerr << "Expected engine.init to fail when weight_dtype is neither FP32 nor Q4_0\n";
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

        // Verify logits after decode forward pass
        float step_max_diff = 0.0f;
        const float* ref_step_logits = &gen_ref_logits[static_cast<size_t>(step) * vocab_size];
        for (uint32_t i = 0; i < vocab_size; ++i) {
            step_max_diff = std::max(step_max_diff, std::fabs(logits[i] - ref_step_logits[i]));
        }
        if (step_max_diff > 1e-3f) {
            std::cerr << "Golden decode logits exceeded tolerance at step " << step
                      << " (max_diff=" << step_max_diff << ")\n";
            return false;
        }
    }

    std::cout << "[SUCCESS] Golden reference parity verified across prompt and "
              << gen_len << " decode steps!\n";
    return true;
}

bool test_real_tinyllama_multiturn_if_present() {
    const std::string model_path = "models/tinyllama-1.1b-q4_0.ie";
    const std::string tok_path = "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model";

    std::ifstream mf(model_path, std::ios::binary);
    std::ifstream tf(tok_path, std::ios::binary);
    if (!mf || !tf) {
        std::cout << "[INFO] Real TinyLlama models not found; skipping live multi-turn test.\n";
        return true;
    }

    std::cout << "[INFO] Running real TinyLlama multi-turn context retention & KV-cache test...\n";

    ModelLoader loader;
    Status st = loader.load_from_file(model_path);
    if (!st.ok()) {
        std::cerr << "Failed to load " << model_path << ": " << st.message << "\n";
        return false;
    }

    Tokenizer tokenizer;
    st = tokenizer.load(tok_path);
    if (!st.ok()) {
        std::cerr << "Failed to load tokenizer: " << st.message << "\n";
        return false;
    }

    TransformerEngine engine;
    st = engine.init(loader.config(), &loader, detect_best_backend(), 4);
    if (!st.ok()) {
        std::cerr << "Failed to init TransformerEngine: " << st.message << "\n";
        return false;
    }

    const uint32_t vocab_size = loader.config().vocab_size;
    std::vector<float> logits(vocab_size, 0.0f);

    // --- Turn 1: Introduce Name ---
    const std::string prompt1 = "<|user|>\nHello, my name is Alex.</s>\n<|assistant|>\n";
    std::vector<int32_t> p1_tokens;
    st = tokenizer.encode(prompt1, true, false, &p1_tokens);
    if (!st.ok() || p1_tokens.empty()) {
        std::cerr << "Failed to encode prompt1\n";
        return false;
    }

    const uint32_t start_len = engine.kv_cache().seq_len();
    if (start_len != 0) {
        std::cerr << "Initial KV cache sequence length must be 0, got " << start_len << "\n";
        return false;
    }

    // Prefill Turn 1
    st = engine.forward_prompt(p1_tokens.data(), p1_tokens.size(), logits.data());
    if (!st.ok()) {
        std::cerr << "Turn 1 forward_prompt failed: " << st.message << "\n";
        return false;
    }
    if (engine.kv_cache().seq_len() != p1_tokens.size()) {
        std::cerr << "KV cache seq_len after Turn 1 prefill mismatch\n";
        return false;
    }

    // Autoregressive decode Turn 1
    std::vector<int32_t> turn1_gen;
    int32_t next_tok = Sampler::sample_greedy(logits.data(), logits.size());
    while (turn1_gen.size() < 40 && next_tok != kEosTokenId) {
        turn1_gen.push_back(next_tok);
        const uint32_t pos = engine.kv_cache().seq_len();
        st = engine.forward_token(next_tok, pos, logits.data());
        if (!st.ok()) {
            std::cerr << "Turn 1 decode forward failed: " << st.message << "\n";
            return false;
        }
        next_tok = Sampler::sample_greedy(logits.data(), logits.size());
    }
    if (next_tok == kEosTokenId) {
        st = engine.forward_token(kEosTokenId, engine.kv_cache().seq_len(), logits.data());
        if (!st.ok()) return false;
    }

    std::string turn1_text;
    tokenizer.decode(turn1_gen, &turn1_text);
    std::cout << "  [Turn 1 Output] " << turn1_text << "\n";
    const uint32_t turn1_end_pos = engine.kv_cache().seq_len();
    std::cout << "  [Turn 1 KV-Cache End Pos] " << turn1_end_pos << "\n";

    // --- Turn 2: Query Name (KV Cache preserved without clearing) ---
    const std::string prompt2 = "\n<|user|>\nWhat is my name?</s>\n<|assistant|>\n";
    std::vector<int32_t> p2_tokens;
    st = tokenizer.encode(prompt2, false, false, &p2_tokens);
    if (!st.ok() || p2_tokens.empty()) {
        std::cerr << "Failed to encode prompt2\n";
        return false;
    }

    // Prefill Turn 2 (starting at turn1_end_pos in the KV cache)
    st = engine.forward_prompt(p2_tokens.data(), p2_tokens.size(), logits.data());
    if (!st.ok()) {
        std::cerr << "Turn 2 forward_prompt failed: " << st.message << "\n";
        return false;
    }
    if (engine.kv_cache().seq_len() != turn1_end_pos + p2_tokens.size()) {
        std::cerr << "KV cache seq_len after Turn 2 prefill mismatch\n";
        return false;
    }

    // Autoregressive decode Turn 2
    std::vector<int32_t> turn2_gen;
    next_tok = Sampler::sample_greedy(logits.data(), logits.size());
    while (turn2_gen.size() < 30 && next_tok != kEosTokenId) {
        turn2_gen.push_back(next_tok);
        const uint32_t pos = engine.kv_cache().seq_len();
        st = engine.forward_token(next_tok, pos, logits.data());
        if (!st.ok()) {
            std::cerr << "Turn 2 decode forward failed: " << st.message << "\n";
            return false;
        }
        next_tok = Sampler::sample_greedy(logits.data(), logits.size());
    }

    std::string turn2_text;
    tokenizer.decode(turn2_gen, &turn2_text);
    std::cout << "  [Turn 2 Output] " << turn2_text << "\n";
    const uint32_t turn2_end_pos = engine.kv_cache().seq_len();
    std::cout << "  [Turn 2 KV-Cache End Pos] " << turn2_end_pos << "\n";

    // Verify context retention: Turn 2 response must contain "Alex"
    if (turn2_text.find("Alex") == std::string::npos) {
        std::cerr << "Turn 2 failed to retain name 'Alex' from Turn 1: " << turn2_text << "\n";
        return false;
    }

    std::cout << "[SUCCESS] Multi-turn context retention & KV cache continuity verified!\n";
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
    if (!test_q4_0_transformer_forward_and_numerical_metrics()) {
        return false;
    }
    if (!test_offline_golden_fixture_if_present()) {
        return false;
    }
    if (!test_real_tinyllama_multiturn_if_present()) {
        return false;
    }
    return true;
}

}  // namespace inference_engine::testing
