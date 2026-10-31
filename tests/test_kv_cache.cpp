#include <iostream>
#include <vector>

#include "inference_engine/kv_cache.hpp"

namespace inference_engine::testing {

bool run_kv_cache_tests() {
    ModelConfig cfg{};
    cfg.num_layers = 4;
    cfg.hidden_dim = 64;
    cfg.intermediate_dim = 128;
    cfg.num_query_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.vocab_size = 128;
    cfg.max_seq_len = 16;

    KvCache cache;
    if (!cache.init(cfg).ok()) {
        std::cerr << "KvCache::init failed\n";
        return false;
    }

    const std::vector<float> k_vec(cfg.kv_dim(), 1.25f);
    const std::vector<float> v_vec(cfg.kv_dim(), -0.75f);

    // Contract verification: calling append() across all layers for position 0
    // MUST NOT increment seq_len() until commit_token() is explicitly called.
    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        if (!cache.append(layer, 0, k_vec.data(), v_vec.data()).ok()) {
            std::cerr << "KvCache::append failed at layer " << layer << "\n";
            return false;
        }
        if (cache.seq_len() != 0) {
            std::cerr << "Contract violation: KvCache::append advanced seq_len() prematurely\n";
            return false;
        }
    }

    if (!cache.commit_token().ok() || cache.seq_len() != 1) {
        std::cerr << "KvCache::commit_token failed to advance seq_len() to 1\n";
        return false;
    }

    const float* stored_k = cache.key_vector(3, 0);
    const float* stored_v = cache.value_vector(3, 0);
    if (stored_k == nullptr || stored_v == nullptr ||
        stored_k[0] != 1.25f || stored_v[0] != -0.75f) {
        std::cerr << "Stored K/V values do not match appended vectors\n";
        return false;
    }

    // Out-of-bounds checks.
    if (cache.append(cfg.num_layers, 0, k_vec.data(), v_vec.data()).ok()) {
        std::cerr << "Expected out-of-bounds layer_idx to fail\n";
        return false;
    }
    if (cache.append(0, cfg.max_seq_len, k_vec.data(), v_vec.data()).ok()) {
        std::cerr << "Expected out-of-bounds pos to fail\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
