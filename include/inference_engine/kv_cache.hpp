#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"

namespace inference_engine {

// KV Cache Contract:
// 1. `append(layer_idx, pos, k_data, v_data)` writes K/V vectors for a single
//    transformer layer and sequence position `pos`.
// 2. `append()` MUST NOT increment the global committed sequence length (`seq_len()`),
//    because all 22 layers execute for the same token position during a forward step.
// 3. `commit_token()` (or `commit_positions(count)`) explicitly advances `seq_len()`
//    ONLY after all layers have completed processing for the token(s).
class KvCache {
public:
    KvCache() = default;

    Status init(const ModelConfig& config);
    void clear() noexcept;

    bool initialized() const noexcept { return initialized_; }
    uint32_t seq_len() const noexcept { return committed_seq_len_; }
    uint32_t max_seq_len() const noexcept { return max_seq_len_; }
    uint32_t num_layers() const noexcept { return num_layers_; }
    uint32_t kv_dim() const noexcept { return kv_dim_; }

    // Writes K and V vectors of length `kv_dim()` for `layer_idx` at `pos`.
    // Does NOT modify `seq_len()`.
    Status append(uint32_t layer_idx,
                  uint32_t pos,
                  const float* k_data,
                  const float* v_data) noexcept;

    // Advances committed sequence length after all layers finish processing `count` tokens.
    Status commit_positions(uint32_t count = 1) noexcept;
    Status commit_token() noexcept { return commit_positions(1); }

    // Rewinds the committed sequence length to `new_len` (must be <= seq_len()).
    // Entries at positions >= new_len become invalid and are overwritten by later appends.
    // This never advances seq_len(), so it preserves the append/commit contract.
    Status truncate(uint32_t new_len) noexcept;

    const float* key_vector(uint32_t layer_idx, uint32_t pos) const noexcept;
    const float* value_vector(uint32_t layer_idx, uint32_t pos) const noexcept;

private:
    bool initialized_ = false;
    uint32_t num_layers_ = 0;
    uint32_t max_seq_len_ = 0;
    uint32_t kv_dim_ = 0;
    uint32_t committed_seq_len_ = 0;
    std::vector<float> k_storage_{};
    std::vector<float> v_storage_{};

    size_t offset_for(uint32_t layer_idx, uint32_t pos) const noexcept;
};

}  // namespace inference_engine
