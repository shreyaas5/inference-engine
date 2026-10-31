#include "inference_engine/kv_cache.hpp"

#include <algorithm>
#include <cstring>

namespace inference_engine {

Status KvCache::init(const ModelConfig& config) {
    Status st = config.validate();
    if (!st.ok()) {
        return st;
    }
    num_layers_ = config.num_layers;
    max_seq_len_ = config.max_seq_len;
    kv_dim_ = config.kv_dim();
    committed_seq_len_ = 0;

    const size_t total_floats =
        static_cast<size_t>(num_layers_) * static_cast<size_t>(max_seq_len_) * static_cast<size_t>(kv_dim_);
    k_storage_.assign(total_floats, 0.0f);
    v_storage_.assign(total_floats, 0.0f);
    initialized_ = true;
    return Status::Ok();
}

void KvCache::clear() noexcept {
    committed_seq_len_ = 0;
    std::fill(k_storage_.begin(), k_storage_.end(), 0.0f);
    std::fill(v_storage_.begin(), v_storage_.end(), 0.0f);
}

size_t KvCache::offset_for(uint32_t layer_idx, uint32_t pos) const noexcept {
    return (static_cast<size_t>(layer_idx) * static_cast<size_t>(max_seq_len_) +
            static_cast<size_t>(pos)) *
           static_cast<size_t>(kv_dim_);
}

Status KvCache::append(uint32_t layer_idx,
                       uint32_t pos,
                       const float* k_data,
                       const float* v_data) noexcept {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "KvCache is not initialized");
    }
    if (k_data == nullptr || v_data == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "k_data and v_data must not be null");
    }
    if (layer_idx >= num_layers_) {
        return Status::Error(StatusCode::OutOfBounds, "layer_idx exceeds num_layers");
    }
    if (pos >= max_seq_len_) {
        return Status::Error(StatusCode::OutOfBounds, "pos exceeds max_seq_len");
    }

    const size_t offset = offset_for(layer_idx, pos);
    const size_t nbytes = static_cast<size_t>(kv_dim_) * sizeof(float);
    std::memcpy(k_storage_.data() + offset, k_data, nbytes);
    std::memcpy(v_storage_.data() + offset, v_data, nbytes);

    // IMPORTANT CONTRACT: append() must NOT advance `committed_seq_len_`.
    return Status::Ok();
}

Status KvCache::commit_positions(uint32_t count) noexcept {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "KvCache is not initialized");
    }
    if (committed_seq_len_ + count > max_seq_len_) {
        return Status::Error(StatusCode::OutOfBounds,
                             "Committing positions would exceed max_seq_len");
    }
    committed_seq_len_ += count;
    return Status::Ok();
}

const float* KvCache::key_vector(uint32_t layer_idx, uint32_t pos) const noexcept {
    if (!initialized_ || layer_idx >= num_layers_ || pos >= max_seq_len_) {
        return nullptr;
    }
    return k_storage_.data() + offset_for(layer_idx, pos);
}

const float* KvCache::value_vector(uint32_t layer_idx, uint32_t pos) const noexcept {
    if (!initialized_ || layer_idx >= num_layers_ || pos >= max_seq_len_) {
        return nullptr;
    }
    return v_storage_.data() + offset_for(layer_idx, pos);
}

}  // namespace inference_engine
