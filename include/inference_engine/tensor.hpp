#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/quant.hpp"

namespace inference_engine {

inline constexpr size_t kMaxTensorDims = 4;
inline constexpr uint64_t kTensorPayloadAlignment = 64;

struct TensorDesc {
    std::string name{};
    DataType dtype = DataType::FP32;
    uint32_t ndim = 0;
    std::array<uint64_t, kMaxTensorDims> shape{0, 0, 0, 0};
    uint64_t payload_offset = 0;
    uint64_t payload_nbytes = 0;

    bool compute_num_elements(uint64_t* out_elements) const noexcept;
    bool compute_expected_bytes(uint64_t* out_bytes) const noexcept;
    Status validate() const noexcept;
};

// Non-owning view into a validated tensor payload managed by ModelLoader.
// Lifetime contract: pointers returned by `raw_data()`, `data_f32()`, and `data_q4_0()`
// remain valid as long as the originating `ModelLoader` remains open/loaded.
class TensorView {
public:
    TensorView() = default;
    TensorView(TensorDesc desc, const uint8_t* data) noexcept
        : desc_(std::move(desc)), data_(data) {}

    const TensorDesc& desc() const noexcept { return desc_; }
    const std::string& name() const noexcept { return desc_.name; }
    DataType dtype() const noexcept { return desc_.dtype; }
    uint32_t ndim() const noexcept { return desc_.ndim; }
    const std::array<uint64_t, kMaxTensorDims>& shape() const noexcept { return desc_.shape; }
    uint64_t payload_offset() const noexcept { return desc_.payload_offset; }
    uint64_t payload_nbytes() const noexcept { return desc_.payload_nbytes; }

    uint64_t num_elements() const noexcept {
        uint64_t elems = 0;
        return desc_.compute_num_elements(&elems) ? elems : 0;
    }

    uint64_t num_blocks() const noexcept {
        uint64_t elems = 0;
        if (!desc_.compute_num_elements(&elems)) {
            return 0;
        }
        const size_t block_elems = dtype_block_elements(desc_.dtype);
        return (block_elems > 0 && (elems % block_elems == 0)) ? (elems / block_elems) : 0;
    }

    const uint8_t* raw_data() const noexcept { return data_; }
    bool empty() const noexcept { return data_ == nullptr || desc_.payload_nbytes == 0; }

    const float* data_f32() const noexcept {
        return desc_.dtype == DataType::FP32 ? reinterpret_cast<const float*>(data_) : nullptr;
    }

    const BlockQ4_0* data_q4_0() const noexcept {
        return desc_.dtype == DataType::Q4_0 ? reinterpret_cast<const BlockQ4_0*>(data_) : nullptr;
    }

private:
    TensorDesc desc_{};
    const uint8_t* data_ = nullptr;
};

}  // namespace inference_engine
