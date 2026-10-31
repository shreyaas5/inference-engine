#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "inference_engine/common.hpp"

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

class TensorView {
public:
    TensorView() = default;
    TensorView(TensorDesc desc, const uint8_t* data) noexcept
        : desc_(std::move(desc)), data_(data) {}

    const TensorDesc& desc() const noexcept { return desc_; }
    const uint8_t* raw_data() const noexcept { return data_; }
    bool empty() const noexcept { return data_ == nullptr || desc_.payload_nbytes == 0; }

    const float* data_f32() const noexcept {
        return desc_.dtype == DataType::FP32 ? reinterpret_cast<const float*>(data_) : nullptr;
    }

private:
    TensorDesc desc_{};
    const uint8_t* data_ = nullptr;
};

}  // namespace inference_engine
