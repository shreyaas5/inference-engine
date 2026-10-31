#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace inference_engine {

enum class StatusCode : uint32_t {
    Ok = 0,
    InvalidArgument = 1,
    CorruptedFile = 2,
    OutOfBounds = 3,
    UnsupportedFormat = 4,
    IoError = 5,
    NotImplemented = 6,
    InternalError = 7,
};

struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message{};

    constexpr bool ok() const noexcept { return code == StatusCode::Ok; }

    static Status Ok() { return Status{StatusCode::Ok, {}}; }
    static Status Error(StatusCode c, std::string msg) {
        return Status{c, std::move(msg)};
    }
};

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
    I32 = 3,
};

enum class SimdBackend : uint32_t {
    Scalar = 0,
    Neon = 1,
    Avx2 = 2,
};

const char* status_code_name(StatusCode code) noexcept;
const char* dtype_name(DataType dtype) noexcept;
size_t dtype_block_elements(DataType dtype) noexcept;
size_t dtype_block_bytes(DataType dtype) noexcept;
bool is_valid_dtype(uint32_t raw_dtype) noexcept;

const char* backend_name(SimdBackend backend) noexcept;
bool is_backend_available(SimdBackend backend) noexcept;
SimdBackend detect_best_backend() noexcept;

inline constexpr bool checked_add_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    if (out == nullptr) {
        return false;
    }
    if (std::numeric_limits<uint64_t>::max() - a < b) {
        return false;
    }
    *out = a + b;
    return true;
}

inline constexpr bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (std::numeric_limits<uint64_t>::max() / a < b) {
        return false;
    }
    *out = a * b;
    return true;
}

inline constexpr bool align_up_u64(uint64_t value, uint64_t alignment, uint64_t* out) noexcept {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return false;
    }
    const uint64_t mask = alignment - 1;
    uint64_t padded = 0;
    if (!checked_add_u64(value, mask, &padded)) {
        return false;
    }
    *out = padded & ~mask;
    return true;
}

}  // namespace inference_engine
