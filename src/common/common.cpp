#include "inference_engine/common.hpp"

#include "inference_engine/quant.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

namespace inference_engine {

const char* status_code_name(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::Ok:
            return "Ok";
        case StatusCode::InvalidArgument:
            return "InvalidArgument";
        case StatusCode::CorruptedFile:
            return "CorruptedFile";
        case StatusCode::OutOfBounds:
            return "OutOfBounds";
        case StatusCode::UnsupportedFormat:
            return "UnsupportedFormat";
        case StatusCode::IoError:
            return "IoError";
        case StatusCode::NotImplemented:
            return "NotImplemented";
        case StatusCode::InternalError:
            return "InternalError";
    }
    return "Unknown";
}

const char* dtype_name(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FP32:
            return "FP32";
        case DataType::FP16:
            return "FP16";
        case DataType::Q4_0:
            return "Q4_0";
        case DataType::I32:
            return "I32";
    }
    return "Unknown";
}

size_t dtype_block_elements(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FP32:
            return 1;
        case DataType::FP16:
            return 1;
        case DataType::Q4_0:
            return kQ4_0BlockElements;
        case DataType::I32:
            return 1;
    }
    return 0;
}

size_t dtype_block_bytes(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
            return 2;
        case DataType::Q4_0:
            return kQ4_0BlockBytes;
        case DataType::I32:
            return 4;
    }
    return 0;
}

bool is_valid_dtype(uint32_t raw_dtype) noexcept {
    return raw_dtype <= static_cast<uint32_t>(DataType::I32);
}

const char* backend_name(SimdBackend backend) noexcept {
    switch (backend) {
        case SimdBackend::Scalar:
            return "Scalar";
        case SimdBackend::Neon:
            return "NEON";
        case SimdBackend::Avx2:
            return "AVX2";
    }
    return "Unknown";
}

bool is_backend_available(SimdBackend backend) noexcept {
    if (backend == SimdBackend::Scalar) {
        return true;
    }
    if (backend == SimdBackend::Neon) {
#if defined(__aarch64__) || defined(_M_ARM64)
        return true;
#else
        return false;
#endif
    }
    if (backend == SimdBackend::Avx2) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
        int info[4] = {0, 0, 0, 0};
        __cpuid(info, 0);
        if (info[0] >= 7) {
            int ext[4] = {0, 0, 0, 0};
            __cpuidex(ext, 7, 0);
            const bool has_avx2 = (ext[1] & (1 << 5)) != 0;
            return has_avx2;
        }
        return false;
#else
        return false;
#endif
#else
        return false;
#endif
    }
    return false;
}

SimdBackend detect_best_backend() noexcept {
    if (is_backend_available(SimdBackend::Neon)) {
        return SimdBackend::Neon;
    }
    if (is_backend_available(SimdBackend::Avx2)) {
        return SimdBackend::Avx2;
    }
    return SimdBackend::Scalar;
}

}  // namespace inference_engine
