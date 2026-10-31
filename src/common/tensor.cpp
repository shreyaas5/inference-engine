#include "inference_engine/tensor.hpp"

namespace inference_engine {

bool TensorDesc::compute_num_elements(uint64_t* out_elements) const noexcept {
    if (out_elements == nullptr || ndim == 0 || ndim > kMaxTensorDims) {
        return false;
    }
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            return false;
        }
        if (!checked_mul_u64(total, shape[i], &total)) {
            return false;
        }
    }
    *out_elements = total;
    return true;
}

bool TensorDesc::compute_expected_bytes(uint64_t* out_bytes) const noexcept {
    if (out_bytes == nullptr) {
        return false;
    }
    uint64_t num_elems = 0;
    if (!compute_num_elements(&num_elems)) {
        return false;
    }
    const size_t block_elems = dtype_block_elements(dtype);
    const size_t block_bytes = dtype_block_bytes(dtype);
    if (block_elems == 0 || block_bytes == 0) {
        return false;
    }
    if (num_elems % block_elems != 0) {
        return false;
    }
    if (block_elems > 1 && (shape[ndim - 1] % block_elems != 0)) {
        return false;
    }
    const uint64_t num_blocks = num_elems / block_elems;
    return checked_mul_u64(num_blocks, static_cast<uint64_t>(block_bytes), out_bytes);
}

Status TensorDesc::validate() const noexcept {
    if (name.empty()) {
        return Status::Error(StatusCode::CorruptedFile, "Tensor name must not be empty");
    }
    if (ndim == 0 || ndim > kMaxTensorDims) {
        return Status::Error(StatusCode::CorruptedFile, "Tensor ndim out of valid range [1, 4]");
    }
    for (uint32_t i = ndim; i < kMaxTensorDims; ++i) {
        if (shape[i] != 0) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Unused tensor shape dimensions must be zero");
        }
    }
    if (payload_offset % kTensorPayloadAlignment != 0) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor payload_offset is not 64-byte aligned");
    }
    uint64_t expected_bytes = 0;
    if (!compute_expected_bytes(&expected_bytes)) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor shape overflows or is incompatible with dtype block size");
    }
    if (payload_nbytes != expected_bytes) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor payload_nbytes does not match exact expected byte size");
    }
    return Status::Ok();
}

}  // namespace inference_engine
