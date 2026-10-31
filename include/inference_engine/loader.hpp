#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference_engine/common.hpp"
#include "inference_engine/config.hpp"
#include "inference_engine/tensor.hpp"

namespace inference_engine {

// Custom `.ie` binary format v1 constants:
// - Little-endian byte order
// - Magic: 0x464D4549 ("IEMF" in ASCII bytes: 'I', 'E', 'M', 'F')
// - Version: 1
// - Header size: 64 bytes (fixed binary layout, serialized field-by-field in little-endian)
// - Payload alignment: 64 bytes
inline constexpr uint32_t kIeMagic = 0x464D4549u;
inline constexpr uint32_t kIeFormatVersion = 1u;
inline constexpr uint64_t kIeHeaderBytes = 64u;
inline constexpr uint64_t kIePayloadAlignment = 64u;
inline constexpr uint32_t kIeMaxTensorNameLen = 256u;
inline constexpr uint32_t kIeMaxTensorCount = 4096u;

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    Status load_from_file(const std::string& path);
    Status load_from_buffer(const uint8_t* data, size_t size);
    void close() noexcept;

    bool is_loaded() const noexcept { return loaded_; }
    const ModelConfig& config() const noexcept { return config_; }
    size_t num_tensors() const noexcept { return tensors_.size(); }
    const std::vector<TensorDesc>& tensor_descriptors() const noexcept { return tensors_; }

    const TensorView* find_tensor(const std::string& name) const noexcept;

    // Serializes a synthetic or converted `.ie` image into `out_bytes` for testing and tools.
    static Status build_ie_image(const ModelConfig& config,
                                 const std::vector<TensorDesc>& descriptors,
                                 const std::vector<std::vector<uint8_t>>& payloads,
                                 std::vector<uint8_t>* out_bytes);

private:
    bool loaded_ = false;
    ModelConfig config_{};
    std::vector<TensorDesc> tensors_{};
    std::unordered_map<std::string, TensorView> views_{};

    // Backing storage: either mmap'd region or owned buffer.
    const uint8_t* mapped_ptr_ = nullptr;
    size_t mapped_size_ = 0;
    int fd_ = -1;
    bool is_mmap_ = false;
    std::vector<uint8_t> owned_buffer_{};
};

}  // namespace inference_engine
