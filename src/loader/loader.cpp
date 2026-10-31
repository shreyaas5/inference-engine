#include "inference_engine/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_set>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace inference_engine {

namespace {

uint32_t read_u32_le(const uint8_t* ptr) noexcept {
    return static_cast<uint32_t>(ptr[0]) |
           (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* ptr) noexcept {
    return static_cast<uint64_t>(read_u32_le(ptr)) |
           (static_cast<uint64_t>(read_u32_le(ptr + 4)) << 32);
}

float read_f32_le(const uint8_t* ptr) noexcept {
    const uint32_t bits = read_u32_le(ptr);
    float val = 0.0f;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

void write_u32_le(std::vector<uint8_t>* buf, uint32_t v) {
    buf->push_back(static_cast<uint8_t>(v & 0xFFu));
    buf->push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    buf->push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    buf->push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

void write_u64_le(std::vector<uint8_t>* buf, uint64_t v) {
    write_u32_le(buf, static_cast<uint32_t>(v & 0xFFFFFFFFu));
    write_u32_le(buf, static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFu));
}

void write_f32_le(std::vector<uint8_t>* buf, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(float));
    write_u32_le(buf, bits);
}

struct ByteRange {
    uint64_t begin = 0;
    uint64_t end = 0;
};

}  // namespace

ModelLoader::~ModelLoader() {
    close();
}

void ModelLoader::close() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    if (is_mmap_ && mapped_ptr_ != nullptr && mapped_size_ > 0) {
        munmap(const_cast<uint8_t*>(mapped_ptr_), mapped_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    mapped_ptr_ = nullptr;
    mapped_size_ = 0;
    is_mmap_ = false;
    owned_buffer_.clear();
    tensors_.clear();
    views_.clear();
    loaded_ = false;
}

Status ModelLoader::load_from_file(const std::string& path) {
    close();
#if defined(__unix__) || defined(__APPLE__)
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status::Error(StatusCode::IoError, "Failed to open model file: " + path);
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return Status::Error(StatusCode::IoError, "Failed to stat model file or file is empty");
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        return Status::Error(StatusCode::IoError, "Failed to mmap model file");
    }
    fd_ = fd;
    mapped_ptr_ = static_cast<const uint8_t*>(mapped);
    mapped_size_ = file_size;
    is_mmap_ = true;
    Status st_parse = load_from_buffer(mapped_ptr_, mapped_size_);
    if (!st_parse.ok()) {
        close();
    }
    return st_parse;
#else
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return Status::Error(StatusCode::IoError, "Failed to open model file: " + path);
    }
    const std::streamsize sz = ifs.tellg();
    if (sz <= 0) {
        return Status::Error(StatusCode::IoError, "Model file is empty");
    }
    ifs.seekg(0, std::ios::beg);
    owned_buffer_.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(owned_buffer_.data()), sz)) {
        owned_buffer_.clear();
        return Status::Error(StatusCode::IoError, "Failed to read model file");
    }
    return load_from_buffer(owned_buffer_.data(), owned_buffer_.size());
#endif
}

Status ModelLoader::load_from_buffer(const uint8_t* data, size_t size) {
    loaded_ = false;
    tensors_.clear();
    views_.clear();

    if (data == nullptr || size < kIeHeaderBytes) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Model buffer is smaller than 64-byte header");
    }

    const uint32_t magic = read_u32_le(data + 0);
    if (magic != kIeMagic) {
        return Status::Error(StatusCode::CorruptedFile, "Invalid .ie magic header");
    }
    const uint32_t version = read_u32_le(data + 4);
    if (version != kIeFormatVersion) {
        return Status::Error(StatusCode::UnsupportedFormat, "Unsupported .ie format version");
    }

    const uint32_t tensor_count = read_u32_le(data + 8);
    if (tensor_count > kIeMaxTensorCount) {
        return Status::Error(StatusCode::CorruptedFile, "Tensor count exceeds maximum limit");
    }

    const uint32_t raw_weight_dtype = read_u32_le(data + 12);
    if (!is_valid_dtype(raw_weight_dtype)) {
        return Status::Error(StatusCode::CorruptedFile, "Invalid model weight dtype in header");
    }
    const DataType header_weight_dtype = static_cast<DataType>(raw_weight_dtype);
    if (header_weight_dtype != DataType::FP32 && header_weight_dtype != DataType::Q4_0) {
        return Status::Error(StatusCode::UnsupportedFormat,
                             "Unsupported model weight_dtype in .ie header; expected FP32 or Q4_0");
    }

    ModelConfig cfg{};
    cfg.weight_dtype = header_weight_dtype;
    cfg.num_layers = read_u32_le(data + 16);
    cfg.hidden_dim = read_u32_le(data + 20);
    cfg.intermediate_dim = read_u32_le(data + 24);
    cfg.num_query_heads = read_u32_le(data + 28);
    cfg.num_kv_heads = read_u32_le(data + 32);
    cfg.head_dim = read_u32_le(data + 36);
    cfg.vocab_size = read_u32_le(data + 40);
    cfg.max_seq_len = read_u32_le(data + 44);
    cfg.rms_norm_eps = read_f32_le(data + 48);
    cfg.rope_theta = read_f32_le(data + 52);

    if (!std::isfinite(cfg.rms_norm_eps) || !std::isfinite(cfg.rope_theta)) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Non-finite floating-point hyperparameter in header");
    }

    const uint64_t total_file_bytes = read_u64_le(data + 56);
    if (total_file_bytes != static_cast<uint64_t>(size)) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Header total_file_bytes does not match actual buffer size");
    }

    Status cfg_status = cfg.validate();
    if (!cfg_status.ok()) {
        return cfg_status;
    }

    uint64_t cursor = kIeHeaderBytes;
    std::vector<TensorDesc> parsed_tensors;
    parsed_tensors.reserve(tensor_count);
    std::unordered_map<std::string, TensorView> parsed_views;
    std::vector<ByteRange> payload_ranges;
    payload_ranges.reserve(tensor_count);

    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint64_t after_name_len = 0;
        if (!checked_add_u64(cursor, 4, &after_name_len) || after_name_len > size) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Truncated tensor directory entry (name_len)");
        }
        const uint32_t name_len = read_u32_le(data + cursor);
        if (name_len == 0 || name_len > kIeMaxTensorNameLen) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Invalid tensor name length in directory");
        }
        cursor = after_name_len;

        // Directory record after name: dtype(4) + ndim(4) + shape(4*8=32) + offset(8) + nbytes(8) = 56 bytes
        uint64_t entry_end = 0;
        if (!checked_add_u64(cursor, static_cast<uint64_t>(name_len) + 56u, &entry_end) ||
            entry_end > size) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Truncated tensor directory entry metadata");
        }

        TensorDesc desc{};
        desc.name.assign(reinterpret_cast<const char*>(data + cursor), name_len);
        cursor += name_len;

        const uint32_t raw_dtype = read_u32_le(data + cursor);
        if (!is_valid_dtype(raw_dtype)) {
            return Status::Error(StatusCode::CorruptedFile, "Invalid tensor dtype in directory");
        }
        desc.dtype = static_cast<DataType>(raw_dtype);
        if (desc.dtype != DataType::FP32 && desc.dtype != DataType::Q4_0) {
            return Status::Error(StatusCode::UnsupportedFormat,
                                 "Unsupported tensor dtype in directory for " + desc.name);
        }
        desc.ndim = read_u32_le(data + cursor + 4);
        cursor += 8;

        for (size_t d = 0; d < kMaxTensorDims; ++d) {
            desc.shape[d] = read_u64_le(data + cursor + d * 8);
        }
        cursor += kMaxTensorDims * 8;

        desc.payload_offset = read_u64_le(data + cursor);
        desc.payload_nbytes = read_u64_le(data + cursor + 8);
        cursor += 16;

        Status desc_status = desc.validate();
        if (!desc_status.ok()) {
            return desc_status;
        }

        if (parsed_views.find(desc.name) != parsed_views.end()) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Duplicate tensor name in directory: " + desc.name);
        }

        uint64_t payload_end = 0;
        if (!checked_add_u64(desc.payload_offset, desc.payload_nbytes, &payload_end) ||
            payload_end > size) {
            return Status::Error(StatusCode::OutOfBounds,
                                 "Tensor payload exceeds file bounds: " + desc.name);
        }

        payload_ranges.push_back(ByteRange{desc.payload_offset, payload_end});
        parsed_views.emplace(desc.name, TensorView(desc, data + desc.payload_offset));
        parsed_tensors.push_back(std::move(desc));
    }

    uint64_t min_payload_offset = 0;
    if (!align_up_u64(cursor, kIePayloadAlignment, &min_payload_offset)) {
        return Status::Error(StatusCode::CorruptedFile, "Directory end alignment overflow");
    }

    // Verify no payload overlaps the header/directory region or another tensor payload.
    std::sort(payload_ranges.begin(), payload_ranges.end(),
              [](const ByteRange& a, const ByteRange& b) { return a.begin < b.begin; });

    for (size_t i = 0; i < payload_ranges.size(); ++i) {
        if (payload_ranges[i].begin < min_payload_offset) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Tensor payload overlaps header or directory region");
        }
        if (i > 0 && payload_ranges[i].begin < payload_ranges[i - 1].end) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Overlapping tensor payload regions detected");
        }
    }

    if (payload_ranges.empty()) {
        if (total_file_bytes != cursor) {
            return Status::Error(StatusCode::CorruptedFile,
                                 "Empty-tensor .ie image has unexpected trailing bytes");
        }
    } else if (payload_ranges.back().end != total_file_bytes) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Final tensor payload end does not match total_file_bytes");
    }

    config_ = cfg;
    tensors_ = std::move(parsed_tensors);
    views_ = std::move(parsed_views);
    loaded_ = true;
    return Status::Ok();
}

const TensorView* ModelLoader::find_tensor(const std::string& name) const noexcept {
    const auto it = views_.find(name);
    if (it == views_.end()) {
        return nullptr;
    }
    return &it->second;
}

Status ModelLoader::validate_tensor_binding(
    const std::string& name,
    uint32_t expected_ndim,
    const std::array<uint64_t, kMaxTensorDims>& expected_shape,
    DataType expected_dtype,
    TensorView* out_view) const noexcept {
    if (!loaded_) {
        return Status::Error(StatusCode::InvalidArgument, "ModelLoader is not loaded");
    }
    const TensorView* view = find_tensor(name);
    if (view == nullptr) {
        return Status::Error(StatusCode::CorruptedFile, "Missing required tensor: " + name);
    }
    const TensorDesc& d = view->desc();
    if (d.ndim != expected_ndim) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor ndim mismatch for " + name);
    }
    if (d.shape != expected_shape) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor shape mismatch for " + name);
    }
    if (d.dtype != expected_dtype) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Tensor dtype mismatch for " + name);
    }
    if (view->raw_data() == nullptr || d.payload_offset % kIePayloadAlignment != 0) {
        return Status::Error(StatusCode::CorruptedFile,
                             "Invalid or unaligned tensor data pointer for " + name);
    }
    if (out_view != nullptr) {
        *out_view = *view;
    }
    return Status::Ok();
}

Status ModelLoader::bind_transformer_weights(TransformerWeightBindings* out_bindings) const {
    if (!loaded_) {
        return Status::Error(StatusCode::InvalidArgument, "ModelLoader is not loaded");
    }
    if (out_bindings == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_bindings must not be null");
    }

    const size_t expected_tensor_count =
        3u + 9u * static_cast<size_t>(config_.num_layers);
    if (tensors_.size() != expected_tensor_count) {
        return Status::Error(
            StatusCode::CorruptedFile,
            "Tensor count does not match expected Llama inventory (3 + 9 * num_layers)");
    }

    const uint64_t vocab = config_.vocab_size;
    const uint64_t hidden = config_.hidden_dim;
    const uint64_t inter = config_.intermediate_dim;
    const uint64_t q_dim = config_.q_dim();
    const uint64_t kv_dim = config_.kv_dim();
    const DataType w_dtype = config_.weight_dtype;

    TransformerWeightBindings bound{};
    bound.layers.resize(config_.num_layers);

    // For Q4_0 models, embed_tokens and lm_head are Q4_0 by default, or FP32 if exported
    // with --keep-embeddings-fp32. For FP32 models, all tensors must be FP32.
    auto validate_embedding_or_head = [&](const std::string& name,
                                          const std::array<uint64_t, kMaxTensorDims>& shape,
                                          TensorView* out_view) -> Status {
        const TensorView* view = find_tensor(name);
        if (view == nullptr) {
            return Status::Error(StatusCode::CorruptedFile, "Missing required tensor: " + name);
        }
        DataType expected_dt = w_dtype;
        if (w_dtype == DataType::Q4_0 && view->dtype() == DataType::FP32) {
            expected_dt = DataType::FP32;
        }
        return validate_tensor_binding(name, 2, shape, expected_dt, out_view);
    };

    Status st = validate_embedding_or_head(
        "model.embed_tokens.weight", {vocab, hidden, 0, 0}, &bound.embed_tokens);
    if (!st.ok()) {
        return st;
    }

    for (uint32_t l = 0; l < config_.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        LayerWeightBindings& lb = bound.layers[l];

        // 1D RMSNorm weights are always FP32 even when 2D matrices use Q4_0.
        st = validate_tensor_binding(prefix + "input_layernorm.weight",
                                     1,
                                     {hidden, 0, 0, 0},
                                     DataType::FP32,
                                     &lb.input_layernorm);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "self_attn.q_proj.weight",
                                     2,
                                     {q_dim, hidden, 0, 0},
                                     w_dtype,
                                     &lb.q_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "self_attn.k_proj.weight",
                                     2,
                                     {kv_dim, hidden, 0, 0},
                                     w_dtype,
                                     &lb.k_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "self_attn.v_proj.weight",
                                     2,
                                     {kv_dim, hidden, 0, 0},
                                     w_dtype,
                                     &lb.v_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "self_attn.o_proj.weight",
                                     2,
                                     {hidden, q_dim, 0, 0},
                                     w_dtype,
                                     &lb.o_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "post_attention_layernorm.weight",
                                     1,
                                     {hidden, 0, 0, 0},
                                     DataType::FP32,
                                     &lb.post_attention_layernorm);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "mlp.gate_proj.weight",
                                     2,
                                     {inter, hidden, 0, 0},
                                     w_dtype,
                                     &lb.gate_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "mlp.up_proj.weight",
                                     2,
                                     {inter, hidden, 0, 0},
                                     w_dtype,
                                     &lb.up_proj);
        if (!st.ok()) return st;

        st = validate_tensor_binding(prefix + "mlp.down_proj.weight",
                                     2,
                                     {hidden, inter, 0, 0},
                                     w_dtype,
                                     &lb.down_proj);
        if (!st.ok()) return st;
    }

    st = validate_tensor_binding("model.norm.weight",
                                 1,
                                 {hidden, 0, 0, 0},
                                 DataType::FP32,
                                 &bound.final_norm);
    if (!st.ok()) {
        return st;
    }

    st = validate_embedding_or_head("lm_head.weight", {vocab, hidden, 0, 0}, &bound.lm_head);
    if (!st.ok()) {
        return st;
    }

    *out_bindings = std::move(bound);
    return Status::Ok();
}

Status ModelLoader::build_ie_image(const ModelConfig& config,
                                   const std::vector<TensorDesc>& descriptors,
                                   const std::vector<std::vector<uint8_t>>& payloads,
                                   std::vector<uint8_t>* out_bytes) {
    if (out_bytes == nullptr) {
        return Status::Error(StatusCode::InvalidArgument, "out_bytes must not be null");
    }
    if (descriptors.size() != payloads.size()) {
        return Status::Error(StatusCode::InvalidArgument,
                             "descriptors and payloads count mismatch");
    }
    if (descriptors.size() > kIeMaxTensorCount) {
        return Status::Error(StatusCode::InvalidArgument, "Too many tensors for .ie format");
    }
    Status cfg_st = config.validate();
    if (!cfg_st.ok()) {
        return cfg_st;
    }

    uint64_t dir_bytes = 0;
    std::unordered_set<std::string> seen_names;
    for (const auto& d : descriptors) {
        if (d.name.empty() || d.name.size() > kIeMaxTensorNameLen) {
            return Status::Error(StatusCode::InvalidArgument, "Invalid tensor name length");
        }
        if (!seen_names.insert(d.name).second) {
            return Status::Error(StatusCode::InvalidArgument,
                                 "Duplicate tensor name in build_ie_image: " + d.name);
        }
        const uint64_t entry_len = 4u + static_cast<uint64_t>(d.name.size()) + 56u;
        if (!checked_add_u64(dir_bytes, entry_len, &dir_bytes)) {
            return Status::Error(StatusCode::InvalidArgument, "Directory byte size overflow");
        }
    }

    uint64_t header_and_dir = 0;
    if (!checked_add_u64(kIeHeaderBytes, dir_bytes, &header_and_dir)) {
        return Status::Error(StatusCode::InvalidArgument, "Header + directory size overflow");
    }

    uint64_t payload_cursor = 0;
    if (descriptors.empty()) {
        payload_cursor = header_and_dir;
    } else if (!align_up_u64(header_and_dir, kIePayloadAlignment, &payload_cursor)) {
        return Status::Error(StatusCode::InvalidArgument, "Offset alignment overflow");
    }

    std::vector<TensorDesc> finalized_descs = descriptors;
    for (size_t i = 0; i < finalized_descs.size(); ++i) {
        if (!align_up_u64(payload_cursor, kIePayloadAlignment, &payload_cursor)) {
            return Status::Error(StatusCode::InvalidArgument, "Payload offset overflow");
        }
        finalized_descs[i].payload_offset = payload_cursor;
        finalized_descs[i].payload_nbytes = payloads[i].size();
        Status st = finalized_descs[i].validate();
        if (!st.ok()) {
            return st;
        }
        if (!checked_add_u64(payload_cursor, payloads[i].size(), &payload_cursor)) {
            return Status::Error(StatusCode::InvalidArgument, "File size overflow");
        }
    }

    const uint64_t total_size = payload_cursor;
    out_bytes->clear();
    out_bytes->reserve(static_cast<size_t>(total_size));

    // Write 64-byte header.
    write_u32_le(out_bytes, kIeMagic);
    write_u32_le(out_bytes, kIeFormatVersion);
    write_u32_le(out_bytes, static_cast<uint32_t>(finalized_descs.size()));
    write_u32_le(out_bytes, static_cast<uint32_t>(config.weight_dtype));
    write_u32_le(out_bytes, config.num_layers);
    write_u32_le(out_bytes, config.hidden_dim);
    write_u32_le(out_bytes, config.intermediate_dim);
    write_u32_le(out_bytes, config.num_query_heads);
    write_u32_le(out_bytes, config.num_kv_heads);
    write_u32_le(out_bytes, config.head_dim);
    write_u32_le(out_bytes, config.vocab_size);
    write_u32_le(out_bytes, config.max_seq_len);
    write_f32_le(out_bytes, config.rms_norm_eps);
    write_f32_le(out_bytes, config.rope_theta);
    write_u64_le(out_bytes, total_size);

    // Write directory entries.
    for (const auto& d : finalized_descs) {
        write_u32_le(out_bytes, static_cast<uint32_t>(d.name.size()));
        out_bytes->insert(out_bytes->end(), d.name.begin(), d.name.end());
        write_u32_le(out_bytes, static_cast<uint32_t>(d.dtype));
        write_u32_le(out_bytes, d.ndim);
        for (size_t dim_idx = 0; dim_idx < kMaxTensorDims; ++dim_idx) {
            write_u64_le(out_bytes, d.shape[dim_idx]);
        }
        write_u64_le(out_bytes, d.payload_offset);
        write_u64_le(out_bytes, d.payload_nbytes);
    }

    // Write aligned payloads.
    for (size_t i = 0; i < finalized_descs.size(); ++i) {
        while (out_bytes->size() < finalized_descs[i].payload_offset) {
            out_bytes->push_back(0);
        }
        out_bytes->insert(out_bytes->end(), payloads[i].begin(), payloads[i].end());
    }

    return Status::Ok();
}

}  // namespace inference_engine
