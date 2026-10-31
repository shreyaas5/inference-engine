#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "inference_engine/loader.hpp"

namespace inference_engine::testing {

namespace {

void write_u32_le_at(uint8_t* ptr, uint32_t v) noexcept {
    ptr[0] = static_cast<uint8_t>(v & 0xFFu);
    ptr[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    ptr[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    ptr[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

void write_u64_le_at(uint8_t* ptr, uint64_t v) noexcept {
    write_u32_le_at(ptr, static_cast<uint32_t>(v & 0xFFFFFFFFu));
    write_u32_le_at(ptr + 4, static_cast<uint32_t>((v >> 32) & 0xFFFFFFFFu));
}

std::vector<uint8_t> pack_f32_Pattern(size_t count, float base, float step) {
    std::vector<uint8_t> bytes(count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        const float val = base + static_cast<float>(i % 251) * step;
        std::memcpy(bytes.data() + i * sizeof(float), &val, sizeof(float));
    }
    return bytes;
}

bool verify_f32_pattern(const TensorView& view, size_t count, float base, float step) {
    const float* ptr = view.data_f32();
    if (ptr == nullptr) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const float expected = base + static_cast<float>(i % 251) * step;
        uint32_t actual_bits = 0;
        uint32_t expected_bits = 0;
        std::memcpy(&actual_bits, ptr + i, sizeof(uint32_t));
        std::memcpy(&expected_bits, &expected, sizeof(uint32_t));
        if (actual_bits != expected_bits) {
            return false;
        }
    }
    return true;
}

bool test_malformed_file_rejection() {
    ModelConfig cfg{};
    cfg.num_layers = 2;
    cfg.hidden_dim = 64;
    cfg.intermediate_dim = 128;
    cfg.num_query_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.vocab_size = 128;
    cfg.max_seq_len = 32;

    TensorDesc t0{};
    t0.name = "model.norm.weight";
    t0.dtype = DataType::FP32;
    t0.ndim = 1;
    t0.shape = {64, 0, 0, 0};

    TensorDesc t1{};
    t1.name = "model.layers.0.self_attn.q_proj.weight";
    t1.dtype = DataType::Q4_0;
    t1.ndim = 2;
    t1.shape = {64, 64, 0, 0};

    std::vector<uint8_t> p0(64 * sizeof(float), 0x11);
    std::vector<uint8_t> p1((64 * 64 / 32) * 18, 0x22);

    std::vector<uint8_t> valid_image;
    Status build_st = ModelLoader::build_ie_image(cfg, {t0, t1}, {p0, p1}, &valid_image);
    if (!build_st.ok()) {
        std::cerr << "Failed to build valid .ie image: " << build_st.message << "\n";
        return false;
    }

    ModelLoader loader;
    if (!loader.load_from_buffer(valid_image.data(), valid_image.size()).ok()) {
        std::cerr << "Failed to load valid baseline .ie image\n";
        return false;
    }

    // 1. Null or undersized header (< 64 bytes).
    ModelLoader bad_loader;
    if (bad_loader.load_from_buffer(nullptr, 64).ok() ||
        bad_loader.load_from_buffer(valid_image.data(), 63).ok()) {
        std::cerr << "Expected null or <64B buffer to fail validation\n";
        return false;
    }

    // 2. Corrupted magic bytes.
    {
        std::vector<uint8_t> img = valid_image;
        img[0] = 'X';
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected corrupted magic to fail validation\n";
            return false;
        }
    }

    // 3. Unsupported format version.
    {
        std::vector<uint8_t> img = valid_image;
        write_u32_le_at(img.data() + 4, 2u);
        if (bad_loader.load_from_buffer(img.data(), img.size()).code !=
            StatusCode::UnsupportedFormat) {
            std::cerr << "Expected unsupported format version to return UnsupportedFormat\n";
            return false;
        }
    }

    // 4. Truncated file / total_file_bytes mismatch.
    if (bad_loader.load_from_buffer(valid_image.data(), valid_image.size() - 4).ok()) {
        std::cerr << "Expected truncated buffer to fail validation\n";
        return false;
    }

    // 5. Trailing garbage bytes where header total_file_bytes is inflated beyond last tensor end.
    {
        std::vector<uint8_t> img = valid_image;
        img.resize(valid_image.size() + 64, 0);
        write_u64_le_at(img.data() + 56, img.size());
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected trailing unreferenced bytes to fail validation\n";
            return false;
        }
    }

    // 6. Non-finite rms_norm_eps (NaN) in header.
    {
        std::vector<uint8_t> img = valid_image;
        write_u32_le_at(img.data() + 48, 0x7FC00000u);  // quiet NaN
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected NaN rms_norm_eps in header to fail validation\n";
            return false;
        }
    }

    // 7. Invalid GQA head divisibility in header (num_query_heads=5, num_kv_heads=2).
    {
        std::vector<uint8_t> img = valid_image;
        write_u32_le_at(img.data() + 28, 5u);
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected invalid GQA head ratio in header to fail validation\n";
            return false;
        }
    }

    // 8. Unaligned payload_offset in directory entry.
    {
        // First directory entry is at offset 64: name_len(4) + name(17) + dtype(4) + ndim(4) + shape(32) = 61 bytes -> offset at 64 + 61 = 125
        std::vector<uint8_t> img = valid_image;
        const size_t t0_offset_field = 64 + 4 + t0.name.size() + 8 + 32;
        write_u64_le_at(img.data() + t0_offset_field, 196u);  // not divisible by 64
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected unaligned payload_offset in directory to fail validation\n";
            return false;
        }
    }

    // 9. Payload overlapping the directory region (payload_offset = 64).
    {
        std::vector<uint8_t> img = valid_image;
        const size_t t0_offset_field = 64 + 4 + t0.name.size() + 8 + 32;
        write_u64_le_at(img.data() + t0_offset_field, 64u);
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected payload overlapping directory region to fail validation\n";
            return false;
        }
    }

    // 10. Overlapping tensor payloads (tensor 1 starts at tensor 0's payload_offset).
    {
        std::vector<uint8_t> img = valid_image;
        const TensorView* v0 = loader.find_tensor(t0.name);
        const size_t entry0_len = 4 + t0.name.size() + 56;
        const size_t t1_offset_field = 64 + entry0_len + 4 + t1.name.size() + 8 + 32;
        write_u64_le_at(img.data() + t1_offset_field, v0->desc().payload_offset);
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected overlapping tensor payloads to fail validation\n";
            return false;
        }
    }

    // 11. Integer overflow in payload_offset + payload_nbytes.
    {
        std::vector<uint8_t> img = valid_image;
        const size_t t0_offset_field = 64 + 4 + t0.name.size() + 8 + 32;
        write_u64_le_at(img.data() + t0_offset_field,
                        std::numeric_limits<uint64_t>::max() - 63u);  // aligned to 64? No, use 0xFFFFFFFFFFFFFFC0u
        write_u64_le_at(img.data() + t0_offset_field, 0xFFFFFFFFFFFFFFC0ULL);
        if (bad_loader.load_from_buffer(img.data(), img.size()).ok()) {
            std::cerr << "Expected overflowing payload_offset + payload_nbytes to fail\n";
            return false;
        }
    }

    // 12. TensorDesc overflow & Q4_0 block size checks.
    {
        TensorDesc overflow_desc = t1;
        overflow_desc.shape = {0x100000000ULL, 0x100000000ULL, 0, 0};
        overflow_desc.payload_offset = 64;
        overflow_desc.payload_nbytes = 64;
        if (overflow_desc.validate().ok()) {
            std::cerr << "Expected 64-bit shape multiplication overflow to fail validation\n";
            return false;
        }

        TensorDesc bad_q4 = t1;
        bad_q4.shape = {64, 30, 0, 0};
        bad_q4.payload_offset = 64;
        bad_q4.payload_nbytes = 1080;
        if (bad_q4.validate().ok()) {
            std::cerr << "Expected non-multiple-of-32 Q4_0 innermost dim to fail validation\n";
            return false;
        }
    }

    return true;
}

bool test_tensor_binding_and_numerical_preservation() {
    ModelConfig cfg{};
    cfg.num_layers = 2;
    cfg.hidden_dim = 64;
    cfg.intermediate_dim = 128;
    cfg.num_query_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.vocab_size = 128;
    cfg.max_seq_len = 32;
    cfg.weight_dtype = DataType::FP32;

    std::vector<TensorDesc> descs;
    std::vector<std::vector<uint8_t>> payloads;

    auto add_fp32_tensor = [&](const std::string& name,
                               uint32_t ndim,
                               std::array<uint64_t, kMaxTensorDims> shape,
                               float base,
                               float step) {
        TensorDesc d{};
        d.name = name;
        d.dtype = DataType::FP32;
        d.ndim = ndim;
        d.shape = shape;
        uint64_t elems = 0;
        d.compute_num_elements(&elems);
        descs.push_back(d);
        payloads.push_back(pack_f32_Pattern(static_cast<size_t>(elems), base, step));
    };

    // Match the exact 3 + 9 * num_layers (21 tensors) inventory.
    add_fp32_tensor("lm_head.weight", 2, {cfg.vocab_size, cfg.hidden_dim, 0, 0}, -1.5f, 0.01f);
    add_fp32_tensor("model.embed_tokens.weight", 2, {cfg.vocab_size, cfg.hidden_dim, 0, 0}, 0.25f, 0.005f);

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const float lb = static_cast<float>(l + 1) * 0.5f;
        add_fp32_tensor(p + "input_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lb + 0.1f, 0.01f);
        add_fp32_tensor(p + "mlp.down_proj.weight", 2, {cfg.hidden_dim, cfg.intermediate_dim, 0, 0}, lb + 0.2f, -0.002f);
        add_fp32_tensor(p + "mlp.gate_proj.weight", 2, {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, lb + 0.3f, 0.003f);
        add_fp32_tensor(p + "mlp.up_proj.weight", 2, {cfg.intermediate_dim, cfg.hidden_dim, 0, 0}, lb + 0.4f, -0.004f);
        add_fp32_tensor(p + "post_attention_layernorm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, lb + 0.5f, 0.01f);
        add_fp32_tensor(p + "self_attn.k_proj.weight", 2, {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, lb + 0.6f, 0.005f);
        add_fp32_tensor(p + "self_attn.o_proj.weight", 2, {cfg.hidden_dim, cfg.q_dim(), 0, 0}, lb + 0.7f, -0.005f);
        add_fp32_tensor(p + "self_attn.q_proj.weight", 2, {cfg.q_dim(), cfg.hidden_dim, 0, 0}, lb + 0.8f, 0.006f);
        add_fp32_tensor(p + "self_attn.v_proj.weight", 2, {cfg.kv_dim(), cfg.hidden_dim, 0, 0}, lb + 0.9f, -0.006f);
    }

    add_fp32_tensor("model.norm.weight", 1, {cfg.hidden_dim, 0, 0, 0}, 1.0f, 0.0025f);

    std::vector<uint8_t> image;
    Status st = ModelLoader::build_ie_image(cfg, descs, payloads, &image);
    if (!st.ok()) {
        std::cerr << "build_ie_image failed for 20-tensor model: " << st.message << "\n";
        return false;
    }

    // Write to a temporary file on disk and load via `load_from_file` (mmap).
    const std::string temp_path = "test_temp_synthetic_model.ie";
    {
        std::ofstream ofs(temp_path, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(image.data()),
                  static_cast<std::streamsize>(image.size()));
    }

    ModelLoader file_loader;
    Status load_st = file_loader.load_from_file(temp_path);
    std::remove(temp_path.c_str());
    if (!load_st.ok()) {
        std::cerr << "load_from_file failed: " << load_st.message << "\n";
        return false;
    }

    // Verify exact descriptor ordering preservation.
    if (file_loader.num_tensors() != descs.size()) {
        std::cerr << "Loaded tensor count mismatch\n";
        return false;
    }
    for (size_t i = 0; i < descs.size(); ++i) {
        const TensorDesc& actual = file_loader.tensor_descriptors()[i];
        if (actual.name != descs[i].name || actual.shape != descs[i].shape ||
            actual.payload_offset % kIePayloadAlignment != 0) {
            std::cerr << "Tensor ordering or shape mismatch at index " << i << "\n";
            return false;
        }
#if defined(__unix__) || defined(__APPLE__)
        const TensorView* v = file_loader.find_tensor(actual.name);
        if (reinterpret_cast<uintptr_t>(v->raw_data()) % kIePayloadAlignment != 0) {
            std::cerr << "mmap virtual address for " << actual.name << " is not 64-byte aligned\n";
            return false;
        }
#endif
    }

    // Bind all transformer weights and verify bit-exact FP32 data preservation.
    TransformerWeightBindings bindings{};
    Status bind_st = file_loader.bind_transformer_weights(&bindings);
    if (!bind_st.ok()) {
        std::cerr << "bind_transformer_weights failed: " << bind_st.message << "\n";
        return false;
    }

    if (!verify_f32_pattern(bindings.lm_head, cfg.vocab_size * cfg.hidden_dim, -1.5f, 0.01f) ||
        !verify_f32_pattern(bindings.embed_tokens, cfg.vocab_size * cfg.hidden_dim, 0.25f, 0.005f) ||
        !verify_f32_pattern(bindings.final_norm, cfg.hidden_dim, 1.0f, 0.0025f)) {
        std::cerr << "Bit-exact FP32 verification failed for top-level tensors\n";
        return false;
    }

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const float lb = static_cast<float>(l + 1) * 0.5f;
        const LayerWeightBindings& b = bindings.layers[l];
        if (!verify_f32_pattern(b.input_layernorm, cfg.hidden_dim, lb + 0.1f, 0.01f) ||
            !verify_f32_pattern(b.down_proj, cfg.hidden_dim * cfg.intermediate_dim, lb + 0.2f, -0.002f) ||
            !verify_f32_pattern(b.gate_proj, cfg.intermediate_dim * cfg.hidden_dim, lb + 0.3f, 0.003f) ||
            !verify_f32_pattern(b.up_proj, cfg.intermediate_dim * cfg.hidden_dim, lb + 0.4f, -0.004f) ||
            !verify_f32_pattern(b.post_attention_layernorm, cfg.hidden_dim, lb + 0.5f, 0.01f) ||
            !verify_f32_pattern(b.k_proj, cfg.kv_dim() * cfg.hidden_dim, lb + 0.6f, 0.005f) ||
            !verify_f32_pattern(b.o_proj, cfg.hidden_dim * cfg.q_dim(), lb + 0.7f, -0.005f) ||
            !verify_f32_pattern(b.q_proj, cfg.q_dim() * cfg.hidden_dim, lb + 0.8f, 0.006f) ||
            !verify_f32_pattern(b.v_proj, cfg.kv_dim() * cfg.hidden_dim, lb + 0.9f, -0.006f)) {
            std::cerr << "Bit-exact FP32 verification failed at layer " << l << "\n";
            return false;
        }
    }

    // Negative binding test: wrong tensor shape on k_proj must be rejected by bind_transformer_weights.
    std::vector<TensorDesc> bad_descs = descs;
    std::vector<std::vector<uint8_t>> bad_payloads = payloads;
    for (size_t i = 0; i < bad_descs.size(); ++i) {
        if (bad_descs[i].name == "model.layers.0.self_attn.k_proj.weight") {
            bad_descs[i].shape = {cfg.q_dim(), cfg.hidden_dim, 0, 0};  // wrong: q_dim instead of kv_dim
            bad_payloads[i].resize(cfg.q_dim() * cfg.hidden_dim * sizeof(float), 0);
        }
    }
    std::vector<uint8_t> bad_bind_img;
    if (!ModelLoader::build_ie_image(cfg, bad_descs, bad_payloads, &bad_bind_img).ok()) {
        return false;
    }
    ModelLoader bad_bind_loader;
    if (!bad_bind_loader.load_from_buffer(bad_bind_img.data(), bad_bind_img.size()).ok()) {
        return false;
    }
    TransformerWeightBindings unused_bindings{};
    if (bad_bind_loader.bind_transformer_weights(&unused_bindings).ok()) {
        std::cerr << "Expected bind_transformer_weights to reject mismatched k_proj shape\n";
        return false;
    }

    return true;
}

}  // namespace

bool run_loader_validation_tests() {
    if (!test_malformed_file_rejection()) {
        return false;
    }
    if (!test_tensor_binding_and_numerical_preservation()) {
        return false;
    }

    const char* external_ie_path = std::getenv("INFERENCE_ENGINE_TEST_IE_PATH");
    if (external_ie_path != nullptr && external_ie_path[0] != '\0') {
        ModelLoader ext_loader;
        Status st = ext_loader.load_from_file(external_ie_path);
        if (!st.ok()) {
            std::cerr << "Failed to load external .ie file (" << external_ie_path
                      << "): " << st.message << "\n";
            return false;
        }
        TransformerWeightBindings ext_bindings{};
        Status bind_st = ext_loader.bind_transformer_weights(&ext_bindings);
        if (!bind_st.ok()) {
            std::cerr << "Failed to bind transformer weights on external .ie file ("
                      << external_ie_path << "): " << bind_st.message << "\n";
            return false;
        }
        std::cout << "[   INFO   ] Verified external .ie model (" << external_ie_path
                  << ") with " << ext_loader.num_tensors() << " bound tensors.\n";
    }

    return true;
}

}  // namespace inference_engine::testing
