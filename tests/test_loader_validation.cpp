#include <cstring>
#include <iostream>
#include <vector>

#include "inference_engine/loader.hpp"

namespace inference_engine::testing {

bool run_loader_validation_tests() {
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

    std::vector<uint8_t> image;
    Status build_st = ModelLoader::build_ie_image(cfg, {t0, t1}, {p0, p1}, &image);
    if (!build_st.ok()) {
        std::cerr << "Failed to build valid .ie image: " << build_st.message << "\n";
        return false;
    }

    ModelLoader loader;
    Status load_st = loader.load_from_buffer(image.data(), image.size());
    if (!load_st.ok() || !loader.is_loaded() || loader.num_tensors() != 2) {
        std::cerr << "Failed to load valid .ie image: " << load_st.message << "\n";
        return false;
    }

    const TensorView* view0 = loader.find_tensor("model.norm.weight");
    const TensorView* view1 = loader.find_tensor("model.layers.0.self_attn.q_proj.weight");
    if (view0 == nullptr || view1 == nullptr) {
        std::cerr << "Expected tensors not found in loaded model\n";
        return false;
    }
    if (view0->desc().payload_offset % kIePayloadAlignment != 0 ||
        view1->desc().payload_offset % kIePayloadAlignment != 0) {
        std::cerr << "Payload offsets are not 64-byte aligned\n";
        return false;
    }

    // Negative test 1: Corrupted magic bytes.
    std::vector<uint8_t> bad_magic = image;
    bad_magic[0] = 'X';
    ModelLoader bad_loader;
    if (bad_loader.load_from_buffer(bad_magic.data(), bad_magic.size()).ok()) {
        std::cerr << "Expected corrupted magic to fail validation\n";
        return false;
    }

    // Negative test 2: Truncated buffer.
    if (bad_loader.load_from_buffer(image.data(), image.size() - 4).ok()) {
        std::cerr << "Expected truncated buffer to fail validation\n";
        return false;
    }

    // Negative test 3: Unaligned payload offset in TensorDesc validation.
    TensorDesc unaligned = t0;
    unaligned.payload_offset = 68;
    unaligned.payload_nbytes = p0.size();
    if (unaligned.validate().ok()) {
        std::cerr << "Expected unaligned payload_offset to fail validation\n";
        return false;
    }

    // Negative test 4: Q4_0 tensor with element count not divisible by 32.
    TensorDesc bad_q4 = t1;
    bad_q4.shape = {64, 30, 0, 0};
    bad_q4.payload_offset = 64;
    bad_q4.payload_nbytes = 1080;
    if (bad_q4.validate().ok()) {
        std::cerr << "Expected non-multiple-of-32 Q4_0 shape to fail validation\n";
        return false;
    }

    return true;
}

}  // namespace inference_engine::testing
