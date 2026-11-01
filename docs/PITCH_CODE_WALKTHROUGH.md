# BIOS V2 — Comprehensive Source Code Walkthrough
## Subsystem-by-Subsystem Implementation Guide for Technical Defense
**Repository:** `inference-engine` (`shreyaas5/inference-engine`)  
**Target Binary:** `inference-engine` (C++17) & `Inference Studio` (Full-Stack Workstation)  
**Standard:** Strict zero-framework architecture (No PyTorch, LibTorch, llama.cpp, GGML, Ollama, MLX)

---

## Guide Overview

This document is the engineering reference manual for technical evaluations and code audits. It details every major subsystem in the repository, providing:
1. **Exact source file paths** and line references.
2. **Key classes, structures, and functions**.
3. **Execution order** and invocation chains.
4. **Important state variables** and data structures.
5. **Exact, verbatim C++17 and TypeScript code excerpts** directly from our source tree.
6. **Subsystem interactions** and architectural interfaces.

---

# Subsystem 1: Model Loader & Verification Engine

### 1.1 Source Files
- Public Interface: [`include/inference_engine/loader.hpp`](../include/inference_engine/loader.hpp)
- Implementation: [`src/loader/loader.cpp`](../src/loader/loader.cpp)
- Unit Tests: [`tests/test_loader_validation.cpp`](../tests/test_loader_validation.cpp)

### 1.2 Classes & Structures
- `ModelLoader`: Primary lifecycle manager responsible for reading, verifying, memory-mapping, and binding model weights.
- `LayerWeightBindings`: Aggregates the 9 projection and normalization `TensorView`s for one Transformer layer.
- `TransformerWeightBindings`: Contains `embed_tokens`, `std::vector<LayerWeightBindings> layers` (size 22), `final_norm`, and `lm_head`.
- `ByteRange`: Private interval struct `{uint64_t begin, end}` for overlap checking.

### 1.3 Execution Order
1. `ModelLoader::load_from_file(path)` is called with a path to a `.ie` binary.
2. Opens file descriptor via POSIX `::open(path.c_str(), O_RDONLY)`.
3. Calls `::fstat(fd, &st)` to verify file existence and obtain total file size.
4. Maps file to virtual address space via `::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0)`.
5. Delegates parsing to `ModelLoader::load_from_buffer(mapped_ptr_, mapped_size_)`.
6. Checks 64-byte file header (Magic `0x464D4549`, Version `1`, hyperparameter sanity, `total_file_bytes`).
7. Iterates through $N$ tensor directory entries, parsing names, shapes, offsets, and descriptors.
8. Sorts and validates payload byte ranges: asserts 64-byte payload alignment, checks that no payload overlaps the header/directory, and verifies zero inter-payload overlap.
9. `ModelLoader::bind_transformer_weights()` binds all 201 expected canonical Llama tensors.

### 1.4 Key Variables & Invariants
- `mapped_ptr_` (`const uint8_t*`): Base pointer to the read-only memory-mapped file region.
- `views_` (`std::unordered_map<std::string, TensorView>`): Fast hash map mapping canonical tensor names (e.g. `"model.layers.0.self_attn.q_proj.weight"`) to non-owning `TensorView`s.
- `tensors_` (`std::vector<TensorDesc>`): In-order sequence of tensor descriptors.
- Invariant: `payload_offset % 64 == 0` for every tensor payload.

### 1.5 Code Excerpts

#### Zero-Copy POSIX `mmap` Setup (`src/loader/loader.cpp:87-109`):
```cpp
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
    return load_from_buffer(mapped_ptr_, mapped_size_);
#endif
}
```

#### Non-Overlap Interval Validation (`src/loader/loader.cpp:276-299`):
```cpp
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
if (payload_ranges.back().end != total_file_bytes) {
    return Status::Error(StatusCode::CorruptedFile,
                         "Final tensor payload end does not match total_file_bytes");
}
```

---

# Subsystem 2: Tensor Representation (`TensorView`)

### 2.1 Source Files
- Public Interface: [`include/inference_engine/tensor.hpp`](../include/inference_engine/tensor.hpp)
- Implementation: [`src/common/tensor.cpp`](../src/common/tensor.cpp)

### 2.2 Classes & Structures
- `TensorDesc`: Metadata descriptor containing `name`, `dtype`, `ndim`, `shape` array ($4 \times \text{u64}$), `payload_offset`, and `payload_nbytes`.
- `TensorView`: Non-owning typed window into a validated byte buffer.

### 2.3 Execution Order & Invariants
- `TensorView` instances are created exclusively by `ModelLoader` during `load_from_buffer()`.
- They allocate zero heap memory and perform zero data copying.
- Lifetime invariant: Pointers returned by `data_f32()` or `data_q4_0()` point directly into the OS page cache via `mapped_ptr_ + payload_offset`. They are valid as long as `ModelLoader` remains open.

### 2.4 Code Excerpt (`include/inference_engine/tensor.hpp:64-71`):
```cpp
const float* data_f32() const noexcept {
    return desc_.dtype == DataType::FP32 ? reinterpret_cast<const float*>(data_) : nullptr;
}

const BlockQ4_0* data_q4_0() const noexcept {
    return desc_.dtype == DataType::Q4_0 ? reinterpret_cast<const BlockQ4_0*>(data_) : nullptr;
}
```

---

# Subsystem 3: Quantization & Offline Converter

### 3.1 Source Files
- Public Interface: [`include/inference_engine/quant.hpp`](../include/inference_engine/quant.hpp)
- C++ Implementation: [`src/kernels/quant_q4_0.cpp`](../src/kernels/quant_q4_0.cpp)
- Offline Converter: [`tools/convert_to_inference_engine.py`](../tools/convert_to_inference_engine.py)
- Unit Tests: [`tests/test_quant_packing.cpp`](../tests/test_quant_packing.cpp)

### 3.2 Classes & Structures
- `BlockQ4_0`: 18-byte packed structure (`#pragma pack(push, 1)`):
  - `uint16_t d`: IEEE 754 half-precision (binary16) scale.
  - `uint8_t qs[16]`: 16 packed bytes storing 32 4-bit unsigned integers.
- Bitwise FP16 functions: `fp32_to_fp16()` and `fp16_to_fp32()`.

### 3.3 Execution Order
1. `quantize_row_q4_0(src, dst_blocks, num_elements)` processes floats in slices of 32 elements.
2. Identifies element with maximum absolute magnitude: $v_{\max} = \arg\max |v_i|$.
3. Computes FP16 scale $d = v_{\max} / -8.0f$, and reciprocal $\text{id} = 1.0f / d$.
4. Clamps quantized values into $[0, 15]$: $q = \text{clamp}(\lfloor x \cdot \text{id} + 8.5f \rfloor, 0, 15)$.
5. Packs low nibbles ($0 \dots 15$) and high nibbles ($16 \dots 31$) into `qs[j]`.
6. At runtime, `dequantize_row_q4_0` reverses the transformation into destination floats.

### 3.4 Code Excerpt (`src/kernels/quant_q4_0.cpp:87-116`):
```cpp
const size_t num_blocks = num_elements / kQ4_0BlockElements;
for (size_t b = 0; b < num_blocks; ++b) {
    const float* block_src = src + b * kQ4_0BlockElements;
    float amax = 0.0f;
    float signed_max = 0.0f;

    for (size_t i = 0; i < kQ4_0BlockElements; ++i) {
        const float v = block_src[i];
        const float av = std::fabs(v);
        if (amax < av) {
            amax = av;
            signed_max = v;
        }
    }

    const float d = signed_max / -8.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    dst_blocks[b].d = fp32_to_fp16(d);

    for (size_t j = 0; j < kQ4_0PackedBytes; ++j) {
        const float x0 = block_src[j] * id;
        const float x1 = block_src[j + kQ4_0PackedBytes] * id;

        const int q0 = std::clamp(static_cast<int>(std::floor(x0 + 8.5f)), 0, 15);
        const int q1 = std::clamp(static_cast<int>(std::floor(x1 + 8.5f)), 0, 15);

        dst_blocks[b].qs[j] =
            pack_q4_0_nibbles(static_cast<uint8_t>(q0), static_cast<uint8_t>(q1));
    }
}
```

---

# Subsystem 4: Transformer Execution Engine

### 4.1 Source Files
- Public Interface: [`include/inference_engine/transformer.hpp`](../include/inference_engine/transformer.hpp)
- Implementation: [`src/transformer/transformer.cpp`](../src/transformer/transformer.cpp)
- Unit Tests: [`tests/test_golden_forward.cpp`](../tests/test_golden_forward.cpp)

### 4.2 Classes & Structures
- `TransformerEngine`: Core inference execution context. Holds pre-allocated activation buffers, the `KvCache`, the precalculated RoPE trigonometric tables, and the `ThreadPool`.

### 4.3 Pre-Allocated Workspace (Zero Runtime Heap Allocations)
Allocated once in `TransformerEngine::init()` (`src/transformer/transformer.cpp:76-87`):
- `x_`: `[2048]` floats (hidden state vector, 8 KB)
- `norm_buf_`: `[2048]` floats (RMSNorm output buffer, 8 KB)
- `q_`: `[2048]` floats (Query vector, 32 heads $\times$ 64, 8 KB)
- `k_`: `[256]` floats (Key vector, 4 heads $\times$ 64, 1 KB)
- `v_`: `[256]` floats (Value vector, 4 heads $\times$ 64, 1 KB)
- `attn_out_`: `[2048]` floats (Multi-head attention output, 8 KB)
- `proj_out_`: `[2048]` floats (Projection scratchpad, 8 KB)
- `gate_`: `[5632]` floats (SwiGLU gate vector, 22.5 KB)
- `up_`: `[5632]` floats (SwiGLU up-projection vector, 22.5 KB)
- `ffn_act_`: `[5632]` floats (SwiGLU activated vector, 22.5 KB)
- `scores_`: `[2048]` floats (Attention score scratchpad, 8 KB)
- `rope_cos_`, `rope_sin_`: `[2048, 32]` floats (Precomputed RoPE trigonometric tables, 524 KB)

### 4.4 16-Step Token Forward Pass Execution
`TransformerEngine::forward_token(token_id, pos, out_logits)` executes:
1. Validates `pos == kv_cache_.seq_len()`.
2. Step 1: Look up token embedding row from `embed_tokens` into `x_` (dequantizes single row if Q4_0).
3. Step 2: Loop `layer_idx = 0 .. 21`:
   - 2a: `rmsnorm_f32` on `x_` with `input_layernorm.weight` $\to$ `norm_buf_`.
   - 2b: Parallel matrix multiplications: $W_q \cdot u \to q$, $W_k \cdot u \to k$, $W_v \cdot u \to v$.
   - 2c: In-place RoPE on $q$ and $k$ via `rope_inplace_cached_f32`.
   - 2d: Append $k$ and $v$ to `kv_cache_` at `(layer_idx, pos)`.
   - 2e: Compute causal GQA attention via `gqa_attention_f32` $\to$ `attn_out_`.
   - 2f: $W_o$ projection $\to$ `proj_out_`, accumulate residual: $x \leftarrow x + \text{proj\_out}$.
   - 2g: `rmsnorm_f32` on `x_` with `post_attention_layernorm.weight` $\to$ `norm_buf_`.
   - 2h: SwiGLU MLP: $W_{\text{gate}} \cdot u \to \text{gate}$, $W_{\text{up}} \cdot u \to \text{up}$, $\text{swiglu\_f32}(\dots) \to \text{ffn\_act}$, $W_{\text{down}} \cdot \text{ffn\_act} \to \text{proj\_out}$, residual: $x \leftarrow x + \text{proj\_out}$.
4. Step 3: Advance sequence length: `kv_cache_.commit_token()`.
5. Step 4: Final RMSNorm with `final_norm.weight` and $W_{\text{lm\_head}}$ projection $\to$ `out_logits` (32000 floats).

### 4.5 Code Excerpt (`src/transformer/transformer.cpp:180-232`):
```cpp
for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
    const LayerWeightBindings& lw = bindings_.layers[layer_idx];

    // 2a. Pre-Attention RMSNorm: u = RMSNorm(x, w_input_layernorm, eps)
    scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.input_layernorm.data_f32(), hidden_dim, eps);

    // 2b. Q/K/V Projections
    matmul(q_.data(), lw.q_proj, norm_buf_.data(), q_dim, hidden_dim);
    matmul(k_.data(), lw.k_proj, norm_buf_.data(), kv_dim, hidden_dim);
    matmul(v_.data(), lw.v_proj, norm_buf_.data(), kv_dim, hidden_dim);

    // 2c. RoPE for Q and K (using precomputed cos/sin tables)
    const size_t half_dim = config_.head_dim / 2;
    const float* cos_row = rope_cos_.data() + static_cast<size_t>(pos) * half_dim;
    const float* sin_row = rope_sin_.data() + static_cast<size_t>(pos) * half_dim;
    scalar::rope_inplace_cached_f32(q_.data(), config_.num_query_heads, config_.head_dim, cos_row, sin_row);
    scalar::rope_inplace_cached_f32(k_.data(), config_.num_kv_heads, config_.head_dim, cos_row, sin_row);

    // 2d. KV-Cache Write (append without advancing committed sequence length)
    kv_cache_.append(layer_idx, pos, k_.data(), v_.data());

    // 2e. Causal Grouped Query Attention (GQA)
    scalar::gqa_attention_f32(attn_out_.data(), q_.data(), kv_cache_, layer_idx,
                              pos, config_.num_query_heads, config_.num_kv_heads,
                              config_.head_dim, scores_.data());

    // 2f. Output Projection and First Residual
    matmul(proj_out_.data(), lw.o_proj, attn_out_.data(), hidden_dim, q_dim);
    for (uint32_t i = 0; i < hidden_dim; ++i) x_[i] += proj_out_[i];

    // 2g. Post-Attention RMSNorm
    scalar::rmsnorm_f32(norm_buf_.data(), x_.data(), lw.post_attention_layernorm.data_f32(), hidden_dim, eps);

    // 2h. SwiGLU Feed-Forward Network and Second Residual
    matmul(gate_.data(), lw.gate_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
    matmul(up_.data(), lw.up_proj, norm_buf_.data(), intermediate_dim, hidden_dim);
    scalar::swiglu_f32(ffn_act_.data(), gate_.data(), up_.data(), intermediate_dim);
    matmul(proj_out_.data(), lw.down_proj, ffn_act_.data(), hidden_dim, intermediate_dim);
    for (uint32_t i = 0; i < hidden_dim; ++i) x_[i] += proj_out_[i];
}
```

---

# Subsystem 5: Key-Value Cache (`KvCache`)

### 5.1 Source Files
- Public Interface: [`include/inference_engine/kv_cache.hpp`](../include/inference_engine/kv_cache.hpp)
- Implementation: [`src/core/kv_cache.cpp`](../src/core/kv_cache.cpp)
- Unit Tests: [`tests/test_kv_cache.cpp`](../tests/test_kv_cache.cpp)

### 5.2 Storage Representation & Layout
- `k_storage_` & `v_storage_`: Two contiguous `std::vector<float>` buffers of size:
  $$\text{size} = 22 \times 2048 \times 256 = 11,534,336\text{ floats } (45.09\text{ MB each})$$
- Offset Calculation (`src/core/kv_cache.cpp:32`):
  ```cpp
  size_t KvCache::offset_for(uint32_t layer_idx, uint32_t pos) const noexcept {
      return (static_cast<size_t>(layer_idx) * static_cast<size_t>(max_seq_len_) +
              static_cast<size_t>(pos)) * static_cast<size_t>(kv_dim_);
  }
  ```

### 5.3 Commit & Rewind Semantics
- `append(layer_idx, pos, k_data, v_data)`: Copies 256 floats into `k_storage_` and `v_storage_` via `std::memcpy`. Enforces the contract: **does NOT modify `committed_seq_len_`**.
- `commit_token()`: Increments `committed_seq_len_` by 1 after all 22 layers complete.
- `truncate(new_len)`: Safely rewinds `committed_seq_len_` to `new_len` for prefix matching across turns.

### 5.4 Code Excerpt (`src/core/kv_cache.cpp:64-86`):
```cpp
Status KvCache::commit_positions(uint32_t count) noexcept {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "KvCache is not initialized");
    }
    if (committed_seq_len_ + count > max_seq_len_) {
        return Status::Error(StatusCode::OutOfBounds, "Committing positions would exceed max_seq_len");
    }
    committed_seq_len_ += count;
    return Status::Ok();
}

Status KvCache::truncate(uint32_t new_len) noexcept {
    if (!initialized_) {
        return Status::Error(StatusCode::InvalidArgument, "KvCache is not initialized");
    }
    if (new_len > committed_seq_len_) {
        return Status::Error(StatusCode::InvalidArgument,
                             "Cannot truncate to a position greater than current seq_len");
    }
    committed_seq_len_ = new_len;
    return Status::Ok();
}
```

---

# Subsystem 6: CPU Mathematical Kernels & SIMD Vectorization

### 6.1 Source Files
- Scalar Reference: [`src/kernels/scalar/kernels_scalar.cpp`](../src/kernels/scalar/kernels_scalar.cpp)
- ARM NEON: [`src/kernels/neon/kernels_neon.cpp`](../src/kernels/neon/kernels_neon.cpp)
- x86 AVX2: [`src/kernels/avx2/kernels_avx2.cpp`](../src/kernels/avx2/kernels_avx2.cpp)
- SIMD Dispatch: [`src/kernels/dispatch.cpp`](../src/kernels/dispatch.cpp)
- Hardware Detection: [`src/common/common.cpp`](../src/common/common.cpp)

### 6.2 Causal Grouped Query Attention (GQA) Implementation
`scalar::gqa_attention_f32` computes scaled dot-product attention over all sequence positions $t \in [0, \text{pos}]$:
```cpp
const uint32_t gqa_group_size = num_query_heads / num_kv_heads; // 32 / 4 = 8
const float head_scale = 1.0f / std::sqrt(static_cast<float>(head_dim)); // 1 / 8 = 0.125f

for (uint32_t hq = 0; hq < num_query_heads; ++hq) {
    const uint32_t hkv = hq / gqa_group_size; // Map query head 0..31 to KV head 0..3
    const float* q_head = q + static_cast<size_t>(hq) * head_dim;

    // 1. Dot-product attention scores across cached positions 0..pos
    for (uint32_t t = 0; t <= pos; ++t) {
        const float* k_head = kv_cache.key_vector(layer_idx, t) + hkv * head_dim;
        scores_scratch[t] = dot_f32(q_head, k_head, head_dim) * head_scale;
    }

    // 2. Numerically stable softmax over scores_scratch[0..pos]
    softmax_f32(scores_scratch, pos + 1);

    // 3. Accumulate weighted values
    float* out_head = out + static_cast<size_t>(hq) * head_dim;
    std::fill(out_head, out_head + head_dim, 0.0f);
    for (uint32_t t = 0; t <= pos; ++t) {
        const float* v_head = kv_cache.value_vector(layer_idx, t) + hkv * head_dim;
        const float weight = scores_scratch[t];
        for (uint32_t d = 0; d < head_dim; ++d) {
            out_head[d] += weight * v_head[d];
        }
    }
}
```

### 6.3 ARM NEON 2-Row Register Tiling (`src/kernels/neon/kernels_neon.cpp:184-277`)
Processes two output rows ($r$ and $r+1$) simultaneously. Loads the 32 floats of activation vector $x$ once into registers `x0..x7` and reuses them across Row 0 and Row 1:
```cpp
for (; r + 2 <= out_dim; r += 2) {
    const BlockQ4_0* row0 = weight_blocks + r * blocks_per_row;
    const BlockQ4_0* row1 = weight_blocks + (r + 1) * blocks_per_row;

    float32x4_t acc_r0 = vdupq_n_f32(0.0f);
    float32x4_t acc_r1 = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const float* x_block = x + b * kQ4_0BlockElements;

        // Load 32 floats of x ONCE for both row0 and row1
        const float32x4_t x0 = vld1q_f32(x_block + 0);
        const float32x4_t x1 = vld1q_f32(x_block + 4);
        const float32x4_t x2 = vld1q_f32(x_block + 8);
        const float32x4_t x3 = vld1q_f32(x_block + 12);
        const float32x4_t x4 = vld1q_f32(x_block + 16);
        const float32x4_t x5 = vld1q_f32(x_block + 20);
        const float32x4_t x6 = vld1q_f32(x_block + 24);
        const float32x4_t x7 = vld1q_f32(x_block + 28);

        // Fused unpack, subtract 8, convert float, and multiply-accumulate for Row 0
        // ... (accumulates unscaled sum in b_acc, scales by fp16_to_fp32(row0[b].d))
        acc_r0 = vfmaq_f32(acc_r0, vaddq_f32(b_acc0, b_acc1), vd0);

        // Compute Row 1 with the SAME x0..x7 registers
        acc_r1 = vfmaq_f32(acc_r1, vaddq_f32(b_acc0_r1, b_acc1_r1), vd1);
    }

    out[r]     = vaddvq_f32(acc_r0);
    out[r + 1] = vaddvq_f32(acc_r1);
}
```

---

# Subsystem 7: Persistent ThreadPool & Dynamic Work Scheduling

### 7.1 Source Files
- Public Interface: [`include/inference_engine/thread_pool.hpp`](../include/inference_engine/thread_pool.hpp)
- Implementation: [`src/kernels/threading/thread_pool.cpp`](../src/kernels/threading/thread_pool.cpp)

### 7.2 Synchronization Architecture
- `generation_` (`uint64_t`): Monotonically increasing counter signaling new work to worker threads.
- `cv_start_` & `cv_done_`: Condition variables governing worker launch and completion.
- `active_workers_`: Atomic count of threads currently processing task chunks.
- `exception_ptr_`: Captures any exception thrown by a thread callback for clean rethrowing on the calling thread.

### 7.3 Dynamic Task Chunking Loop (`src/kernels/threading/thread_pool.cpp:55-82`):
```cpp
const size_t chunk_size = std::max<size_t>(1, num_tasks / (num_threads_ * 4));

// Thread 0 participates in executing tasks alongside workers
while (true) {
    size_t start_idx = 0;
    size_t end_idx = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_task_ >= total_tasks_) break;
        start_idx = next_task_;
        next_task_ = std::min(total_tasks_, start_idx + chunk_size);
        end_idx = next_task_;
    }
    try {
        for (size_t t = start_idx; t < end_idx; ++t) {
            task_fn(t, 0);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!exception_ptr_) exception_ptr_ = std::current_exception();
        next_task_ = total_tasks_;
        break;
    }
}
```

---

# Subsystem 8: Tokenizer & Token Sampler

### 8.1 Source Files
- Tokenizer: [`include/inference_engine/tokenizer.hpp`](../include/inference_engine/tokenizer.hpp), [`src/tokenizer/tokenizer.cpp`](../src/tokenizer/tokenizer.cpp)
- Sampler: [`include/inference_engine/sampler.hpp`](../include/inference_engine/sampler.hpp), [`src/sampler/sampler.cpp`](../src/sampler/sampler.cpp)
- Unit Tests: [`tests/test_tokenizer.cpp`](../tests/test_tokenizer.cpp), [`tests/test_sampler.cpp`](../tests/test_sampler.cpp)

### 8.2 Tokenizer Dual Processor & Dummy Prefix Handling
In `src/tokenizer/tokenizer.cpp:71-82`, `Tokenizer::load()` creates two processor instances from `tokenizer.model`:
1. `processor_`: Default SentencePiece processor (adds dummy prefix space `▁`).
2. `processor_no_prefix_`: Cloned from `model_proto()` with `mutable_normalizer_spec()->set_add_dummy_prefix(false)`.
When encoding a chat template, text segments following control tokens (`<|user|>`) are encoded with `processor_no_prefix_`, guaranteeing 100% token ID parity with Hugging Face transformers (`tools/verify_tokenizer_parity.py`).

### 8.3 Sampler Nucleus & Top-K Pipeline (`src/sampler/sampler.cpp:73-123`)
```cpp
struct Candidate { int32_t id; double weight; };
std::vector<Candidate> candidates;
// 1. Subtract max logit for numerical stability and apply temperature
for (size_t i = 0; i < vocab_size; ++i) {
    const double scaled = (static_cast<double>(logits[i]) - max_logit) / config_.temperature;
    candidates.push_back({static_cast<int32_t>(i), std::exp(scaled)});
}

// 2. Sort descending by probability weight
std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    return a.weight > b.weight;
});

// 3. Top-K Candidate Truncation
if (config_.top_k > 0 && candidates.size() > config_.top_k) {
    candidates.resize(config_.top_k);
}

// 4. Top-P (Nucleus) Cumulative Distribution Filtering
const double nucleus_limit = config_.top_p * total_weight;
size_t nucleus_size = 0;
double nucleus_weight = 0.0;
do {
    nucleus_weight += candidates[nucleus_size].weight;
    ++nucleus_size;
} while (nucleus_size < candidates.size() && nucleus_weight < nucleus_limit);

// 5. Deterministic Random Draw via 64-bit xorshift* PRNG
const double target = uniform_random() * nucleus_weight;
double cumulative = 0.0;
for (size_t i = 0; i < nucleus_size; ++i) {
    cumulative += candidates[i].weight;
    if (target < cumulative) return candidates[i].id;
}
return candidates[nucleus_size - 1].id;
```

---

# Subsystem 9: Studio Persistent Worker (`--studio-worker`)

### 9.1 Source Files
- Header & Implementation: [`src/app/studio_worker.hpp`](../src/app/studio_worker.hpp)
- CLI Integration: [`src/app/main.cpp:928`](../src/app/main.cpp)

### 9.2 Thread Architecture
- **Thread 1 (`main` thread):** Pops parsed JSON commands from `cmd_queue_` and executes inference.
- **Thread 2 (`stdin_reader_loop` thread):** Reads standard input line-by-line. Intercepts `stop` signals immediately to set `stop_requested_ = true`; pushes other JSON commands to `cmd_queue_`.
- **Stdout Protocol Purity:** Every outgoing message is wrapped in a JSON object under `stdout_mutex_`:
  ```cpp
  void emit_event(const std::string& event_json) {
      std::lock_guard<std::mutex> lock(stdout_mutex_);
      std::cout << event_json << "\n";
      std::cout.flush();
  }
  ```

### 9.3 OS Telemetry Instrumentation (`src/app/studio_worker.hpp:50-80`):
```cpp
inline uint64_t current_resident_bytes() noexcept {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t size_pages = 0, resident_pages = 0;
    if (statm >> size_pages >> resident_pages) {
        return resident_pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    }
    return 0;
#endif
}
```

---

# Subsystem 10: Node.js API Gateway & Process Supervisor

### 10.1 Source Files
- Process Supervisor: [`studio/server/engineManager.ts`](../studio/server/engineManager.ts)
- API Server: [`studio/server/index.ts`](../studio/server/index.ts)

### 10.2 Child Process Management
`EngineManager` uses Node.js `child_process.spawn()` to launch `./build/inference-engine --studio-worker`:
- Attaches `readline.createInterface({ input: this.worker.stdout })` to process line-delimited events.
- Emits events over Node's internal `EventEmitter`.
- Translates client requests into single-line JSON commands written to `this.worker.stdin.write(JSON.stringify(cmd) + '\n')`.
- Handles cancellation by sending `{"cmd":"stop"}\n`.

### 10.3 Server-Sent Events (SSE) Dispatch (`studio/server/index.ts:182-205`):
```typescript
app.get('/api/events', (req: Request, res: Response) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });

  const onEngineEvent = (event: any) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  engineManager.on('engine_event', onEngineEvent);
  req.on('close', () => {
    engineManager.off('engine_event', onEngineEvent);
  });
});
```

---

# Subsystem 11: React 19 Observability Studio Frontend

### 11.1 Source Files
- Main Application Shell: [`studio/src/App.tsx`](../studio/src/App.tsx)
- Live Chat & Metrics: [`studio/src/components/ChatPlayground.tsx`](../studio/src/components/ChatPlayground.tsx)
- Telemetry & Gauges: [`studio/src/components/ObservabilityPanel.tsx`](../studio/src/components/ObservabilityPanel.tsx)
- 201-Tensor Inspector: [`studio/src/components/ModelInspector.tsx`](../studio/src/components/ModelInspector.tsx)
- Multi-Run Benchmark Lab: [`studio/src/components/BenchmarkLab.tsx`](../studio/src/components/BenchmarkLab.tsx)
- Engine Log Streamer: [`studio/src/components/RuntimeLogViewer.tsx`](../studio/src/components/RuntimeLogViewer.tsx)

### 11.2 Real-Time SVG Throughput Charting
In `ObservabilityPanel.tsx`, throughput history is tracked in an array of points `(time, decode_tps)`. An SVG `<path>` dynamically updates in real time using cubic bezier curves without external heavy charting libraries, visualizing smooth generation bursts and prefill transitions.
