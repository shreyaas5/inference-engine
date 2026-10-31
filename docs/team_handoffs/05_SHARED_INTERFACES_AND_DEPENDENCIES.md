# `05_SHARED_INTERFACES_AND_DEPENDENCIES.md` — Cross-Module C++17 & Tooling Contracts

**Document Role:** Complete Interface Reference Across All Subsystems (`docs/team_handoffs/05_SHARED_INTERFACES_AND_DEPENDENCIES.md`)
**Consolidated Branch:** `integration-ps08-team-sync-2026-10-31`

---

## 1. Public Header Dependency DAG

```
include/inference_engine/common.hpp
  ├── include/inference_engine/config.hpp
  ├── include/inference_engine/quant.hpp
  │     └── include/inference_engine/tensor.hpp
  │           └── include/inference_engine/loader.hpp
  ├── include/inference_engine/thread_pool.hpp
  │     └── include/inference_engine/kernels.hpp   (also includes quant.hpp)
  ├── include/inference_engine/kv_cache.hpp        (includes config.hpp)
  │     └── include/inference_engine/transformer.hpp (includes loader.hpp, kv_cache.hpp, thread_pool.hpp)
  ├── include/inference_engine/tokenizer.hpp
  └── include/inference_engine/sampler.hpp
```

---

## 2. Exact C++17 Subsystem Interfaces

### 2.1 Common Status, DataType & Checked Arithmetic ([`common.hpp`](../../include/inference_engine/common.hpp))
- `enum class StatusCode : uint32_t { Ok = 0, InvalidArgument, IoError, FormatError, OutOfBounds, Unsupported, NotImplemented, InternalError };`
- `struct Status { StatusCode code; std::string message; bool ok() const noexcept; };`
- `enum class DataType : uint32_t { FP32 = 0, Q4_0 = 1 };`
  - `dtype_block_bytes(FP32) == 4`, `dtype_block_elements(FP32) == 1`
  - `dtype_block_bytes(Q4_0) == 18`, `dtype_block_elements(Q4_0) == 32`
- `enum class SimdBackend : uint32_t { Scalar = 0, Neon = 1, Avx2 = 2 };`
  - `SimdBackend detect_best_backend() noexcept;`
  - `bool is_backend_available(SimdBackend backend) noexcept;`
- Checked arithmetic helpers: `checked_add_u64(a, b, &out)`, `checked_mul_u64(a, b, &out)`, `align_up_u64(value, alignment, &out)`.

### 2.2 TensorView & ModelLoader ([`tensor.hpp`](../../include/inference_engine/tensor.hpp), [`loader.hpp`](../../include/inference_engine/loader.hpp))
- `class TensorView`:
  - `const float* data_f32() const noexcept;` (returns `nullptr` unless `dtype() == DataType::FP32`)
  - `const BlockQ4_0* data_q4_0() const noexcept;` (returns `nullptr` unless `dtype() == DataType::Q4_0`)
  - `DataType dtype() const noexcept;`, `uint32_t ndim() const noexcept;`, `const std::array<uint64_t, 4>& shape() const noexcept;`
  - `uint64_t num_elements() const noexcept;`, `uint64_t num_blocks() const noexcept;`, `uint64_t payload_nbytes() const noexcept;`
- `struct TransformerWeightBindings`:
  - `const TensorView* tok_embeddings`, `const TensorView* final_norm`, `const TensorView* lm_head`
  - `std::vector<LayerWeightBindings> layers` (`attn_norm`, `wq`, `wk`, `wv`, `wo`, `ffn_norm`, `w_gate`, `w_up`, `w_down`)
  - `DataType linear_dtype` (`FP32` or `Q4_0`)
- `class ModelLoader`:
  - `Status load_from_file(const std::string& path);` (POSIX `mmap` on macOS/Linux)
  - `Status load_from_buffer(const uint8_t* data, size_t size);`
  - `Status bind_transformer_weights(TransformerWeightBindings* out_bindings) const;`

### 2.3 Quantisation, SIMD Kernels & ThreadPool ([`quant.hpp`](../../include/inference_engine/quant.hpp), [`kernels.hpp`](../../include/inference_engine/kernels.hpp), [`thread_pool.hpp`](../../include/inference_engine/thread_pool.hpp))
- `struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };` (`18` bytes)
- `void quantize_row_q4_0(const float* src, BlockQ4_0* dst, size_t n);`
- `void dequantize_row_q4_0(const BlockQ4_0* src, float* dst, size_t n);`
- `class ThreadPool`:
  - `explicit ThreadPool(uint32_t num_threads);`
  - `void parallel_for(size_t num_tasks, const std::function<void(size_t, uint32_t)>& fn);` (dynamically chunks tasks into `num_workers * 4` ranges and rethrows worker exceptions via `std::exception_ptr`)
- Dispatched matrix-vector entry points used by `TransformerEngine`:
  - `void dispatch_matmul_f32(const float* weights, const float* input, float* output, size_t rows, size_t cols, SimdBackend backend, ThreadPool* pool = nullptr);`
  - `void dispatch_matmul_q4_0_f32(const BlockQ4_0* weights, const float* input, float* output, size_t rows, size_t cols, SimdBackend backend, ThreadPool* pool = nullptr);`

### 2.4 KV Cache & TransformerEngine ([`kv_cache.hpp`](../../include/inference_engine/kv_cache.hpp), [`transformer.hpp`](../../include/inference_engine/transformer.hpp))
- `class KvCache`:
  - `Status init(const ModelConfig& config);`
  - `Status append(uint32_t layer_idx, uint32_t pos, const float* k_data, const float* v_data);` (never increments `seq_len()`)
  - `Status commit_token() noexcept;` (advances `seq_len()` by 1 after all layers finish)
- `class TransformerEngine`:
  - `Status init(const ModelConfig& config, const ModelLoader* loader, SimdBackend backend = detect_best_backend(), uint32_t num_threads = 1);`
  - `Status forward_token(int32_t token_id, uint32_t pos, float* logits);`
  - `void reset_cache() noexcept;`

### 2.5 Tokenizer & Sampler ([`tokenizer.hpp`](../../include/inference_engine/tokenizer.hpp), [`sampler.hpp`](../../include/inference_engine/sampler.hpp))
- `class Tokenizer`:
  - `Status load(const std::string& model_path);`
  - `Status encode(const std::string& text, bool add_bos, bool add_eos, std::vector<int32_t>* out_ids) const;`
  - `Status decode(const std::vector<int32_t>& ids, std::string* out_text) const;`
  - `static std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true);`
- `class Sampler`:
  - `explicit Sampler(const SamplerConfig& config = {});`
  - `int32_t sample(const float* logits, size_t vocab_size);`
  - `static int32_t sample_greedy(const float* logits, size_t vocab_size) noexcept;`
