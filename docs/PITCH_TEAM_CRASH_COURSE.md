# BIOS V2 — Team Pitch Crash Course & Rehearsal Manual
## Final Technical Rehearsal & Engineering Defense Plan
**Project:** `inference-engine` | **Problem Statement:** PS08 (Own Inference Engine)  
**Total Rehearsal Duration:** 2 Hours 15 Minutes (4 Focused Modules)  
**Objective:** Prepare all four team members to independently explain, whiteboard, and defend every engineering decision to technical judges.

---

## 1. Team Speaker Assignments & Ownership Matrix

Every team member owns specific subsystems based strictly on their actual implementation history and committed code:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                TEAM RESPONSIBILITY MATRIX                              │
├──────────────┬──────────────────┬───────────────────────┬──────────────────────────────┤
│ Team Member  │ Role             │ Hardware Target       │ Technical Defense Ownership  │
├──────────────┼──────────────────┼───────────────────────┼──────────────────────────────┤
│ Member S     │ Team Lead &      │ Apple MacBook Air M4  │ 1. Project identity & scope  │
│ (Shreyaas)   │ System Architect │ (ARM64 NEON, 24 GB)   │ 2. Custom .ie v1 format      │
│              │                  │                       │ 3. Zero-copy mmap loader     │
│              │                  │                       │ 4. Inference Studio & Worker │
├──────────────┼──────────────────┼───────────────────────┼──────────────────────────────┤
│ Member I     │ Quantisation &   │ AMD Ryzen 7 7745HX    │ 1. Q4_0 block packing math   │
│ (Ishaan)     │ CPU Optimisation │ (x86 AVX2/FMA, 16 GB) │ 2. NEON & AVX2 intrinsics    │
│              │                  │                       │ 3. 2-row register tiling     │
│              │                  │                       │ 4. ThreadPool task chunking  │
├──────────────┼──────────────────┼───────────────────────┼──────────────────────────────┤
│ Member U     │ Transformer &    │ Intel Core Ultra 7    │ 1. 16-step forward pass      │
│ (Utkarsh)    │ Scalar Reference │ (x86 AVX2/FMA, 16 GB) │ 2. Causal 8:1 GQA math       │
│              │                  │                       │ 3. RoPE precomputed tables   │
│              │                  │                       │ 4. KvCache commit contract   │
├──────────────┼──────────────────┼───────────────────────┼──────────────────────────────┤
│ Member P     │ Tokenizer,       │ Intel Core i5-10210U  │ 1. SentencePiece integration │
│ (Pratyush)   │ Runtime & Sampler│ (x86 AVX2, 8 GB)      │ 2. Zephyr chat template      │
│              │                  │                       │ 3. Sampler (top-p, top-k)    │
│              │                  │                       │ 4. REPL & CI automation      │
└──────────────┴──────────────────┴───────────────────────┴──────────────────────────────┘
```

---

# MODULE 1: The 30-Minute Complete Overview
*(Goal: Align all 4 members on the high-level narrative, project numbers, and core pitch positioning).*

### Minutes 0–5: The Opening Hook & Problem Statement
- **Speaker:** **Member S**
- **Core Message:** "We built `inference-engine`, a zero-framework, hardware-optimised C++17 CPU inference engine running a real 1.1-billion-parameter language model (`TinyLlama-1.1B-Chat-v1.0`) on commodity laptops. We achieve **35.5 tokens/sec decode throughput** and a **600 ms TTFT** in **658 MiB of RAM** without linking or wrapping PyTorch, llama.cpp, Ollama, or ONNX."
- **Why It Matters:** Most "local AI" projects are shallow Electron shells around llama.cpp. We own the binary format, virtual memory mapping, SIMD vector assembly, KV cache, and thread scheduling from scratch.

### Minutes 5–12: The Five Numbers Every Member Must Know by Heart
1. **Model Parameters:** **$1,100,048,384$** (1.100 billion parameters, 22 layers, hidden dim 2048, intermediate dim 5632, GQA 32:4, vocab 32000).
2. **Q4_0 Compression Ratio:** **$7.11\times$** ($4.40\text{ GB}$ FP32 down to **$619.11\text{ MB}$** pure Q4_0; $4.5\text{ bits/weight}$).
3. **Decode Throughput:** **$\sim 35.5\text{ tokens/sec}$** on consumer laptop CPUs (M4 NEON 4 threads).
4. **Time To First Token (TTFT):** **$\sim 600\text{ ms}$** for 20-token prompt prefill.
5. **Peak Physical RSS:** **$658.31\text{ MiB}$** (measured from OS kernel via `mach_task_basic_info` / `/proc/self/statm`).

### Minutes 12–20: Subsystem Division of Labor
- **Member S:** Focuses on how data enters the engine: `.ie` binary layout, 64-byte payload alignment, `mmap` zero-copy virtual memory, and the persistent worker architecture connecting to Inference Studio.
- **Member I:** Focuses on how math executes in hardware: 18-byte `BlockQ4_0` structs, split-half nibble unpacking, fused SIMD multiply-accumulate on NEON and AVX2, and 2-row register tiling.
- **Member U:** Focuses on how neural activations flow: the 16-step Transformer decoder pass, 8:1 Grouped Query Attention, in-place half-dim RoPE, RMSNorm, SwiGLU, and `KvCache` commit semantics.
- **Member P:** Focuses on the user interaction: SentencePiece tokenization, Zephyr chat formatting, Top-P/Top-K sampling with deterministic `xorshift*`, and multi-turn KV cache prefix reuse.

### Minutes 20–30: What We Must NEVER Claim (Anti-Hallucination Rules)
- **Rule 1:** NEVER claim we support GPU, Metal, or CUDA. We are proudly **100% CPU-only**.
- **Rule 2:** NEVER claim we wrote SentencePiece from scratch. We integrated Google's `sentencepiece-static` library and wrote our own dual-processor wrapper for chat tokens.
- **Rule 3:** NEVER claim Q4_0 is 8x smaller. It is **7.1x smaller** due to the FP16 scale per 32 weights and preserving RMSNorm in FP32.
- **Rule 4:** NEVER claim AVX2 full-model inference throughput is identical to NEON until physically run on x86 hardware. AVX2 kernel microbenchmarks are verified on Ryzen 7, but E2E numbers come from our Apple Silicon runs.

---

# MODULE 2: The 60-Minute Deep Technical Walkthrough
*(Goal: Trace the entire source code line-by-line so any member can answer deep architectural questions).*

### Minutes 30–45: Binary Format, Loader & Virtual Memory
- **Lead Explainer:** **Member S**
- **Files to Open:** `include/inference_engine/loader.hpp`, `src/loader/loader.cpp:87-300`.
- **Key Concepts to Master:**
  - *Why not read safetensors directly?* Safetensors has a variable-length JSON header without 64-byte alignment, forcing memory copies to align Q4_0 vector loads.
  - *Why 64-byte alignment?* Matches CPU L1 cache line boundaries and 512-bit SIMD registers, preventing split-cache-line penalties.
  - *The 10-Point Validation Gauntlet:* Walk through magic `0x464D4549`, version `1`, `checked_add_u64()`, non-overlap interval sorting, and exact payload size checks.
  - *Zero-Copy `mmap`:* Explain how `mmap` returns a pointer into the OS page cache with zero heap allocations, loading the model in 0.34 ms.

### Minutes 45–60: Q4_0 Quantization & SIMD Vector Kernels
- **Lead Explainer:** **Member I**
- **Files to Open:** `include/inference_engine/quant.hpp`, `src/kernels/quant_q4_0.cpp`, `src/kernels/neon/kernels_neon.cpp`, `src/kernels/avx2/kernels_avx2.cpp`.
- **Key Concepts to Master:**
  - *Block Structure:* 18 bytes = 2-byte FP16 scale $d$ + 16 packed bytes `qs[0..15]`.
  - *Split-Half Nibble Layout:* Low nibble `qs[j] & 0x0F` stores weight $j$ ($0..15$); high nibble `qs[j] >> 4` stores weight $j+16$ ($16..31$). Enables single-instruction 16-byte SIMD loads.
  - *Dequantization Formula:* $w_i = (q_i - 8) \times \text{fp16\_to\_fp32}(d)$.
  - *Vectorized Scale Accumulation:* Explain why deferring horizontal addition to the row end eliminates 64 reduction stalls per row.
  - *2-Row Register Tiling:* Explain how loading 32 floats of activation vector $x$ once into registers and computing Row 0 and Row 1 concurrently cuts cache load port traffic by 50%.
  - *ThreadPool Dynamic Chunking:* Explain why chunking tasks as $\text{tasks} / (\text{threads} \times 4)$ eliminates mutex convoying across 32,000 matrix rows.

### Minutes 60–75: 16-Step Transformer Forward Pass & KV Cache
- **Lead Explainer:** **Member U**
- **Files to Open:** `src/transformer/transformer.cpp:124-245`, `src/core/kv_cache.cpp`, `src/kernels/scalar/kernels_scalar.cpp`.
- **Key Concepts to Master:**
  - *Pre-allocated Workspace:* Explain that $x$, $q$, $k$, $v$, gate, up, and scores buffers are allocated once in `init()` (~393 KB total), guaranteeing zero heap allocation during `forward_token()`.
  - *The 16 Steps:* Recite the 16 steps in exact chronological order (Embed $\to$ RMSNorm $\to$ Q/K/V Projections $\to$ RoPE $\to$ KV Append $\to$ GQA $\to$ O Projection $\to$ Residual $\to$ RMSNorm $\to$ Gate/Up Projections $\to$ SwiGLU $\to$ Down Projection $\to$ Residual $\to$ Commit Token $\to$ Final RMSNorm $\to$ LM Head).
  - *GQA 8:1 Math:* Show that query head $h_q$ reads KV head $\lfloor h_q / 8 \rfloor$, reducing KV cache from 738 MB to 88 MB.
  - *RoPE Precomputed Tables:* Explain how precalculating $2048 \times 32$ floats of $\cos$ and $\sin$ in `init()` eliminates runtime transcendental calls.
  - *KV Cache Commit Contract:* Explain why `append()` never modifies sequence length, and only `commit_token()` advances it after all 22 layers finish.

### Minutes 75–90: Tokenizer, Sampler, Multi-Turn REPL & Inference Studio
- **Lead Explainer:** **Member P** (with Member S on Studio)
- **Files to Open:** `src/tokenizer/tokenizer.cpp`, `src/sampler/sampler.cpp`, `src/app/main.cpp:600-720`, `src/app/studio_worker.hpp`.
- **Key Concepts to Master:**
  - *SentencePiece Dual Processor:* Explain why `processor_no_prefix_` is required to prevent spurious space insertion after chat control tokens (`<|user|>`).
  - *Sampling Pipeline:* Sanitization $\to$ Temperature scaling $\to$ Top-K candidate truncation $\to$ Top-P nucleus cumulative filtering $\to$ Seeded 64-bit `xorshift*` random draw.
  - *Incremental Prefix Detokenization:* Explain how finding the common prefix between previous and current decoded strings prevents broken multi-byte UTF-8 glyphs.
  - *KV Cache Prefix Matching:* Walk through `committed_tokens` prefix length matching and `KvCache::truncate()`, proving how Turn 2 reuses 30+ tokens in $<30\text{ ms}$.
  - *Studio Worker & Node.js Gateway:* Explain the persistent worker subprocess (`--studio-worker`), JSON-lines stdio protocol, atomic cancellation, and Server-Sent Events.

---

# MODULE 3: The 30-Minute Judge-Question Drill
*(Goal: Rapid-fire Q&A practice simulating aggressive, adversarial technical judge interrogations).*

### Drill Format:
- The team takes turns firing questions at each other.
- The responder must give a crisp **20-second pitch answer**, followed by the **deep architectural proof**, and name the exact file.

#### Drill Round 1 (Member S Defends System Architecture):
1. *"Why didn't you just wrap llama.cpp?"*  
   **Response:** "Problem statement PS08 mandates building our **own** engine. Wrapping llama.cpp defeats the competition. Our binary links to zero external ML libraries; every parser, SIMD loop, and cache manager is written from scratch in our repository."
2. *"Does mmap mean the entire 619 MB model is occupying physical RAM?"*  
   **Response:** "No. `mmap` assigns virtual memory addresses only. The OS demand-pages 4 KB blocks into the physical page cache on demand. Our resident RAM is 658 MiB, measured directly from kernel Mach task ports."

#### Drill Round 2 (Member I Defends Quantization & CPU SIMD):
3. *"Why is Q4_0 not exactly 8x smaller than FP32?"*  
   **Response:** "Because each 32-weight block requires an 18-byte struct including an FP16 scale factor ($18 \times 8 / 32 = 4.5\text{ bits/weight} = 7.11\times$), and all 45 RMSNorm tensors are kept in full 32-bit float to preserve scale precision."
4. *"If Q4_0 requires CPU cycles to unpack nibbles, why is it faster than FP32?"*  
   **Response:** "Single-token decoding is memory-bandwidth bound. FP32 must stream 4.40 GB across the memory bus per token, bottlenecking at 9 tok/s. Q4_0 streams only 0.62 GB per token. The CPU has abundant spare ALU cycles to unpack nibbles while waiting for RAM, yielding 35.5 tok/s."

#### Drill Round 3 (Member U Defends Transformer & Numerical Parity):
5. *"How do you prove your output is mathematically correct?"*  
   **Response:** "Automated golden tests in `test_golden_forward.cpp` compare C++ forward activations against Hugging Face PyTorch references with $<10^{-6}$ error on FP32 and $>0.996$ cosine similarity on Q4_0."
6. *"What happens if sequence length reaches 2048?"*  
   **Response:** "The engine checks `pos >= config_.max_seq_len` and returns `StatusCode::OutOfBounds`. We never silently overwrite older tokens with a naive ring buffer because that corrupts RoPE positional angles."

#### Drill Round 4 (Member P Defends Tokenizer, Sampler & REPL):
7. *"How do you ensure your SentencePiece token IDs match Hugging Face?"*  
   **Response:** "We run `tools/verify_tokenizer_parity.py` against Hugging Face across 15 complex prompts. Our dual-processor wrapper handles special token boundaries, ensuring 100% bit-exact parity."
8. *"How does conversational memory work across multiple turns?"*  
   **Response:** "We track committed token IDs. On each turn, we compute the common prefix between history and cache, truncate the cache if necessary, and prefill only the new tokens. On Turn 2, 30+ tokens are reused in $<30\text{ ms}$."

---

# MODULE 4: The 15-Minute Architecture Whiteboard Exercise
*(Goal: Every team member can walk up to a physical whiteboard and draw the system architecture, tensor flow, and Q4_0 block layout from memory).*

### Whiteboard Drawing 1: System Hierarchy (Member S)
```
┌─────────────────────────────────────────────────────────────┐
│                 REACT 19 OBSERVABILITY STUDIO               │
│      Chat Playground  │  Benchmark Lab  │  Model Inspector  │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP SSE (/api/events) + REST
┌──────────────────────────────▼──────────────────────────────┐
│            NODE.JS GATEWAY & PROCESS SUPERVISOR             │
│            studio/server/index.ts & engineManager.ts        │
└──────────────────────────────┬──────────────────────────────┘
                               │ Bidirectional Stdio Pipes (JSON-Lines)
┌──────────────────────────────▼──────────────────────────────┐
│            STANDALONE C++17 INFERENCE ENGINE                │
│            build/inference-engine --studio-worker           │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │     ModelLoader       │       │       Tokenizer       │  │
│  │ POSIX mmap / .ie v1   │       │ SentencePiece Static  │  │
│  └───────────┬───────────┘       └───────────┬───────────┘  │
│              │ Typed Views                   │ Token IDs    │
│              ▼                               ▼              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  TransformerEngine                    │  │
│  │  16-Step Forward Pass (22 Layers, GQA 8:1, SwiGLU)    │  │
│  └───────────┬───────────────────────────────┬───────────┘  │
│              │ Intermediate KV               │ Projections  │
│              ▼                               ▼              │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │        KvCache        │       │   SIMD Dispatcher     │  │
│  │ Flat Float [22,2048,  │       │ ARM NEON / x86 AVX2   │  │
│  │             4,64]     │       │ 2-Row Tiled + Threads │  │
│  └───────────────────────┘       └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Whiteboard Drawing 2: One Transformer Decoder Layer (Member U)
```
Input Activation: x [2048]
  │
  ├─────────────────────────────────────────────┐  (Residual Stream)
  ▼                                             │
Pre-Attn RMSNorm: u = RMSNorm(x, w_norm, eps)   │
  │                                             │
  ├─► W_q · u [2048] ──► RoPE ──┐               │
  ├─► W_k · u [256]  ──► RoPE ──┼─► KvCache     │
  └─► W_v · u [256]  ───────────┼─► KvCache     │
                                │               │
Causal 8:1 GQA Attention ◄──────┘               │
  │  (AttnOut = Softmax(q·k / 8) · v)           │
  ▼                                             │
Attention Output Projection: W_o · AttnOut      │
  │                                             │
  ▼                                             │
Residual Addition 1: x = x + W_o_proj ◄─────────┘
  │
  ├─────────────────────────────────────────────┐  (Residual Stream)
  ▼                                             │
Post-Attn RMSNorm: m = RMSNorm(x, w_post, eps)  │
  │                                             │
  ├─► Gate Projection: W_gate · m [5632] ──┐    │
  └─► Up Projection:   W_up · m   [5632] ──┼─► SwiGLU: SiLU(gate) * up
                                           │
Down Projection: W_down · ffn_act [2048] ◄─┘
  │
  ▼
Residual Addition 2: x = x + W_down_proj ◄──────┘
  │
  ▼
Output to Layer i + 1: x [2048]
```

### Whiteboard Drawing 3: Q4_0 Block Struct & Nibble Layout (Member I)
```
BlockQ4_0 (Total: 18 Bytes packed, representing 32 weights)
┌──────────────────┬────────────────────────────────────────────────────────┐
│  d (Bytes 0..1)  │                  qs[16] (Bytes 2..17)                  │
│  IEEE 754 FP16   │      16 bytes containing 32 packed 4-bit nibbles       │
└──────────────────┴────────────────────────────────────────────────────────┘

Inside qs[j] (Byte index j from 0 to 15):
┌──────────────────────────────────┬──────────────────────────────────┐
│     High Nibble: qs[j] >> 4      │      Low Nibble: qs[j] & 0x0F    │
│    Weight Index j + 16 (16..31)  │      Weight Index j (0..15)      │
└──────────────────────────────────┴──────────────────────────────────┘

Dequantization Formula:
w_j      = ( (qs[j] & 0x0F) - 8 ) * fp16_to_fp32(d)
w_{j+16} = ( (qs[j] >> 4)   - 8 ) * fp16_to_fp32(d)
```

---

## Pitch Day Checklist
- [ ] Clean build passes: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4` (0 warnings).
- [ ] All 9 CTest suites pass: `ctest --test-dir build --output-on-failure` (100% pass rate).
- [ ] Model files verified in `models/`: `tinyllama-1.1b-q4_0.ie` (619 MB) and `tokenizer.model` (584 KB).
- [ ] Standalone CLI tested: `./build/inference-engine --info`.
- [ ] Workstation verified: `cd studio && npm run build && npm start` (serving on `http://localhost:3000`).
- [ ] Live demo chat verified: 2 turns showing instant prefix reuse on Turn 2.
- [ ] Benchmark Lab verified: 3-run sweep executing live showing ~35.5 tok/s decode.
