# ggmlc Architecture Decisions & Strategic Roadmap

This document outlines the architectural foundation, core design decisions, and future technical roadmap for `ggmlc`.

---

## 1. Core Architectural Pillars

### A. Intended Use & Target Scenarios
`ggmlc` bridges the gap between high-level neural network research (PyTorch, JAX) and high-performance, zero-dependency CPU/edge deployment (GGML).

- **Pain Point Addressed**: Eliminates the weeks of manual C++ graph-coding and stride-math currently required to port new or custom neural network architectures to GGML/`llama.cpp`.
- **Target Persona**: AI researchers, edge ML engineers, and application developers deploying custom architectures across servers, laptops, mobile devices, and WebAssembly.

---

### B. Strategic Architectural Decisions

#### Decision 1: GGUF as the Primary Container Format
- **Decision**: Adopt the standard **GGUF binary format** as the official packaging format rather than an isolated proprietary container.
- **Implementation**:
  - `general.architecture = "ggmlc"`
  - `ggmlc.version = "1.0.0"`
  - `ggmlc.graph_spec = "{ ... }"` (graph metadata stored in GGUF header)
  - Tensors stored using standard GGUF memory-aligned buffers (FP32, Q8_0, Q4_0, Q4_K).
- **Rationale**: Immediate compatibility with Hugging Face Hub, `gguf-py`, existing quantization tooling, and zero-copy memory mapping (`mmap`).

#### Decision 2: Dual Output Engine (Generic Runtime + Human-Readable C++ Codegen)
- **Decision**: Provide two compilation targets:
  1. **Direct GGUF Compilation**: Portable `.gguf` executable by the generic C++ runtime without recompilation.
  2. **Human-Readable C++ Code Generation**: Clean, structured `model.cpp` source code implementing the model's `build_graph()` function.
- **Rationale**:
  - Complete transparency and inspectability for engineers.
  - Serves as the foundation for **Bring Your Own Kernel (BYOK)** and **Agentic Custom Kernel Optimization**.
  - Provides a drafting tool for upstreaming model architectures directly to `llama.cpp`.

#### Decision 3: Clean Separation of Compiler and Runtime
- **Decision**:
  - **`ggmlc` (Compiler)**: Python-based frontend, IR passes, shape inferencing, quantization, and codegen.
  - **`ggmlc-runtime` (Engine)**: Pure C++17 library depending exclusively on GGML with **zero Python / PyTorch dependencies**.
- **Rationale**: Allows packaging a lightweight standalone C++ shared library (`libggmlc_runtime.so` / `.dll`) and creating lightweight bindings across Python, Rust (`ggmlc-rs`), Swift, Android (JNI), and WebAssembly.

#### Decision 4: Cross-Framework Semantic Convergence via Transformation Passes (Not Ingestion Isomorphism)
- **Decision**: Target **strict runtime numerical parity and semantic convergence** across disparate ML frontends (PyTorch ATen vs JAX/XLA primitives) rather than forcing 100% graph isomorphism at the raw Canonical IR ingestion boundary. Exploit structural differences to drive robust pattern matching and operator fusion optimization passes.
- **Rationale**:
  - `torch.export` captures graphs at the mid-to-coarse ATen operator level (`aten.layer_norm`, `aten._softmax`, `aten.relu`, `aten.scaled_dot_product_attention`).
  - `jax.make_jaxpr` decomposes graphs into RISC-style mathematical primitives (`lax.reduce_sum`, `lax.max`, `lax.exp`, `lax.div`, `lax.broadcast_in_dim`).
  - Forcing the frontend importer to synthetically reconstruct all fine-grained math into coarse ATen-like nodes creates brittle, non-composable heuristics.
  - Instead, Canonical IR natively supports multi-level representations. The **Optimization Pipeline (`transforms/fusion.py`)** pattern-matches decomposed subgraphs (Softmax, LayerNorm, RMSNorm, SwiGLU, BiasGELU, Conv+ReLU) into high-performance fused operators, benefiting both JAX and PyTorch execution paths.
  - Guarantees exact runtime numerical parity ($\Delta = 0.0$) on the GGML engine while maximizing hardware execution throughput.

---

## 2. Technical Roadmap & Strategic Priorities

```
+---------------------------------------------------------------------------------------+
|                                        ggmlc                                          |
+---------------------------------------------------------------------------------------+
       |                               |                               |
       v                               v                               v
 [ Phase 1: Core & Hardware ]   [ Phase 2: Massive Model Zoo ]   [ Phase 3: Runtime Pipelines ]
 - PyTorch & JAX Frontends      - Extensive PyTorch Zoo          - Pre-Processing Pipeline
 - Canonical IR & Optimizations - Comprehensive JAX/Flax Zoo     - Post-Processing & Sampling
 - GGUF v3 Binary Container     - 25+ Diverse Architectures      - Tokenizer / Audio Mel Frontend
 - Native CPU & CUDA Engine     - Edge-Cases & Dyn Shapes        - Low-Level Nanobind Tensor API
                                                                        |
                                                                        v
                                                         [ Phase 4: Long-Term Ecosystem ]
                                                         - Deferred Foreign Bindings (Rust)
                                                         - Speculative Decoding & Paged Attn
                                                         - BYOK Agentic Custom Kernels
```

---

### Phase 1: Core Compiler & Hardware Foundations (Completed ✅)
- **PyTorch (`torch.export`) & JAX (`jaxpr`) Frontends**: Standardized functional ingestion.
- **Canonical IR & Dialect Lowering**: Strongly typed graph DAG, symbolic shape engine, operator fusion passes.
- **GGUF v3 Container**: Streamed serialization with embedded execution graph metadata.
- **Dual Hardware Execution**: Native Windows & Linux/WSL support for multi-threaded CPU and NVIDIA CUDA GPU.
- **Continuous Benchmarking Harness**: Automated performance latency and differential accuracy testing.

---

### Phase 2: Massive Multi-Model Expansion & Differential Testing (Active Priority 🎯)

Our primary value proposition is that **any PyTorch, JAX, or Flax neural network can be automatically compiled into high-performance GGML execution graphs with guaranteed 100% numerical parity**. 11 architectures are a strong foundation, but establishing true industrial confidence requires extensive coverage across all major model families in both PyTorch and JAX/Flax.

#### A. Expanded PyTorch Model Coverage
1. **Advanced Vision Architectures**:
   - **ConvNeXt** (`convnext_tiny` / `small`): 7x7 depthwise separable convolutions, inverted bottleneck, LayerNorm in channels-last/first formats.
   - **EfficientNet** (`efficientnet_b0` / `v2_s`): MBConv blocks, fused squeeze-and-excitation, Swish activations.
   - **Swin Transformer** (`swin_t`): Shifted window self-attention, cyclic rolling, relative position bias.
2. **Language & Modern SLMs**:
   - **Llama 3 / 3.2 (1B)**: GQA, RoPE, SwiGLU, RMSNorm with extended context windows.
   - **Phi-3 / Phi-3.5 Mini**: Partial RoPE embeddings, fused MLP projections.
   - **Gemma / Gemma 2 (2B)**: GeGLU, query pre-scaling, sliding window attention.
3. **Audio, Speech & Multimodal**:
   - **Wav2Vec 2.0 / HuBERT**: Raw waveform 1D feature extractors, multi-head contextual attention.
   - **CLIP** (Vision Transformer + Text Transformer): Multi-modal dual embeddings and cosine projection heads.

#### B. Comprehensive JAX / Flax Model Coverage
1. **Flax ResNet-18 / 50**: Convolutional residual blocks, BatchNorm folding, global average pooling.
2. **Flax Vision Transformer (ViT)**: Patch tokenization, class token prepend, multi-head self-attention.
3. **Flax GPT-2 / RoBERTa**: Causal language modeling and bidirectional masked token representations.
4. **Flax T5 / Dense Multi-Layer Models**: Relative position embeddings, gated MLP blocks.

---

### Phase 3: Runtime Pre/Post-Processing Pipeline Subsystem

Enable fully self-contained inference pipelines directly in `ggmlc` without requiring users to maintain heavy external image/audio/token manipulation scripts:

1. **Pre-Processing Pipeline**:
   - **Vision**: Bilinear/bicubic image resizing, aspect-ratio letterboxing, normalization (`(x - mean) / std`), channel permutations (NHWC $\leftrightarrow$ NCHW).
   - **Text**: Tokenization (BPE, WordPiece, Byte-level BPE) integrated with model inputs.
   - **Audio**: Log-Mel spectrogram computation directly from raw waveform audio.
2. **Post-Processing Pipeline**:
   - **Generation & Sampling**: Top-K, Top-P (nucleus), temperature scaling, repetition penalties, min-p sampling.
   - **Vision & Detection**: Softmax probabilities, bounding box regression unflattening, Non-Maximum Suppression (NMS).
   - **Text Detokenization**: Output token ID stream to UTF-8 decoded text.

---

### Phase 4: Low-Level Nanobind Tensor Operations API

In addition to the high-level `ModelRunner` (`runner(x)`), expose granular tensor-level GGML operators and execution contexts via Nanobind:

- **Interactive Experimentation & Debugging**: Call individual operators (`ggmlc.ops.matmul(a, b)`, `ggmlc.ops.norm(x, eps)`, `ggmlc.ops.flash_attn(q, k, v)`) directly on NumPy/CPU/GPU buffers.
- **Lightweight ML Operations**: Perform high-performance tensor mathematics on edge devices without loading PyTorch or heavy frameworks.
- **Micro-Benchmarking**: Profile individual operator kernels (e.g. testing different gemm/conv tile sizes on custom hardware).

---

### Phase 5: Long-Term Ecosystem Bindings & Advanced Optimizations (Deferred)

Foreign language bindings (Rust `ggmlc-rs`, WebAssembly, Swift) are strictly deferred until the core Python and C++ APIs are fully stabilized:

1. **Foreign Language Bindings**: Rust crate, WebAssembly browser runner, and C-FFI header once core features solidify.
2. **Advanced Transformer Optimizations**: Paged KV-cache block allocation and speculative decoding draft-target verification loops.
3. **Agentic Custom Kernel Optimizer (BYOK)**: Automated synthesis and compilation of custom SIMD/AVX2/CUDA micro-kernels for hot fused subgraphs.
