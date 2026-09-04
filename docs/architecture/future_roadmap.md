# ggmlc Architecture & Future Roadmap

This document covers how `ggmlc` is built today, why it was designed this way, and what we're building next.

---

## 1. Core Architecture Decisions

### GGUF as the Native Container
Instead of inventing a custom file format, `ggmlc` outputs standard **GGUF v3** files:
- Header metadata stores the entire execution graph spec (`ggmlc.graph_spec` JSON), tensor descriptors, dynamic shapes, and storage classes.
- Weight tensors are stored in memory-aligned binary buffers (supporting FP32, Q8_0, Q4_0, and k-quants).
- **Why this matters**: Immediate compatibility with existing tooling (`gguf-py`, Hugging Face Hub), zero-copy memory mapping (`mmap`), and fast load times.

### Dual Targets: Generic Runtime + Standalone C++ Project Export
When you compile a model in `ggmlc`, you can target two modes:
1. **GGUF Binary**: Streamed to disk and executed by the generic `ggmlc-runtime` / Python runner without any recompilation.
2. **Standalone C++ Project (`ggmlc.codegen`)**: Generates a self-contained directory containing `<Model>.h`, `ggmlc_main.cpp`, and `CMakeLists.txt`.
- **Why this matters**:
  - Full transparency: You can inspect every line of C++ graph construction and stride math.
  - Zero Python dependencies at deployment: Compile directly with MSVC, GCC, or Clang.
  - Foundation for **BYOK (Bring Your Own Kernel)** and **Agentic Custom Kernel Optimization**, letting developers (or AI coding agents) swap in hand-tuned or synthesized micro-kernels for hot fused subgraphs.

### Clean Separation Between Compiler and Runtime
- **`ggmlc` (Compiler)**: Python-based frontend, Canonical IR functional DAG, shape inferencing, quantization, optimization passes, and C++ code generator.
- **`ggmlc-runtime` (Execution Engine)**: Pure C++17 library linking against GGML. Zero dependency on Python, PyTorch, or libtorch.
- **Why this matters**: Makes it trivial to embed into native desktop apps, server binaries, mobile apps (Android/iOS), or foreign language bindings (Rust, Swift, WASM).

### Semantic Convergence via Passes (Not Brittle Ingestion Hacks)
Different ML frameworks represent the same computation differently:
- PyTorch (`torch.export`) emits mid-level ATen ops (`aten.layer_norm`, `aten._softmax`, `aten.scaled_dot_product_attention`).
- JAX (`jax.make_jaxpr`) decomposes math into RISC-style primitives (`lax.reduce_sum`, `lax.max`, `lax.exp`, `lax.div`).

Rather than trying to force both frontends into identical ASTs at the ingestion boundary, `ggmlc` ingests them cleanly into Canonical IR and relies on the **Optimization Pipeline (`transforms/fusion.py`)** to pattern-match decomposed subgraphs (LayerNorm, RMSNorm, Softmax, SwiGLU, BiasGELU, Conv+ReLU) into fused execution kernels. Both frontends converge to the same high-performance GGML execution graph with exact numerical parity.

---

## 2. Living Baseline: Continuous Model Zoo & Differential Parity

- **Current Zoo (30 validated models)**: Spans Vision-CNN (ResNet, MobileNetV3, ConvNeXt, EfficientNet, DenseNet, RegNet), Detection (SSDLite), Vision Transformers (ViT-B/16), Text Embeddings (MiniLM, BGE-M3), Encoders (BERT, DistilBERT), SLMs (GPT-2, Qwen-2.5-0.5B, Gemma 3), Audio Seq2Seq (Whisper-Tiny Encoder & Decoder), and Multimodal Vision-Language (OpenAI CLIP Vision, Text, and End-to-End Similarity).

---

## 3. Implemented Capabilities & What We're Building Next

### ✅ Completed: End-to-End Pipelines & Pre/Post-Processing
- **Vision Preprocessing Subsystem (`ggmlc.pipeline.vision`)**:
  - Continuous bicubic, bilinear, nearest resampling.
  - Aspect-ratio letterboxing, center crop, normalization (`(x - mean) / std`), channel permutations (NCHW $\leftrightarrow$ NHWC).
  - Introspection and mathematical parity verification against `torchvision.transforms` (`from_torchvision`, `verify_torchvision_parity`).
  - Native C++ engine (`stb_image` decode + fast SIMD bicubic interpolation + normalization) in `runtime/src/pipeline/image.cpp`.
- **Tokenization Subsystem (`ggmlc.pipeline.tokenizer`)**:
  - Byte-level Byte-Pair Encoding (BPE) for CLIP and GPT-2, WordPiece for BERT.
  - Automatic metadata export to standard GGUF tokenizer keys (`tokenizer.ggml.*`).
  - Native C++ tokenizer engine with regex pre-tokenization in `runtime/src/pipeline/tokenizer.cpp`.
  - Introspection from Hugging Face processors/tokenizers (`from_huggingface`, `from_huggingface_tokenizer`).
- **Raw Input Ingestion in `ggmlc-run`**:
  - Direct execution from CLI with `--image <path>` and `--text <string>`.

---

## 4. Next Priorities On The Roadmap

### B. Low-Level Nanobind Tensor Ops API
In addition to the high-level model runner (`runner(x)`), we will expose granular, direct Python bindings to GGML tensor operations:
- **Interactive Prototyping & Debugging**: Call individual operators (`ggmlc.ops.matmul(a, b)`, `ggmlc.ops.norm(x, eps)`, `ggmlc.ops.flash_attn(q, k, v)`) directly on NumPy / host / CUDA buffers.
- **Micro-Benchmarking**: Profile and test standalone kernel efficiency and memory bandwidth on custom hardware without setting up full graphs.
- **Lightweight Tensor Math**: Fast native CPU/GPU array operations on edge devices without needing heavy frameworks installed.

### C. Bring Your Own Kernel (BYOK) & Agentic Kernel Optimization
The standalone C++ export creates a direct path for custom hardware acceleration:
- **Manual BYOK**: Drop custom SIMD (AVX2, AVX-512, NEON) or CUDA kernels directly into the generated C++ project.
- **Agentic Optimization**: AI coding agents can analyze the exported C++ graph, detect performance bottlenecks in hot fused subgraphs, synthesize optimized micro-kernels, compile, benchmark them in a loop, and integrate the fastest variant back into the project.

### D. Hardware Targets & Ecosystem Integration
1. **Apple Silicon Metal (MPS)**:
   - Native `ggml-metal` backend with unified memory zero-copy buffers on macOS and iOS.
2. **WebAssembly & WebGPU**:
   - In-browser execution via Emscripten and WebGPU compute shaders for zero-server-cost client inference.
3. **Rust Bindings (`ggmlc-rs`)**:
   - Safe, idiomatic Rust crate wrapping the native C++ runtime for systems and backend applications.
4. **Mobile Platforms (Android NDK & iOS)**:
   - Packaged static/shared libraries with JNI and Swift wrappers for on-device mobile deployment.
5. **AMD ROCm (HIP) & Vulkan**:
   - Broad GPU compatibility across AMD Radeon, Intel Arc, and integrated GPUs.

### E. Advanced Quantization & Graph Optimizations
1. **Advanced PTQ (AWQ & GPTQ)**:
   - Activation-aware weight quantization and second-order error compensation integrated directly into Canonical IR passes.
2. **Mixed-Precision K-Quants**:
   - Automated per-tensor sensitivity analysis selecting the optimal GGML k-quant mix (`Q4_K_M`, `Q5_K_M`, `Q6_K`) to maximize throughput while minimizing perplexity loss.
3. **Attention Kernel Specialization**:
   - FlashAttention v2/v3 kernels for long-context execution ($L > 2048$).
   - Sliding-window and chunked local attention lowerings (e.g. for Gemma 3 and Mistral architectures).
4. **Static Arena Planning**:
   - Interval-graph lifetime analysis to pre-allocate a single static scratchpad buffer, eliminating all runtime memory allocations during forward passes.

### F. High-Throughput Serving & Dynamic KV-Cache
1. **Paged KV-Cache**:
   - Block-allocated virtual memory management for autoregressive decoders, eliminating VRAM fragmentation during multi-turn generation.
2. **Continuous Dynamic Batching**:
   - Iteration-level scheduling and request preemption for concurrent multi-user serving.
3. **Speculative Decoding Loops**:
   - Paired execution of lightweight draft SLMs (e.g. Qwen-2.5-0.5B) with larger target models to accelerate token generation throughput.

### G. Multimodal & Extended Frontends
1. **ONNX & SafeTensors Ingestion**:
   - Direct parsing of standard ONNX computational graphs into Canonical IR.
2. **Vision-Language & Multimodal Models**:
   - Cross-attention and vision-projection architectures (CLIP, LLaVA, Florence-2).
3. **Diffusion Architectures**:
   - Ingestion and optimization of UNet and Diffusion Transformer (DiT) backbones.

---

> [!NOTE]
> The tracks and features outlined above represent our technical priorities and intended design direction. They will not necessarily be implemented in this exact sequential order and will adapt flexibly based on hardware updates, ecosystem shifts, and community contributions.



