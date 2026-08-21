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

---

## 2. Technical Roadmap

```
+-------------------------------------------------------------------------------+
|                                    ggmlc                                      |
+-------------------------------------------------------------------------------+
       |                               |                                |
       v                               v                                v
 [ Phase 1: Core (Done) ]   [ Phase 2: Packaging & Codegen ]   [ Phase 3: Hardware & Bindings ]
 - PyTorch / JAX Importers   - GGUF Container Integration       - Hardware Backends (CUDA, Metal)
 - Canonical IR & Dialect    - Human-Readable C++ Codegen       - Multi-Language Bindings (Rust, JS)
 - Fused SIMD Kernels        - BYOK & Agentic Kernel Pipeline   - Paged KV Cache Attention
 - 100% Numerical Parity     - CLI One-Line Compilation         - Speculative Decoding
```

---

### Phase 2: GGUF Integration & Human-Readable C++ Codegen (Next Milestones)

1. **GGUF Container Serialization & Loader**:
   - Save execution graph and tensor weights into standard GGUF files.
   - Update C++ `ModelLoader` to parse `ggmlc.graph_spec` metadata and bind GGUF tensor data directly to GGML execution graphs.
2. **Human-Readable C++ Code Generator (`ggmlc.codegen`)**:
   - Transpile Canonical/GGML dialect graphs into formatted, commented C++ code.
   - Layer-scoped variable naming (`x_norm`, `q_proj`, `attn_scores`, `mlp_out`).
   - Ready-to-compile standalone CMake project.
3. **Single-Command CLI Workflow**:
   - `ggmlc compile model.pt --format gguf -o model.gguf`
   - `ggmlc compile model.pt --format cpp -o ./build_graph.cpp`

---

### Phase 3: Hardware Acceleration & Ecosystem Bindings

1. **Multi-Language Runtime Bindings**:
   - `ggmlc-rs`: Rust bindings to `libggmlc_runtime`.
   - `ggmlc-py-runtime`: Minimal Python C-FFI runtime without PyTorch.
   - Mobile & WASM wrappers.
2. **Hardware Acceleration**:
   - Integration with `ggml-cuda`, `ggml-metal`, and `ggml-vulkan` backends.
3. **Agentic Kernel Optimizer**:
   - Interactive optimization loop: autonomous profiling, generating custom C++ SIMD kernels, and verifying numerical parity.
