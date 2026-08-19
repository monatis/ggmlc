# ggmlc Architecture Decisions & Future Roadmap

This document outlines key design decisions taken in `ggmlc`, architectural invariants, and the technical roadmap for future iterations.

---

## 1. Key Architectural Decisions

### Decision 1: Generic Graph Interpreter vs. Code Generation
- **Decision**: Rather than transpiling neural network graphs into architecture-specific C++ source files that require compilation with `g++` / `clang++` for each model, `ggmlc` compiles graphs into a **portable binary container (`.ggmlc`)** executed by a single, generic runtime binary (`ggmlc-run` / `ggmlc::Executor`).
- **Rationale**:
  1. Instant compilation (< 0.5s for 24-layer LLMs).
  2. Safe dynamic rank/symbol interpretation without runtime recompilation.
  3. Single hardened C++ binary embedded cleanly into applications.

---

### Decision 2: Dialect Isolation & Frontend Independence
- **Decision**: Frontends produce Canonical IR only. Dialect lowerings translate Canonical IR to GGML semantics.
- **Rationale**:
  - Allows adding new frontends (e.g. ONNX, StableHLO, MLX) without modifying GGML lowering passes.
  - Allows adding new execution backends (e.g. ExecuTorch, WebGPU) without modifying PyTorch or JAX importers.

---

### Decision 3: StorageClass & Explicit Stateful Tensors
- **Decision**: All tensors explicitly specify their lifetime via `StorageClass`:
  - `INPUT`: Activation provided by caller.
  - `OUTPUT`: Activation returned to caller.
  - `PARAMETER`: Read-only weight from model checkpoint.
  - `CONSTANT`: Inlined tensor constant.
  - `ACTIVATION`: Ephemeral intermediate allocated in scratch arena.
  - `STATE`: Persistent buffer preserved across consecutive executions (e.g. KV Cache).
- **Rationale**: Enables zero-copy weight memory mapping, single-buffer scratch arena planning, and state persistence for multi-step autoregressive generation.

---

## 2. Future Technical Roadmap

```
  +-------------------------------------------------------------------------+
  |                               ggmlc                                     |
  +-------------------------------------------------------------------------+
       |                         |                          |
       v                         v                          v
  [ Phase 1: Core ]      [ Phase 2: Optimizations ]  [ Phase 3: Hardware ]
  - PyTorch FX export    - Pass Pipeline & Fusion   - CUDA / cuBLAS
  - JAX jaxpr export     - Constant Folding         - Apple Metal (MPS)
  - GGML Dialect         - Dead Code Elimination    - Vulkan / Kompute
  - Generic C++ Runtime  - INT4/INT8 Quantization   - Distributed Sharding
  - 5 Full Hub Models    - KV-Cache Paged Attention - Speculative Decoding
  (Completed)            (Next Milestones)          (Future Expansion)
```

---

### Phase 2: Compiler Optimization Pass Pipeline
1. **Pass Infrastructure**:
   - Sequential rewrite passes on Canonical IR:
     - `ConstantFoldingPass`: Precomputes constant subgraphs at compile time.
     - `DeadCodeEliminationPass`: Prunes unused ops and intermediate activations.
     - `OperatorFusionPass`: Detects pattern subgraphs (e.g., `RMSNorm`, `SwiGLU`, `Conv2D+Bias+ReLU`) and replaces them with fused Canonical nodes.
2. **Quantization Engine (`ggmlc-quantize`)**:
   - Support weight-only quantization to `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_K`, `GGML_TYPE_Q8_0`.
   - Lowers memory footprint by $4\times$ to $8\times$ with near-zero perplexity loss.
3. **Continuous Batching & Paged Attention**:
   - Chunked prefill and dynamic block tables for managing memory across concurrent requests.

---

### Phase 3: Hardware Acceleration & GPU Backends
1. **GGML Backend Dispatch**:
   - GGML provides native hardware backends (`ggml-cuda`, `ggml-metal`, `ggml-vulkan`, `ggml-sycl`).
   - Extend `ggmlc::Executor` to initialize `ggml_backend_t` (e.g. `ggml_backend_cuda_init(0)`) and execute the `.ggmlc` execution graph directly on GPU memory.
2. **Speculative Decoding Runtime**:
   - Co-execute a small draft model (e.g. `Qwen2.5-0.5B`) and target model in the generic runtime with fused verification kernels.
