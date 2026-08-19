# ggmlc Compilation Workflow & Architecture

`ggmlc` is an optimizing tensor program compiler that translates high-level neural network graphs from PyTorch (`torch.export`) and JAX (`jaxpr`) into GGML dialect execution graphs executed by a lightweight C++ generic runtime.

---

## 1. End-to-End Compilation Pipeline

```
  +--------------------+        +--------------------+
  | PyTorch (Exported) |        |    JAX (jaxpr)     |
  +---------+----------+        +---------+----------+
            |                             |
            v                             v
  +---------+-----------------------------+----------+
  |              Frontend Importers                  |
  |  - PyTorch FX Importer (importer.py)             |
  |  - JAX jaxpr Importer (importer.py)              |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |               Canonical IR Graph                 |
  |  - Framework-independent semantic tensor ops     |
  |  - Symbolic shape arithmetic (SymbolDim, Add...) |
  |  - Explicit StorageClass (Param, State, Input)   |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |      Transformation & Optimization Passes        |
  |  - Constant Folding (ConstantFoldingPass)        |
  |  - Dead Code Elimination (DeadCodeElimination)   |
  |  - Operator Fusion (Conv2D+ReLU, SwiGLU, MatMul) |
  |  - Redundant Cast & Permutation Pruning          |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |             GGML Dialect Lowering                |
  |  - Column-major axis & stride translation        |
  |  - Contraction dimension alignment for MatMul   |
  |  - High-level op mapping (Norm, Softmax, RoPE)   |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |         Block Quantization Engine                |
  |  - Q8_0 (34 bytes/32 floats, 3.76x compression)  |
  |  - Q4_0 (18 bytes/32 floats, 7.11x compression)  |
  |  - Selective parameter weight quantization       |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |           Binary Serialization (.ggmlc)          |
  |  - Compact header, symbol table, tensor table    |
  |  - 16-byte aligned parameter buffer embedding    |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |          C++ Generic Runtime (ggmlc-run)         |
  |  - Dynamic symbol evaluation                     |
  |  - Static single-buffer memory planning          |
  |  - Persistent state buffers (KV cache)           |
  |  - Native quantized SIMD kernel execution        |
  +--------------------------------------------------+
```

---

## 2. Compilation Stages

### Stage 1: Framework Ingestion (Frontends)
- **PyTorch**: Ingests `torch.export.ExportedProgram` (AOTInductor / Dynamo capture). Extracts operator schemas, lifted parameters, persistent buffers, dynamic symbolic shape expressions (`torch.export.Dim`), and ATen targets.
- **JAX**: Traces pure functions via JAX core primitives into `ClosedJaxpr`. Flattens nested equations, lifts constants and parameters, and maps JAX primitives (`add`, `dot_general`, `reshape`, `transpose`, `convert_element_type`) into Canonical IR.

### Stage 2: Canonical IR Construction
The Canonical IR represents the neural network as a purely functional, framework-independent directed acyclic graph (DAG):
- Nodes represent strongly typed mathematical operations.
- Edges represent value tensors with symbolic or static multi-dimensional shapes.
- Tensor lifetimes and mutability are captured cleanly by `StorageClass`.

### Stage 3: Optimization Pass Pipeline
Graph optimizations are performed on the Canonical IR by `PassManager`:
- **`ConstantFoldingPass`**: Pre-evaluates deterministic constant subgraphs at compile time.
- **`OperatorFusionPass`**: Fuses composite patterns (Conv2D+ReLU, Linear+Bias, SwiGLU) into monolithic operations.
- **`DeadCodeEliminationPass`**: Prunes unreachable nodes and orphan activations via backward reachability from outputs and states.
- **`RedundantCastPruner`**: Removes identity transpositions and identical dtype casts.

### Stage 4: Target Dialect Lowering (GGML Dialect)
Translates Canonical IR semantics into GGML computational structures:
- **Memory Order Mapping**: PyTorch/JAX row-major (C-order) semantics are mapped onto GGML's column-major (Fortran-order) layout:
  $$\text{shape}_{GGML} = \text{reverse}(\text{shape}_{Canonical})$$
- **Permutation Invariant**: Forward permutation vectors are computed according to:
  $$\text{dest\_axis}_i = R - 1 - P.\text{index}(R - 1 - i)$$
- **Matrix Multiplication Alignment**: Parameters and activations are aligned so inner contraction dimensions strictly match before `ggml_mul_mat` evaluation.

### Stage 5: Block Quantization Engine
Compresses 2D parameter matrices into hardware-aligned block formats:
- **`Q8_0`**: 8-bit block quantization ($3.76\times$ compression, cosine similarity $> 0.9999$).
- **`Q4_0`**: 4-bit nibble-packed block quantization ($7.11\times$ compression, cosine similarity $> 0.9850$).
- 1D biases and activation paths remain in full `F32` for numerical stability.

### Stage 6: Binary Serialization (`.ggmlc`)
The execution graph is compiled into a standalone, portable binary container:
- Magic header (`GGMLC\x01\x00\x00`) and format version.
- Global dynamic symbol registry (`symbol_table`).
- Input, output, parameter, and state tensor metadata.
- Recursive symbolic dimension trees for dynamic rank/extent evaluation.
- Opcode schedule with operator-specific attributes.
- 16-byte aligned binary weight data buffer for parameters and constants.

### Stage 7: Generic C++ Execution
Rather than generating brittle, model-specific C++ source code that requires recompilation for each architecture, `ggmlc` utilizes a **universal graph interpreter**:
- Reads `.ggmlc` artifacts directly via `ModelLoader`.
- Evaluates dynamic dimensions for the current inference batch/sequence size.
- Pre-allocates single continuous tensor memory pools without runtime memory allocations in the hot inference loop.
- Manages persistent state tensors for multi-step autoregressive generation (KV-cache).
- Dispatches quantized matrix multiplications directly to AVX-512 / ARM NEON kernels.
