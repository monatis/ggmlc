# ggmlc Compilation Workflow & Architecture

`ggmlc` is an optimizing tensor program compiler that translates high-level neural network graphs from PyTorch (`torch.export`) and JAX (`jaxpr`) into GGML dialect execution graphs executed by a lightweight C++ generic runtime or emitted as standalone C++ projects.

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
  |  - Operator Fusion (Conv2D+ReLU, SwiGLU, Norms)  |
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
  |           GGUF v3 Binary Serialization           |
  |  - Standard GGUF container with 32-byte padding  |
  |  - Lossless DAG JSON spec in ggmlc.graph_spec    |
  +---------------------+----------------------------+
            |                                    |
            v                                    v
  +---------+---------------+          +---------+----------------+
  | C++ Generic Interpreter |          | Standalone C++ Codegen   |
  |  - ggmlc-run binary     |          |  - <Model>.h             |
  |  - Dynamic symbol eval  |          |  - ggmlc_main.cpp        |
  |  - Zero-compile deploy  |          |  - CMakeLists.txt        |
  +-------------------------+          +--------------------------+
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
- **`OperatorFusionPass`**: Fuses composite patterns (Conv2D+ReLU, Linear+Bias, SwiGLU, LayerNorm, RMSNorm) into monolithic operations.
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

### Stage 6: Standard GGUF v3 Serialization
The execution graph is serialized into a standard, universal **GGUF v3** binary container:
- Magic header (`GGUF\x03\x00\x00\x00`) and standard metadata tables.
- Complete execution DAG (nodes, shapes, symbols, storage classes, and operator attributes) encoded losslessly as JSON string metadata under `ggmlc.graph_spec`.
- Parameters and constants stored with strict 32-byte alignment.

### Stage 7: Dual Execution Modes
`ggmlc` supports two deployment options:
1. **Generic C++ Runtime Interpreter (`ggmlc-run`)**:
   - Reads `.gguf` artifacts directly via `ModelLoader`.
   - Evaluates dynamic dimensions for the current inference batch/sequence size.
   - Pre-allocates single continuous tensor memory pools without runtime allocations in the inference loop.
   - Manages persistent state tensors for multi-step autoregressive generation (KV-cache).
2. **Ahead-Of-Time (AOT) C++ Code Generation (`ggmlc.codegen`)**:
   - Emits a standalone, human-readable C++ directory containing `<ModelName>.h`, `ggmlc_main.cpp`, and `CMakeLists.txt`.
   - Ideal for embedding into native mobile or embedded environments without external dependencies.
