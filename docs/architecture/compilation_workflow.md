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
  |             GGML Dialect Lowering                |
  |  - Column-major axis & stride translation        |
  |  - Contraction dimension alignment for MatMul   |
  |  - High-level op mapping (Norm, Softmax, RoPE)   |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |           Binary Serialization (.ggmlc)          |
  |  - Compact header, symbol table, tensor table    |
  |  - Zero-copy parameter buffer embedding          |
  +---------------------+----------------------------+
                        |
                        v
  +---------------------+----------------------------+
  |          C++ Generic Runtime (ggmlc-run)         |
  |  - Dynamic symbol evaluation                     |
  |  - Static single-buffer memory planning          |
  |  - Persistent state buffers (KV cache)           |
  |  - High-performance GGML thread execution       |
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

### Stage 3: Target Dialect Lowering (GGML Dialect)
Translates Canonical IR semantics into GGML computational structures:
- **Memory Order Mapping**: PyTorch/JAX row-major (C-order) semantics are mapped onto GGML's column-major (Fortran-order) layout:
  $$\text{shape}_{GGML} = \text{reverse}(\text{shape}_{Canonical})$$
- **Permutation Invariant**: Forward permutation vectors are computed according to:
  $$\text{dest\_axis}_i = R - 1 - P.\text{index}(R - 1 - i)$$
- **Matrix Multiplication Alignment**: Parameters and activations are aligned so inner contraction dimensions strictly match before `ggml_mul_mat` evaluation.
- **Operator Fusion**: Fuses common patterns (e.g. SwiGLU, LayerNorm, RMSNorm) into native GGML kernel dispatches.

### Stage 4: Binary Serialization (`.ggmlc`)
The execution graph is compiled into a standalone, portable binary container:
- Magic header (`GGMLC\x01\x00\x00`) and format version.
- Global dynamic symbol registry (`symbol_table`).
- Input, output, parameter, and state tensor metadata.
- Recursive symbolic dimension trees for dynamic rank/extent evaluation.
- Opcode schedule with operator-specific attributes.
- Contiguous binary weight data buffer for parameters and constants.

### Stage 5: Generic C++ Execution
Rather than generating brittle, model-specific C++ source code that requires recompilation for each architecture, `ggmlc` utilizes a **universal graph interpreter**:
- Reads `.ggmlc` artifacts directly via `ModelLoader`.
- Evaluates dynamic dimensions for the current inference batch/sequence size.
- Pre-allocates single continuous tensor memory pools without runtime memory allocations in the hot inference loop.
- Manages persistent state tensors for multi-step autoregressive generation (KV-cache).
