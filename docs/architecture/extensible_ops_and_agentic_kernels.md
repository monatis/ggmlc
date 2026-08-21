# Extensible Custom Operators & Agentic Kernel Developer Architecture

## 1. Executive Summary & Design Philosophy

A critical flaw in previous neural network compilers (such as `executorch-ggml`) is their reliance on **PyTorch ATen runtime fallbacks** when encountering unsupported operators. While this achieves functional completeness on paper, it completely defeats the primary goal of edge/embedded deployment by dragging the entire multi-gigabyte `libtorch` runtime, complex dynamic memory allocators, and PyTorch ABI dependencies into what was meant to be a lightweight, zero-dependency C++ execution engine.

`ggmlc` adopts a strictly **Zero-PyTorch-Runtime** architectural philosophy. Extensibility is achieved through 4 unified pillars:

```
                      +-----------------------------+
                      |     Canonical IR Graph      |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
    +------------+-------------+           +-------------+------------+
    | Standard GGML Lowering   |           | Unsupported / Custom Op  |
    +------------+-------------+           +-------------+------------+
                 |                                       |
                 |                     +-----------------+-----------------+
                 |                     |                 |                 |
                 |                     v                 v                 v
                 |               +-----------+     +-----------+     +------------+
                 |               |  ggmlc    |     | User BYOK |     |  Agentic   |
                 |               |  StdLib   |     |  Plugin   |     |   Kernel   |
                 |               |  Kernels  |     |  (.so)    |     | Developer  |
                 |               +-----+-----+     +-----+-----+     +-----+------+
                 |                     |                 |                 |
                 +---------------------+-----------------+-----------------+
                                       |
                                       v
                        +--------------+--------------+
                        |  Generic C++ Executor Engine|
                        |  (Zero PyTorch Runtime Dep) |
                        +-----------------------------+
```

---

## 2. The Four Pillars of Extensibility

### Pillar 1: In-House Standalone Standard Library (`ggmlc-stdlib`)
For operators not natively supported in upstream GGML (such as `Hardswish`, `Mish`, `GroupNorm`, `ScatterAdd`, `Roll`, `Mamba/SSM Selective Scan`, `Gated Linear Units`), `ggmlc` provides standalone, pure C++ implementations with AVX-512 / ARM NEON vectorization.
- **Header-only / Standalone**: Zero external dependencies.
- **Thread-safe**: Operates directly within GGML's thread-pool work distribution.

### Pillar 2: Aggressively Fused Compound Kernels (`ggmlc-fused`)
Certain operator sequences produce massive memory bandwidth bottlenecks if executed as discrete primitives (e.g. `LayerNorm -> Add -> GELU` or `Q @ K^T -> Softmax -> V`). 
- Compile-time fusion passes in `ggmlc.transforms.fusion` group contiguous subgraphs into single compound nodes (`FusedSDPA`, `FusedSwiGLU`, `FusedBiasGELU`).
- Dispatched to monolithic fused micro-kernels, eliminating intermediate activation buffers and memory traffic.

### Pillar 3: User-Defined Custom Op Plugin Interface (BYOK)
Developers can supply custom domain-specific kernels (e.g. custom DSP filters, proprietary activations, custom quantization decoders) without recompiling `ggmlc-run`.

#### C ABI Interface Specification (`ggmlc/custom_op.h`):
```cpp
typedef struct {
    const char* name;
    int version;
    void (*infer_shape)(const int64_t* in_shapes[], int n_in, int64_t* out_shape, void* user_data);
    void (*compute)(const void* in_ptrs[], int n_in, void* out_ptr, const int64_t* out_shape, int n_threads, void* user_data);
} ggmlc_custom_op_t;

// Exported by user dynamic library (.so / .dll)
extern "C" ggmlc_custom_op_t* ggmlc_register_custom_op(void);
```

#### Python Registration API:
```python
from ggmlc.ir.op import register_custom_opcode
from ggmlc.frontend.pytorch import register_aten_lowering

register_custom_opcode(
    name="my_custom_filter",
    infer_shape_fn=my_shape_inference,
    plugin_path="/path/to/libmy_op.so",
)
```

---

## 3. Pillar 4: The "Agentic Kernel Developer" Subsystem

One of `ggmlc`'s most innovative capabilities is the **Agentic Kernel Developer** — an autonomous AI agent integrated into the compilation pipeline that synthesizes, optimizes, tests, and compiles missing C++ kernels on-the-fly.

```
       +---------------------------------------------+
       |   Unsupported ATen / JAX Primitive Detected |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |   Kernel Synthesis Agent (LLM Prompting)     |
       |   - Analyzes mathematical semantics         |
       |   - Emits pure C++ / SIMD micro-kernel      |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |   Automated JIT Compilation (GCC / Clang)   |
       |   - Compiles kernel into scratch .so plugin |
       +----------------------+----------------------+
                              |
                              v
       +----------------------+----------------------+
       |   Differential Fuzzing & Numerical Verifier |
       |   - Generates random synthetic tensor inputs|
       |   - Compares C++ plugin vs PyTorch ATen ref |
       |   - Verifies cosine similarity > 0.9999     |
       +----------------------+----------------------+
                              |
                 +------------+------------+
                 |                         |
            [PASSED]                   [FAILED]
                 |                         |
                 v                         v
        +---------+----------+    +---------+----------+
        | Register Plugin in |    | Auto-Feedback Loop |
        | Compiled .gguf     |    | to Synthesis Agent |
        +--------------------+    +--------------------+
```

### Synthesis Workflow:
1. **Semantic Extraction**: When `torch.export` or `jaxpr` encounters an unmapped node (e.g. `aten.special_erf`), the importer captures input dtypes, ranks, and exact PyTorch reference semantics.
2. **Code Generation**: The agent generates a standalone C++ function with OpenMP / SIMD multi-threading.
3. **JIT Compilation & Sandbox Execution**: The generated code is compiled via `g++ -O3 -shared -fPIC` into the cache directory.
4. **Differential Fuzzing**: The compiler runs $1000$ synthetic test cases with edge values (NaNs, subnormals, extremes) comparing the JIT kernel against PyTorch reference.
5. **Persistence**: Validated kernels are added to the project's local custom op cache for permanent reuse.

---

## 4. Subgraph Partitioning & Diagnostic Engine

When complete lowering is impossible or unsupported, the **Subgraph Partitioner** (`ggmlc.partitioning`) segments the Canonical IR DAG into maximum supported clusters:

```
  Graph: [A] -> [B] -> [C] (unsupported) -> [D] -> [E]

  Partitioned:
    Subgraph 1 (GGML Cluster): [A] -> [B]
    Subgraph 2 (Custom / Fallback Plugin): [C]
    Subgraph 3 (GGML Cluster): [D] -> [E]
```

### Diagnostic Output Example:
```text
================================================================================
ggmlc Compilation Diagnostic Report:
--------------------------------------------------------------------------------
Node #42: 'aten.custom_dequantize'
  Status:      NOT DIRECTLY LOWERABLE TO STANDARD GGML
  Reason:      Non-standard 3-bit sub-block packaging
  Decomposition: No standard decomposition rule found
  Resolution:
    [1] Agentic Kernel Developer synthesized JIT plugin: 'cache/kernels/dequant_3bit.so'
    [2] Differential Numerical Parity: PASSED (Max Diff: 1.2e-7, Cosine Sim: 1.000000)
    [3] Emitted standalone .gguf with embedded custom plugin dependency.
================================================================================
```
