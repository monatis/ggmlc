# ggmlc Documentation

Welcome to the documentation for **ggmlc** — a high-performance, framework-agnostic neural network tensor program compiler targeting GGML and native C++ execution.

```
                    ┌─────────────────────────┐
                    │   PyTorch / JAX Model   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Canonical IR DAG     │
                    └────────────┬────────────┘
                                 │ (Passes: CF, DCE, Fusion)
                                 ▼
                    ┌─────────────────────────┐
                    │  GGML Target Execution  │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     ┌───────────────────────┐       ┌───────────────────────┐
     │ Standard GGUF v3 File │       │ Standalone C++ Proj   │
     │   (Zero-Copy Native)  │       │  (<Model>.h + CMake)  │
     └───────────────────────┘       └───────────────────────┘
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

guides/python_api_guide
guides/developer_guide
guides/quantization_guide
guides/autoregressive_generation
guides/troubleshooting_and_debugging
```

```{toctree}
:maxdepth: 2
:caption: Architecture & Design

architecture/compilation_workflow
architecture/extensible_ops_and_agentic_kernels
architecture/quantization_and_optimizations
architecture/future_roadmap
ir/canonical_ir
dialect/ggml_dialect
codegen/cpp_codegen
runtime/runtime_architecture
reference/operator_reference
```

```{toctree}
:maxdepth: 2
:caption: Benchmarks & Analysis

benchmarks/model_benchmark_suite
benchmarks/operator_fusion_speedup_analysis
benchmarks/sequence_length_scaling_analysis
```

```{toctree}
:maxdepth: 2
:caption: Python API Reference

api/compiler
api/runtime
api/ir
api/dialect
api/transforms
api/quantization
api/visualization
```

## Quick Installation

```bash
# Minimal Runtime (High-performance inference only)
pip install ggmlc

# With PyTorch compiler frontend
pip install "ggmlc[torch]"

# With JAX/Flax compiler frontend
pip install "ggmlc[jax]"

# Complete development environment
pip install "ggmlc[all]"
```
