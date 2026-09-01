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

Pre-built binary wheels (~130 MB each with bundled CUDA and C++ runtimes) are hosted on our custom PyPI index:

```bash
# Minimal Runtime (High-performance inference only)
pip install ggmlc --extra-index-url https://monatis.github.io/ggmlc-index/

# With PyTorch compiler frontend
pip install "ggmlc[torch]" --extra-index-url https://monatis.github.io/ggmlc-index/

# With JAX/Flax compiler frontend
pip install "ggmlc[jax]" --extra-index-url https://monatis.github.io/ggmlc-index/

# Complete development environment
pip install "ggmlc[all]" --extra-index-url https://monatis.github.io/ggmlc-index/
```

## Interactive Google Colab Demo

Run the interactive demo notebook covering benchmarking, PyTorch/JAX model compilation, graph visualization, and standalone C++ export directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jD5Pr4ObD9CGoRoC7_LQmAGvh0AZZ6KW?usp=sharing)

