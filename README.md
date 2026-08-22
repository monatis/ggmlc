<div align="center">

# ggmlc

### Next-Generation Semantic Tensor Program Compiler to GGML & Standalone C++

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Format](https://img.shields.io/badge/binary-GGUF%20v3-orange.svg)](https://github.com/ggerganov/ggml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

*Compile neural network graphs from PyTorch and JAX into ultra-fast, portable GGUF binaries and human-readable C++ projects with GGML execution.*

---

</div>

## 🚀 Why ggmlc?

Deploying modern neural networks on edge devices, CPU servers, and embedded platforms often requires writing brittle, hand-crafted C++ inference code for each new model architecture.

`ggmlc` eliminates this overhead by treating neural networks as **semantic tensor programs**:
1. **Zero Hand-Written C++ Glue**: Ingests models directly from **PyTorch** (`torch.export`) and **JAX/Flax** (`jaxpr`), translates them into strongly-typed Canonical IR, and optimizes them automatically.
2. **Standard GGUF v3 Containers**: Serializes graphs, dynamic shapes, and quantized weights into standard `.gguf` binaries — no proprietary file formats or runtime lock-in.
3. **Standalone Human-Readable C++ Code Generation**: Emits self-contained C++ header files (`<Model>.h`), native entry points (`ggmlc_main.cpp`), and `CMakeLists.txt` for direct embedding into native applications.
4. **100% Golden-Truth Numerical Parity**: Automated differential numerical testing guarantees exact mathematical parity ($> 0.99999$ cosine similarity) against PyTorch and JAX reference runs.
5. **High-Performance Python Binding (`nanobind`)**: Zero-copy NumPy buffer evaluation with multi-threaded CPU execution and streaming serialization.

---

## 🏗️ Compiler Architecture

```mermaid
graph TD
    subgraph Frontends["1. Multi-Framework Ingestion"]
        PT["PyTorch 2.x (torch.export)"]
        JX["JAX / Flax (jaxpr)"]
    end

    subgraph IR["2. Canonical Intermediate Representation (IR)"]
        DAG["Semantic Functional DAG<br/><i>Symbolic Shapes & Storage Classes</i>"]
    end

    subgraph Passes["3. Compile-Time Optimization Passes"]
        CF["Constant Folding"]
        DCE["Dead Code Elimination"]
        FUS["Pattern-Based Operator Fusion<br/><i>(Conv+ReLU, SwiGLU, LayerNorm, RMSNorm)</i>"]
        PRN["Redundant Cast & Permute Pruning"]
    end

    subgraph Lowering["4. Target Dialect Lowering"]
        GGML["GGML Dialect Graph<br/><i>(Block Quantization: Q8_0, Q4_0)</i>"]
    end

    subgraph Outputs["5. Deployment Targets"]
        GGUF["Standard GGUF v3 Binary<br/><i>(High-Speed nanobind Runner / ggmlc-run)</i>"]
        CPP["Standalone C++ Project Folder<br/><i>(&lt;Model&gt;.h, ggmlc_main.cpp, CMakeLists.txt)</i>"]
    end

    PT --> DAG
    JX --> DAG
    DAG --> CF --> DCE --> FUS --> PRN
    PRN --> GGML
    GGML --> GGUF
    GGML --> CPP

    classDef frontend fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#004d40;
    classDef ir fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef passes fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef target fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef deploy fill:#e8f8f5,stroke:#26a69a,stroke-width:2px,color:#004d40;

    class PT,JX frontend;
    class DAG ir;
    class CF,DCE,FUS,PRN passes;
    class GGML target;
    class GGUF,CPP deploy;
```

---

## ⚡ 3-Line Quickstarts

### 1. Compile and Run a PyTorch Model
```python
import ggmlc
import torch
import torchvision.models as models

# 1. Take any PyTorch model
model = models.resnet18(weights=None).eval()
example_x = torch.randn(1, 3, 224, 224)

# 2. Compile directly to a standard GGUF binary file
model_path = ggmlc.compile(model, (example_x,), output="resnet18.gguf")

# 3. Load into high-performance native runtime and evaluate
runner = ggmlc.load(model_path, n_threads=4)
output = runner(example_x.numpy())
print("Output shape:", output.shape)
```

### 2. Compile and Run JAX / Flax
```python
import ggmlc
import jax.numpy as jnp
from examples.models.flax_models import FlaxTransformerLayer

# 1. Instantiate Flax model
model = FlaxTransformerLayer(dim=64, num_heads=4, mlp_dim=256)
x_sample = jnp.ones((1, 8, 64), dtype=jnp.float32)
params = model.init(jax.random.PRNGKey(0), x_sample)

# 2. Compile JAX forward function to GGUF
model_path = ggmlc.compile(lambda x: model.apply(params, x), (x_sample,), output="transformer.gguf")

# 3. Fast native execution with zero-copy NumPy buffers
runner = ggmlc.load(model_path)
out = runner(x_sample)
```

### 3. Generate Standalone C++ Project
```python
# Emit a complete, standalone C++ project linking against GGML
ggmlc.codegen(
    model=model,
    sample_inputs=(example_x,),
    output_dir="./build/resnet18_cpp",
    model_name="ResNet18",
)
```
Generates:
- `ResNet18.h`: Self-contained C++ header with model tensor descriptors and computation graph.
- `ggmlc_main.cpp`: Standalone CLI executable driver for inference and benchmarking.
- `CMakeLists.txt`: Build configuration ready for compilation with MSVC, GCC, or Clang.

---

## 📊 Verified Pretrained Model Zoo

All models are validated end-to-end against real Hugging Face & TorchVision weights with **differential numerical testing**:

| Architecture | Framework | Key Features | Compression (Q4_0) | Parity Status |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18 / 50** | PyTorch / TorchVision | Residual Convolutions, AdaptiveAvgPool2D | $6.8\times$ | **PASSED** ($1.000$) |
| **MiniLM-L6-v2** | PyTorch / Transformers | Bidirectional Multi-Head Attention, Embeddings | $7.11\times$ | **PASSED** ($0.9999$) |
| **GPT-2** | PyTorch / Transformers | Causal Self-Attention, WTE/WPE, Autoregressive LM Head | $7.05\times$ | **PASSED** ($0.9999$) |
| **Qwen-2.5 (0.5B)** | PyTorch / Transformers | Grouped Query Attention (GQA), RoPE, SwiGLU, RMSNorm | $7.15\times$ | **PASSED** ($0.9999$) |
| **BGE-M3-Distill** | PyTorch / Transformers | Multilingual Embeddings, Dense Vector Pooling | $7.08\times$ | **PASSED** ($0.9999$) |
| **Flax Transformer** | JAX / Flax | Pre-LN Self-Attention, GELU Feed-Forward Network | $6.9\times$ | **PASSED** ($1.0000$) |
| **Flax MLP Classifier** | JAX / Flax | LayerNorm, Dense, GELU / ReLU | $7.0\times$ | **PASSED** ($1.0000$) |

---

## 🛠️ Installation & Building

### Python Package Installation
```bash
# High-performance lightweight runtime (Inference only)
pip install ggmlc

# With PyTorch compiler frontend
pip install "ggmlc[torch]"

# With JAX/Flax compiler frontend
pip install "ggmlc[jax]"

# Complete development suite
pip install "ggmlc[all]"
```

### Native C++ Runtime Compilation

#### Windows (MSVC)
```powershell
cmake -B build-win
cmake --build build-win --target _runtime --config Release
```

#### Linux / WSL (GCC / Clang)
```bash
cmake -B build
cmake --build build --target _runtime -j$(nproc)
```

---

## 🔍 Graph & IR Pass Visualization

`ggmlc` includes built-in Mermaid diagram export for inspectable compiler graph visualization:

```python
# Export interactive HTML with zoom/pan or Mermaid markdown
graph_path = ggmlc.visualize(graph, output_path="model_graph.html")
```

---

## 📖 Documentation

Comprehensive guides, tutorials, and API references are available in the [`docs/`](docs/) directory:

- **[Python API Guide](docs/guides/python_api_guide.md)**: Detailed Python usage with `ggmlc.compile`, `ggmlc.load`, and `ggmlc.codegen`.
- **[Developer & Contributor Guide](docs/guides/developer_guide.md)**: Adding new operators, lowering rules, and C++ kernels.
- **[Quantization Subsystem Guide](docs/guides/quantization_guide.md)**: Q8_0 and Q4_0 block quantization details and precision benchmarks.
- **[Autoregressive Text Generation](docs/guides/autoregressive_generation.md)**: Multi-token KV-cache generation and parity verification.
- **[Troubleshooting & Debugging](docs/guides/troubleshooting_and_debugging.md)**: Common issues, tensor stride semantics, and memory alignments.

---

## 📄 License

`ggmlc` is released under the [MIT License](LICENSE).
