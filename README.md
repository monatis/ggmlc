# ggmlc — Neural Network Tensor Program Compiler to GGML

`ggmlc` is a high-performance tensor program compiler and execution runtime that ingests neural network graphs from **PyTorch** (`torch.export`) and **JAX** (`jaxpr`), translates them into a strongly-typed **Canonical Intermediate Representation (IR)**, applies compile-time graph optimization passes, quantizes parameters to high-performance block formats (**Q8_0**, **Q4_0**), lowers to target-specific **GGML execution graphs**, and serializes them into standard **GGUF v3** binary containers (`.gguf`) executed by a lightweight zero-dependency generic C++ runtime or emitted as standalone human-readable C++ projects.

---

## Key Features

- **Multi-Framework Ingestion**:
  - Full support for PyTorch 2.x `torch.export` (AOTInductor / Dynamo graph capture).
  - Native JAX `jaxpr` tracing and equation lowering.
- **Strongly-Typed Canonical Intermediate Representation**:
  - Framework-independent functional Directed Acyclic Graph (DAG).
  - Explicit tensor lifetimes and storage classes (`INPUT`, `OUTPUT`, `PARAMETER`, `CONSTANT`, `ACTIVATION`, `STATE`).
  - Symbolic shape arithmetic (`SymbolDim`, `StaticDim`, `AddDim`, `MulDim`).
- **Graph Optimization Transformation Passes**:
  - Compile-time **Constant Folding** for static subgraphs.
  - Backward reachability **Dead Code Elimination (DCE)**.
  - Pattern-based **Operator Fusion** (Conv2D+ReLU, Linear+Bias, SwiGLU, LayerNorm, RMSNorm).
  - **Redundant Cast and Permutation Pruning**.
- **Block Quantization Subsystem**:
  - **`Q8_0` Format**: 34 bytes per 32 floats ($\mathbf{3.76\times}$ compression, cosine similarity $> 0.9999$).
  - **`Q4_0` Format**: 18 bytes per 32 floats ($\mathbf{7.11\times}$ compression, cosine similarity $> 0.9850$).
  - Unified one-shot quantization CLI: `python -m ggmlc.cli.quantize`.
- **Standard GGUF v3 Binary Containers (`.gguf`)**:
  - Zero proprietary formats: Models are serialized to standard GGUF v3 files with 32-byte tensor alignment.
  - Complete graph topology, tensor metadata, dynamic shapes, and attributes stored losslessly in `ggmlc.graph_spec` metadata.
- **Flexible Execution Modes**:
  - **Generic Dynamic C++ Runner (`ggmlc-run`)**: Execute any compiled `.gguf` model directly with zero compilation.
  - **Standalone C++ Code Generation (`ggmlc.codegen`)**: Emit self-contained, human-readable C++ header files (`<Model>.h`), runner (`ggmlc_main.cpp`), and `CMakeLists.txt` for native app embedding.
- **Verified Pretrained Hub Architectures**:
  - Validated end-to-end against real pretrained checkpoints from Hugging Face & TorchVision with differential numerical testing.

---

## Verified Pretrained Hub Models

| Architecture | Source / Library | Layers & Features | Compression (Q4_0) | Parity Status |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** | `torchvision` (`ImageNet1K_V1`) | 18 layers, Conv2D, Residuals, AdaptiveAvgPool2D | $6.8\times$ | **PASSED** |
| **ResNet-50** | `torchvision` (`ImageNet1K_V1`) | 50 layers, Bottleneck Residuals, Conv2D | $7.0\times$ | **PASSED** |
| **MiniLM-L6-v2** | `sentence-transformers` | 6 layers, Bidirectional Attention, Embeddings | $7.11\times$ | **PASSED** |
| **GPT-2** | `transformers` (`gpt2` 124M) | 12 layers, Causal Attention, WTE/WPE, LM Head | $7.05\times$ | **PASSED** |
| **Qwen2.5-0.5B** | `transformers` (`Qwen/Qwen2.5-0.5B`) | 24 layers, Grouped Query Attention (GQA), RoPE, SwiGLU | $7.15\times$ | **PASSED** |
| **BGE-M3-Distill-8L** | `sentence-transformers` (`altaidevorg`) | 8 layers, XLM-RoBERTa Embeddings, Mean Pooling | $7.08\times$ | **PASSED** |

---

## Quickstart

### 1. Python Environment Setup
```bash
# Using uv or pip
uv sync
```

### 2. Building the C++ Generic Runtime
```bash
# Build C++ runtime binary
cmake -B build && cmake --build build -j$(nproc)
```

### 3. Model Quantization & GGUF Export CLI
Export, optimize, and quantize a model in one command to a standard GGUF file:
```bash
# Quantize MiniLM to Q4_0 with optimization passes
python -m ggmlc.cli.quantize --model minilm --dtype q4_0 --optimize --output minilm_q4.gguf
```

### 4. Running Models via Generic C++ Runner
Execute the compiled `.gguf` artifact with the generic runtime:
```bash
./build/runtime/ggmlc-run minilm_q4.gguf --input input_ids:in.bin --threads 4 --output 128:out.bin
```

### 5. Generating Standalone C++ Projects
Compile any neural network into a self-contained C++ project:
```python
from examples.models.hub_models import load_minilm_model
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.codegen import generate_cpp_project

model, sample_inputs, _ = load_minilm_model()
exported = export_torch_model(model, sample_inputs, model_name="MiniLM")

generate_cpp_project(
    exported_program=exported,
    output_dir="./build/generated/minilm",
    model_name="MiniLM",
    enable_fusion=True,
)
```

### 6. Autoregressive Text Generation
Generate tokens interactively using the compiled `ggmlc` runtime with differential PyTorch verification:
```bash
python examples/generate.py --model gpt2 --prompt "The capital of France is" --max-tokens 16 --verify-pytorch
```

### 7. Running the Comprehensive Test Suite
```bash
# Run all unit, numerical, transform, dynamic shape, and quantization tests
pytest -v
```

---

## Documentation Index

- **Architecture & System Design**:
  - [Compilation Workflow](docs/architecture/compilation_workflow.md): End-to-end compilation stages from frontends to GGUF container.
  - [Optimization & Quantization Pipeline](docs/architecture/quantization_and_optimizations.md): Transformation passes and block quantization math.
  - [Future Roadmap & Design Decisions](docs/architecture/future_roadmap.md): Architectural decisions and GPU backend plans.
- **Specifications**:
  - [Canonical IR Specification](docs/ir/canonical_ir.md): Type system, symbolic shapes, storage classes, and operator schemas.
  - [GGML Dialect Specification](docs/dialect/ggml_dialect.md): Memory layouts, permutation formulas, and operator mappings.
  - [Operator Reference](docs/reference/operator_reference.md): Full operator catalog and shape inference rules.
- **Code Generation & Runtime**:
  - [C++ Code Generation Guide](docs/codegen/cpp_codegen.md): Standalone C++ source generator, project structure, and CMake integration.
  - [C++ Generic Runtime Architecture](docs/runtime/runtime_architecture.md): GGUF loader, dynamic symbol evaluator, and memory manager.
- **Developer & User Guides**:
  - [Quantization User Guide](docs/guides/quantization_guide.md): CLI commands, APIs, compression benchmarks, and numerical evaluation.
  - [Autoregressive Generation Guide](docs/guides/autoregressive_generation.md): Dynamic symbol generation and KV-cache persistence.
  - [Troubleshooting & Debugging Guide](docs/guides/troubleshooting_and_debugging.md): Permutation math, layout pitfalls, and debugging tips.
  - [Developer & Contributor Guide](docs/guides/developer_guide.md): Step-by-step checklist for adding new operators, passes, and models.
