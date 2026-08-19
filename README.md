# ggmlc — Neural Network Tensor Program Compiler to GGML

`ggmlc` is a high-performance tensor program compiler and execution runtime that ingests neural network graphs from **PyTorch** (`torch.export`) and **JAX** (`jaxpr`), translates them into a strongly-typed **Canonical Intermediate Representation (IR)**, lowers them to target-specific **GGML execution graphs**, and serializes them into compact binary containers (`.ggmlc`) executed by a lightweight, zero-dependency generic C++ runtime.

---

## Key Features

- **Multi-Framework Ingestion**:
  - Full support for PyTorch 2.x `torch.export` (AOTInductor / Dynamo graph capture).
  - Native JAX `jaxpr` tracing and equation lowering.
- **Strongly-Typed Canonical Intermediate Representation**:
  - Framework-independent functional Directed Acyclic Graph (DAG).
  - Explicit tensor lifetimes and storage classes (`INPUT`, `OUTPUT`, `PARAMETER`, `CONSTANT`, `ACTIVATION`, `STATE`).
  - Symbolic shape arithmetic (`SymbolDim`, `StaticDim`, `AddDim`, `MulDim`).
- **GGML Dialect Lowering**:
  - Automatic column-major $\leftrightarrow$ row-major stride and memory mapping.
  - Contraction dimension alignment for linear and dynamic attention matrix multiplications.
  - Native kernel fusion for SwiGLU, RMSNorm, LayerNorm, Rotary Position Embeddings (RoPE), and 2D Spatial Convolutions.
- **Lightweight Generic C++ Execution Engine (`ggmlc-run`)**:
  - Single hardened C++ binary interpreter (`ggmlc::Executor`) — no per-model C++ code generation or recompilation required.
  - Dynamic symbol evaluation for variable sequence and batch dimensions.
  - Multi-step stateful KV-cache persistence for autoregressive LLM decoding.
- **Verified Pretrained Hub Architectures**:
  - Validated end-to-end against real pretrained checkpoints from Hugging Face & TorchVision with differential numerical testing.

---

## Verified Pretrained Hub Models

| Architecture | Source / Library | Layers & Features | Max Abs Diff | Parity Status |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** | `torchvision` (`ImageNet1K_V1`) | 18 layers, Conv2D, Residuals, AdaptiveAvgPool2D | $0.0067$ | **PASSED** |
| **ResNet-50** | `torchvision` (`ImageNet1K_V1`) | 50 layers, Bottleneck Residuals, Conv2D | $0.0053$ | **PASSED** |
| **MiniLM-L6-v2** | `sentence-transformers` | 6 layers, Bidirectional Attention, Embeddings | $0.0024$ | **PASSED** |
| **GPT-2** | `transformers` (`gpt2` 124M) | 12 layers, Causal Attention, WTE/WPE, LM Head | $0.0031$ | **PASSED** |
| **Qwen2.5-0.5B** | `transformers` (`Qwen/Qwen2.5-0.5B`) | 24 layers, Grouped Query Attention (GQA), RoPE, SwiGLU | $5.4 \times 10^{-5}$ | **PASSED** |
| **BGE-M3-Distill-8L** | `sentence-transformers` (`altaidevorg`) | 8 layers, XLM-RoBERTa Embeddings, Mean Pooling | $0.1386$ | **PASSED** |

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

### 3. Autoregressive Text Generation CLI
Generate tokens interactively using the compiled `ggmlc` runtime with differential PyTorch verification:
```bash
python examples/generate.py --model gpt2 --prompt "The capital of France is" --max-tokens 16 --verify-pytorch
```

Output:
```
Loaded 'openai-community/gpt2' successfully.
Compiling forward graph via ggmlc...
Compiled ggmlc model in 0.48s.

=== Autoregressive Generation ===
Prompt: 'The capital of France is'

[PyTorch Reference Generation]
Step 1: token 262 -> ' the'
Step 2: token 3139 -> ' capital'
Step 3: token 286 -> ' of'
Step 4: token 262 -> ' the'
Step 5: token 4731 -> ' French'
Step 6: token 4719 -> ' Republic'
Full PyTorch Text: The capital of France is the capital of the French Republic

[ggmlc C++ Runtime Generation]
Step 1: token 262 -> ' the' (logits max_diff=0.0019)
Step 2: token 3139 -> ' capital' (logits max_diff=0.0022)
Step 3: token 286 -> ' of' (logits max_diff=0.0025)
Step 4: token 262 -> ' the' (logits max_diff=0.0024)
Step 5: token 4731 -> ' French' (logits max_diff=0.0028)
Step 6: token 4719 -> ' Republic' (logits max_diff=0.0029)
Full ggmlc Text: The capital of France is the capital of the French Republic

=== Verification Result: MATCH [OK] ===
Generated tokens are 100% identical.
```

### 4. Running the Comprehensive Test Suite
```bash
# Run all 54 tests
pytest -v

# Run only e2e hub model tests
pytest tests/e2e/test_full_models.py -v
```

---

## Documentation Roadmap

- **Architecture & Pipeline**:
  - [Compilation Workflow](docs/architecture/compilation_workflow.md): Complete 4-stage compiler lifecycle.
  - [Architecture Decisions & Future Roadmap](docs/architecture/future_roadmap.md): Design rationale, quantization, and GPU backends.
- **Dialect & IR**:
  - [Canonical IR Specification](docs/ir/canonical_ir.md): Graph, Tensor, Operation, and Symbolic Shape rules.
  - [GGML Dialect Specification](docs/dialect/ggml_dialect.md): Memory layouts, stride inversion, GQA, and convolutions.
  - [Operator Reference Map](docs/reference/operator_reference.md): Master ATen / JAX $\to$ Canonical IR $\to$ GGML lookup table.
- **Runtime & Developer Guides**:
  - [Runtime Architecture & Embedding](docs/runtime/runtime_architecture.md): C++ embedding API and memory planning.
  - [Developer & Contributor Guide](docs/guides/developer_guide.md): Step-by-step checklist for adding new operators and frontends.
  - [Troubleshooting & Debugging Guide](docs/guides/troubleshooting_and_debugging.md): Top gotchas, mathematical pitfalls, and numerical debugging strategies.
  - [Autoregressive Generation & KV-Cache](docs/guides/autoregressive_generation.md): Dynamic sequence symbols and token-by-token greedy decoding.
