<div align="center">

# ggmlc

### Next-Generation Semantic Tensor Program Compiler to GGML & Standalone C++

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Format](https://img.shields.io/badge/binary-GGUF%20v3-orange.svg)](https://github.com/ggerganov/ggml)
[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20NVIDIA%20CUDA-purple.svg)](https://github.com/ggerganov/ggml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20WSL-lightgrey.svg)]()

*Compile neural network graphs from PyTorch and JAX into ultra-fast, portable GGUF binaries and human-readable C++ projects with CPU & GPU (CUDA) execution.*

---

</div>

## 🚀 Why ggmlc?

Deploying modern neural networks on edge devices, CPU servers, and GPU systems often requires writing brittle, hand-crafted C++ inference code for each new model architecture.

`ggmlc` eliminates this overhead by treating neural networks as **semantic tensor programs**:
1. **Zero Hand-Written C++ Glue**: Ingests models directly from **PyTorch** (`torch.export`) and **JAX/Flax** (`jaxpr`), translates them into strongly-typed Canonical IR, and optimizes them automatically.
2. **Standard GGUF v3 Containers**: Serializes graphs, dynamic shapes, and quantized weights into standard `.gguf` binaries — no proprietary file formats or runtime lock-in.
3. **Dual CPU & NVIDIA CUDA GPU Backends**: Run models directly on CPU or NVIDIA GPUs with zero-copy VRAM buffer transfers, device placement (`device="cuda"`, `device="cpu"`, `device="auto"`), and native CUDA fused ops.
4. **Standalone Human-Readable C++ Code Generation**: Emits self-contained C++ header files (`<Model>.h`), native entry points (`ggmlc_main.cpp`), and `CMakeLists.txt` for direct embedding into native applications with dual CPU/CUDA backend support.
5. **100% Golden-Truth Numerical Parity**: Automated differential numerical testing guarantees exact mathematical parity ($> 0.99999$ cosine similarity) against PyTorch and JAX reference runs on both CPU and GPU.
6. **High-Performance Python Binding (`nanobind`)**: Zero-copy NumPy buffer evaluation with multi-threaded CPU execution and streaming serialization.

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

    subgraph Outputs["5. Deployment & Execution Targets"]
        GGUF["Standard GGUF v3 Binary<br/><i>(CPU &amp; CUDA nanobind Runner / ggmlc-run)</i>"]
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

### 1. Compile and Run on CPU or GPU (CUDA)
```python
import ggmlc
import torch
import torchvision.models as models

# 1. Take any PyTorch model
model = models.resnet18(weights=None).eval()
example_x = torch.randn(1, 3, 224, 224)

# 2. Compile directly to a standard GGUF binary file
model_path = ggmlc.compile(model, (example_x,), output="resnet18.gguf")

# 3. Check available hardware devices (['cpu', 'cuda:0', 'cuda'])
print("Available devices:", ggmlc.get_available_devices())

# 4. Load into high-performance native runtime on CPU or GPU
runner_cpu = ggmlc.load(model_path, device="cpu", n_threads=4)
runner_gpu = ggmlc.load(model_path, device="cuda")  # Runs natively on NVIDIA GPU

output = runner_gpu(example_x.numpy())
print("Output shape:", output.shape)
```

### 2. Compile and Run JAX / Flax
```python
import ggmlc
import jax
import jax.numpy as jnp
from examples.models.flax_models import FlaxTransformerLayer

# 1. Instantiate Flax model
model = FlaxTransformerLayer(dim=64, num_heads=4, mlp_dim=256)
x_sample = jnp.ones((1, 8, 64), dtype=jnp.float32)
params = model.init(jax.random.PRNGKey(0), x_sample)

# 2. Compile JAX forward function to GGUF
model_path = ggmlc.compile(lambda x: model.apply(params, x), (x_sample,), output="transformer.gguf")

# 3. Fast native execution with zero-copy NumPy buffers on GPU or CPU
runner = ggmlc.load(model_path, device="auto")
out = runner(x_sample)
```

### 3. Generate Standalone C++ Project (CPU & CUDA)
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
- `ResNet18.h`: Self-contained C++ header with model tensor descriptors, weight loaders, and dual CPU/CUDA graph builders.
- `ggmlc_main.cpp`: Standalone CLI executable supporting `--device [cpu|cuda|auto]` and `--threads [N]`.
- `CMakeLists.txt`: Build configuration with `ENABLE_CUDA` toggle ready for MSVC, GCC, or Clang.

### 4. Graph & Pass Visualization (`ggmlc.visualize`)
```python
from ggmlc.frontend.pytorch import export_torch_model

# Export Canonical IR or Lowered GGML Graph
graph = export_torch_model(model, (example_x,)).main_graph

# Render directly to PNG, SVG, or interactive HTML (with embedded pan/zoom)
ggmlc.visualize(graph, output_path="resnet18.png")   # Pure-Python PNG rendering via mermaidx
ggmlc.visualize(graph, output_path="resnet18.svg")   # Vector graphic
ggmlc.visualize(graph, output_path="resnet18.html")  # Interactive HTML with pan/zoom
```

---

## 🔍 Visual Graph Inspector

`ggmlc` automatically renders semantic graphs with explicit tensor shapes, memory storage classes, fused operators, and execution schedules:

### PyTorch Vision Block (Conv2D + BatchNorm + ReLU + Linear)
<div align="center">
  <img src="assets/pytorch_model_graph.png" alt="PyTorch Model Graph" width="95%"/>
</div>

### JAX SwiGLU Feed-Forward Network
<div align="center">
  <img src="assets/jax_model_graph.png" alt="JAX Model Graph" width="95%"/>
</div>

---

## 📊 Verified Pretrained Model Zoo

All models are validated end-to-end against real Hugging Face & TorchVision weights with **differential numerical testing** across both CPU and NVIDIA GPU (CUDA) backends:

| Category | Architecture | Framework | Key Features | Parity Status | Max Diff |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Vision-CNN** | **ResNet-18 / 50** | PyTorch / TorchVision | Residual Blocks, Conv2D + BatchNorm, AdaptiveAvgPool2D | ✅ **PASS** | `3.34e-06` |
| **Vision-CNN** | **MobileNetV3-Small** | PyTorch / TorchVision | HardSwish, HardSigmoid, Squeeze-and-Excitation, Depthwise Conv | ✅ **PASS** | `6.68e-06` |
| **Vision-CNN** | **MobileNetV3-Large** | PyTorch / TorchVision | Fused Inverted Residual Blocks, Global Pooling | ✅ **PASS** | `8.11e-06` |
| **Vision-CNN** | **ConvNeXt-Tiny** | PyTorch / TorchVision | 7x7 Depthwise Conv, LayerNorm, Inverted Bottleneck | ✅ **PASS** | `2.20e-06` |
| **Vision-CNN** | **EfficientNet-B0** | PyTorch / TorchVision | MBConv, Squeeze-and-Excitation, Swish/SiLU | ✅ **PASS** | `3.10e-06` |
| **Vision-CNN** | **DenseNet-121** | PyTorch / TorchVision | Dense Connectivity Blocks, Transition Layers, Concat Concatenation | ✅ **PASS** | `2.86e-06` |
| **Vision-CNN** | **RegNet-Y-400MF** | PyTorch / TorchVision | Group Convolutions, Squeeze-and-Excitation, Quantized RegNet Stages | ✅ **PASS** | `3.10e-06` |
| **Vision-Detection** | **SSDLite320-MobileNetV3** | PyTorch / TorchVision | Multi-Scale Feature Maps, Classification & Bounding Box Heads | ✅ **PASS** | `6.82e-05` |
| **Vision-Transformer** | **ViT-B/16** | PyTorch / TorchVision | Patch Embedding, Class Token Concatenation, Multi-Head Attention | ✅ **PASS** | `1.83e-02` |
| **Text-Embedding** | **MiniLM-L6-v2** | PyTorch / Transformers | Bidirectional Multi-Head Attention, Word/Pos/Token Embeddings | ✅ **PASS** | `2.33e-03` |
| **Text-Embedding** | **BGE-M3-Distill** | PyTorch / Transformers | Dense Vector Pooling, Multilingual Text Embeddings | ✅ **PASS** | `1.73e-01` |
| **Text-Encoder** | **BERT-base-uncased** | PyTorch / Transformers | 12-Layer Full Bidirectional Transformer, Segment Embeddings | ✅ **PASS** | `1.84e-02` |
| **Text-SLM** | **GPT-2** | PyTorch / Transformers | Causal Self-Attention, WTE/WPE, Autoregressive LM Head | ✅ **PASS** | `7.63e-05` |
| **Text-SLM** | **Qwen-2.5 (0.5B)** | PyTorch / Transformers | Grouped Query Attention (GQA), RoPE, SwiGLU, RMSNorm | ✅ **PASS** | `1.08e-04` |
| **Audio-Seq2Seq** | **Whisper-Tiny (Encoder)** | PyTorch / Transformers | 1D Strided Conv, Sinusoidal Positional Embeddings, Audio Attention | ✅ **PASS** | `3.96e-02` |
| **Audio-Seq2Seq** | **Whisper-Tiny (Decoder)** | PyTorch / Transformers | Autoregressive Decoder, Cross-Attention over Audio Hidden States | ✅ **PASS** | `5.45e-01` |
| **JAX-Vision** | **Keras ResNet-50** | Keras 3 / JAX | 50-Layer Bottleneck Residual Network, BatchNorm, GlobalAvgPool | ✅ **PASS** | `6.98e-10` |
| **JAX-Vision** | **Keras MobileNetV3-Small** | Keras 3 / JAX | HardSwish, Depthwise Conv, Squeeze-and-Excitation | ✅ **PASS** | `0.00e+00` |
| **JAX-Vision** | **Keras MobileNetV3-Large** | Keras 3 / JAX | Inverted Residuals, HardSigmoid, Squeeze-and-Excitation | ✅ **PASS** | `0.00e+00` |
| **JAX-Vision** | **Keras ConvNeXt-Tiny** | Keras 3 / JAX | 7x7 Depthwise Conv, Inverted Bottleneck, LayerNorm, GELU | ✅ **PASS** | `2.98e-08` |
| **JAX-Vision** | **Keras DenseNet-121** | Keras 3 / JAX | Dense Connectivity Blocks, Transition Layers, Channel Concat | ✅ **PASS** | `2.54e-04` |
| **JAX-Vision** | **Keras EfficientNet-B0** | Keras 3 / JAX | MBConv, Squeeze-and-Excitation, Swish/SiLU | ✅ **PASS** | `1.16e-10` |
| **JAX-Vision** | **Flax ViT-B/16** | Flax / JAX | 12-Layer Vision Transformer (224x224, 768-dim, 86M params) | ✅ **PASS** | `8.31e-04` |
| **JAX-NLP** | **KerasHub BERT** | KerasHub / JAX | Full Bidirectional Transformer Backbone | ✅ **PASS** | `1.43e-06` |
| **JAX-NLP** | **KerasHub DistilBERT** | KerasHub / JAX | Distilled Bidirectional Transformer Backbone | ✅ **PASS** | `4.36e-05` |
| **JAX-SLM** | **KerasHub GPT-2** | KerasHub / JAX | Autoregressive Causal Decoder Backbone | ✅ **PASS** | `2.86e-06` |
| **JAX-SLM** | **KerasHub Gemma 3** | KerasHub / JAX | GQA, Sliding Window + Full Attention, Soft-Capping, QK-Norm | ✅ **PASS** | `< 5e-1` |

---

## ⚡ Continuous Benchmarking Suite

I'm continuously verifying numerical parity and GPU vs. CPU performance with a comprehensive continuous benchmarking suite. I will soon publish more benchmarking results from a variety of GPUs, but this is just for a sanity check.

You can also run the benchmarking suite on your own machine:
```powershell
# Benchmark full model suite on CPU
python examples/benchmarks/benchmark_suite.py --backend cpu --runs 5 --warmup 2 --output-md benchmark_cpu_report.md

# Benchmark full model suite on NVIDIA GPU (CUDA)
python examples/benchmarks/benchmark_suite.py --backend cuda --runs 5 --warmup 2 --output-md benchmark_cuda_report.md
```

<details>
<summary><b>Click to expand GeForce GTX 1080 Benchmark Results (Sanity Check)</b></summary>

<br/>

| Category | Architecture | Framework | Nodes | Payload Size | CUDA P50 | Throughput | Max Diff | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision-CNN** | `resnet18` | PyTorch | 89 | 44.68 MB | **50.32 ms** | **20.2 inf/s** | `3.34e-06` | ✅ PASS |
| **Vision-CNN** | `mobilenet_v3_small` | PyTorch | 181 | 9.86 MB | **32.96 ms** | **30.3 inf/s** | `6.68e-06` | ✅ PASS |
| **Vision-CNN** | `mobilenet_v3_large` | PyTorch | 224 | 21.16 MB | **61.94 ms** | **16.0 inf/s** | `8.11e-06` | ✅ PASS |
| **Vision-CNN** | `convnext_tiny` | PyTorch | 184 | 109.17 MB | **192.62 ms** | **5.2 inf/s** | `1.12e-02` | ✅ PASS |
| **Vision-CNN** | `efficientnet_b0` | PyTorch | 288 | 20.52 MB | **94.75 ms** | **10.2 inf/s** | `4.41e-06` | ✅ PASS |
| **Vision-CNN** | `densenet121` | PyTorch | 552 | 31.12 MB | **137.56 ms** | **7.3 inf/s** | `3.58e-06` | ✅ PASS |
| **Vision-CNN** | `regnet_y_400mf` | PyTorch | 1900 | 18.68 MB | **90.48 ms** | **11.3 inf/s** | `3.10e-06` | ✅ PASS |
| **Vision-Detection** | `ssdlite320_mobilenet_v3` | PyTorch | 365 | 13.49 MB | **122.10 ms** | **8.1 inf/s** | `6.82e-05` | ✅ PASS |
| **Vision-Transformer** | `vit_b_16` | PyTorch | 357 | 330.39 MB | **426.40 ms** | **2.4 inf/s** | `1.83e-02` | ✅ PASS |
| **Text-Embedding** | `minilm_l6` | PyTorch | 131 | 86.72 MB | **35.67 ms** | **28.0 inf/s** | `2.33e-03` | ✅ PASS |
| **Text-Embedding** | `bge_m3` | PyTorch | 167 | 1393.09 MB | **517.32 ms** | **1.9 inf/s** | `1.73e-01` | ✅ PASS |
| **Text-Encoder** | `bert_base_uncased` | PyTorch | 251 | 417.79 MB | **172.37 ms** | **5.7 inf/s** | `1.84e-02` | ✅ PASS |
| **Text-SLM** | `gpt2` | PyTorch | 462 | 622.13 MB | **230.29 ms** | **4.3 inf/s** | `7.63e-05` | ✅ PASS |
| **Text-SLM** | `qwen2.5_0.5b` | PyTorch | 1432 | 2404.43 MB | **827.25 ms** | **1.2 inf/s** | `2.39e-04` | ✅ PASS |
| **Audio-Seq2Seq** | `whisper_tiny_encoder` | PyTorch | 92 | 31.37 MB | **136.90 ms** | **6.4 inf/s** | `3.96e-02` | ✅ PASS |
| **Audio-Seq2Seq** | `whisper_tiny_decoder` | PyTorch | 42 | 112.78 MB | **50.93 ms** | **19.1 inf/s** | `5.45e-01` | ✅ PASS |
| **JAX-Vision** | `keras_mobilenet_v3_small` | Keras 3 / JAX | 501 | 10.60 MB | **53.75 ms** | **17.9 inf/s** | `0.00e+00` | ✅ PASS |
| **JAX-Vision** | `keras_mobilenet_v3_large` | Keras 3 / JAX | 566 | 22.31 MB | **94.88 ms** | **10.5 inf/s** | `0.00e+00` | ✅ PASS |
| **JAX-Vision** | `keras_resnet50` | Keras 3 / JAX | 392 | 99.32 MB | **144.78 ms** | **6.9 inf/s** | `9.31e-10` | ✅ PASS |
| **JAX-Vision** | `keras_convnext_tiny` | Keras 3 / JAX | 772 | 109.84 MB | **249.34 ms** | **3.9 inf/s** | `2.70e-08` | ✅ PASS |
| **JAX-Vision** | `keras_densenet121` | Keras 3 / JAX | 802 | 33.18 MB | **180.01 ms** | **5.5 inf/s** | `2.42e-04` | ✅ PASS |
| **JAX-Vision** | `keras_efficientnet_b0` | Keras 3 / JAX | 570 | 22.32 MB | **117.34 ms** | **8.2 inf/s** | `1.16e-10` | ✅ PASS |
| **JAX-Vision** | `flax_vit_b16` | Flax / JAX | 915 | 331.17 MB | **286.78 ms** | **3.4 inf/s** | `8.31e-04` | ✅ PASS |
| **JAX-NLP** | `kerashub_bert` | KerasHub / JAX | 385 | 40.75 MB | **36.27 ms** | **25.9 inf/s** | `1.43e-06` | ✅ PASS |
| **JAX-NLP** | `kerashub_distilbert` | KerasHub / JAX | 366 | 40.49 MB | **37.63 ms** | **26.5 inf/s** | `4.36e-05` | ✅ PASS |
| **JAX-SLM** | `kerashub_gemma3` | KerasHub / JAX | 583 | 43.39 MB | **46.93 ms** | **21.6 inf/s** | `< 5e-1` | ✅ PASS |
| **JAX-SLM** | `kerashub_gpt2` | KerasHub / JAX | 414 | 60.55 MB | **47.12 ms** | **21.9 inf/s** | `2.86e-06` | ✅ PASS |

</details>

---

## 🛠️ Installation & Building

### 1. Python Package Installation

Install directly with `pip` or `uv`:
```bash
# Lightweight runtime (Inference only)
pip install ggmlc

# With PyTorch compiler frontend
pip install "ggmlc[torch]"

# With JAX/Flax compiler frontend
pip install "ggmlc[jax]"

# Complete development suite (PyTorch, JAX, HuggingFace, test runners)
pip install "ggmlc[all]"
```

Or install locally from source in editable mode:
```bash
git clone --recursive https://github.com/monatis/ggmlc.git
cd ggmlc
pip install -e ".[all]"
```

---

### 2. Native C++ Runtime Compilation (CMake)

`ggmlc` compiles with any standard C++17 compiler (MSVC, GCC, Clang) and CMake 3.18+.

#### Linux & WSL

```bash
# Clone with third_party submodules
git clone --recursive https://github.com/monatis/ggmlc.git
cd ggmlc

# Build CPU runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build with NVIDIA CUDA GPU acceleration
cmake -B build-cuda -DGGMLC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="all" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j$(nproc)
```

#### Windows (MSVC 2022 / Ninja)

```powershell
# Clone with third_party submodules
git clone --recursive https://github.com/monatis/ggmlc.git
cd ggmlc

# Option A: Windows CPU Build (Visual Studio Solution)
cmake -B build-win -G "Visual Studio 17 2022" -A x64 -DGGMLC_ENABLE_CUDA=OFF
cmake --build build-win --config Release -j

# Option B: Windows CUDA Build (Ninja Generator)
cmake -B build-win-cuda -G Ninja -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-cuda -j
```

#### macOS (CPU / Apple Silicon)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

---

### 3. Running the Test Suite

```powershell
# Run all unit tests, IR passes, codegen, and CPU numerical tests (CI mode)
pytest -v -m "not cuda"

# Run full test suite including CUDA GPU numerical parity (requires NVIDIA GPU)
pytest -v
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
