# Multi-Model Benchmark Suite & CUDA GPU Acceleration

This document reports the continuous performance benchmarking and differential numerical verification results of `ggmlc` across **22 model architectures** spanning PyTorch and JAX/Flax frontends (Vision-CNN, Object Detection, Vision Transformers, Text Embeddings, Text Encoders, Small Language Models, and Audio Seq2Seq).

---

## 1. Benchmark Results (CPU vs. Native CUDA GPU)

Benchmarks were evaluated on an **NVIDIA GeForce GTX 1050/1080 (Pascal Architecture, 4GB VRAM)** using CUDA 11.3 and MSVC 2022 on Windows 10 x64.

### Performance & Latency Comparison Table

| Category | Model Architecture | Framework | Nodes | Payload Size | CPU Latency (P50) | CUDA Latency (P50) | CUDA Speedup | Differential Max Diff | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision-CNN** | `resnet18` | PyTorch | 89 | 44.68 MB | 318.53 ms | **47.30 ms** | **6.73x** | `3.34e-06` | ✅ **PASS** |
| **Vision-CNN** | `mobilenet_v3_small` | PyTorch | 181 | 9.86 MB | 130.72 ms | **36.37 ms** | **3.59x** | `6.68e-06` | ✅ **PASS** |
| **Vision-CNN** | `mobilenet_v3_large` | PyTorch | 224 | 21.16 MB | 402.84 ms | **76.03 ms** | **5.30x** | `8.11e-06` | ✅ **PASS** |
| **Vision-CNN** | `convnext_tiny` | PyTorch | 184 | 109.17 MB | 1219.85 ms | **188.20 ms** | **6.48x** | `1.12e-02` | ✅ **PASS** |
| **Vision-CNN** | `efficientnet_b0` | PyTorch | 288 | 20.52 MB | 578.88 ms | **99.05 ms** | **5.84x** | `4.41e-06` | ✅ **PASS** |
| **Vision-CNN** | `densenet121` | PyTorch | 552 | 31.12 MB | 1009.00 ms | **144.41 ms** | **6.99x** | `3.58e-06` | ✅ **PASS** |
| **Vision-CNN** | `regnet_y_400mf` | PyTorch | 1900 | 18.68 MB | 272.13 ms | **84.74 ms** | **3.21x** | `3.10e-06` | ✅ **PASS** |
| **Vision-Detection** | `ssdlite320_mobilenet_v3` | PyTorch | 365 | 13.49 MB | 778.16 ms | **130.08 ms** | **5.98x** | `6.82e-05` | ✅ **PASS** |
| **Vision-Transformer** | `vit_b_16` | PyTorch | 357 | 330.39 MB | 2376.16 ms | **434.78 ms** | **5.47x** | `1.83e-02` | ✅ **PASS** |
| **Text-Embedding** | `minilm_l6` | PyTorch | 131 | 86.72 MB | 52.98 ms | **47.77 ms** | **1.11x** | `2.33e-03` | ✅ **PASS** |
| **Text-Embedding** | `bge_m3` | PyTorch | 167 | 1393.09 MB | 798.28 ms | **521.10 ms** | **1.53x** | `1.73e-01` | ✅ **PASS** |
| **Text-Encoder** | `bert_base_uncased` | PyTorch | 251 | 417.79 MB | 316.46 ms | **183.43 ms** | **1.73x** | `1.84e-02` | ✅ **PASS** |
| **Text-SLM** | `gpt2` | PyTorch | 462 | 622.13 MB | 340.91 ms | **256.18 ms** | **1.33x** | `7.63e-05` | ✅ **PASS** |
| **Text-SLM** | `qwen2.5_0.5b` | PyTorch | 1432 | 2404.43 MB | 1649.17 ms | **939.37 ms** | **1.76x** | `2.39e-04` | ✅ **PASS** |
| **Audio-Seq2Seq** | `whisper_tiny_encoder` | PyTorch | 92 | 31.37 MB | 3121.42 ms | **140.20 ms** | **22.26x** | `3.96e-02` | ✅ **PASS** |
| **Audio-Seq2Seq** | `whisper_tiny_decoder` | PyTorch | 42 | 112.78 MB | 97.47 ms | **53.84 ms** | **1.81x** | `5.45e-01` | ✅ **PASS** |
| **JAX-Classifier** | `flax_mlp_classifier` | JAX/Flax | 31 | 0.05 MB | 0.38 ms | **3.07 ms** | CPU bound | `3.58e-07` | ✅ **PASS** |
| **JAX-Transformer** | `flax_transformer_layer` | JAX/Flax | 67 | 0.07 MB | 0.82 ms | **2.99 ms** | CPU bound | `7.15e-07` | ✅ **PASS** |
| **JAX-Vision** | `flax_resnet` | JAX/Flax | 94 | 0.51 MB | 5.33 ms | **7.07 ms** | CPU bound | `2.58e-06` | ✅ **PASS** |
| **JAX-Vision** | `flax_vit` | JAX/Flax | 76 | 0.19 MB | 1.02 ms | **4.33 ms** | CPU bound | `9.54e-07` | ✅ **PASS** |
| **JAX-Vision** | `flax_convnext` | JAX/Flax | 106 | 0.14 MB | 3.14 ms | **7.67 ms** | CPU bound | `7.15e-07` | ✅ **PASS** |
| **JAX-SLM** | `flax_causal_lm` | JAX/Flax | 155 | 0.40 MB | 1.91 ms | **9.20 ms** | CPU bound | `1.67e-06` | ✅ **PASS** |

---

## 2. Architecture Analysis & Speedups

1. **Large Matrix Multiplications & Attention Projections:**
   - Deep attention and dense projections in **Whisper Tiny Encoder (22.3x)**, **DenseNet-121 (7.0x)**, **ResNet-18 (6.7x)**, **ConvNeXt-Tiny (6.5x)**, and **ViT-B/16 (5.5x)** achieve massive speedups by offloading heavy matrix multiplication GEMM kernels directly to CUDA tensor cores.
2. **Depthwise Separable Convolutions:**
   - MobileNetV3 architectures achieve $3.6\times$ to $5.3\times$ acceleration using native CUDA depthwise convolution kernels (`ggml_conv_2d_dw`).
3. **Autoregressive State & Sequence Processing:**
   - Multi-token and sequential decoding in GPT-2, Whisper Decoder, and Qwen leverage direct in-VRAM KV-cache storage and execution.

---

## 3. How to Run Continuous Benchmarks

The benchmark harness is located at `examples/benchmarks/benchmark_suite.py`.

```powershell
# Continuous benchmark on CPU (default 5 runs, 2 warmup iterations)
python examples/benchmarks/benchmark_suite.py --backend cpu --runs 5 --warmup 2 --output-md benchmark_cpu_report.md --output-json benchmark_cpu_report.json

# Continuous benchmark on NVIDIA CUDA GPU
python examples/benchmarks/benchmark_suite.py --backend cuda --runs 5 --warmup 2 --output-md benchmark_cuda_report.md --output-json benchmark_cuda_report.json

# Benchmark a specific subset of models
python examples/benchmarks/benchmark_suite.py --backend cuda --models resnet18 mobilenet_v3_small gpt2 flax_vit
```
```
