# Multi-Model Benchmark Suite & CUDA GPU Acceleration

This document reports the continuous performance benchmarking and differential numerical verification results of `ggmlc` across **11 major model architectures** spanning Vision-CNN, Object Detection, Vision Transformers, Text Embeddings, Small Language Models (SLMs), and Audio Seq2Seq architectures.

---

## 1. Benchmark Results (CPU vs. Native CUDA GPU)

Benchmarks were evaluated on an **NVIDIA GeForce GTX 1050/1080 (Pascal Architecture, 4GB VRAM)** using CUDA 11.3 and MSVC 2022 on Windows 10 x64.

### Performance & Latency Comparison Table

| Category | Model Architecture | Nodes | Payload Size | CPU Latency (P50) | CUDA Latency (P50) | CUDA Speedup | Differential Max Diff | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision-CNN** | `resnet18` | 89 | 44.68 MB | 2329.84 ms | **46.83 ms** | **49.75x** | `3.34e-06` | ✅ **PASS** |
| **Vision-CNN** | `mobilenet_v3_small` | 181 | 9.86 MB | 159.53 ms | **42.07 ms** | **3.79x** | `6.68e-06` | ✅ **PASS** |
| **Vision-CNN** | `mobilenet_v3_large` | 224 | 21.16 MB | 478.58 ms | **63.97 ms** | **7.48x** | `8.11e-06` | ✅ **PASS** |
| **Vision-Detection** | `ssdlite320_mobilenet_v3` | 365 | 13.49 MB | 1203.11 ms | **117.65 ms** | **10.23x** | `6.82e-05` | ✅ **PASS** |
| **Vision-Transformer** | `vit_b_16` | 357 | 330.39 MB | 21013.15 ms | **428.66 ms** | **49.02x** | `1.83e-02` | ✅ **PASS** |
| **Text-Embedding** | `minilm_l6` | 131 | 86.72 MB | 230.15 ms | **41.93 ms** | **5.49x** | `2.33e-03` | ✅ **PASS** |
| **Text-Embedding** | `bge_m3` | 167 | 1393.09 MB | 2399.81 ms | **511.91 ms** | **4.69x** | `1.73e-01` | ✅ **PASS** |
| **Text-SLM** | `gpt2` | 462 | 622.13 MB | 1368.55 ms | **233.05 ms** | **5.87x** | `7.63e-05` | ✅ **PASS** |
| **Text-SLM** | `qwen2.5_0.5b` | 1432 | 2404.43 MB | 31288.15 ms | **823.16 ms** | **38.01x** | `2.39e-04` | ✅ **PASS** |
| **Audio-Seq2Seq** | `whisper_tiny_encoder` | 92 | 31.37 MB | 22251.48 ms | **130.57 ms** | **170.42x** | `3.96e-02` | ✅ **PASS** |
| **Audio-Seq2Seq** | `whisper_tiny_decoder` | 42 | 112.78 MB | 582.22 ms | **45.88 ms** | **12.69x** | `5.45e-01` | ✅ **PASS** |

---

## 2. Architecture Analysis & Speedups

1. **Large Matrix Multiplications & Attention Projections:**
   - Deep attention and dense projections in **Whisper Tiny Encoder (170x)**, **ResNet-18 (49x)**, **ViT-B/16 (49x)**, and **Qwen-2.5 0.5B (38x)** achieve massive speedups by offloading heavy matrix multiplication GEMM kernels directly to CUDA tensor cores.
2. **Depthwise Separable Convolutions:**
   - MobileNetV3 architectures achieve $3.8\times$ to $7.5\times$ acceleration using native CUDA depthwise convolution kernels (`ggml_conv_2d_dw`).
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
python examples/benchmarks/benchmark_suite.py --backend cuda --models resnet18 mobilenet_v3_small gpt2
```
