# `tab_completion.cpp` — Offline Continuous Diffusion Tab Completion Engine

A high-performance, 100% offline AI code autocompletion engine modeled after modern agentic code editor completions (**Antigravity Tab**, **Cursor Tab**), powered by **PlaidQ** (continuous latent diffusion for code).

Developed as part of the `ggmlc` compiler project to dogfood C++ code generation, cross-backend operator lowerings (CPU, CUDA, and Apple Metal), and realistic offline developer tools.

---

## Key Features

1. **Continuous Latent Diffusion Architecture**:
   - Uses continuous latent space ($D = 16$, unit-norm codebook $E \in \mathbb{R}^{V \times 16}$) instead of traditional autoregressive token-by-token loops.
   - Supports **few-step (8-step & 16-step) high-accuracy sampling** as well as fast 1-step/4-step evaluation.
   - Generates entire multi-token completions ($H = 16 \dots 64$ tokens) across a 256-token canvas simultaneously.
2. **Pure Mathematical Fill-In-The-Middle (FIM)**:
   - Zero prompt tagging hacks (e.g. `<PRE>`, `<SUF>`, `<MID>`).
   - Clean latent substitution: prefix and suffix token positions are pinned directly to their clean codebook embeddings $E[t] \cdot \sqrt{D}$ on the canvas; hole positions are initialized with Gaussian noise $z \sim \mathcal{N}(0, I)$.
   - The non-causal bidirectional transformer trunk attends simultaneously to both preceding prefix context and succeeding suffix context.
3. **Multi-Backend Acceleration & CUDA Graphs**:
   - **CPU**: Hand-tuned AVX2/FMA GEMV microkernels with OpenMP multi-threading.
   - **NVIDIA CUDA**: Unified `CUDAGraphManager` and Driver-VMM virtual memory page mapping, supporting all Compute Capabilities $\ge 6.0$ (Pascal through Blackwell). Static 256-token canvas captured into a single `cudaGraph_t` stream launch.
   - **Apple Metal**: Native Metal backend via `-DENABLE_METAL=ON` linking Apple `Foundation`, `Metal`, and `MetalKit`.
4. **Hole-Selective & Parallel Sampler**:
   - Computes Softmax and codebook projections $\hat{x}_0 = \text{Softmax}(S) @ E$ selectively only over active hole tokens $[P, P+H)$, skipping 224 invariant prefix/suffix tokens ($8\times$ reduction in host FLOPs).
   - Fully parallelized across CPU cores with `#pragma omp parallel for`.
5. **IDE Daemon Mode (`--daemon`)**:
   - High-throughput newline-delimited JSON-RPC interface over `stdin`/`stdout` for zero-overhead integration with VS Code, Cursor, Neovim, and Antigravity IDE extensions.
6. **Multiple Quantization Schemes**:
   - **FP16**: Full floating-point precision for reference parity and high-accuracy code infilling.
   - **Q4_0**: Ultra-compact 4-bit block quantization slashing memory footprint down to ~400 MB for lightweight background execution.
   - **Unsloth Dynamic & Q4_K_M**: Selective role-based quantization keeping embedding and attention projections in F16 while quantizing MLP projections to Q4, with strict 1D F32 preservation.

---

## Architectural Layout

```
examples/tab_completion/
├── CMakeLists.txt             # Standalone & unified CMake build configuration
├── include/
│   ├── infilling.h            # Discrete-to-latent canvas & context manager
│   ├── sampler.h              # Continuous diffusion schedule, DDIM solver & hole sampler
│   ├── engine.h               # Core TabCompletionEngine coordinator
│   └── plaidq_model.h         # ggmlc-generated C++ computation graph
├── src/
│   ├── infilling.cpp          # Prefix/suffix stitching & codebook pinning
│   ├── sampler.cpp            # OpenMP DDIM step, score temp, top-p/greedy sampling
│   ├── engine.cpp             # Model loading, CUDA graph execution & benchmark
│   └── main.cpp               # CLI entry point & JSON daemon mode
├── handoff.md                 # Deep technical architecture & learnings guide
└── README.md                  # This document
```

---

## Building

### A. Windows (MSVC 2022 + CUDA + Ninja)
```powershell
# Inside ggmlc build environment
cmake -B build-win-cuda -G Ninja -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-cuda --target tab_completion -j8
```

### B. Linux / WSL (CUDA or CPU)
```bash
cmake -B build -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tab_completion -j8
```

### C. macOS (Apple Metal)
```bash
cmake -B build-metal -DENABLE_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --target tab_completion -j8
```

---

## Usage & CLI Modes

### 1. Standalone Code Autocompletion (8-Step High Quality)
```bash
tab_completion \
  --model scratch/plaidq_0.7b_16step_q4_0.gguf \
  --steps 8 \
  --score-temp 0.5 \
  --prefix "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    " \
  --suffix "\n    return quicksort(left) + [pivot] + quicksort(right)\n" \
  --max-tokens 32 \
  --device cuda
```

### 2. Full 16-Step Maximum Convergence
```bash
tab_completion \
  --model scratch/plaidq_0.7b_16step_q4_0.gguf \
  --steps 16 \
  --score-temp 0.5 \
  --prefix "def quicksort(arr):\n    " \
  --max-tokens 32 \
  --device cuda
```

### 3. IDE Daemon Server (`--daemon`)
Run the persistent background process in your IDE extension:
```bash
tab_completion --model scratch/plaidq_0.7b_16step_q4_0.gguf --steps 8 --daemon --device cuda
```

#### JSON-RPC Interface Protocol:
**Ready handshake from engine:**
```json
{"status":"ready","device":"cuda","canvas_len":256}
```

**Request from IDE (send on stdin):**
```json
{"id": 1, "prefix": "def quicksort(arr):\n    ", "suffix": "", "max_tokens": 32}
```

**Response from engine (streamed to stdout):**
```json
{"id": 1, "completion": "if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    less = [x for x in arr[1:] if x <= pivot]\n    greater = [x for x in arr[1:] if x > pivot]\n    return quicksort(less) + [pivot] + quicksort(greater)\n", "total_ms": 3392.3, "tokens_per_second": 8.7}
```

---

## Empirical Latency & Throughput Measurements

Benchmarked on **NVIDIA GeForce GTX 1050 (4GB VRAM)** and **Intel CPU (4 Threads)** with canvas length $L = 256$ and infill hole $H = 32$ tokens across model quantizations and sampling step budgets:

| Model / Configuration | Steps ($N$) | Precision | Model Size | Hardware | Total Latency | Compute Pass | Infill Rate | Usability / Quality |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Qwen3-0.6B (Autoregressive)** | 32 tok | `Q4_0` | 409.0 MB | GTX 1050 | **1.53 s** | 1.53 s | **20.8 tok/s** | High (Coherent baseline) |
| **PlaidQ-0.7B (Diffusion)** | **1 step** | `Q4_0` | 399.8 MB | GTX 1050 | **1.16 s** | 0.55 s | **27.5 tok/s** | Mode collapse / Repetitive |
| **PlaidQ-0.7B (Diffusion)** | **4 steps** | `Q4_0` | 399.8 MB | GTX 1050 | **2.55 s** | 2.25 s | **12.5 tok/s** | Partial structure |
| **PlaidQ-0.7B (Diffusion)** | **8 steps** | `Q4_0` | 399.8 MB | GTX 1050 | **3.69 s** | 3.39 s | **8.7 tok/s** | **High (Usable algorithms, ~40% faster)** |
| **PlaidQ-0.7B (Diffusion)** | **16 steps** | `Q4_0` | 399.8 MB | GTX 1050 | **6.58 s** | 6.28 s | **4.9 tok/s** | **Highest quality code completion** |
| **PlaidQ-0.7B (Diffusion)** | **8 steps** | `Q4_0` | 399.8 MB | CPU (4T) | 56.90 s | 56.90 s | 0.6 tok/s | High (CPU fallback) |
| **PlaidQ-0.7B (Diffusion)** | **16 steps** | `FP16` | 1379.3 MB| GTX 1050 | 22.40 s | 22.10 s | 1.4 tok/s | Exact FP16 reference |

### Key Optimization Gains:
1. **CUDA Graph Stream Replay**: Slashed forward pass execution from **950 ms down to 369 ms per step** by pre-capturing the static 256-canvas transformer graph.
2. **Hole-Selective Softmax Sampler**: Reduced categorical reconstruction and argmaxing from **250 ms down to 47 ms per step** by focusing only on active hole tokens.
3. **8-Step vs 16-Step Operating Point**: 8 steps delivers virtually identical code quality to 16 steps while cutting generation time from **6.58 s down to 3.69 s** on consumer GPUs.

---

## Architectural Guidelines for Building Standalone C++ Tools with `ggmlc.codegen`

`tab_completion.cpp` serves as a blueprint for creating specialized, standalone C++ AI applications compiled with `ggmlc`. The following principles apply to upcoming domain-specific projects (such as **Google TimesFM 3.0 PyTorch Time-Series Engine**):

### 1. Maximize Core `ggmlc` Capabilities
Delegate all neural computation and target-specific execution to `ggmlc`:
- **IR Lowering & Operator Fusion**: Frontends (`torch.export`, JAX `jaxpr`) lower to Canonical IR and the GGML dialect, collapsing multi-head projections into fused kernels.
- **Dynamic Quantization**: Use `ggmlc` quantization policies (`UNSLOTH_DYNAMIC`, `Q4_K_M`, `Q8_0`, `Q4_0`) to halve or quarter VRAM footprints while strictly preserving 1D vectors in F32.
- **Runtime Acceleration**: Leverage `ModelExecutor`, static / multi-bucket `CUDAGraphManager`, Driver-VMM paged memory, and OpenMP CPU dispatch with zero external dependencies.

### 2. Encapsulate Domain-Specific Math in Modular Helpers
Keep the core neural trunk clean while implementing specialized mathematical frameworks in companion C++ classes:
- **PlaidQ**: Implemented continuous diffusion schedules, log-SNR curves, and DDIM reverse steps in `sampler.cpp` and canvas geometry in `infilling.cpp`.
- **TimesFM 3.0 (Time-Series)**: Implement Reversible Instance Normalization (RevIN), dynamic patching/unpatching (`patch_len=32`, `stride=32`), frequency indicator embedding, and quantile loss projections in dedicated `timesfm_pipeline.cpp` modules.

### 3. Application-Level Utilities for True Standalone Binaries
To deliver a single, portable executable (`.exe` on Windows, ELF binary on Linux) that functions without Python or complex runtime environments:
- **Data Ingestion**: High-throughput parsing for standard data formats (CSV, TSV, JSON, Parquet) directly in C++.
- **Multi-Format Export**: Support serializing predictions to JSON, CSV, or raw binary arrays.
- **Embedded Visualization**: Generate standalone SVG / HTML vector graphics directly from predictions (e.g. historical vs forecast curves with quantile confidence bands).
- **Compile-Time Embedded Web Dashboard & REST API**: Embed static HTML5 / CSS3 / Chart.js assets inside the C++ binary (via raw string literals or binary resource packing) and serve an interactive dashboard alongside a REST API over a lightweight embedded micro HTTP server.
