# AGENTS.md — Guidelines for Coding Agents in `ggmlc`

## Core Philosophy
1. **A neural network is a semantic tensor program.**
2. **Framework Independence**: Frontends (`torch.export`, JAX `jaxpr`) produce Canonical IR.
3. **Target-Specific Lowering**: The GGML Dialect translates Canonical IR into GGML execution semantics.
4. **Correctness Before Optimization**: Differential testing against reference frameworks is the golden source of truth.
5. **Small Vertical Slices**: Every feature/op must have: IR definition, lowering, shape inference, serialization, C++ runtime support, and differential numerical tests.
6. **Git Workflow & AGENTS.md Hygiene**:
    - **NEVER EVER commit `AGENTS.md` to git history**, and **DO NOT include it in `.gitignore`** (it must remain untracked in the workspace).
    - Small, incremental local commits are encouraged for features and fixes, but **NEVER EVER push to remote**.

## Current Focus (2026-09-20)
- **Phase 3 / 3b compile-time bake is done and merged.** Do not reopen RMS/affine bake unless a regression appears.
- **Phase 4 (quantized KV) and Phase 5 (swarm serving) are deferred.** Do not start them in the next chat.
- **Current work**: third production showcase — **Laya / Jev-style System 1 decisions** under `examples/laya`. See **§17**. Do not start Qwen3-TTS, Flux, Phase 4, or Phase 5.

## Adding a New Operator Checklist
1. Identify source operator semantics (PyTorch ATen or JAX primitive).
2. Add or verify Canonical IR Op in `python/ggmlc/ir/op.py`.
3. Add shape inference rule in `python/ggmlc/ir/shape.py` or op schema.
4. Add frontend importer rule in `python/ggmlc/frontend/`.
5. Add GGML dialect lowering in `python/ggmlc/dialect/ggml/lowering.py`.
6. Add opcode / kernel handling in C++ runtime `runtime/src/`.
7. Add Python unit test in `tests/ops/`.
8. Add differential numerical test in `tests/numerical/`.

## Common Commands & Build Workflows

### 1. Python Environment & Package Management
- **ALWAYS use `uv`** (never run bare `pip`):
  - Install packages: `uv pip install <package>` (e.g., `uv pip install keras keras-hub ninja pytest ruff`)
  - List packages: `uv pip list`
- Python executable: `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux)
- **Scratch Exploration Scripts**: Always write exploration and debugging scripts into the `scratch/` directory (e.g. `scratch/explore_jax_pad.py`) instead of running complex multi-line inline strings with `python -c`.

### 2. Multi-Framework Model Sources & Verification Status
The framework supports **27 production model architectures** across PyTorch, Keras 3, KerasHub, and Flax frontends:
1. **Batch 1 (Keras 3 JAX Applications)**:
   - Full-scale vision models compiled from JAX (`KERAS_BACKEND=jax`): `ResNet50`, `MobileNetV3Large`, `MobileNetV3Small`, `ConvNeXtTiny`, `EfficientNetB0`, `DenseNet121`.
2. **Batch 2 (KerasHub & Full Flax Architectures)**:
   - Modern NLP, SLM, and Vision backbones from `keras_hub` and Flax (`KERAS_BACKEND=jax`): `Gemma 3`, `BERT`, `DistilBERT`, `GPT-2`, and Flax `ViT-B/16`.
3. **Batch 3 (Keras 3 Multi-Backend Cross-Frontend Differential Verification)**:
   - Verified exact IR canonicalization and cross-backend numerical parity across models exported from both PyTorch (`KERAS_BACKEND=torch` -> `torch.export`) and JAX (`KERAS_BACKEND=jax` -> `jax.make_jaxpr`):
     - MLP Classifier: Bitwise exact parity (`max_diff = 0.00e+00`, Cosine Similarity = `1.000000`).
     - Conv2D + BatchNorm + Activation: `max_diff = 4.77e-07`, Cosine Similarity = `0.999999`.
     - ResNet Residual Block: `max_diff = 8.94e-07`, Cosine Similarity = `0.999999`.
     - LayerNorm / Normalization: `max_diff = 0.00e+00`, Cosine Similarity = `1.000000`.

### 3. Benchmarking Principle: Realistic Models Only
- **Benchmarking is NOT unit testing.**
- Micro/toy networks (e.g. 50 KB, 16x16 images, single blocks) are dominated by CUDA kernel launch and stream synchronization overhead (~2-3 ms on Windows), making them appear artificially slower on GPU than CPU.
- Continuous benchmarks must ONLY run realistic, full-scale production models (e.g. standard 224x224 image resolutions, full 12-layer / 768-dim Transformers, multi-megabyte checkpoints) to accurately reflect true CPU vs GPU hardware throughput and acceleration.

### 4. Native Windows Build Commands

#### A. Native Windows CUDA Build (MSVC 2022 BuildTools + CUDA 12.8 + Ninja)
When building on Windows with CUDA support:
```powershell
# Set up MSVC, Windows SDK, and CUDA Toolkit paths in environment (RTX 4050 Laptop / MSVC 2022 BuildTools / CUDA 12.8)
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Users\Gigabyte\ggmlc\.venv\Scripts;" + $env:PATH
$env:INCLUDE = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;" + $env:INCLUDE
$env:LIB = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64;" + $env:LIB
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

# Configure with Ninja and MSVC STL version bypass
cmake -B build-win-cuda -G Ninja -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" -DPython_EXECUTABLE="C:\Users\Gigabyte\ggmlc\.venv\Scripts\python.exe"

# Compile runtime, shared library, and Python nanobind module
cmake --build build-win-cuda -j8
```

#### B. Native Windows CPU Build
```powershell
cmake -B build-win -G "Visual Studio 17 2022" -A x64 -DGGMLC_ENABLE_CUDA=OFF -DPython_EXECUTABLE="C:\Users\Gigabyte\ggmlc\.venv\Scripts\python.exe"
cmake --build build-win --config Release -j8
```

### 5. Linux / WSL Build (Reserved for Linux validation)
```bash
wsl cmake -B build -DGGMLC_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
wsl cmake --build build -j8
```

### 6. Testing Commands
- **Run all unit & numerical tests (Windows):**
  ```powershell
  pytest -v --ignore=tests/numerical/test_ggml_ops_differential.py
  ```
- **Run E2E Full Models Test Suite:**
  ```powershell
  pytest tests/e2e/test_full_models.py -v
  ```
- **Run Native CUDA E2E Tests:**
  ```powershell
  pytest tests/e2e/test_cuda_models.py -v
  ```
- **Run JAX / Flax / Keras Tests:**
  ```powershell
  pytest tests/e2e/test_jax_flax_models.py -v
  ```
- **Run Continuous Batching & VMM Page Manager Tests:**
  ```powershell
  pytest tests/e2e/test_continuous_batching.py -v
  ```
- **Run Radix Tree Prefix Caching Tests:**
  ```powershell
  pytest tests/e2e/test_prefix_caching.py -v
  ```

### 7. Continuous Benchmarking Commands
- **Benchmark Suite on CPU:**
  ```powershell
  python examples/benchmarks/benchmark_suite.py --backend cpu --runs 5 --warmup 2 --output-md benchmark_cpu_report.md --output-json benchmark_cpu_report.json
  ```
- **Benchmark Suite on CUDA GPU:**
  ```powershell
  python examples/benchmarks/benchmark_suite.py --backend cuda --runs 5 --warmup 2 --output-md benchmark_cuda_report.md --output-json benchmark_cuda_report.json
  ```
- **KV Cache & Autoregressive Decode Benchmark:**
  ```powershell
  python examples/benchmarks/benchmark_kv_cache.py --device both --threads 4
  ```
- **vs `llama.cpp` (pp/tg matrix, Q8_0, ubatch 512):**
  ```powershell
  python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --backend cuda --cuda-graph --ubatch 512 --runs 5 --skip-numerical-check
  ```
- **vs `llama.cpp` end-to-end wall clock (prefill+decode one timed shot):**
  ```powershell
  python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --backend cuda --cuda-graph --e2e --skip-numerical-check
  ```
- **Fusion A/B compile (never overwrite baseline `*_q8_0.gguf`):**
 ```powershell
 python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --force-recompile --fusion-no-horizontal-mlp --gguf-suffix no_hmlp ...
 python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --force-recompile --fusion-no-horizontal-qkv --gguf-suffix no_hqkv ...
 python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --force-recompile --fusion-no-bake-rms --gguf-suffix no_bake_rms ...
 python examples/benchmarks/compare_ggmlc_vs_llama_cpp.py --force-recompile --fusion-no-bake-affine --gguf-suffix no_bake_affine ...
 ```
- Latest reference artifacts: `scratch/compare_ggmlc_vs_llama_report.{md,json}` (full pp/tg matrix, Phase 2), `scratch/compare_ggmlc_vs_llama_e2e_*`, `scratch/compare_ggmlc_vs_llama_fusion_ab_summary.md`, `scratch/phase2_*`, `scratch/affine_bake_ab_summary.md`. Details: `docs/benchmarks/ggmlc_vs_llama_cpp.md`.

### 8. Linting and Formatting
- Format code: `ruff format python/ tests/`
- Lint and auto-fix: `ruff check --fix python/ tests/`

### 9. Documentation Maintenance Rule
- **Always update documentation to reflect reality**: After every development cycle, feature addition, or benchmark run, immediately update `README.md` and relevant documentation pages in `docs/` (such as `docs/benchmarks/model_benchmark_suite.md`, `docs/runtime/runtime_architecture.md`, etc.). Never leave documentation stale or out-of-sync with active code and benchmark results.

### 10. Standalone C++ Binary Runner (`ggmlc-run`) Reference

`ggmlc-run` is the native standalone C++ runner that executes any compiled GGUF model without Python or LibTorch dependencies.

- **Executable Locations**:
  - Windows CUDA Build: `.\build-win-cuda\runtime\ggmlc-run.exe`
  - Windows CPU Build: `.\build-win\runtime\Release\ggmlc-run.exe`
  - Linux/WSL Build: `./build/runtime/ggmlc-run`

#### A. Core Architectural Behaviors
1. **Capability-Driven Validation**: Reads GGUF KV metadata directly (`tokenizer.*`, `pipeline.*`, `ggmlc.tasks`). Rejects incompatible operations with actionable error messages (e.g., passing `--image` to a model without vision preprocessing metadata).
2. **Compile-Time Tasks (`ggmlc.tasks`)**: Models declare task strings during compilation via `ggmlc.compile(..., tasks=["classification"])` or multi-task `tasks=["similarity", "embedding"]`. `ggmlc-run` automatically formats output accordingly (Softmax + Top-5 classes, vector + L2 norm preview, cosine score) without brittle shape heuristics.
3. **Clean Streaming in `--chat`**: By default, streams **only** the assistant's generated response directly to `stdout`. Prompt scaffolding and special control tokens are suppressed unless explicitly requested.
4. **Special Token & Whitespace Preservation**: C++ `BPETokenizer` scans special tokens (`<|im_start|>`, `<|im_end|>`, `<start_of_turn>`, etc.) before BPE word splitting to emit exact token IDs, preserves byte-level whitespace (`\n` -> 198), and never appends `eos_token` on prompt continuation.
5. **Dynamic Sequence Length Deduction**: Automatically binds dynamic sequence symbols (e.g. `s`) to `current_tokens.size()` on every autoregressive step.
6. **Planned Arena Memory Reuse**: Memory arena reuse planning is active by default. Use `--unplanned` for debugging tensor memory reuse or corruption.

#### B. CLI Flags & Options Matrix
| Category | Flag / Option | Description & Usage |
| :--- | :--- | :--- |
| **Inspection** | `-h, --help` | Displays categorized help menu with copy-pasteable examples. |
| | `--info` | Inspects GGUF metadata, tensor graph, dynamic symbols, declared tasks, and capabilities. |
| **Text Gen & Chat** | `--chat <msg>` | Formats prompt with embedded chat template (ChatML, Gemma, Llama-3) and streams assistant response. |
| | `--prompt <str>` | Autoregressive continuation from a raw prompt string. |
| | `--system <msg>` | System instructions for chat template. |
| | `--generate` | Explicitly enables autoregressive generation loop. |
| | `--max-tokens <N>` | Maximum new tokens to generate (default: `32`). |
| | `--temperature <T>`| Sampling temperature (`0.0` = greedy argmax, default: `0.0`). |
| | `--top-p <P>` | Nucleus sampling probability cutoff (default: `0.9`). |
| | `--echo-prompt` | Echoes prompt before streaming generated output (for debugging). |
| | `--show-special` | Prints special control tokens (e.g. `<|im_end|>`) instead of stopping silently. |
| | `--chunk-size, --ubatch <C>` | Uniform prompt prefill chunk size (default: `128`, `0` = disabled single-pass). |
| **Preprocessing** | `--image <name:file>` | Preprocesses image (bicubic resize, normalize) via GGUF pipeline and binds to input tensor `name`. |
| | `--text <name:str>` | Tokenizes text string via GGUF tokenizer and binds to input tensor `name`. |
| **Execution** | `--device <cpu\|cuda>` | Hardware execution target (default: `cpu`). |
| | `--threads <N>` | Number of CPU worker threads (default: `1`). |
| | `--cuda-graph` | Enables low-latency CUDA graph capture and execution. |
| | `--unplanned` | Disables memory arena reuse planning (for debugging). |
| | `--symbol <k=v>` | Manually binds dynamic dimension symbols (e.g. `--symbol s=128`). |
| **Serving & Paging** | `--serve` | Starts continuous batching interactive server session with iteration-level dynamic scheduling. |
| | `--paged-kv` | Enables Driver-VMM virtual page mapping for dynamic on-demand KV cache allocation. |
| | `--max-batch <N>` | Sets maximum concurrent requests in continuous batching (default: `8`). |
| | `--gpu-utilization <R>` | Eagerly pre-allocates physical VRAM blocks upfront up to memory ratio (e.g. `0.9`). |
| | `--warm-blocks <N>` | Recycles freed physical 2 MB pages in memory pool up to ceiling `N` (default: `64`). |
| | `--no-prefix-cache` | Disables automated Paged Radix Tree prefix caching. |
| **Raw Tensor I/O** | `--input <name:file>` | Ingests raw input tensor data from binary file. |
| | `--output <id:file>` | Saves raw computed output tensor ID to binary file. |
| | `--state-in / --state-out` | Loads / saves recurrent hidden states for stateful networks. |

#### C. Ready-to-Run Verification Commands
```powershell
# 1. Inspect model capabilities and metadata
.\build-win-cuda\runtime\ggmlc-run.exe scratch\smollm2_chat.gguf --info

# 2. Instruction chat generation (CPU, 4 threads, assistant-only stream)
.\build-win-cuda\runtime\ggmlc-run.exe scratch\smollm2_chat.gguf --chat "What is the capital of France?" --max-tokens 16 --threads 4

# 3. Offload chat generation to NVIDIA CUDA GPU with CUDA graph capture
.\build-win-cuda\runtime\ggmlc-run.exe scratch\smollm2_chat.gguf --chat "What is the capital of France?" --max-tokens 16 --device cuda --cuda-graph

# 4. Interactive Continuous Batching Server with Paged Radix Tree prefix caching
.\build-win-cuda\runtime\ggmlc-run.exe scratch\smollm2_chat.gguf --serve --device cuda --cuda-graph --warm-blocks 64

# 5. Multimodal vision inference (task-aware classification)
.\build-win-cuda\runtime\ggmlc-run.exe resnet50.gguf --image x:assets\sample.png --threads 4
```

### 11. Autoregressive Inference Optimization & Empirical Findings

#### A. Roofline Analysis, Arithmetic Intensity & Memory Bandwidth
- **Arithmetic Intensity = 1.0 FLOP / Byte**:
  - In single-token autoregressive decode ($S = 1$, batch size 1), every model parameter must be fetched once from memory per token to compute a Matrix-Vector product (GEMV).
  - For a 135M parameter model in F16 (~270 MB), compute is $\approx 270\text{ MFLOPs}$.
  - The arithmetic intensity is $\frac{270\text{ MFLOPs}}{270\text{ MB}} = 1.0\text{ FLOP / Byte}$.
  - Decode is **100% memory bandwidth-bound** (not compute-bound). Theoretical TFLOPS have virtually zero impact on single-stream decode rate.
- **Hardware Bus Ceilings (SmolLM2-135M F16, 270 MB & LLaMA-3.2-1B Q8_0, 1.32 GB)**:
  - **Host DDR4/DDR5 Dual-Channel (~20–40 GB/s practical)**: Minimum physical transfer time is ~7–13 ms. With native C++ function calls (~0.1 $\mu$s dispatch overhead) and hand-tuned AVX2/FMA GEMV microkernels (`GGML_LLAMAFILE=ON`), CPU decode achieves **15.3 – 16.6 ms/tok** (~60–65 tok/s, ~85–90% bus saturation).
  - **NVIDIA GeForce RTX 4050 Laptop GPU 96-bit GDDR6 (~192 GB/s theoretical, ~162 GB/s practical saturated)**: Minimum physical transfer time for 270 MB is $\approx 1.6\text{ ms}$. For a 1.23B model (LLaMA-3.2-1B, ~1.32 GB Q8_0), decode executes in **8.15 ms/tok (122.7 tok/s)**, achieving **162 GB/s memory bandwidth** (~85–90% theoretical bus saturation).
- **The Windows WDDM CUDA Launch Bottleneck**:
  - On Windows consumer GPUs, each CUDA kernel dispatch incurs a DirectX/WDDM driver queue latency of **15 – 25 $\mu$s per launch**.
  - A 30-layer model dispatches **~350 to 450 distinct kernels per token** (GEMVs, RMSNorms, RoPE, Flash Attention, Residual adds, activations).
  - $400 \times 20\,\mu\text{s} \approx \mathbf{8.0\text{ ms of pure driver launch latency!}}$
  - Total naive GPU decode time = $4.0\text{ ms (VRAM streaming)} + 8.0\text{ ms (WDDM launch)} \approx \mathbf{12.0\text{ ms/tok}}$.
  - This explains why CPU (~15 ms) and entry-level GPU (~12 ms) exhibit similar latency on Windows without kernel reduction or graph capture.

#### B. Profiling Methodology & Rigorous Benchmarking
1. **Never Trust Third-Party Python Wrappers Unconditionally**:
   - `llama-cpp-python` wheels from pip are frequently **CPU-only** (`llama_supports_gpu_offload() == False`).
   - Passing `n_gpu_layers=99` to a CPU-only build silently executes on the CPU, causing false comparisons.
   - Always verify active hardware backends with device telemetry, `nvidia-smi`, or standalone binaries (`ggmlc-run.exe` / `llama-cli.exe`).
2. **Steady-State vs. Cache Initialization**:
   - Step 1 decode compiles/prepares the static decode graph cache (~25 – 35 ms).
   - Steps 2+ execute from the cached graph (~12 – 13 ms). Always isolate steady-state decode steps when calculating tokens/second.
3. **Thermal Throttling Awareness**:
   - Running back-to-back CPU benchmarks can cause CPU cores to throttle from 4.2 GHz down to 3.2 GHz, producing 15% variance across test phases.

#### C. Optimization Hierarchy & Empirical Validation
1. **Weight & Activation Dtypes**:
   - Use F16 weights for multi-dimensional matrices to halve memory bus traffic.
   - **Strict 1D F32 Rule**: Keep 1D vectors (RMSNorm weights, biases, RoPE frequencies) in F32 to preserve numerical stability and avoid activation divergence.
2. **Native Grouped Query Attention (GQA)**:
   - Use `torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)` to emit a single `OpCode.SDPA` node and avoid 240+ unrolled `repeat_kv` slice/concat ops.
3. **Horizontal Operator Fusion (Compiler Pass)**:
   - Pattern match parallel GEMVs consuming identical activations:
     - `Gate + Up`: $\text{Linear}(x, W_{\text{gate}})$ & $\text{Linear}(x, W_{\text{up}}) \to \text{Linear}(x, [W_{\text{gate}} ; W_{\text{up}}])$
     - `Q + K + V`: $\text{Linear}(x, W_q)$, $\text{Linear}(x, W_k)$, $\text{Linear}(x, W_v) \to \text{Linear}(x, [W_q ; W_k ; W_v])$
   - Reduces GEMV kernel launches from 5 down to 2 per layer (slashing 90 CUDA launches per token).
4. **CUDA Graph Capture (`CUDAGraphManager`)**:
   - **Falsified Initial Assumption**: The initial assumption that we should delegate CC $\ge 7.0$ to GGML's built-in CUDA graph and only write custom capture for CC $< 7.0$ turned out to be flawed. Upstream GGML CUDA graph management has rigid allocation constraints that break dynamic paged KV contexts and hardcodes compute capability restrictions.
   - **Implemented Reality**: `ggmlc` built its own unified `CUDAGraphManager` runtime bridge (`runtime/src/cuda_graph.cu`) interfacing directly with CUDA stream capture (`cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, and `cudaGraphLaunch`).
   - Zero modification to `third_party/ggml`. Works uniformly across **all CC $\ge 6.0$ architectures (Pascal CC 6.1 through Ada Lovelace CC 8.9 / Hopper / Blackwell)**.
   - Replays static decode topologies in **steady-state GPU execution** and achieves **122.7 tok/s on LLaMA-3.2-1B (0.98x of official `llama.cpp`)** and **660 tok/s on GPT-2**.

### 12. Chunked Prefill & CUDA Graph Prompt Batching

#### A. Core Motivation & Problem Statement
1. **Dynamic Shape vs. Static Graph**:
   - CUDA Graphs require fixed tensor shapes and memory topologies.
   - Raw prompt prefill ingests variable-length prompts ($N = 17, 83, 345, \dots$), forcing dynamic graph compilation or disabling CUDA graphs.
2. **Quadratic Activation Memory Spike**:
    - Full prefill computes self-attention across all $N$ tokens at once, allocating $O(N^2)$ transient attention score matrices. For long prompts (e.g. 2k–8k tokens), this creates massive VRAM allocation spikes and can trigger out-of-memory (OOM) failures on consumer laptop GPUs (e.g. 6GB RTX 4050 Laptop).
3. **Serving & Decode Stalling**:
   - Long full-sequence prefills monopolize the GPU for hundreds of milliseconds, blocking concurrent decode requests.

#### B. Chunked Prefill Architectural Mechanics
1. **Uniform Chunk Size ($C$)**:
   - Slice any prompt of length $N$ into sequential chunks of fixed size $C$ (`--chunk-size <C>`, alias `--ubatch`, default: `128`):
     $$\text{Chunk } k: \text{tokens }[k \cdot C \dots \min((k+1) \cdot C, N) - 1], \quad \text{pos } = k \cdot C$$
   - Every full chunk has **identical tensor dimensions ($s_q = C$)**, identical activation shapes, and identical kernel grids.
2. **Sequential KV Cache Accumulation**:
   - For chunk $k$ at offset $pos = k \cdot C$:
     - Key/Value projections for the chunk's $s_q$ tokens are written to `kv_cache` at slot `pos * nb[1]`.
     - Active key/value context spans all tokens accumulated so far: $s_{kv} = pos + s_q$.
   - **Causal Masking**:
     - Inside the chunk, attention scores are masked with `ggml_diag_mask_inf(ctx, scores, pos)`.
     - Tokens at local index $j$ within the chunk attend fully to all $pos$ prior tokens in the cache plus tokens $\le j$ within the chunk.
3. **Integration with Prefix Caching**:
   - When Paged Radix Tree prefix caching matches $M$ prefix tokens, the prompt prefill skips those $M$ tokens completely.
   - Prefill executes solely for the remaining suffix ($N - M$ tokens) starting from `pos = M`.
   - If $M == N$ (100% prefix cache hit), prefill only evaluates the very last prompt token ($s=1, pos=N-1$) to extract logits and sample the initial decode token.

### 13. High-Throughput Agent Serving & Runtime Architecture

#### A. Paradigm Shift: From Single-User Chatbot to Agentic Swarms
1. **The 2023 Assumption (Obsolete)**:
   - Early `ggml` / `llama.cpp` assumed a single human user waiting for a single streaming response ($B = 1$).
   - Concurrency was treated as an edge case; fixed memory allocations were partitioned per slot.
2. **The 2026 Reality (Agentic Concurrency)**:
   - Modern agentic frameworks (coding assistants, autonomous subagents) trigger **3 to 10 concurrent requests simultaneously** for a single human user (planning, code exploration, semantic retrieval, unit testing, browser interactions).
   - Local serving engines on desktop/workstation hardware must provide **high-throughput concurrent decoding ($B = 2 \dots 16$)** without multi-gigabyte cloud framework bloat.

#### B. Paged Attention via Driver-VMM vs. Software Page Tables
1. **The Upstream Trap (Falsified Consideration)**:
   - Upstream llama.cpp attempts at software page tables (PR #22569) introduced custom non-affine gather kernels. This broke contiguous memory addressing (`nb`), sacrificed FlashAttention, and ran 3–5x slower.
2. **The Driver-VMM Solution**:
   - Pushed page management to the **GPU Memory Management Unit (MMU)** via CUDA Driver Virtual Memory Management (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`).
   - Physical 2 MB device memory pages are mapped dynamically into a continuous 64-bit virtual address window.
   - Unmodified, hand-tuned CUDA kernels (including `ggml_flash_attn_ext` and GEMV microkernels) execute over the mapped virtual space with full hardware acceleration.
3. **Empirical Benchmarking Ceilings**:
   - **Zero Bandwidth Penalty**: Verified D2D memory bandwidth under `cuMemMap` achieves **162 GB/s** on NVIDIA RTX 4050 Laptop GPU, matching standard `cudaMalloc`.
   - **Zero-Copy Context Expansion & Pointer Invariance**: Expanding sequence length from 2 MB to 16 MB executes in **1.28 ms** with zero bytes copied and zero device pointer relocation. Because device virtual base pointers never change, static CUDA Graphs remain 100% valid across dynamic allocations.
   - **Immediate VRAM Reclamation**: Freeing finished request slots unmaps and releases physical pages instantly, reducing active VRAM to 0 bytes without requiring graph teardown or device synchronization.

#### C. Continuous Batch Scheduler & Multi-Bucket Graphs
1. **Iteration-Level Dynamic Scheduling (`ContinuousBatchScheduler`)**:
   - Managed in `runtime/src/batch_scheduler.cpp`.
   - Admitted requests are scheduled dynamically across discrete batch buckets ($B \in \{1, 2, 4, 8, 16\}$).
   - **Separation of Concerns**: When a request is first admitted, the scheduler executes its prefill phase for uncached tokens, registers newly formed 2 MB pages into the Radix Tree, and extracts the first token.
   - Once admitted, all active requests participate in uniform single-token decode steps ($S=1$) within the selected batch bucket $B$. This avoids heterogeneous kernel grids and ensures CUDA graphs remain 100% static and valid.
2. **Multi-Bucket CUDA Graphs**:
   - Autoregressive decode is strictly $S = 1$. The tensor dimensions for batch sizes $B \in [1, 2, 4, 8, 16]$ are completely static (`[B, 1, hidden]`).
   - Pre-capture discrete graphs for each batch bucket ($B \in \{1, 2, 4, 8, 16\}$) to eliminate driver dispatch latency under concurrent agent loads.

#### D. Paged Radix Tree Prefix Caching & Warm Block Recycling
1. **Zero-Kernel Radix Attention via Driver-VMM (`PagedRadixTree`)**:
   - Host-side trie structure (`runtime/src/radix_tree.cpp`) indexing completed 2 MB physical page handles (`cuMemGenericAllocationHandle`) by prompt token IDs.
   - Matched physical page handles are mapped directly into contiguous virtual memory slots of the request's VA window via `cuMemMap`.
   - FlashAttention and GEMV microkernels execute over the mapped virtual memory with zero overhead and zero kernel changes.
2. **Dual-Mode Physical Page Lifecycle (`VMMBlockManager`)**:
   - **Desktop / Elastic Mode (Default, `--warm-blocks N`)**: Dynamically creates physical 2 MB pages via `cuMemCreate` on-demand and recycles freed pages into an in-memory pool (`free_pages_pool_`) up to ceiling $N$. Any pages beyond $N$ are released immediately back to the OS via `cuMemRelease`, leaving desktop GPU VRAM unencumbered.
   - **Dedicated Server Mode (`--gpu-utilization <ratio>`)**: Eagerly pre-allocates $N$ physical 2 MB blocks during runtime initialization (vLLM style), ensuring zero driver syscalls during high-throughput agent swarm serving.

### 14. Cross-Platform CI & Runtime Engineering Lessons

#### A. Apple Silicon & ARM64 Architecture Linkage
1. **Llamafile AVX2 Assembly Exclusivity**:
   - `llamafile/sgemm.cpp` contains hand-crafted x86/x86_64 AVX2/AVX-512 assembly kernels. It is excluded from compilation on ARM64 architectures.
   - If `GGML_USE_LLAMAFILE` is defined on ARM64, `ggml-cpu.c` references `_llamafile_sgemm`, leading to linker errors:
     ```text
     Undefined symbols for architecture arm64: "_llamafile_sgemm", referenced from: _ggml_compute_forward_mul_mat
     ```
   - **Rule**: In `CMakeLists.txt`, always guard `GGML_USE_LLAMAFILE` with architecture detection:
     ```cmake
     if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
         target_compile_definitions(ggml_lib PUBLIC GGML_USE_LLAMAFILE)
     endif()
     ```

#### B. MSVC Standard Library Iterator Invariants & Memory Safety
1. **Cross-Container Iterator Comparison UB**:
   - In MSVC (and standard C++), comparing an iterator initialized from container $A$ (e.g. `ggml_tensors_`) against container $B$'s `.end()` (e.g. `state_tensors_.end()`) is undefined behavior.
   - On Windows, `it != state_tensors_.end()` evaluates to `true`, leading code to dereference `it->second` on an invalid iterator, resulting in fatal Windows SEH exception `0xC0000005: Access Violation`.
   - **Rule**: Keep iterators strictly scoped to their respective containers. Never re-assign an iterator across maps or test against a different container's `.end()`.

#### C. Positional Embeddings vs. KV Cache Applicability
1. **Absolute vs. Rotary/Dynamic Positional Indexing**:
   - Single-token autoregressive KV cache decode ($S=1$) requires dynamic position tracking ($pos$).
   - Models with learned absolute position embeddings (such as standard GPT-2 exports using `wpe(torch.arange(0, input_ids.shape[-1]))`) do not take a dynamic `pos` symbol. When executed with $S=1$, they evaluate position 0 on every step (`wpe([0])`), corrupting generation.
   - **Rule**: In `ModelExecutor::init_kv_cache`, only allocate KV cache if the model supports position offsets (uses `GGML_OP_ROPE` or binds an explicit dynamic `"pos"` symbol). Models with static absolute position embeddings must decode with full context to maintain bit-for-bit numerical and generation parity with reference frameworks.

#### D. Non-CUDA Cross-Platform Fallback Stubs
1. **Header & Symbol Isolation**:
   - Classes managing hardware-specific features (such as `CUDAGraphManager` and `NativeVMMBlockManager`) must compile cleanly on non-CUDA targets (CPU builds on Windows, macOS, Linux).
   - Maintain dedicated CPU stubs (`cuda_graph_stub.cpp`, `vmm_pool_stub.cpp`) so that tools like `ggmlc-run` and Python bindings link without unresolved symbols across all platforms.

#### E. Strict 64-bit Format String Portability
1. **Format Specifiers across Clang and MSVC**:
    - `ggml_blck_size` and tensor dimensions are 64-bit integers (`int64_t`). Passing them to `%zu` causes `-Wformat` compiler warnings/errors on Clang on macOS and Linux.
    - **Rule**: Explicitly cast 64-bit integer arguments to `(long long)` and use `%lld` (or `<cinttypes>` `PRId64`) in all debug print and format strings.

#### F. Memory Allocation (galloc) and Unallocated View Tensors
1. **galloc vs. Backend Allocated Contexts**:
    - Under graph arena memory reuse (`enable_arena_reuse=True`), memory allocation for the computational graph (`cgraph_`) is handled by `ggml_gallocr_alloc_graph`. `galloc` only allocates memory for tensors that are reached/computed in `cgraph_` and are part of its topological sort.
    - When arena reuse is disabled, the engine falls back to `ggml_backend_alloc_ctx_tensors(ctx_, backend_)`, which automatically allocates a backend buffer for *every* tensor initialized in `ctx_`, regardless of whether it's dead code, pruned, or a leaf.
2. **Intermediate Sliced Views and Null Buffers**:
    - In models with complex operations like Grouped Convolution (e.g., RegNet), weight parameters are sliced/split during lowering into multiple sub-tensors using `GGML_OP_VIEW`.
    - These sub-tensors are classified as `StorageClass::ACTIVATION` (since they are generated by operations) and put into `compute_tensors_`. However, they are serialized with static data in Python, resulting in non-null `data_ptr` during model loading.
    - Since they are view operations whose parent tensors (the static weights) reside in a separate `weight_buffer_` outside `galloc`, `ggml_gallocr_alloc_graph` does not allocate separate buffers for them, leaving their buffer pointers as `nullptr`.
    - **Rule**: During model preparation or tensor state/input binding, always verify `pair.second->buffer != nullptr` before calling `ggml_backend_tensor_set`. Attempting to copy data to unallocated view tensors (or pruned/dead-code tensors) will trigger a native GGML assertion abort (`buf != NULL && "tensor buffer not set"`). Always skip initialization for null-buffer tensors (their parent parameters hold the valid data) and throw explicit `std::runtime_error` exceptions in input/state bindings to avoid native crashes.
3. **Lowering-created copies must be pinned for gallocr**:
    - `ModelExecutor::prepare` builds new ggml nodes (`VIEW`/`CONT`/`CAST`/`CLAMP` of inputs and constants, `GET_ROWS` sources, FA masks) that do **not** inherit `INPUT`/`OUTPUT` flags from IR placeholders. `gallocr` then reuses those buffers while later layers still need them → NaN logits (Laya).
    - **Rule**: After `ggml_build_forward_expand` and before `ggml_gallocr_alloc_graph`, call `pin_live_graph_tensors`: `ggml_set_input`+`ggml_set_output` on remapped IR inputs; `ggml_set_output` on all `cgraph_` leaves, view-like chains whose root is an input or weight, `GET_ROWS` `src0`, and `FLASH_ATTN_EXT` mask. Do not pin `ADD`/`MUL` descendants of `input_ids` (that would freeze the whole encoder). Arena-off (`ggml_backend_alloc_ctx_tensors`) is the debug path, not the default.

#### G. Apple Clang / macOS Target Architecture Flag Gotchas (`GGML_NATIVE` / `-mcpu=native`)
1. **Unsupported `-mcpu=native` on Apple Clang**:
   - On macOS with Apple Clang (both Apple Silicon ARM64 and Intel x86_64), passing `-mcpu=native` causes a fatal compilation error:
     ```text
     clang: error: unsupported argument 'native' to option '-mcpu='
     ```
   - Upstream GGML default options enable `GGML_NATIVE=ON`, which unconditionally injects `-mcpu=native` into compiler flags when `CMAKE_SYSTEM_PROCESSOR` is ARM.
   - **Rule**: In `CMakeLists.txt` and `third_party/ggml/src/ggml-cpu/CMakeLists.txt`, detect Apple Clang (`CMAKE_CXX_COMPILER_ID MATCHES "AppleClang"`) or macOS (`APPLE`) and force `GGML_NATIVE=OFF` before configuring GGML dependencies.

#### H. FX Importer & Frontend Parity Lessons (TimesFM, Pax `PerDimScale`, Softplus & RoPE)
1. **Constant Folding Parameter Ops (`aten.softplus.default`)**:
   - Non-standard layer scaling (such as Pax-style `PerDimScale` in TimesFM: $y = x \cdot \frac{1.442695}{\sqrt{d}} \cdot \text{softplus}(w)$) creates FX nodes on constant parameter weights.
   - **Rule**: When `torch.export` emits parameter-only math (e.g. `aten.softplus.default`), evaluate the operation stably during graph ingestion in `python/ggmlc/frontend/pytorch/importer.py` using NumPy/PyTorch constant folding rather than lowering to unsupported runtime activation ops.
2. **RoPE Tensor Dtype Alignment for CUDA Broadcast Kernels**:
   - Creating position tensors as `torch.int32` causes downstream arithmetic to emit integer tensors that trigger CUDA assertion failures in `binbcast.cu` (`GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16)`).
   - **Rule**: Always construct RoPE position vectors in `torch.float32`.
3. **Flax vs. PyTorch SDPA Attention Scaling Discrepancy**:
   - In models ported from Flax/Pax (e.g., TimesFM), attention uses `rescale_logits=False`, passing queries multiplied by $\sqrt{d_{head}}$ into SDPA with `scale = math.sqrt(d_{head})` (rather than standard PyTorch $\frac{1}{\sqrt{d_{head}}}$).
   - **Rule**: When validating differential numerical parity on Flax/Pax ports, check attention logit scaling. Explicitly setting `scale=math.sqrt(head_dim)` resolves what would otherwise appear as an $80\times$ attention divergence.

#### I. Embedded C++ Web Studio & Socket Networking Gotchas
1. **HTTP Keep-Alive & `ERR_CONNECTION_RESET`**:
   - In lightweight embedded C++ HTTP servers (like `timesfm.exe` Web Studio), browsers using `fetch()` keep connections open or expect exact `Content-Length` headers.
   - **Rule**: Always read the full request body up to `Content-Length`, include explicit `Content-Length` and `Connection: close` headers in the response, and cleanly close/shutdown client sockets after responding.
2. **Client-Side Blob Downloads over Server Endpoints**:
   - For exporting tabular/matrix data (CSV, JSON), generate files client-side via JavaScript `Blob` (`URL.createObjectURL(new Blob([data], {type: 'text/csv'}))`). This completely eliminates redundant server-side serialization logic and network round-trips.

#### J. Peak RAM Optimization & Garbage Collection in E2E Verification
1. **Avoiding Redundant Model File Duplication in Memory**:
   - Passing file paths to `ModelRunner(model_path)` should load directly via C++ `_runtime.ModelLoader.load_from_file(path)` without reading the entire binary into a Python `bytes` object (`read_bytes()`).
   - Reading multi-gigabyte GGUF files into Python memory doubles peak RAM consumption and can cause `std::bad_alloc` on memory-constrained systems.
2. **Proactive Resource Release in Differential Tests**:
   - In full-model E2E tests, always explicitly `del` intermediate PyTorch reference outputs (`ref_out`), FX export objects (`exported`), and IR graphs (`ggml_graph`) before instantiating C++ `ModelRunner`.
   - Call `gc.collect()` before C++ memory arena allocation to ensure memory blocks are free for contiguous buffer allocation.

#### K. Graph-Output `VIEW` Must Be Contiguous (Fusion / Slice Host Reads)
1. **Symptom**: Horizontal fusion and `aten.slice` tests fail with nonsense shapes (e.g. fused gate/up `(448,)` vs `(1,1,4,64)`, slice dim1 `120` vs `64`).
2. **Cause**: Slicing fused `[gate;up]` / `[Q;K;V]` along GGML `ne[0]` keeps the parent's larger `nb[1]`, so the view is **non-contiguous**. `ggml_nbytes` returns the strided *span*, and `ggml_backend_tensor_get` linearly copies that span — not the logical tensor.
3. **Rule**: In `ModelExecutor` `GGML_OP_VIEW`, if `out_id` is in `model_graph_.outputs`, always `ggml_cont` the view before returning. Intermediate QKV→RoPE→FA views stay non-contiguous (zero-copy). `get_tensor_size_bytes` / `get_output_data` must use **logical** `nelements × type_size`, not span `ggml_nbytes`, and require contiguous outputs.
4. **Local editable installs**: After `cmake --build`, copy/rebuild so `.venv` loads the new `_runtime*.pyd` (site-packages can stay stale while `python/ggmlc/` is updated).

#### L. `FusionOptions` Must Reach the Exporter
1. Horizontal fusion runs inside `export_torch_model` → `create_standard_optimization_pipeline(...)` **before** `ggmlc.compile`'s later passes.
2. **Rule**: Thread `fusion_options` from `ggmlc.compile(..., fusion_options=...)` into `export_torch_model`. Flags only on `compile` are no-ops if the exporter always fuses with defaults — A/B GGUFs silently keep `fused_gate_up` / `fused_qkv` names.
3. A/B harness: `--fusion-no-horizontal-mlp` / `--fusion-no-horizontal-qkv` + `--gguf-suffix` + `--force-recompile`. Confirm with weight-name histograms (`gate_up` vs `gate_proj`/`up_proj`, `qkv` vs `q_proj`).

#### M. Prefill/Decode Metrics vs User-Facing E2E
1. Separate `pp` / `tg` rates can disagree with chat UX: decode often dominates once $N \gtrsim 32$–64.
2. **Rule**: For go/no-go on parity PRs, measure the full pp/tg matrix **and** `--e2e` / `ggmlc-bench -pg` / `llama-bench -pg`. After Phase 2, pp512/pp1024 should also be ≥ ~0.97× llama when `ggmlc-bench` uses last-token logits (default); do not compare against `--full-logits` or a Python `ModelExecutor` without `set_logits_last_only(True)`.
3. On Windows WDDM, `llama-bench` CUDA graphs can abort (`GGML_ASSERT` uncaught exception). The compare harness sets `GGML_CUDA_DISABLE_GRAPHS=1` for llama unless `GGMLC_LLAMA_ENABLE_CUDA_GRAPHS=1`.
4. HF / official GGUFs store **separate** `gate_proj`/`up_proj` (and Q/K/V); fused Gate+Up / QKV in ggmlc is compile-time only.

---

## 15. Future Development Roadmap: Interactive Serving & High-Throughput HTTP API

Building upon the completed Driver-VMM paged KV cache, Paged Radix Tree prefix caching, and continuous batch scheduler, future runtime development will focus on interactive user experience and standards-compliant serving:

### A. Interactive Serve TUI & Terminal UX (`ggmlc-run --serve`)
Currently, `ggmlc-run --serve` implements a synchronous terminal prompt loop. The next evolution of the interactive serve mode will deliver a rich, persistent Terminal User Interface (TUI):

1. **Sticky Bottom Status Bar**:
   - Pin a non-scrolling status bar to the bottom of the terminal window using ANSI/VT100 escape sequences (or a lightweight embedded library like FTXUI).
   - Generated token text continues to stream smoothly in the main scrolling pane above without being overwritten by status updates.
2. **Real-Time Activity Indicator**:
   - Dynamic spinner / pulse animation indicating current hardware state:
     - `[PREFILL]` (amber): Prompt ingestion and KV cache allocation in progress.
     - `[DECODE]` (green): Active autoregressive token generation across active slots.
     - `[IDLE]` (blue): Waiting for incoming requests with warm memory pages maintained.
3. **Comprehensive Live Generation Telemetry**:
   - **Concurrency & Batching**: Active request count vs. max batch (`Active: 3/8 [Bucket B=4]`).
   - **Driver-VMM Page Utilization**: Physical pages mapped vs. warm pool vs. total reserved VRAM (`Pages: 14/64 (28 MB mapped, 100 MB warm pool)`).
   - **Prefix Cache Hit Rate**: Rolling prompt cache hit percentage and cumulative tokens saved (`Prefix Cache: 78.4% hits, 1,420 tokens saved`).
   - **Latency & Throughput Telemetry**:
     - TTFT (Time To First Token / prefill latency in ms).
     - Aggregate decode throughput (tokens/sec across all active slots).
     - Single-stream inter-token latency (ms/tok).
4. **Asynchronous Non-Blocking Input Thread**:
   - Decouple terminal input from the continuous batching step loop via a background input worker thread.
   - Allows users (and automated agent scripts) to submit new prompt queries while previous requests are actively streaming output, dynamically queuing requests into `ContinuousBatchScheduler` mid-flight.

### B. High-Throughput OpenAI-Compatible HTTP Server
To enable seamless drop-in integration with agentic frameworks (Antigravity, Cursor, Cline, AutoGen, LangChain, LiteLLM, and LlamaIndex):

1. **Standard REST Endpoints**:
   - `POST /v1/chat/completions`: Full support for ChatML / instruct chat templates, system prompts, sampling parameters (`temperature`, `top_p`, `max_tokens`, `stop`), and Server-Sent Events (`stream: true` -> `data: {"choices": [{"delta": {"content": "..."}}]}`).
   - `POST /v1/completions`: Raw text continuation and token completion.
   - `GET /v1/models`: Returns metadata for the loaded GGUF model artifact.
2. **Implementation Pathways**:
   - **Native C++ Standalone Server**:
     - Embedded lightweight HTTP/SSE library (e.g. `cpp-httplib` or `Boost.Beast`) directly driving `ContinuousBatchScheduler`.
     - Zero Python dependency, ultra-low memory footprint (<10 MB server binary overhead), direct kernel dispatch.
   - **Python FastAPI / Uvicorn Service Wrapper**:
     - Lightweight async service importing `ggmlc._runtime.ContinuousBatchScheduler` and `ModelRunner`.
     - Enables rapid customization, custom tokenizers, middleware, and authentication.
3. **vLLM-Class Throughput on Consumer Hardware**:
   - Deliver vLLM-grade continuous batching, prefix caching, and dynamic memory reclamation on consumer GPUs (e.g. RTX 4050 Laptop / RTX 3060 / RTX 4090) without requiring multi-gigabyte PyTorch/Ray dependencies or datacenter GPU clusters.

### C. Speculative Decoding & Advanced Serving Techniques
1. **Speculative Decoding with Physical Page Aliasing**:
   - Run a compact draft model (e.g. SmolLM2-135M) alongside a larger target model (e.g. Qwen2.5-1.5B/3B).
   - Use Driver-VMM to map draft KV pages directly for parallel target verification with zero memory copying.
2. **Dynamic Logit Processors & Token Healing**:
   - Dynamic grammar constraints (GBNF / regex matching) applied during `sample_next_token`.
   - Token healing to resolve tokenization boundary artifacts on prompt suffix continuations.

---

## 16. Optimization Roadmap & Phased Milestones

### Phase 1: Affine View & Metadata Folding + 1B Scale Alignment (Completed)
- Eliminating redundant `VIEW`, `RESHAPE`, and `PERMUTE` nodes from Canonical IR.
- SDPA-transpose elimination (`fused_transpose=1`).
- Fused native SwiGLU via `ggml_swiglu_split` and single-input `ggml_swiglu`.
- Added `gpt2_medium` (355M) and `llama3.2_1b` (1.23B) to benchmark suite; phased out micro-models.
- Aligned physical prompt prefill chunking (`--ubatch 512`) across `ggml-bench` and `llama-bench`.
- Result: LLaMA-3.2-1B decode reaches **122.7 tok/s** (98% parity with `llama.cpp` at 126.8 tok/s, 162 GB/s bandwidth), with **+19% prefill speedup** on short prompts.

### Phase 1b: llama.cpp Prefill/Decode Parity Stack (Completed — this PR / #22)
- **SET_ROWS + padded `n_kv`**: Decode and chunked prefill write K/V without per-token FA `ne[]` mutation so CUDA graphs stay warm.
- **Strided fused-QKV `VIEW→RESHAPE`**: Kill CONT tax on RoPE→FA path; unary kernels still CONT at consumer when required.
- **`(s_q, n_kv)` prepared-graph buckets**: Stash/activate (llama `can_reuse` analogue) instead of mutating live graphs across pad strides; `reset_kv_cache` must not tear down buckets.
- **Multi-chunk pp**: SmolLM2 / GPT-2 reached **~1.00x** vs llama on pp1024 under buckets alone; residual Qwen/LLaMA **~0.84x–0.90x** was **not** bucket-switch tax (closed in Phase 2 via last-token logits).
- **Horizontal fusion A/B**: `no_hmlp` and `no_hqkv` falsified (parity models regress; gap models do not win enough). **Keep default fusion ON.**
- **E2E wall clock (`--e2e` / `-pg`)**: Chat-like $N=128$ turns **embraced** (~1.0x–1.3x vs llama). Prefill-heavy tiny-$N$ still shows the residual pp gap.
- **CI fix**: Materialize CONT for graph-output `VIEW`s (fusion/slice host reads).

### Phase 2: Wide-FFN Q8 Prefill Parity — Qwen / LLaMA (Completed — last-token logits)
- **Problem**: After buckets + fusion A/Bs + e2e embrace, Qwen (`I≈4864`) and LLaMA (`I≈8192`) remained ~0.84x–0.90x on prefill vs `llama.cpp`. Initially attributed to fat-FFN Q8 GEMM tiles.
- **Root cause (falsified tile hypothesis)**: Op/CONT dumps showed `n_cont_or_dup=0`; physical MMQ tiles were `J=128`, contiguous Q8_0, and gap models already on the MMQ **fast** path. The real miss vs `llama-bench` was **full-sequence `lm_head`**: ggmlc wrote `V×P` logits (Qwen V=152k → ~311 MB write) while llama uses `n_outputs=1` (`llama_batch_get_one` → last token only; `ggml_get_rows` before last layer).
- **Fix**: `ModelExecutor::set_logits_last_only` gathers the last token before graph-output `MUL_MAT` (tied embed / lm_head).
- **Full matrix (2026-09-19, ubatch 512, Q8_0, CUDA)**: **31/32** cells ≥ **1.01×** vs llama — Qwen pp512 **1.08×** / pp1024 **1.16×**; LLaMA pp512/pp1024 **1.05×**; SmolLM **1.07×**; decode **1.01×–1.37×**. Only GPT-2 `pp16` at **0.92×**.
- **Follow-ups (optional)**: llama also gathers before the *last transformer layer*; multi-request output-id gather for continuous batching (see `set_logits_last_only` note below).

#### `ModelExecutor::set_logits_last_only` — origin, defaults, serving A/B
- **Origin**: Mirrors `llama.cpp` `n_outputs=1` for sampling/bench (`llama_batch_get_one` leaves `logits==nullptr` → only last token is an output). ggmlc implements the `lm_head` half: before the graph-output `MUL_MAT`, `ggml_view_2d` the last column of the activation (`ne[1] > 1` → `ne[1] = 1`). Does **not** yet skip the last transformer layer the way llama's `ggml_get_rows` before `il == n_layer-1` does.
- **API**: `executor.set_logits_last_only(bool)`; `executor.logits_last_only()`. Toggling invalidates `prepared_` so the next `prepare()` rebuilds the gather.
- **Enabled by default today**:
  - `ggmlc-bench` (opt out: `--full-logits`) — required for apples-to-apples vs `llama-bench`.
  - `ggmlc-run` chat / generate / `--serve` paths.
  - `ContinuousBatchScheduler` constructor.
- **Disabled by default**: raw `ModelExecutor` construction (Python bindings / numerical differential tests that need full-seq logits). Call `set_logits_last_only(True)` explicitly when measuring sampling prefill or comparing to llama.
- **Upcoming phases (continuous batching / swarm serving)**: keep this flag as an **A/B knob**. Single-stream decode ($B=1$, $S=1$) is a no-op (already one column). Prefill of one request still wants last-token only for sampling. Concurrent prefill or “logits for every slot in a bucket” may need **per-request output ids** (llama `out_ids` vector) instead of a single last-column view — measure TTFT / tok/s with gather ON vs OFF before changing the default for `--serve`.

### Phase 3: Vertical Operator Fusion — Bake RMS into Linear (Default ON)
- **Discovery**: stock CUDA already fuses adjacent `RMS_NORM→MUL`. Residual
  `ADD→RMS` cannot drop the ADD write (pre-norm). Runtime kernel prototypes for
  ADD+RMS and norm-into-MMQ were tried and **removed** (flat / decode-regressing).
- **Ship path (no new kernels)**: bake RMSNorm `gamma` into following Linear
  weight columns at compile time (`W'[j,k]=W[j,k]*gamma[k]`), emit weightless
  `RMS_NORM`. Flag: `FusionOptions.enable_bake_rms_into_linear` (default **ON**);
  A/B opt-out: `--fusion-no-bake-rms`. Skips tied embed/lm_head. Must bake
  **before** Q8 requant (recompile already-quant GGUFs).
- **A/B (CUDA graph, Q8_0)**:
  - SmolLM2-360M: −64 kernels (`MUL` 65→1); tg128 **1.05×**; pp **~1.01–1.02×**
  - Qwen2.5-0.5B: −48 kernels (`MUL` 49→1); F16 cosine **0.999996**; tg/pp **~flat**
  - LLaMA-3.2-1B: −32 kernels (`MUL` 33→1); Q8 cosine **0.9993**; tg128 **1.05×**,
    pp128 **1.05×**, pp512 **1.14×**
- Artifacts: `scratch/phase3_vertical_discovery.md`, `scratch/phase3_bake_rms_ab.md`,
  `scratch/phase3_bake_rms_parity.md`, `scratch/phase3_bake_rms_qwen_llama.md`.

### Phase 3b: Generic Const-Affine → Linear / MatMul / Conv (Completed — merged)
- Same compile-time identity as RMS bake, frontend-agnostic: static
  $y = a \odot x + b$ is absorbed into the adjacent `LINEAR` / `MATMUL` /
  `CONV2D` weight (and bias). No new kernels.
- Covers Conv+BatchNorm, LayerScale, LayerNorm $\gamma/\beta$ → GEMM, and
  stray channel scales. Flag: `FusionOptions.enable_bake_affine` (default
  **ON**); A/B opt-out: `--fusion-no-bake-affine`.
- **Must not** eat `RMS_NORM/LAYER_NORM → MUL(γ) → ROPE` (QK-Norm) or
  `→ residual ADD` (Gemma peri-norm); those keep stock CUDA fused kernels.
- Post-norm residual fanout (BERT/MiniLM `LN` output is both next Linear and
  residual) correctly refuses the LN bake — the affine is still live on the
  residual path.
- A/B (RTX 4050 Laptop, CUDA, 3 runs, F32):
  - ResNet-18: 89 → 40 nodes (`MUL+ADD` 48 → 8); parity `max_diff ≈ 5e-3`
  - ConvNeXt-Tiny: 184 → 166 nodes; P50 35.4 → 33.3 ms
  - GPT-2: 24/25 LayerNorms weightless (final `ln_f` tied); P50 8.26 → 7.31 ms
  - MiniLM-L6: no LN bake (post-norm residual fanout); parity unchanged
- Artifacts: `scratch/affine_bake_on.{md,json}`, `scratch/affine_bake_off.{md,json}`,
  `scratch/ab_bake_affine_nodes.py`.

### Phase 4: Quantized KV Cache (FP8 / Q8_0) (Deferred)
- **Do not start.** Resume after the Laya showcase (§17).
- **Context**: Long-context / multi-turn KV footprint and fetch bandwidth dominate past ~1k–2k tokens. Q8/FP8 KV halves traffic and extends max context under fixed VRAM.
- **Deliverables**: `Q8_0` (+ Ada FP8) KV pages in `VMMBlockManager`; wire into `ggml_flash_attn_ext` without FP16 materialize in global memory; perplexity gate (<0.1% vs FP16).
- **Benchmark rule**: Match llama `-ctk/-ctv` types exactly — never mix quantized KV vs f16 baseline.

### Phase 5: High-Throughput Agentic Swarm Serving ($B \in [2, 4, 8, 16]$) (Deferred)
- **Do not start.** Resume after the Laya showcase (§17).
- Continuous batch scheduler + multi-bucket CUDA graphs + Driver-VMM + radix prefix cache (much already landed for `--serve`); close the loop with swarm bench (TTFT, aggregate tok/s, P50/P90/P99, VRAM) at concurrency $1\ldots16$ on RTX 4050 Laptop.

---

## 17. Current Work: Laya System 1 Decision Showcase (`examples/laya`)

Ship a third production-grade standalone C++ application in the same class as `tab_completion` and `timesfm`. Goal: a demo people will actually share (local Jev-style System 1 decisions) that proves ggmlc compiles **more than llama-style chat**.

Closed-source TypeSafe **Jev** scores typed questions (choice / score / noul) over a state in one parallel pass — no autoregressive tokens. Open reproduction is **Laya** (`convaiinnovations/laya`, ModernBERT-large 421M DecisionModel, Apache-style weights on HF). Default artifact: English `convaiinnovations/laya` F16 (~845 MB) on the RTX 4050 Laptop 6 GB. Stretch later: `laya-multilingual`, `laya-typed-decisions`.

### Pattern (do not invent a new app shape)

| Piece | `tab_completion` | `timesfm` | Laya (do this) |
| :--- | :--- | :--- | :--- |
| Dir | `examples/tab_completion/` | `examples/timesfm/` | `examples/laya/` |
| Binary | `tab_completion.exe` | `timesfm.exe` | `laya.exe` |
| Neural trunk | PlaidQ GGUF via `ggmlc_runtime` | TimesFM 3.0 GGUF | ModernBERT + typed head GGUF (`LayaCleanTrunk`) |
| Domain math in C++ | FIM canvas, DDIM, hole sampler | RevIN, detrend, patching, quantiles | `build_sequence`, option markers, temperature / softmax / Shannon confidence, noul=p[true] |
| UX | CLI + `--daemon` JSON-RPC | CLI + `--serve` Web Studio | CLI + `--daemon` JSON-RPC + `--serve` Web Studio |
| CMake | `GGMLC_BUILD_EXAMPLE_TAB_COMPLETION` | `GGMLC_BUILD_EXAMPLE_TIMESFM` | `GGMLC_BUILD_EXAMPLE_LAYA` |
| Tests | `tests/numerical/test_plaidq_differential.py` | `tests/numerical/test_timesfm3_differential.py` | `tests/numerical/test_laya_differential.py` |
| Compile | — | — | `examples/laya/compile_laya.py` → `scratch/laya_english_f16.gguf` |

### Architecture facts (from Laya sources, not assumptions)
- Sequence: `[CLS] {type} question: {ins} [SEP] [MASK] opt0 … [SEP] state [SEP]`. Noul options always `[false, true]`.
- Special ids (English ModernBERT): cls=50281 sep=50282 pad=50283 mask=50284. max_len=512, head_max_len=192, max_opts=16 static for CUDA graphs.
- Encoder: RoPE (full θ=160000, sliding θ=10000), GeGLU, local_attention=128 → window 64, layer0 attn_norm Identity, type_emb after encoder.
- Head: 2× pre-LN TransformerEncoder (ReLU FFN), [MASK] scorer, act/escalate MLP. **Head SDPA has no RoPE.**
- Temperature per qtype / option-count applied **after** logits. Confidence = `1 - H(p)/log(K)`; noul confidence = `max(p[1], 1-p[1])`.
- **Do not** `torch.gather` markers (importer GATHER → prefix SLICE). Use `F.embedding` / GET_ROWS. **Do not** enable `FusionOptions.enable_rope` — precomputed dual-theta cos/sin is not LLaMA `GGML_OP_ROPE`.
- Comparisons `ge/gt` constant-fold from example meta — pad bias is `(mask-1)*1e4`.
- **Do not** use a 5D QKV view `(B,S,3,H,D)`. The importer folds 5D permutes by dropping batch and scrambles attention. Split on the last 4D axis: `qkv.split(hidden, dim=-1)` then `view(B,S,H,D).transpose(1,2)`.
- **Do not** add I32 batch offsets into `marker_pos` (CUDA `binbcast` is F32/F16 only). Flatten `b*S` on the host.
- FlashAttention needs an `n_kv × n_q` mask, not a broadcast `[1,1,1,S]` pad vector. Add `full_zeros[1,1,S,S] + pad_bias`.
- `aten.contiguous` is an importer identity. Prefer `transpose(1,2).reshape` after SDPA so `fuse_sdpa_transpose` can match.
- English ModernBERT tokenizer is **byte-level BPE** (GPT-2 merges, specials `[CLS]/[SEP]/[PAD]/[MASK]` at 50281–50284), not WordPiece.
- `prepare` uses `ggml_gallocr` (arena on). `pin_live_graph_tensors` flags remapped inputs, graph leaves, view/cont of weights/inputs, FA masks, and GET_ROWS sources so gallocr does not overwrite them. Arena-off was the NaN workaround and is no longer the Laya default.
- Export with `Dim("b")` / `Dim("s")`; torch.export may rename them (`s22`, `s53`). Bind symbols from `input_ids` GGML `ne[0]=S, ne[1]=B`, not from the names `b`/`s`.
- RoPE and sliding-window buffers are `max_len+1` so `[:, :s]` is never identity at `s=512` (export would guard `s != 512`).
- Host flattens markers: `marker_pos[b,k] = b * S + pos`. CUDA batch cap `B*S <= 1024` with OOM-halve fallback.
- **Unpadded = pad-to-max-in-batch**, matching `laya.common.collate_items` (`L = max(len_i)`, `attention_mask` zeros pads). That is **not** concat packing (one `S=sum` sequence + block-diagonal mask) and **not** FlashAttention varlen/`cu_seqlens`. Concat packing would make FA `O((Σ S_i)²)` and overflow `max_len=512` on the email preset (~648 tokens). The official Agent encoder is SDPA, not unpadded flash.
- CUDA graphs recapture when `(B,S)` changes. Dummy-pad later chunks to the first chunk's `B` so a preset's forwards stay one shape. Live `S` is `max(len_i)` in the length group, clamped to `[min_seq, max_len]` (export `Dim` min=64).
- Importer `aten.slice` end may be a symbolic FX Node; treat as `-1` and let runtime VIEW use the concrete output shape.
- Latency (RTX 4050 Laptop, 2026-09-20): C++ F16 **S=84 B=1 ~25 ms**/noul (best 23.5); email 7q **S=124 B=7 one forward ~143 ms** (best 134). Python `Agent` pad-to-max-in-batch CUDA ~57 ms / ~143 ms. Static `[1,512]` was ~230 ms/q. Numbers in `examples/laya/README.md`.

### Vertical slices
1. Explore PyTorch graph in `scratch/` (done). Clean trunk vs live Agent option-logit parity.
2. IR gaps only as needed (`aten.amax.dim` mapped). Compile F16 GGUF. Strict 1D F32 for norms/biases. Bake RMS/affine stays default ON except **rope fusion OFF**.
3. Differential test vs `laya.Agent.system_one` on the model-card email (billing / urgency / churn / refund).
4. C++ CLI `--preset` / `--state` / `--text` / `--json`.
5. `--daemon` newline JSON-RPC (tab_completion pattern).
6. `--serve` Web Studio + `POST /api/decide` (timesfm keep-alive rule §14.I).
7. Presets from Laya (`email`, `triage`, `guard`, `moderation`, `router`) and Jev/LangChain evals (`expense`, `security`, `invoice`, `customer_service`, `harness`).
8. README + root showcase list. Never commit `AGENTS.md`. Never push.
9. `--bench` latency/throughput (done): per-question vs full-preset, CUDA vs CUDA-graph vs CPU, Python Agent reference.
10. Pad-to-max-in-batch + gallocr pin (done): Python `collate_items` equivalent; `pin_live_graph_tensors` so arena reuse is on; email is one `B=7` forward. Do not start Phase 4/5 unless asked.

### Constraints
- Hardware ceiling is the **RTX 4050 Laptop 6 GB**. Do not pick 1.7B as the default demo.
- Zero Python at runtime for the shipped `.exe`.
- CUDA graphs only on a stable `(B, S)` per request (dummy-pad later chunks; recapture if live `S` changes). CUDA `B·S ≤ 1024` with OOM-halve fallback.
- Scratch scripts in `scratch/`. Use `uv pip install laya` if needed.
- Never commit `AGENTS.md`. Never push.

### Out of scope for this example
- Phase 4 quantized KV, Phase 5 swarm serving, OpenAI HTTP `/v1/chat/completions`.
- `laya-typed-decisions` / multilingual GGUFs until English CustomVoice-class demo ships.
- Qwen3-TTS, video, FLUX — different domain; not this chat.

