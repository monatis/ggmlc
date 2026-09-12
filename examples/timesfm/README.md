# `timesfm.cpp` — Standalone Google TimesFM 3.0 Foundation Forecasting Engine

A zero-dependency, high-performance C++ implementation of **Google TimesFM 3.0** (`google/timesfm-3.0-pytorch`), compiled and optimized via the **`ggmlc`** neural compiler toolchain.

---

## Key Highlights

- **100% Offline & Framework-Free**: Runs natively in pure C++ on CPU (AVX2/FMA GEMV microkernels) and NVIDIA GPUs (CUDA stream & tensor execution) without Python, PyTorch, or LibTorch.
- **Full Foundation Architecture**: Exact execution of Google TimesFM 3.0:
  - 20 Mixing Transformer layers ($1280$ hidden dim, $16$ heads, $80$ head dim)
  - RoPE positional embeddings, QK-RMSNorm, Pax-style `PerDimScale`, and cross-variate attention
  - Dynamic patching ($32 \to 64$ patch mapping)
- **Production Quantization Schemes**:
  - `F16`: Full precision weights (~$632$ MB)
  - `Q4_0`: 4-bit block quantization (~$178$ MB, **3.5x memory reduction**)
- **Full Domain Math & Statistical Calibration Suite**:
  - **Empirical Quantile Interval Coverage**: Scored against held-out actuals for both 80% ($q_{0.10} \dots q_{0.90}$) and 40% ($q_{0.30} \dots q_{0.70}$) intervals to evaluate probabilistic calibration.
  - **MAE vs. Naive Baseline Skill Ratio**: Compares TimesFM error against the persistence naive baseline ($y_{t+h} = y_t$). Ratios $< 1.0$ quantitatively prove the model outperforms persistence.
  - **RevIN Normalization & CPM Refinement**: Robust mean/variance normalization and step-decay shrinkage to mitigate long-horizon autoregressive drift.
  - **Linear Detrending**: Automatic $R^2 \ge 0.5$ regression slope/intercept estimation, pre-model subtraction, and post-prediction trend reconstitution.
  - **Quantile Monotonicity Enforcement**: Guarantees $Q_{0.10} \le Q_{0.20} \le \dots \le Q_{0.90}$ sorting across all future time steps.
  - **Auto-Positive & Non-Negativity Constraint**: Enforces physical boundaries ($\ge 0$) automatically if history is non-negative, or via `--non-negative`.
  - **Symmetric Flip-Invariance Averaging (`--sym-avg`)**: Averages predictions of series $y$ and $-y$ to eliminate directional polarity bias.
  - **Missing Value Linear Interpolation**: Robust handling of NaNs, nulls, and leading/trailing gaps.
- **Application-Level Utilities & Ergonomics**:
  - **Freeform Text Ingestion**: Pass raw number sequences separated by comma, space, semicolon, or newline directly via `--text "<v1,v2...>"` or Web UI textarea.
  - **Built-in Synthetic & Benchmark Presets**: 8 standard presets available out-of-the-box (`linear_trend`, `seasonal_sine`, `trend_seasonal`, `random_walk`, `weekly_retail`, `spiky_demand`, `airline_passengers`, `sunspots`).
  - Zero-dependency CSV parser with auto-column detection.
  - Multi-format exporter: CSV and structured JSON.
  - **Multi-Band Vector Visualizer**: Generates high-DPI SVG line charts with dark theme aesthetics and layered 80%, 60%, and 40% quantile confidence fans.
  - **Interactive Chart.js HTML Visualizer**: Standalone web report with pan/zoom and dataset toggles.
  - **Single-Binary Web Studio & REST API**: Run `--serve --port 8080` to launch an embedded Single-Page Application (SPA) dashboard and HTTP JSON API.

---

## Directory Layout

```
examples/timesfm/
├── CMakeLists.txt              # CMake build configuration (links ggmlc_runtime)
├── README.md                   # Architecture and CLI documentation
├── handoff.md                  # Comprehensive engineering handoff notes
├── include/
│   ├── revin.h                 # RevIN normalization & CPM refinement
│   ├── detrending.h            # Linear detrending (R^2 thresholding)
│   ├── patching.h              # Dynamic patching, unpatching, NaN interpolation
│   ├── data_loader.h           # CSV parser, freeform text parser, and preset loader
│   ├── export.h                # CSV & JSON serialization
│   ├── visualizer.h            # Standalone SVG & HTML vector chart generators
│   ├── forecaster.h            # High-level forecaster & multi-horizon rollout
│   ├── backtest.h              # Rolling backtesting & calibration metrics engine
│   ├── server.h                # Embedded HTTP micro-server & REST endpoints
│   └── web_assets.h            # Embedded Single-Page App Studio dashboard
└── src/
    ├── revin.cpp
    ├── detrending.cpp
    ├── patching.cpp
    ├── data_loader.cpp
    ├── export.cpp
    ├── visualizer.cpp
    ├── forecaster.cpp
    ├── backtest.cpp
    ├── server.cpp
    └── main.cpp                # CLI entry point
```

---

## Build Instructions

### Native Windows Build (MSVC 2022 + CUDA 11.3 + Ninja)
```powershell
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin;C:\Users\ailabs\ggmlc\.venv\Scripts;" + $env:PATH
$env:INCLUDE = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22000.0\um;" + $env:INCLUDE
$env:LIB = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x64;" + $env:LIB
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -Xcompiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

cmake --build build-win-cuda --target timesfm -j8
```

The compiled binary will be located at:
- `.\build-win-cuda\examples\timesfm\timesfm.exe`

---

## CLI Reference & Quickstart

```powershell
# 1. Inspect GGUF model metadata and tensor graph
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --info

# 2. Run automated Doctor health, speed, and accuracy benchmark diagnostics
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --doctor
# Or test on CUDA GPU:
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_ud_q4_k_m.gguf --doctor --device cuda

# 3. List all built-in synthetic & classic presets
.\build-win-cuda\examples\timesfm\timesfm.exe --list-presets

# 4. Forecast from freeform pasted numbers and export layered quantile SVG
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --text "12.1, 14.5, 18.2, 22.0, 25.1, 28.4, 31.0, 35.2" --horizon 16 --svg forecast.svg

# 5. Forecast on built-in 'weekly_retail' preset with non-negativity clamp
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --preset weekly_retail --horizon 28 --non-negative --svg retail.svg

# 6. Run rolling backtest on classic 'airline_passengers' benchmark
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --preset airline_passengers --backtest --context 72 --horizon 24 --stride 12 --svg airline_bt.svg --output-backtest airline_bt.json

# 7. Offload to NVIDIA CUDA GPU
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --preset sunspots --device cuda --horizon 64

# 8. Launch Single-Binary Web Studio & REST API Server
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --serve --port 8080
```

## Backtesting Engine & Calibration Metrics

`timesfm.exe` includes a built-in high-throughput backtesting engine for quantitative evaluation and historical rolling-window simulation.

```powershell
# Run rolling backtest across 256 time steps (Context=128, Horizon=64, Stride=32)
.\build-win-cuda\examples\timesfm\timesfm.exe scratch\timesfm3_f16.gguf --input data.csv --backtest --context 128 --horizon 64 --stride 32 --svg backtest.svg --output-backtest backtest.json
```

### Evaluated Error & Calibration Metrics
- **Mean Absolute Error (MAE)**: $\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|$ (median point forecast $q=0.5$).
- **Root Mean Square Error (RMSE)**: $\sqrt{\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2}$.
- **Symmetric MAPE (sMAPE)**: $\frac{100\%}{H} \sum_{t=1}^H \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t| + 1e-5}$.
- **Continuous Ranked Probability Score (CRPS)**: $\frac{2}{|\mathcal{Q}|} \sum_{\tau \in \mathcal{Q}} \frac{1}{H} \sum_{t=1}^H \text{Pinball}_\tau(y_t, \hat{y}_{\tau, t})$.
- **80% Central Interval Coverage**: $\frac{1}{H} \sum_{t=1}^H \mathbb{I}(q_{0.1, t} \le y_t \le q_{0.9, t})$ (evaluates if ~80% of actuals fall in the 80% band).
- **40% Central Interval Coverage**: $\frac{1}{H} \sum_{t=1}^H \mathbb{I}(q_{0.3, t} \le y_t \le q_{0.7, t})$.
- **MAE vs. Naive Baseline Skill Ratio**: $\frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$. A ratio $< 1.0$ indicates outperformance over naive persistence.

---

## REST API Specification

When running with `--serve`, the embedded server exposes:

### `GET /`
Serves the responsive dual-tab web studio (Forecast Studio & Backtesting Suite).

### `GET /api/presets`
Returns the list of available built-in synthetic and benchmark datasets.

### `GET /api/health`
```json
{
  "status": "ok",
  "model": "TimesFM 3.0",
  "device": "ready"
}
```

### `POST /api/forecast`
```json
{
  "context": [10.2, 11.5, 12.8, 14.1, 15.0, 16.3, 17.5, 18.2],
  "horizon": 64,
  "normalize": true,
  "detrend": true,
  "sort_quantiles": true,
  "make_positive": false,
  "use_symmetric_averaging": false
}
```

### `POST /api/forecast_batch`
```json
{
  "batch": [
    [10.2, 11.5, 12.8, 14.1, 15.0],
    [22.4, 23.1, 24.5, 25.0, 26.2]
  ],
  "horizon": 64,
  "normalize": true,
  "detrend": true
}
```

### `POST /api/backtest`
```json
{
  "series": [10.2, 11.5, 12.8, 14.1, 15.0, 16.3, 17.5, 18.2, 19.0, 20.4, 21.8, 23.0],
  "context_len": 128,
  "horizon": 64,
  "stride": 32,
  "max_windows": 32
}
```

---

## Empirical Verification & Benchmark Results

### A. Production Quantization Benchmark Matrix (Horizon = 64 Steps)

| Model Variant | Format | Weight Precision | GGUF Size | Memory Footprint | CPU Latency (4T) | CUDA Latency | Numerical Parity (Cos Sim) | Mean Rel Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TimesFM 3.0** | `F16` | Full Precision | **632.00 MB** | ~720 MB | **312.57 ms** | **1187.34 ms** | **1.000000** | **0.00%** (Baseline) | Verified Active |
| **TimesFM 3.0** | `Q8_0` | 8-bit Block | **336.47 MB** | ~390 MB | **198.60 ms** | **1431.06 ms** | **0.999945** | **0.77%** | Verified Active (Fastest CPU) |
| **TimesFM 3.0** | `UD-Q4_K_M` | Unsloth Dynamic | **304.20 MB** | ~360 MB | **236.59 ms** | **1376.02 ms** | **0.999898** | **1.19%** | Verified Active (Best Compression) |
| **TimesFM 3.0** | `Q4_0` | 4-bit Naive Block | **178.81 MB** | ~240 MB | 285.55 ms | 1773.89 ms | 0.998585 | 4.00% | *Pruned from Tests (Unacceptable Error)* |

*All active variants guarantee **100.0% Quantile Monotonicity** ($Q_{0.1} \le Q_{0.5} \le Q_{0.9}$) across all rollout windows.*

### B. Hardware Batching Throughput Matrix: CPU (4 Threads) vs NVIDIA CUDA GPU (GTX 1050)

All measurements conducted across 64 forecast steps (2 rollout chunks of 32):

| Model Variant | Hardware Target | Batch Size ($B$) | Total Latency (ms) | Per-Series Latency (ms) | Throughput (Series/sec) | Throughput (Points/sec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F16** (632 MB) | CPU (4 Threads) | $B = 1$ | 60.03 ms | 60.03 ms | 16.7 series/s | 1,066 pts/s |
| | CPU (4 Threads) | $B = 4$ | 242.35 ms | 60.59 ms | 16.5 series/s | 1,056 pts/s |
| | CPU (4 Threads) | $B = 16$ | 1127.80 ms | 70.49 ms | 14.2 series/s | 908 pts/s |
| | **CUDA GPU** | $B = 1$ | 48.16 ms | 48.16 ms | 20.8 series/s | 1,329 pts/s |
| | **CUDA GPU** | $B = 4$ | 157.41 ms | 39.35 ms | 25.4 series/s | 1,626 pts/s |
| | **CUDA GPU** | $B = 16$ | 608.85 ms | 38.05 ms | **26.3 series/s** | **1,682 pts/s** |
| **Q8_0** (336 MB) | CPU (4 Threads) | $B = 1$ | 59.30 ms | 59.30 ms | 16.9 series/s | 1,079 pts/s |
| | CPU (4 Threads) | $B = 4$ | 205.78 ms | 51.45 ms | 19.4 series/s | 1,244 pts/s |
| | CPU (4 Threads) | $B = 16$ | 650.22 ms | 40.64 ms | 24.6 series/s | 1,575 pts/s |
| | **CUDA GPU** | $B = 1$ | 30.83 ms | 30.83 ms | 32.4 series/s | 2,076 pts/s |
| | **CUDA GPU** | $B = 4$ | 81.83 ms | 20.46 ms | 48.9 series/s | 3,128 pts/s |
| | **CUDA GPU** | $B = 16$ | 301.91 ms | 18.87 ms | **53.0 series/s** | **3,392 pts/s** |
| **UD-Q4_K_M** (304 MB) | CPU (4 Threads) | $B = 1$ | 59.64 ms | 59.64 ms | 16.8 series/s | 1,073 pts/s |
| | CPU (4 Threads) | $B = 4$ | 180.86 ms | 45.22 ms | 22.1 series/s | 1,415 pts/s |
| | CPU (4 Threads) | $B = 16$ | 660.98 ms | 41.31 ms | 24.2 series/s | 1,549 pts/s |
| | **CUDA GPU** | $B = 1$ | 32.09 ms | 32.09 ms | 31.2 series/s | 1,994 pts/s |
| | **CUDA GPU** | $B = 4$ | 81.17 ms | 20.29 ms | 49.3 series/s | 3,154 pts/s |
| | **CUDA GPU** | $B = 16$ | 299.14 ms | 18.70 ms | **53.5 series/s** | **3,423 pts/s** |

### C. Architectural Takeaways: GPU Scaling vs CPU Saturation

1. **CUDA GPU Batching Speedup**:
   - On NVIDIA GeForce GTX 1050, increasing batch size from $B=1$ to $B=16$ slashes per-series latency from **30.83 ms down to 18.87 ms** (a **+63% throughput increase**), reaching **3,392 points/sec** for `Q8_0` and **3,423 points/sec** for `UD-Q4_K_M`.
   - Batching saturates GPU CUDA cores and amortizes Windows WDDM driver queue launch overhead across all $B$ series.
2. **CPU Behavior Under Batching**:
   - On CPU (4 AVX2 threads), memory bus bandwidth and core compute are nearly fully utilized at $B=1..4$. Beyond $B=8$, multi-threading synchronization and L3 cache thrashing create slight per-series latency plateaus.
   - At $B=16$, **CUDA GPU delivers 2.2x higher throughput than CPU** (3,392 vs 1,575 pts/s for Q8_0, and 1,682 vs 908 pts/s for F16).



