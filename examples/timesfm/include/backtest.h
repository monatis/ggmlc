#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "forecaster.h"

namespace timesfm {

struct BacktestConfig {
    int64_t context_len = 128;
    int64_t horizon = 64;
    int64_t stride = 64;
    int64_t max_windows = 32;
    bool normalize = true;
    bool detrend = true;
    bool sort_quantiles = true;
    bool make_positive = false;
    std::string device = "cpu";
    int n_threads = 4;
};

struct WindowMetrics {
    int64_t window_idx = 0;
    int64_t start_t = 0;
    int64_t cut_t = 0;
    int64_t end_t = 0;
    float mae = 0.0f;
    float rmse = 0.0f;
    float smape = 0.0f;
    float mape = 0.0f;
    float wape = 0.0f;
    float crps = 0.0f;
    float coverage_80 = 0.0f;
    float coverage_40 = 0.0f;
    float naive_mae = 0.0f;
    float mae_vs_naive_ratio = 1.0f;
};

struct BacktestResult {
    int64_t total_windows = 0;
    int64_t context_len = 0;
    int64_t horizon = 0;
    int64_t stride = 0;
    float avg_mae = 0.0f;
    float avg_rmse = 0.0f;
    float avg_smape = 0.0f;
    float avg_mape = 0.0f;
    float avg_wape = 0.0f;
    float avg_crps = 0.0f;
    float avg_coverage_80 = 0.0f;
    float avg_coverage_40 = 0.0f;
    float avg_naive_mae = 0.0f;
    float avg_mae_vs_naive_ratio = 1.0f;
    double total_time_ms = 0.0;
    std::vector<WindowMetrics> windows;
    std::vector<ForecastResult> window_forecasts;
};

class BacktestEngine {
public:
    // Slices full_series into rolling windows, batches them into GPU/CPU forward passes, and computes probabilistic metrics.
    static BacktestResult run_backtest(
        TimesFMForecaster& forecaster,
        const std::vector<float>& full_series,
        const BacktestConfig& config = {}
    );

    static std::string to_json_string(
        const BacktestResult& result,
        const std::vector<float>& full_series = {}
    );

    static bool save_json(
        const std::string& filepath,
        const BacktestResult& result,
        const std::vector<float>& full_series = {}
    );

    // Generates a comprehensive multi-window backtest visualization SVG
    static bool generate_svg(
        const std::string& filepath,
        const std::vector<float>& full_series,
        const BacktestResult& result,
        const std::string& title = "TimesFM 3.0 Backtesting Evaluation",
        int width = 1200,
        int height = 600
    );
};

} // namespace timesfm
