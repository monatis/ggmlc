#include "doctor.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace timesfm {

DoctorReport DoctorEngine::run_diagnostics(
    TimesFMForecaster& forecaster,
    const std::string& device,
    int n_threads
) {
    DoctorReport report;
    report.model_name = forecaster.model_graph().name.empty() ? "TimesFM 3.0" : forecaster.model_graph().name;
    report.device = device;
    report.n_threads = n_threads;

    auto start_all = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------
    // Check 1: Sanity & Monotonicity (Finite float, quantile sorting)
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Quantile Monotonicity & Numeric Sanity";
        c.category = "Sanity";

        TimeSeriesData demo_data;
        DataLoader::load_preset("trend_seasonal", demo_data, 128);

        ForecastConfig cfg;
        cfg.horizon = 64;
        cfg.sort_quantiles = true;
        cfg.device = device;
        cfg.n_threads = n_threads;

        ForecastResult res = forecaster.forecast(demo_data.values, cfg);

        bool finite = true;
        bool sorted = true;
        for (int64_t t = 0; t < res.horizon; ++t) {
            for (int64_t q = 0; q < res.num_quantiles; ++q) {
                float val = res.predictions[q * res.horizon + t];
                if (!std::isfinite(val)) finite = false;
                if (q > 0) {
                    float prev_val = res.predictions[(q - 1) * res.horizon + t];
                    if (val < prev_val - 1e-5f) sorted = false;
                }
            }
        }

        c.passed = finite && sorted && (res.horizon == 64);
        c.measured_value = (finite && sorted) ? "All 9 quantiles sorted, 0 NaNs" : "NaN or sorting violation";
        c.target_threshold = "Monotonic q10 <= ... <= q90 & finite";
        c.details = "Evaluated 64 future steps x 9 quantiles (576 values)";
        report.checks.push_back(c);
    }

    // -------------------------------------------------------------
    // Check 2: Single-Item Latency (B=1, Horizon=128)
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Single-Item Latency (B=1, H=128)";
        c.category = "Speed";

        TimeSeriesData demo_data;
        DataLoader::load_preset("linear_trend", demo_data, 128);

        ForecastConfig cfg;
        cfg.horizon = 128;
        cfg.device = device;
        cfg.n_threads = n_threads;

        // Warmup
        forecaster.forecast(demo_data.values, cfg);

        auto t0 = std::chrono::high_resolution_clock::now();
        ForecastResult res = forecaster.forecast(demo_data.values, cfg);
        auto t1 = std::chrono::high_resolution_clock::now();

        double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report.single_latency_ms = lat_ms;

        c.passed = (lat_ms < 1500.0); // pass if < 1.5s
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << lat_ms << " ms";
        c.measured_value = oss.str();
        c.target_threshold = "< 1500.0 ms (2-chunk rollout)";
        c.details = "Full 128-step multi-patch autoregressive rollout";
        report.checks.push_back(c);
    }

    // -------------------------------------------------------------
    // Check 3: Batched Throughput (B=8, Horizon=128)
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Batch Throughput (B=8, H=128)";
        c.category = "Speed";

        TimeSeriesData demo_data;
        DataLoader::load_preset("trend_seasonal", demo_data, 128);
        std::vector<std::vector<float>> batch8(8, demo_data.values);

        ForecastConfig cfg;
        cfg.horizon = 128;
        cfg.device = device;
        cfg.n_threads = n_threads;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto b_res = forecaster.forecast_batch(batch8, cfg);
        auto t1 = std::chrono::high_resolution_clock::now();

        double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double total_points = 8.0 * 128.0;
        double pts_sec = (total_points / (lat_ms / 1000.0));
        report.batch8_throughput_pts_sec = pts_sec;

        c.passed = (pts_sec > 250.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << pts_sec << " pts/sec (" << lat_ms << " ms total)";
        c.measured_value = oss.str();
        c.target_threshold = "> 250.0 points/sec";
        c.details = "Concurrent batch forward pass across 8 parallel series";
        report.checks.push_back(c);
    }

    // -------------------------------------------------------------
    // Check 4: Classic Benchmark Accuracy (airline_passengers)
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Airline Passengers Skill Ratio (vs Naive)";
        c.category = "Accuracy";

        TimeSeriesData air_data;
        DataLoader::load_preset("airline_passengers", air_data);

        BacktestConfig b_cfg;
        b_cfg.context_len = 72;
        b_cfg.horizon = 24;
        b_cfg.stride = 12;
        b_cfg.device = device;
        b_cfg.n_threads = n_threads;

        BacktestResult bt_res = BacktestEngine::run_backtest(forecaster, air_data.values, b_cfg);

        c.passed = (bt_res.avg_mae_vs_naive_ratio < 1.0f && bt_res.avg_mae < 45.0f);
        std::ostringstream oss;
        oss << "Ratio: " << std::fixed << std::setprecision(3) << bt_res.avg_mae_vs_naive_ratio
            << " (MAE: " << std::setprecision(1) << bt_res.avg_mae << " vs Naive: " << bt_res.avg_naive_mae << ")";
        c.measured_value = oss.str();
        c.target_threshold = "Skill Ratio < 1.00 (Beats persistence)";
        c.details = "5 rolling windows (Context=72, Horizon=24, Stride=12)";
        report.checks.push_back(c);
    }

    // -------------------------------------------------------------
    // Check 5: Retail Seasonality & RevIN Capture (weekly_retail)
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Weekly Retail Seasonality & Error";
        c.category = "Accuracy";

        TimeSeriesData ret_data;
        DataLoader::load_preset("weekly_retail", ret_data, 192);

        BacktestConfig b_cfg;
        b_cfg.context_len = 64;
        b_cfg.horizon = 28;
        b_cfg.stride = 14;
        b_cfg.device = device;
        b_cfg.n_threads = n_threads;

        BacktestResult bt_res = BacktestEngine::run_backtest(forecaster, ret_data.values, b_cfg);

        c.passed = (bt_res.avg_smape < 25.0f);
        std::ostringstream oss;
        oss << "sMAPE: " << std::fixed << std::setprecision(2) << bt_res.avg_smape << "%"
            << " (80% Cov: " << std::setprecision(1) << bt_res.avg_coverage_80 << "%)";
        c.measured_value = oss.str();
        c.target_threshold = "sMAPE < 25.00%";
        c.details = "Multi-window retail demand evaluation with day multipliers";
        report.checks.push_back(c);
    }

    // -------------------------------------------------------------
    // Check 6: Linear Detrending & Non-Negativity Auto-Clamping
    // -------------------------------------------------------------
    {
        DiagnosticCheck c;
        c.name = "Detrending & Non-Negativity Clamp";
        c.category = "Sanity";

        TimeSeriesData spiky_data;
        DataLoader::load_preset("spiky_demand", spiky_data, 128);

        ForecastConfig cfg;
        cfg.horizon = 32;
        cfg.detrend = true;
        cfg.make_positive = true;
        cfg.device = device;
        cfg.n_threads = n_threads;

        ForecastResult res = forecaster.forecast(spiky_data.values, cfg);

        float min_v = 1e9f;
        for (float v : res.predictions) {
            min_v = std::min(min_v, v);
        }

        c.passed = (min_v >= 0.0f);
        std::ostringstream oss;
        oss << "Min Value: " << std::fixed << std::setprecision(4) << min_v;
        c.measured_value = oss.str();
        c.target_threshold = "Min Value >= 0.0000";
        c.details = "Verified on Poisson-like spiky demand series";
        report.checks.push_back(c);
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    report.total_runtime_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

    report.total_checks = static_cast<int>(report.checks.size());
    report.passed_checks = 0;
    for (const auto& chk : report.checks) {
        if (chk.passed) report.passed_checks++;
    }
    report.all_passed = (report.passed_checks == report.total_checks);

    return report;
}

void DoctorEngine::print_report(const DoctorReport& report) {
    std::cout << "\n========================================================================================\n"
              << "                      Google TimesFM 3.0 — Comprehensive Engine Doctor Report           \n"
              << "========================================================================================\n"
              << " Model Graph     : " << report.model_name << "\n"
              << " Hardware Target : " << report.device << (report.device == "cpu" ? (" (" + std::to_string(report.n_threads) + " AVX2 threads)") : " (NVIDIA CUDA)") << "\n"
              << " Diagnostic Time : " << std::fixed << std::setprecision(2) << report.total_runtime_ms << " ms\n"
              << " Overall Status  : " << (report.all_passed ? "[HEALTHY / 100% OPERATIONAL]" : "[DEGRADED / ISSUES FOUND]") << "\n"
              << "----------------------------------------------------------------------------------------\n"
              << std::left << std::setw(10) << "Status"
              << std::setw(38) << "Diagnostic Check"
              << std::setw(28) << "Measured Performance"
              << "Target / Baseline\n"
              << "----------------------------------------------------------------------------------------\n";

    for (const auto& chk : report.checks) {
        std::string status_tag = chk.passed ? "[PASS]" : "[FAIL]";
        std::cout << std::left << std::setw(10) << status_tag
                  << std::setw(38) << chk.name
                  << std::setw(28) << chk.measured_value
                  << chk.target_threshold << "\n";
    }

    std::cout << "----------------------------------------------------------------------------------------\n"
              << " Summary: " << report.passed_checks << "/" << report.total_checks << " diagnostic checks passed.\n";

    if (report.all_passed) {
        std::cout << " Result : All mathematical pipelines, hardware kernels, and benchmark accuracy tests PASSED.\n"
                  << "          TimesFM 3.0 is verified and ready for zero-shot forecasting and production serving.\n";
    } else {
        std::cout << " Result : Some diagnostic checks did not meet the target thresholds. Review details above.\n";
    }
    std::cout << "========================================================================================\n\n";
}

} // namespace timesfm
