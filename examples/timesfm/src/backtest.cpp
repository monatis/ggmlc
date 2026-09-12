#include "backtest.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

namespace timesfm {

BacktestResult BacktestEngine::run_backtest(
    TimesFMForecaster& forecaster,
    const std::vector<float>& full_series,
    const BacktestConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    BacktestResult res;
    res.context_len = config.context_len;
    res.horizon = config.horizon;
    res.stride = config.stride;

    int64_t total_len = static_cast<int64_t>(full_series.size());
    if (total_len < config.context_len + config.horizon) {
        std::cerr << "Backtest warning: full series length (" << total_len
                  << ") is shorter than context_len + horizon ("
                  << config.context_len + config.horizon << ")." << std::endl;
        return res;
    }

    // 1. Identify all rolling evaluation cut-off points
    struct WindowTask {
        int64_t window_idx;
        int64_t start_t;
        int64_t cut_t;
        int64_t end_t;
        std::vector<float> context;
        std::vector<float> ground_truth;
    };

    std::vector<WindowTask> tasks;
    int64_t cut_t = config.context_len;
    int64_t w_idx = 0;

    while (cut_t + config.horizon <= total_len && static_cast<int64_t>(tasks.size()) < config.max_windows) {
        int64_t start_t = cut_t - config.context_len;
        int64_t end_t = cut_t + config.horizon;

        WindowTask task;
        task.window_idx = w_idx++;
        task.start_t = start_t;
        task.cut_t = cut_t;
        task.end_t = end_t;

        task.context.assign(full_series.begin() + start_t, full_series.begin() + cut_t);
        task.ground_truth.assign(full_series.begin() + cut_t, full_series.begin() + end_t);

        tasks.push_back(std::move(task));
        cut_t += config.stride;
    }

    res.total_windows = static_cast<int64_t>(tasks.size());
    if (tasks.empty()) return res;

    // 2. Batch forecast all window contexts
    std::vector<std::vector<float>> batch_contexts;
    batch_contexts.reserve(tasks.size());
    for (const auto& t : tasks) {
        batch_contexts.push_back(t.context);
    }

    ForecastConfig f_cfg;
    f_cfg.horizon = config.horizon;
    f_cfg.normalize = config.normalize;
    f_cfg.detrend = config.detrend;
    f_cfg.sort_quantiles = config.sort_quantiles;
    f_cfg.make_positive = config.make_positive;
    f_cfg.device = config.device;
    f_cfg.n_threads = config.n_threads;

    res.window_forecasts = forecaster.forecast_batch(batch_contexts, f_cfg);
    res.windows.resize(tasks.size());

    // 3. Compute Probabilistic & Point Metrics for each window
    float sum_mae = 0.0f;
    float sum_rmse = 0.0f;
    float sum_smape = 0.0f;
    float sum_crps = 0.0f;
    float sum_cov80 = 0.0f;
    float sum_cov40 = 0.0f;
    float sum_naive_mae = 0.0f;
    float sum_ratio = 0.0f;

    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& t = tasks[i];
        const auto& f = res.window_forecasts[i];
        auto& wm = res.windows[i];

        wm.window_idx = t.window_idx;
        wm.start_t = t.start_t;
        wm.cut_t = t.cut_t;
        wm.end_t = t.end_t;

        float w_ae = 0.0f;
        float w_se = 0.0f;
        float w_smape = 0.0f;
        float w_mape = 0.0f;
        float w_sum_actual = 0.0f;
        float w_pinball_sum = 0.0f;
        float w_naive_ae = 0.0f;
        int64_t in_80_count = 0;
        int64_t in_40_count = 0;

        int64_t H = config.horizon;
        int64_t Q = f.num_quantiles;
        int64_t median_idx = Q / 2; // q50
        int64_t q10_idx = 0;       // q10
        int64_t q90_idx = Q - 1;   // q90
        int64_t q30_idx = std::min<int64_t>(2, Q - 1); // q30
        int64_t q70_idx = std::min<int64_t>(6, Q - 1); // q70

        float last_observed = t.context.empty() ? 0.0f : t.context.back();

        for (int64_t h = 0; h < H; ++h) {
            float y = t.ground_truth[h];
            float y_hat = f.predictions[median_idx * H + h];

            float diff = y - y_hat;
            float abs_diff = std::abs(diff);
            w_ae += abs_diff;
            w_se += diff * diff;
            w_sum_actual += std::abs(y);
            w_naive_ae += std::abs(y - last_observed);

            float denom_smape = std::abs(y) + std::abs(y_hat) + 1e-5f;
            w_smape += (2.0f * abs_diff / denom_smape) * 100.0f;

            float denom_mape = std::abs(y) + 1e-5f;
            w_mape += (abs_diff / denom_mape) * 100.0f;

            // Pinball loss across all quantiles
            for (int64_t q_i = 0; q_i < Q; ++q_i) {
                float tau = f.quantiles[q_i];
                float y_q = f.predictions[q_i * H + h];
                float e = y - y_q;
                float loss = (e >= 0.0f) ? (tau * e) : ((tau - 1.0f) * e);
                w_pinball_sum += loss;
            }

            // 80% coverage check (q10 to q90)
            float q10 = f.predictions[q10_idx * H + h];
            float q90 = f.predictions[q90_idx * H + h];
            if (y >= q10 && y <= q90) {
                in_80_count++;
            }

            // 40% coverage check (q30 to q70)
            float q30 = f.predictions[q30_idx * H + h];
            float q70 = f.predictions[q70_idx * H + h];
            if (y >= q30 && y <= q70) {
                in_40_count++;
            }
        }

        wm.mae = w_ae / static_cast<float>(H);
        wm.rmse = std::sqrt(w_se / static_cast<float>(H));
        wm.smape = w_smape / static_cast<float>(H);
        wm.mape = w_mape / static_cast<float>(H);
        wm.wape = (w_ae / (w_sum_actual + 1e-5f)) * 100.0f;
        wm.crps = (2.0f * w_pinball_sum) / static_cast<float>(Q * H);
        wm.coverage_80 = (static_cast<float>(in_80_count) / static_cast<float>(H)) * 100.0f;
        wm.coverage_40 = (static_cast<float>(in_40_count) / static_cast<float>(H)) * 100.0f;
        wm.naive_mae = w_naive_ae / static_cast<float>(H);
        wm.mae_vs_naive_ratio = (wm.naive_mae > 1e-6f) ? (wm.mae / wm.naive_mae) : 1.0f;

        sum_mae += wm.mae;
        sum_rmse += wm.rmse;
        sum_smape += wm.smape;
        sum_crps += wm.crps;
        sum_cov80 += wm.coverage_80;
        sum_cov40 += wm.coverage_40;
        sum_naive_mae += wm.naive_mae;
        sum_ratio += wm.mae_vs_naive_ratio;
    }

    float N = static_cast<float>(tasks.size());
    float sum_mape = 0.0f;
    float sum_wape = 0.0f;
    for (const auto& w : res.windows) {
        sum_mape += w.mape;
        sum_wape += w.wape;
    }
    res.avg_mae = sum_mae / N;
    res.avg_rmse = sum_rmse / N;
    res.avg_smape = sum_smape / N;
    res.avg_mape = sum_mape / N;
    res.avg_wape = sum_wape / N;
    res.avg_crps = sum_crps / N;
    res.avg_coverage_80 = sum_cov80 / N;
    res.avg_coverage_40 = sum_cov40 / N;
    res.avg_naive_mae = sum_naive_mae / N;
    res.avg_mae_vs_naive_ratio = sum_ratio / N;

    auto end_time = std::chrono::high_resolution_clock::now();
    res.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return res;
}

std::string BacktestEngine::to_json_string(
    const BacktestResult& result,
    const std::vector<float>& full_series
) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "{\n";
    oss << "  \"total_windows\": " << result.total_windows << ",\n";
    oss << "  \"context_len\": " << result.context_len << ",\n";
    oss << "  \"horizon\": " << result.horizon << ",\n";
    oss << "  \"stride\": " << result.stride << ",\n";
    oss << "  \"avg_mae\": " << result.avg_mae << ",\n";
    oss << "  \"avg_rmse\": " << result.avg_rmse << ",\n";
    oss << "  \"avg_smape\": " << result.avg_smape << ",\n";
    oss << "  \"avg_mape\": " << result.avg_mape << ",\n";
    oss << "  \"avg_wape\": " << result.avg_wape << ",\n";
    oss << "  \"avg_crps\": " << result.avg_crps << ",\n";
    oss << "  \"avg_coverage_80\": " << result.avg_coverage_80 << ",\n";
    oss << "  \"avg_coverage_40\": " << result.avg_coverage_40 << ",\n";
    oss << "  \"avg_naive_mae\": " << result.avg_naive_mae << ",\n";
    oss << "  \"avg_mae_vs_naive_ratio\": " << result.avg_mae_vs_naive_ratio << ",\n";
    oss << "  \"total_time_ms\": " << result.total_time_ms << ",\n";

    oss << "  \"windows\": [\n";
    for (size_t i = 0; i < result.windows.size(); ++i) {
        const auto& w = result.windows[i];
        oss << "    {\n";
        oss << "      \"window_idx\": " << w.window_idx << ",\n";
        oss << "      \"start_t\": " << w.start_t << ",\n";
        oss << "      \"cut_t\": " << w.cut_t << ",\n";
        oss << "      \"end_t\": " << w.end_t << ",\n";
        oss << "      \"mae\": " << w.mae << ",\n";
        oss << "      \"rmse\": " << w.rmse << ",\n";
        oss << "      \"smape\": " << w.smape << ",\n";
        oss << "      \"mape\": " << w.mape << ",\n";
        oss << "      \"wape\": " << w.wape << ",\n";
        oss << "      \"crps\": " << w.crps << ",\n";
        oss << "      \"coverage_80\": " << w.coverage_80 << ",\n";
        oss << "      \"coverage_40\": " << w.coverage_40 << ",\n";
        oss << "      \"naive_mae\": " << w.naive_mae << ",\n";
        oss << "      \"mae_vs_naive_ratio\": " << w.mae_vs_naive_ratio << "\n";
        oss << "    }" << (i + 1 < result.windows.size() ? "," : "") << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";

    return oss.str();
}

bool BacktestEngine::save_json(
    const std::string& filepath,
    const BacktestResult& result,
    const std::vector<float>& full_series
) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;
    ofs << to_json_string(result, full_series);
    return true;
}

bool BacktestEngine::generate_svg(
    const std::string& filepath,
    const std::vector<float>& full_series,
    const BacktestResult& result,
    const std::string& title,
    int width,
    int height
) {
    if (full_series.empty()) return false;

    float min_val = full_series[0];
    float max_val = full_series[0];
    for (float v : full_series) {
        if (!std::isnan(v)) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }
    for (const auto& f : result.window_forecasts) {
        for (float v : f.predictions) {
            if (!std::isnan(v)) {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
    }

    float pad = (max_val - min_val) * 0.08f;
    if (pad < 1e-4f) pad = 1.0f;
    min_val -= pad;
    max_val += pad;

    int pad_left = 70;
    int pad_right = 40;
    int pad_top = 70;
    int pad_bottom = 50;

    int plot_w = width - pad_left - pad_right;
    int plot_h = height - pad_top - pad_bottom;

    int64_t total_points = static_cast<int64_t>(full_series.size());
    auto map_x = [&](int64_t idx) -> float {
        return pad_left + (static_cast<float>(idx) / std::max<int64_t>(1, total_points - 1)) * plot_w;
    };
    auto map_y = [&](float val) -> float {
        return pad_top + (1.0f - (val - min_val) / (max_val - min_val)) * plot_h;
    };

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;

    ofs << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << width << " " << height << "\" width=\"100%\" height=\"100%\">\n";
    ofs << "  <defs>\n";
    ofs << "    <linearGradient id=\"bgGrad\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">\n";
    ofs << "      <stop offset=\"0%\" stop-color=\"#0d1117\"/>\n";
    ofs << "      <stop offset=\"100%\" stop-color=\"#161b22\"/>\n";
    ofs << "    </linearGradient>\n";
    ofs << "  </defs>\n";
    ofs << "  <rect width=\"" << width << "\" height=\"" << height << "\" fill=\"url(#bgGrad)\" rx=\"12\"/>\n";

    // Grid lines
    ofs << "  <g stroke=\"#21262d\" stroke-width=\"1\" stroke-dasharray=\"3 3\">\n";
    for (int i = 0; i <= 5; ++i) {
        float y = pad_top + (static_cast<float>(i) / 5.0f) * plot_h;
        float val = max_val - (static_cast<float>(i) / 5.0f) * (max_val - min_val);
        ofs << "    <line x1=\"" << pad_left << "\" y1=\"" << y << "\" x2=\"" << (width - pad_right) << "\" y2=\"" << y << "\"/>\n";
        ofs << "    <text x=\"" << (pad_left - 10) << "\" y=\"" << (y + 4) << "\" fill=\"#8b949e\" font-size=\"11\" font-family=\"monospace\" text-anchor=\"end\">"
            << std::fixed << std::setprecision(1) << val << "</text>\n";
    }
    ofs << "  </g>\n";

    // Title & Summary Metrics
    ofs << "  <text x=\"" << pad_left << "\" y=\"36\" fill=\"#58a6ff\" font-size=\"18\" font-family=\"sans-serif\" font-weight=\"bold\">" << title << "</text>\n";
    ofs << "  <text x=\"" << (width - pad_right) << "\" y=\"36\" fill=\"#7ee787\" font-size=\"13\" font-family=\"monospace\" text-anchor=\"end\">"
        << "MAE: " << std::fixed << std::setprecision(2) << result.avg_mae
        << " | RMSE: " << result.avg_rmse
        << " | sMAPE: " << result.avg_smape << "%"
        << " | 80% Cov: " << result.avg_coverage_80 << "%</text>\n";

    // Ground Truth Line
    ofs << "  <polyline fill=\"none\" stroke=\"#8b949e\" stroke-width=\"2\" points=\"";
    for (int64_t i = 0; i < total_points; ++i) {
        ofs << map_x(i) << "," << map_y(full_series[i]) << " ";
    }
    ofs << "\"/>\n";

    // Forecast Windows
    const std::vector<std::string> window_colors = {"#f0883e", "#79c0ff", "#a371f7", "#56d364", "#ff7b72", "#e3b341"};
    for (size_t w_i = 0; w_i < result.windows.size(); ++w_i) {
        const auto& w = result.windows[w_i];
        const auto& f = result.window_forecasts[w_i];
        std::string col = window_colors[w_i % window_colors.size()];

        int64_t H = f.horizon;
        int64_t Q = f.num_quantiles;
        int64_t median_idx = Q / 2;
        int64_t q10_idx = 0;
        int64_t q90_idx = Q - 1;

        // Draw 80% band (q10 to q90)
        ofs << "  <polygon fill=\"" << col << "\" fill-opacity=\"0.2\" points=\"";
        for (int64_t h = 0; h < H; ++h) {
            int64_t t = w.cut_t + h;
            ofs << map_x(t) << "," << map_y(f.predictions[q90_idx * H + h]) << " ";
        }
        for (int64_t h = H - 1; h >= 0; --h) {
            int64_t t = w.cut_t + h;
            ofs << map_x(t) << "," << map_y(f.predictions[q10_idx * H + h]) << " ";
        }
        ofs << "\"/>\n";

        // Draw median prediction line
        ofs << "  <polyline fill=\"none\" stroke=\"" << col << "\" stroke-width=\"2.5\" points=\"";
        for (int64_t h = 0; h < H; ++h) {
            int64_t t = w.cut_t + h;
            ofs << map_x(t) << "," << map_y(f.predictions[median_idx * H + h]) << " ";
        }
        ofs << "\"/>\n";

        // Draw vertical cut marker
        float cut_x = map_x(w.cut_t);
        ofs << "  <line x1=\"" << cut_x << "\" y1=\"" << pad_top << "\" x2=\"" << cut_x << "\" y2=\"" << (height - pad_bottom) << "\" stroke=\"" << col << "\" stroke-width=\"1.5\" stroke-dasharray=\"4 2\" stroke-opacity=\"0.6\"/>\n";
    }

    ofs << "</svg>\n";
    return true;
}

} // namespace timesfm
