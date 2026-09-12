#include "visualizer.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace timesfm {

bool Visualizer::generate_svg(
    const std::string& filepath,
    const std::vector<float>& history,
    const ForecastResult& forecast,
    const std::string& title,
    int width,
    int height
) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    int pad_left = 70;
    int pad_right = 40;
    int pad_top = 60;
    int pad_bottom = 50;

    int plot_w = width - pad_left - pad_right;
    int plot_h = height - pad_top - pad_bottom;

    // Determine min/max values
    float min_val = 1e9f;
    float max_val = -1e9f;

    for (float v : history) {
        if (!std::isnan(v)) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }

    for (float v : forecast.predictions) {
        if (!std::isnan(v)) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }

    if (min_val >= max_val) {
        min_val -= 1.0f;
        max_val += 1.0f;
    }

    float val_range = max_val - min_val;
    min_val -= val_range * 0.05f;
    max_val += val_range * 0.05f;
    val_range = max_val - min_val;

    int total_points = static_cast<int>(history.size() + forecast.horizon);
    auto get_x = [&](int idx) -> float {
        return pad_left + static_cast<float>(idx) / std::max(1, total_points - 1) * plot_w;
    };

    auto get_y = [&](float val) -> float {
        return pad_top + (1.0f - (val - min_val) / val_range) * plot_h;
    };

    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << width << " " << height << "\" width=\"100%\" height=\"100%\">\n";
    file << "  <defs>\n";
    file << "    <linearGradient id=\"bgGrad\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">\n";
    file << "      <stop offset=\"0%\" stop-color=\"#0b0f19\"/>\n";
    file << "      <stop offset=\"100%\" stop-color=\"#111827\"/>\n";
    file << "    </linearGradient>\n";
    file << "  </defs>\n";

    // Background
    file << "  <rect width=\"" << width << "\" height=\"" << height << "\" fill=\"url(#bgGrad)\" rx=\"10\"/>\n";

    // Title
    file << "  <text x=\"" << pad_left << "\" y=\"36\" fill=\"#f3f4f6\" font-family=\"system-ui, -apple-system, sans-serif\" font-size=\"18\" font-weight=\"600\">" << title << "</text>\n";
    file << "  <text x=\"" << (width - pad_right) << "\" y=\"36\" text-anchor=\"end\" fill=\"#9ca3af\" font-family=\"monospace\" font-size=\"12\">"
         << "Horizon: " << forecast.horizon << " | Latency: " << std::fixed << std::setprecision(1) << forecast.inference_time_ms << "ms</text>\n";

    // Grid lines
    int num_y_ticks = 5;
    for (int i = 0; i <= num_y_ticks; ++i) {
        float y_val = min_val + (val_range / num_y_ticks) * i;
        float y_pos = get_y(y_val);
        file << "  <line x1=\"" << pad_left << "\" y1=\"" << y_pos << "\" x2=\"" << (pad_left + plot_w) << "\" y2=\"" << y_pos << "\" stroke=\"#1f2937\" stroke-width=\"1\" stroke-dasharray=\"3,3\"/>\n";
        file << "  <text x=\"" << (pad_left - 10) << "\" y=\"" << (y_pos + 4) << "\" text-anchor=\"end\" fill=\"#6b7280\" font-family=\"monospace\" font-size=\"11\">" << std::fixed << std::setprecision(2) << y_val << "</text>\n";
    }

    // Split vertical line
    if (!history.empty()) {
        float split_x = get_x(static_cast<int>(history.size() - 1));
        file << "  <line x1=\"" << split_x << "\" y1=\"" << pad_top << "\" x2=\"" << split_x << "\" y2=\"" << (pad_top + plot_h) << "\" stroke=\"#6b7280\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\"/>\n";
        file << "  <text x=\"" << (split_x - 6) << "\" y=\"" << (pad_top + 16) << "\" fill=\"#9ca3af\" font-size=\"11\" font-family=\"sans-serif\" text-anchor=\"end\">History</text>\n";
        file << "  <text x=\"" << (split_x + 6) << "\" y=\"" << (pad_top + 16) << "\" fill=\"#ea580c\" font-size=\"11\" font-family=\"sans-serif\">Forecast</text>\n";
    }

    int hist_offset = static_cast<int>(history.size());
    int q_0 = 0; // q10
    int q_8 = 8; // q90
    int q_1 = 1; // q20
    int q_7 = 7; // q80
    int q_2 = 2; // q30
    int q_6 = 6; // q70
    int q_median = 4; // q50

    // Multi-Layer Nested Quantile Fans
    if (forecast.num_quantiles >= 9) {
        // Outer 80% Band (q10 to q90)
        file << "  <polygon points=\"";
        for (int t = 0; t < forecast.horizon; ++t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_8 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        for (int t = static_cast<int>(forecast.horizon) - 1; t >= 0; --t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_0 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        file << "\" fill=\"#ea580c\" fill-opacity=\"0.14\"/>\n";

        // Middle 60% Band (q20 to q80)
        file << "  <polygon points=\"";
        for (int t = 0; t < forecast.horizon; ++t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_7 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        for (int t = static_cast<int>(forecast.horizon) - 1; t >= 0; --t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_1 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        file << "\" fill=\"#ea580c\" fill-opacity=\"0.20\"/>\n";

        // Inner 40% Band (q30 to q70)
        file << "  <polygon points=\"";
        for (int t = 0; t < forecast.horizon; ++t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_6 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        for (int t = static_cast<int>(forecast.horizon) - 1; t >= 0; --t) {
            float x = get_x(hist_offset + t);
            float y = get_y(forecast.predictions[q_2 * forecast.horizon + t]);
            file << x << "," << y << " ";
        }
        file << "\" fill=\"#ea580c\" fill-opacity=\"0.28\"/>\n";
    }

    // Historical context path
    if (!history.empty()) {
        file << "  <path d=\"M ";
        for (size_t i = 0; i < history.size(); ++i) {
            float x = get_x(static_cast<int>(i));
            float y = get_y(history[i]);
            file << (i == 0 ? "" : "L ") << x << " " << y << " ";
        }
        file << "\" fill=\"none\" stroke=\"#38bdf8\" stroke-width=\"2.4\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n";
    }

    // Forecast median path
    file << "  <path d=\"M ";
    if (!history.empty()) {
        file << get_x(hist_offset - 1) << " " << get_y(history.back()) << " L ";
    }
    for (int t = 0; t < forecast.horizon; ++t) {
        float x = get_x(hist_offset + t);
        float y = get_y(forecast.predictions[q_median * forecast.horizon + t]);
        file << (t == 0 && history.empty() ? "" : "L ") << x << " " << y << " ";
    }
    file << "\" fill=\"none\" stroke=\"#ea580c\" stroke-width=\"2.6\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n";

    // Legend
    int leg_y = height - 16;
    file << "  <line x1=\"" << pad_left << "\" y1=\"" << leg_y << "\" x2=\"" << (pad_left + 18) << "\" y2=\"" << leg_y << "\" stroke=\"#38bdf8\" stroke-width=\"2.5\"/>\n";
    file << "  <text x=\"" << (pad_left + 24) << "\" y=\"" << (leg_y + 4) << "\" fill=\"#9ca3af\" font-family=\"sans-serif\" font-size=\"11\">Observed History</text>\n";

    file << "  <line x1=\"" << (pad_left + 140) << "\" y1=\"" << leg_y << "\" x2=\"" << (pad_left + 158) << "\" y2=\"" << leg_y << "\" stroke=\"#ea580c\" stroke-width=\"2.5\"/>\n";
    file << "  <text x=\"" << (pad_left + 164) << "\" y=\"" << (leg_y + 4) << "\" fill=\"#9ca3af\" font-family=\"sans-serif\" font-size=\"11\">Forecast (q50)</text>\n";

    file << "  <rect x=\"" << (pad_left + 270) << "\" y=\"" << (leg_y - 6) << "\" width=\"14\" height=\"12\" fill=\"#ea580c\" fill-opacity=\"0.14\" stroke=\"#ea580c\" stroke-width=\"0.5\"/>\n";
    file << "  <text x=\"" << (pad_left + 290) << "\" y=\"" << (leg_y + 4) << "\" fill=\"#9ca3af\" font-family=\"sans-serif\" font-size=\"11\">80% Interval (q10-q90)</text>\n";

    file << "  <rect x=\"" << (pad_left + 440) << "\" y=\"" << (leg_y - 6) << "\" width=\"14\" height=\"12\" fill=\"#ea580c\" fill-opacity=\"0.28\" stroke=\"#ea580c\" stroke-width=\"0.5\"/>\n";
    file << "  <text x=\"" << (pad_left + 460) << "\" y=\"" << (leg_y + 4) << "\" fill=\"#9ca3af\" font-family=\"sans-serif\" font-size=\"11\">40% Interval (q30-q70)</text>\n";

    file << "</svg>\n";
    return true;
}

bool Visualizer::generate_html_chart(
    const std::string& filepath,
    const std::vector<float>& history,
    const ForecastResult& forecast,
    const std::string& title
) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "<!DOCTYPE html>\n<html>\n<head>\n";
    file << "  <meta charset=\"utf-8\">\n";
    file << "  <title>" << title << "</title>\n";
    file << "  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n";
    file << "  <style>\n";
    file << "    body { background: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 24px; }\n";
    file << "    .card { background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); max-width: 1200px; margin: 0 auto; }\n";
    file << "    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 12px; }\n";
    file << "    .stats { display: flex; gap: 20px; font-size: 14px; color: #94a3b8; }\n";
    file << "    .badge { background: #0f172a; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; }\n";
    file << "    .chart-container { position: relative; height: 550px; width: 100%; }\n";
    file << "  </style>\n";
    file << "</head>\n<body>\n";
    file << "<div class=\"card\">\n";
    file << "  <div class=\"header\">\n";
    file << "    <h2>" << title << "</h2>\n";
    file << "    <div class=\"stats\">\n";
    file << "      <div class=\"badge\">Horizon: <b>" << forecast.horizon << " steps</b></div>\n";
    file << "      <div class=\"badge\">Latency: <b>" << std::fixed << std::setprecision(1) << forecast.inference_time_ms << " ms</b></div>\n";
    file << "      <div class=\"badge\">RevIN Mean/Std: <b>" << std::fixed << std::setprecision(2) << forecast.mean << " / " << forecast.std << "</b></div>\n";
    file << "    </div>\n";
    file << "  </div>\n";
    file << "  <div class=\"chart-container\"><canvas id=\"forecastChart\"></canvas></div>\n";
    file << "</div>\n";

    // Chart.js script
    file << "<script>\n";
    file << "  const ctx = document.getElementById('forecastChart').getContext('2d');\n";
    file << "  const historyData = [";
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0) file << ",";
        file << history[i];
    }
    file << "];\n";

    file << "  const horizon = " << forecast.horizon << ";\n";
    file << "  const totalLen = historyData.length + horizon;\n";
    file << "  const labels = Array.from({length: totalLen}, (_, i) => i < historyData.length ? 'T-' + (historyData.length - i) : 'T+' + (i - historyData.length + 1));\n";

    file << "  const q10 = Array(historyData.length).fill(null).concat([";
    for (int t = 0; t < forecast.horizon; ++t) {
        if (t > 0) file << ",";
        file << forecast.predictions[0 * forecast.horizon + t];
    }
    file << "]);\n";

    file << "  const q50 = Array(historyData.length - 1).fill(null).concat([historyData[historyData.length-1], ";
    for (int t = 0; t < forecast.horizon; ++t) {
        if (t > 0) file << ",";
        file << forecast.predictions[4 * forecast.horizon + t];
    }
    file << "]);\n";

    file << "  const q90 = Array(historyData.length).fill(null).concat([";
    for (int t = 0; t < forecast.horizon; ++t) {
        if (t > 0) file << ",";
        file << forecast.predictions[8 * forecast.horizon + t];
    }
    file << "]);\n";

    file << "  new Chart(ctx, {\n";
    file << "    type: 'line',\n";
    file << "    data: {\n";
    file << "      labels: labels,\n";
    file << "      datasets: [\n";
    file << "        { label: 'History', data: historyData.concat(Array(horizon).fill(null)), borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 },\n";
    file << "        { label: 'Forecast (q50)', data: q50, borderColor: '#10b981', borderWidth: 2.5, pointRadius: 0 },\n";
    file << "        { label: 'q90 Upper', data: q90, borderColor: 'transparent', backgroundColor: 'rgba(16, 185, 129, 0.15)', fill: '+1', pointRadius: 0 },\n";
    file << "        { label: 'q10 Lower', data: q10, borderColor: 'transparent', fill: false, pointRadius: 0 }\n";
    file << "      ]\n";
    file << "    },\n";
    file << "    options: {\n";
    file << "      responsive: true,\n";
    file << "      maintainAspectRatio: false,\n";
    file << "      scales: { x: { grid: { color: '#334155' } }, y: { grid: { color: '#334155' } } }\n";
    file << "    }\n";
    file << "  });\n";
    file << "</script>\n</body>\n</html>\n";

    return true;
}

} // namespace timesfm
