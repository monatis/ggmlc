#include "export.h"

#include <fstream>
#include <sstream>
#include <iomanip>

namespace timesfm {

bool Exporter::save_csv(
    const std::string& filepath,
    const ForecastResult& result,
    const std::string& value_name
) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "step,timestamp";
    for (size_t q = 0; q < result.quantiles.size(); ++q) {
        file << ",q" << std::fixed << std::setprecision(2) << result.quantiles[q];
    }
    file << "\n";

    for (int64_t t = 0; t < result.horizon; ++t) {
        std::string ts = (static_cast<size_t>(t) < result.future_timestamps.size())
                             ? result.future_timestamps[t]
                             : std::to_string(t + 1);
        file << (t + 1) << "," << ts;

        for (int64_t q = 0; q < result.num_quantiles; ++q) {
            int64_t idx = q * result.horizon + t;
            file << "," << std::fixed << std::setprecision(6) << result.predictions[idx];
        }
        file << "\n";
    }

    return true;
}

std::string Exporter::to_json_string(
    const ForecastResult& result,
    const std::vector<float>& history,
    const std::vector<std::string>& history_timestamps
) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\n";
    ss << "  \"horizon\": " << result.horizon << ",\n";
    ss << "  \"num_quantiles\": " << result.num_quantiles << ",\n";
    ss << "  \"inference_time_ms\": " << result.inference_time_ms << ",\n";
    ss << "  \"revin\": {\n";
    ss << "    \"mean\": " << result.mean << ",\n";
    ss << "    \"std\": " << result.std << "\n";
    ss << "  },\n";
    ss << "  \"detrending\": {\n";
    ss << "    \"applied\": " << (result.detrend_applied ? "true" : "false") << ",\n";
    ss << "    \"slope\": " << result.trend_slope << ",\n";
    ss << "    \"r2\": " << result.trend_r2 << "\n";
    ss << "  },\n";

    // Quantiles
    ss << "  \"quantiles\": [";
    for (size_t i = 0; i < result.quantiles.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << result.quantiles[i];
    }
    ss << "],\n";

    // Forecasts by quantile
    ss << "  \"forecast\": {\n";
    for (int64_t q = 0; q < result.num_quantiles; ++q) {
        std::string q_key = (static_cast<size_t>(q) < result.quantiles.size())
                                ? "q" + std::to_string(static_cast<int>(result.quantiles[q] * 100))
                                : "q_" + std::to_string(q);
        ss << "    \"" << q_key << "\": [";
        for (int64_t t = 0; t < result.horizon; ++t) {
            if (t > 0) ss << ", ";
            ss << result.predictions[q * result.horizon + t];
        }
        ss << "]";
        if (q + 1 < result.num_quantiles) ss << ",";
        ss << "\n";
    }
    ss << "  },\n";

    // History (if present)
    ss << "  \"history_len\": " << history.size() << ",\n";
    ss << "  \"history\": [";
    for (size_t i = 0; i < history.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << history[i];
    }
    ss << "]\n";
    ss << "}\n";

    return ss.str();
}

bool Exporter::save_json(
    const std::string& filepath,
    const ForecastResult& result,
    const std::vector<float>& history,
    const std::vector<std::string>& history_timestamps
) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;
    file << to_json_string(result, history, history_timestamps);
    return true;
}

} // namespace timesfm
