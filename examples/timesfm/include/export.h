#pragma once

#include <string>
#include <vector>

namespace timesfm {

struct ForecastResult {
    int64_t horizon = 0;
    int64_t num_quantiles = 9;
    std::vector<float> quantiles; // size = 9 (e.g. 0.1 .. 0.9)
    std::vector<float> predictions; // shape [num_quantiles, horizon]
    std::vector<std::string> future_timestamps;
    float mean = 0.0f;
    float std = 1.0f;
    float trend_slope = 0.0f;
    float trend_r2 = 0.0f;
    bool detrend_applied = false;
    double inference_time_ms = 0.0;
};

class Exporter {
public:
    static bool save_csv(
        const std::string& filepath,
        const ForecastResult& result,
        const std::string& value_name = "forecast"
    );

    static bool save_json(
        const std::string& filepath,
        const ForecastResult& result,
        const std::vector<float>& history = {},
        const std::vector<std::string>& history_timestamps = {}
    );

    static std::string to_json_string(
        const ForecastResult& result,
        const std::vector<float>& history = {},
        const std::vector<std::string>& history_timestamps = {}
    );
};

} // namespace timesfm
