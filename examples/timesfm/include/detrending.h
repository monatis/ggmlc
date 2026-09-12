#pragma once

#include <cstdint>
#include <vector>

namespace timesfm {

struct LinearTrend {
    float slope = 0.0f;
    float intercept = 0.0f;
    float r_squared = 0.0f;
    bool applied = false;
};

// Fits a linear regression line y = slope * t + intercept over t in [0, n-1].
// If r_squared >= r2_threshold (default 0.5), sets applied = true and subtracts trend from input.
LinearTrend fit_and_remove_linear_trend(
    const float* input,
    int64_t len,
    float r2_threshold = 0.5f,
    float* detrended_output = nullptr
);

// Adds the projected future linear trend back to forecasted quantiles:
// trend(t_future) = intercept + slope * (context_len + t)
void add_linear_trend_to_forecast(
    float* forecast_quantiles,
    int64_t horizon,
    int64_t num_quantiles,
    int64_t context_len,
    const LinearTrend& trend
);

} // namespace timesfm
