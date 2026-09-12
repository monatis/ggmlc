#include "detrending.h"

#include <cmath>
#include <algorithm>

namespace timesfm {

LinearTrend fit_and_remove_linear_trend(
    const float* input,
    int64_t len,
    float r2_threshold,
    float* detrended_output
) {
    LinearTrend result;
    if (!input || len < 2) {
        if (detrended_output && input) {
            std::copy(input, input + len, detrended_output);
        }
        return result;
    }

    double n = static_cast<double>(len);
    double sum_t = 0.0;
    double sum_y = 0.0;
    double sum_tt = 0.0;
    double sum_ty = 0.0;
    double sum_yy = 0.0;

    for (int64_t i = 0; i < len; ++i) {
        double t = static_cast<double>(i);
        double y = static_cast<double>(input[i]);
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
        sum_yy += y * y;
    }

    double denom_t = n * sum_tt - sum_t * sum_t;
    if (std::abs(denom_t) < 1e-12) {
        if (detrended_output) {
            std::copy(input, input + len, detrended_output);
        }
        return result;
    }

    double slope = (n * sum_ty - sum_t * sum_y) / denom_t;
    double intercept = (sum_y - slope * sum_t) / n;

    // Compute R^2
    double mean_y = sum_y / n;
    double ss_tot = 0.0;
    double ss_res = 0.0;

    for (int64_t i = 0; i < len; ++i) {
        double y = static_cast<double>(input[i]);
        double y_pred = intercept + slope * static_cast<double>(i);
        ss_tot += (y - mean_y) * (y - mean_y);
        ss_res += (y - y_pred) * (y - y_pred);
    }

    double r2 = 0.0;
    if (ss_tot > 1e-12) {
        r2 = 1.0 - (ss_res / ss_tot);
        if (r2 < 0.0) r2 = 0.0;
        if (r2 > 1.0) r2 = 1.0;
    }

    result.slope = static_cast<float>(slope);
    result.intercept = static_cast<float>(intercept);
    result.r_squared = static_cast<float>(r2);

    if (result.r_squared >= r2_threshold) {
        result.applied = true;
        if (detrended_output) {
            for (int64_t i = 0; i < len; ++i) {
                float trend_val = result.intercept + result.slope * static_cast<float>(i);
                detrended_output[i] = input[i] - trend_val;
            }
        }
    } else {
        result.applied = false;
        if (detrended_output) {
            std::copy(input, input + len, detrended_output);
        }
    }

    return result;
}

void add_linear_trend_to_forecast(
    float* forecast_quantiles,
    int64_t horizon,
    int64_t num_quantiles,
    int64_t context_len,
    const LinearTrend& trend
) {
    if (!forecast_quantiles || horizon <= 0 || num_quantiles <= 0 || !trend.applied) return;

    for (int64_t t = 0; t < horizon; ++t) {
        float future_t = static_cast<float>(context_len + t);
        float trend_val = trend.intercept + trend.slope * future_t;

        for (int64_t q = 0; q < num_quantiles; ++q) {
            int64_t idx = q * horizon + t;
            forecast_quantiles[idx] += trend_val;
        }
    }
}

} // namespace timesfm
