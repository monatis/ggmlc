#include "revin.h"

#include <cmath>
#include <algorithm>

namespace timesfm {

void revin_normalize(const float* input, int64_t len, const float* weights, float eps, RevINState* state, float* output) {
    if (!input || len <= 0 || !state) return;

    double sum_w = 0.0;
    double sum_xw = 0.0;

    for (int64_t i = 0; i < len; ++i) {
        float w = weights ? weights[i] : 1.0f;
        if (w <= 0.0f) continue;
        sum_w += w;
        sum_xw += input[i] * w;
    }

    if (sum_w <= 0.0) {
        state->mean = 0.0f;
        state->std = 1.0f;
        state->eps = eps;
        state->applied = false;
        if (output && output != input) {
            std::copy(input, input + len, output);
        }
        return;
    }

    double mean = sum_xw / sum_w;
    double sum_var = 0.0;

    for (int64_t i = 0; i < len; ++i) {
        float w = weights ? weights[i] : 1.0f;
        if (w <= 0.0f) continue;
        double diff = input[i] - mean;
        sum_var += w * diff * diff;
    }

    double var = sum_var / sum_w;
    double std_val = std::sqrt(var + static_cast<double>(eps));

    state->mean = static_cast<float>(mean);
    state->std = static_cast<float>(std_val);
    state->eps = eps;
    state->applied = true;

    if (output) {
        for (int64_t i = 0; i < len; ++i) {
            output[i] = static_cast<float>((input[i] - mean) / std_val);
        }
    }
}

void revin_denormalize(const float* norm_forecast, int64_t horizon, int64_t num_quantiles, const RevINState& state, float* out_forecast) {
    if (!norm_forecast || !out_forecast || horizon <= 0 || num_quantiles <= 0) return;

    if (!state.applied) {
        std::copy(norm_forecast, norm_forecast + (horizon * num_quantiles), out_forecast);
        return;
    }

    for (int64_t q = 0; q < num_quantiles; ++q) {
        for (int64_t t = 0; t < horizon; ++t) {
            int64_t idx = q * horizon + t;
            out_forecast[idx] = norm_forecast[idx] * state.std + state.mean;
        }
    }
}

void cpm_iterative_revin_refine(
    float* forecast_quantiles,
    int64_t horizon,
    int64_t num_quantiles,
    const RevINState& base_state,
    float decay_factor
) {
    if (!forecast_quantiles || horizon <= 0 || num_quantiles <= 0 || !base_state.applied) return;

    // Apply smooth step-decay shrinkage to std scaling across long horizons
    for (int64_t t = 0; t < horizon; ++t) {
        float step_ratio = static_cast<float>(t) / static_cast<float>(std::max<int64_t>(1, horizon));
        float factor = 1.0f + (decay_factor - 1.0f) * step_ratio;

        for (int64_t q = 0; q < num_quantiles; ++q) {
            int64_t idx = q * horizon + t;
            float diff = forecast_quantiles[idx] - base_state.mean;
            forecast_quantiles[idx] = base_state.mean + diff * factor;
        }
    }
}

} // namespace timesfm
