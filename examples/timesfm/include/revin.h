#pragma once

#include <cstdint>
#include <vector>

namespace timesfm {

struct RevINState {
    float mean = 0.0f;
    float std = 1.0f;
    float eps = 1e-5f;
    bool applied = false;
};

// Computes RevIN mean and standard deviation, and normalizes input data in-place or to output buffer.
// If weights is non-null, computes weighted mean and variance.
void revin_normalize(const float* input, int64_t len, const float* weights, float eps, RevINState* state, float* output);

// De-normalizes forecasted quantiles: output[q, t] = norm_forecast[q, t] * state.std + state.mean
void revin_denormalize(const float* norm_forecast, int64_t horizon, int64_t num_quantiles, const RevINState& state, float* out_forecast);

// CPM (Continuous Patch Mixing) iterative RevIN refinement across multi-horizon steps.
// Applies step-decay shrinkage to mitigate drift over long forecast horizons.
void cpm_iterative_revin_refine(
    float* forecast_quantiles,
    int64_t horizon,
    int64_t num_quantiles,
    const RevINState& base_state,
    float decay_factor = 0.95f
);

} // namespace timesfm
