#pragma once

#include <cstdint>
#include <vector>

namespace timesfm {

constexpr int INPUT_PATCH_LEN = 32;
constexpr int OUTPUT_PATCH_LEN = 64;
constexpr int FEATURE_DIM = 192;
constexpr int DEFAULT_NUM_QUANTILES = 9;

// Interpolates missing or NaN values linearly in a time series.
void interpolate_nans(std::vector<float>& series);

// Packages historical time series into [1, 1, num_patches, 192] feature tensor format expected by TimesFM 3.0.
// Returns the number of patches N.
int64_t create_model_patches(
    const std::vector<float>& history,
    std::vector<float>& out_tensor,
    int64_t input_patch_len = INPUT_PATCH_LEN,
    int64_t feature_dim = FEATURE_DIM
);

} // namespace timesfm
