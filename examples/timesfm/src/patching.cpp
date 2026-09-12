#include "patching.h"

#include <cmath>
#include <algorithm>

namespace timesfm {

void interpolate_nans(std::vector<float>& series) {
    int64_t n = static_cast<int64_t>(series.size());
    if (n == 0) return;

    // Find first valid value
    int64_t first_valid = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isnan(series[i]) && !std::isinf(series[i])) {
            first_valid = i;
            break;
        }
    }

    if (first_valid == -1) {
        // All values are NaN -> fill with 0
        std::fill(series.begin(), series.end(), 0.0f);
        return;
    }

    // Fill leading NaNs with first valid
    for (int64_t i = 0; i < first_valid; ++i) {
        series[i] = series[first_valid];
    }

    // Interpolate internal NaNs
    int64_t last_valid = first_valid;
    for (int64_t i = first_valid + 1; i < n; ++i) {
        if (!std::isnan(series[i]) && !std::isinf(series[i])) {
            if (i > last_valid + 1) {
                float start_val = series[last_valid];
                float end_val = series[i];
                float steps = static_cast<float>(i - last_valid);
                for (int64_t j = last_valid + 1; j < i; ++j) {
                    float alpha = static_cast<float>(j - last_valid) / steps;
                    series[j] = start_val + alpha * (end_val - start_val);
                }
            }
            last_valid = i;
        }
    }

    // Fill trailing NaNs with last valid
    for (int64_t i = last_valid + 1; i < n; ++i) {
        series[i] = series[last_valid];
    }
}

int64_t create_model_patches(
    const std::vector<float>& history,
    std::vector<float>& out_tensor,
    int64_t input_patch_len,
    int64_t feature_dim
) {
    int64_t hist_len = static_cast<int64_t>(history.size());
    if (hist_len == 0) return 0;

    int64_t rem = hist_len % input_patch_len;
    int64_t pad_left = (rem == 0) ? 0 : (input_patch_len - rem);
    int64_t padded_len = hist_len + pad_left;
    int64_t num_patches = padded_len / input_patch_len;

    out_tensor.assign(num_patches * feature_dim, 0.0f);

    for (int64_t p = 0; p < num_patches; ++p) {
        float* patch_ptr = &out_tensor[p * feature_dim];

        for (int64_t i = 0; i < input_patch_len; ++i) {
            int64_t global_idx = p * input_patch_len + i;
            if (global_idx < pad_left) {
                // Left-padded slot
                patch_ptr[i] = 0.0f; // values
                patch_ptr[96 + i] = 1.0f; // mask = 1 (masked/padded)
            } else {
                int64_t src_idx = global_idx - pad_left;
                patch_ptr[i] = history[src_idx]; // values
                patch_ptr[96 + i] = 0.0f; // mask = 0 (valid historical data)
            }
        }
        // Indices 32..95 are values_fcov (0.0f)
        // Indices 128..191 are masks_fcov (0.0f)
    }

    return num_patches;
}

} // namespace timesfm
