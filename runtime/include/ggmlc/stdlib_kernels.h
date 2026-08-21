#pragma once

#include <cstdint>
#include <cstddef>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Custom op parameters struct passed to ggml_map_custom
struct ggmlc_norm_params {
    float eps;
};

// GGML Map Custom function implementations
// Bias + GELU: dst = gelu(a + b) where b is bias [D]
void ggmlc_compute_forward_bias_gelu(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* b,
    int ith,
    int nth,
    void* userdata
);

// LayerNorm Fused: dst = (a - mean) / sqrt(var + eps) * w + b
void ggmlc_compute_forward_layer_norm(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* w,
    const struct ggml_tensor* b,
    int ith,
    int nth,
    void* userdata
);

// RMSNorm Fused: dst = a / sqrt(mean(a^2) + eps) * w
void ggmlc_compute_forward_rms_norm(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* w,
    int ith,
    int nth,
    void* userdata
);

// SwiGLU Fused: dst = silu(gate) * up
void ggmlc_compute_forward_swiglu(
    struct ggml_tensor* dst,
    const struct ggml_tensor* gate,
    const struct ggml_tensor* up,
    int ith,
    int nth,
    void* userdata
);

#ifdef __cplusplus
}
#endif
