#include "ggmlc/stdlib_kernels.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

// Fast, vectorizable GELU approximation matching ATen/GGML precision
static inline float fast_gelu_f32(float x) {
    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    constexpr float GELU_COEF_A    = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEF_A * x3);
    // Rational Pade-approximation for fast vectorizable tanh
    float inner_clamped = std::max(-9.0f, std::min(9.0f, inner));
    float exp2 = std::exp(2.0f * inner_clamped);
    float tanh_val = (exp2 - 1.0f) / (exp2 + 1.0f);
    return 0.5f * x * (1.0f + tanh_val);
}

static inline float fast_silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

} // anonymous namespace

extern "C" {

void ggmlc_compute_forward_bias_gelu(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* b,
    int ith,
    int nth,
    void* userdata
) {
    (void)userdata;
    const int64_t ne00 = a->ne[0];
    const int64_t ne01 = a->ne[1];
    const int64_t ne02 = a->ne[2];
    const int64_t ne03 = a->ne[3];

    const size_t nb00 = a->nb[0];
    const size_t nb01 = a->nb[1];
    const size_t nb02 = a->nb[2];
    const size_t nb03 = a->nb[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const float* bias = (const float*)b->data;
    const int64_t b_ne00 = b->ne[0];

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const char* src_row = (const char*)a->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
                char* dst_row = (char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3;

                if (nb00 == sizeof(float) && nb0 == sizeof(float) && b_ne00 == ne00) {
                    const float* x_ptr = (const float*)src_row;
                    float* y_ptr = (float*)dst_row;

                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        float val = x_ptr[i00] + bias[i00];
                        y_ptr[i00] = fast_gelu_f32(val);
                    }
                } else {
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        float x_val = *(const float*)(src_row + i00 * nb00);
                        float b_val = (i00 < b_ne00) ? bias[i00] : 0.0f;
                        *(float*)(dst_row + i00 * nb0) = fast_gelu_f32(x_val + b_val);
                    }
                }
            }
        }
    }
}

void ggmlc_compute_forward_layer_norm(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* w,
    const struct ggml_tensor* b,
    int ith,
    int nth,
    void* userdata
) {
    const int64_t ne00 = a->ne[0];
    const int64_t ne01 = a->ne[1];
    const int64_t ne02 = a->ne[2];
    const int64_t ne03 = a->ne[3];

    const size_t nb01 = a->nb[1];
    const size_t nb02 = a->nb[2];
    const size_t nb03 = a->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    float eps = 1e-5f;
    if (userdata) {
        eps = ((const struct ggmlc_norm_params*)userdata)->eps;
    }

    const float* weight = w ? (const float*)w->data : nullptr;
    const float* bias = b ? (const float*)b->data : nullptr;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float* x_row = (const float*)((const char*)a->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                float* y_row = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                // 1. Vectorized Mean
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    sum += x_row[i00];
                }
                const float mean = sum / static_cast<float>(ne00);

                // 2. Vectorized Variance
                float var_sum = 0.0f;
                #pragma omp simd reduction(+:var_sum)
                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    float diff = x_row[i00] - mean;
                    var_sum += diff * diff;
                }
                const float variance = var_sum / static_cast<float>(ne00);
                const float inv_std = 1.0f / std::sqrt(variance + eps);

                // 3. Vectorized Fused Normalization + Weight + Bias
                if (weight && bias) {
                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        y_row[i00] = (x_row[i00] - mean) * inv_std * weight[i00] + bias[i00];
                    }
                } else if (weight) {
                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        y_row[i00] = (x_row[i00] - mean) * inv_std * weight[i00];
                    }
                } else {
                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        y_row[i00] = (x_row[i00] - mean) * inv_std;
                    }
                }
            }
        }
    }
}

void ggmlc_compute_forward_rms_norm(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* w,
    int ith,
    int nth,
    void* userdata
) {
    const int64_t ne00 = a->ne[0];
    const int64_t ne01 = a->ne[1];
    const int64_t ne02 = a->ne[2];
    const int64_t ne03 = a->ne[3];

    const size_t nb01 = a->nb[1];
    const size_t nb02 = a->nb[2];
    const size_t nb03 = a->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    float eps = 1e-5f;
    if (userdata) {
        eps = ((const struct ggmlc_norm_params*)userdata)->eps;
    }

    const float* weight = w ? (const float*)w->data : nullptr;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float* x_row = (const float*)((const char*)a->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                float* y_row = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                // 1. Vectorized Mean of Squares
                float sqr_sum = 0.0f;
                #pragma omp simd reduction(+:sqr_sum)
                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    sqr_sum += x_row[i00] * x_row[i00];
                }
                const float mean_sqr = sqr_sum / static_cast<float>(ne00);
                const float inv_rms = 1.0f / std::sqrt(mean_sqr + eps);

                // 2. Fused Scale + Weight
                if (weight) {
                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        y_row[i00] = x_row[i00] * inv_rms * weight[i00];
                    }
                } else {
                    #pragma omp simd
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        y_row[i00] = x_row[i00] * inv_rms;
                    }
                }
            }
        }
    }
}

void ggmlc_compute_forward_swiglu(
    struct ggml_tensor* dst,
    const struct ggml_tensor* gate,
    const struct ggml_tensor* up,
    int ith,
    int nth,
    void* userdata
) {
    (void)userdata;
    const int64_t ne00 = gate->ne[0];
    const int64_t ne01 = gate->ne[1];
    const int64_t ne02 = gate->ne[2];
    const int64_t ne03 = gate->ne[3];

    const size_t nb01 = gate->nb[1];
    const size_t nb02 = gate->nb[2];
    const size_t nb03 = gate->nb[3];

    const size_t up_nb01 = up->nb[1];
    const size_t up_nb02 = up->nb[2];
    const size_t up_nb03 = up->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float* gate_row = (const float*)((const char*)gate->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
                const float* up_row = (const float*)((const char*)up->data + i01 * up_nb01 + i02 * up_nb02 + i03 * up_nb03);
                float* y_row = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                #pragma omp simd
                for (int64_t i00 = 0; i00 < ne00; ++i00) {
                    y_row[i00] = fast_silu_f32(gate_row[i00]) * up_row[i00];
                }
            }
        }
    }
}

} // extern "C"
