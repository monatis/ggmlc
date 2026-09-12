#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

namespace tab_completion {

struct DDIMStepParams {
    float gamma_t;
    float gamma_s;
    float score_temp = 0.99f;
};

class DiffusionSampler {
public:
    DiffusionSampler(
        float gamma_0 = -3.0f,
        float gamma_1 = 6.0f
    );

    // Compute log-SNR gamma from normalized time t in [0.0, 1.0]
    float gamma_from_t(float t) const;

    // Generate time pairs (t, s) for n_steps
    std::vector<std::pair<float, float>> get_time_schedule(int n_steps) const;

    // Execute one DDIM reverse step: z_t -> z_s using predicted x_reconst
    void ddim_step(
        const float* z_t,          // [L x D]
        const float* x_reconst,    // [L x D]
        float gamma_t,
        float gamma_s,
        float score_temp,
        int length,
        int embed_dim,
        float* z_s_out             // [L x D]
    ) const;

    // Compute x_reconst = softmax(logits) @ embedding_matrix (optionally only for [start_pos, end_pos))
    void compute_x_reconst_from_logits(
        const float* logits,           // [L x V]
        const float* embedding_matrix, // [V x D]
        int length,
        int vocab_size,
        int embed_dim,
        float* x_reconst_out,          // [L x D]
        int start_pos = 0,
        int end_pos = -1
    ) const;

    // Extract predicted token IDs from logits (greedy argmax)
    std::vector<int32_t> sample_tokens_greedy(
        const float* logits, // [L x V]
        int length,
        int vocab_size
    ) const;

    // Extract predicted token IDs with temperature and top-p sampling
    std::vector<int32_t> sample_tokens_top_p(
        const float* logits, // [L x V]
        int length,
        int vocab_size,
        float temperature = 0.7f,
        float top_p = 0.9f,
        uint64_t seed = 42
    ) const;

private:
    float gamma_0_;
    float gamma_1_;

    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

} // namespace tab_completion
