#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tab_completion {

DiffusionSampler::DiffusionSampler(
    float gamma_0,
    float gamma_1
) : gamma_0_(gamma_0), gamma_1_(gamma_1) {}

float DiffusionSampler::gamma_from_t(float t) const {
    // Clamp t to [0.0, 1.0]
    float tc = std::max(0.0f, std::min(1.0f, t));
    return gamma_0_ + (gamma_1_ - gamma_0_) * tc;
}

std::vector<std::pair<float, float>> DiffusionSampler::get_time_schedule(int n_steps) const {
    std::vector<std::pair<float, float>> schedule;
    if (n_steps <= 1) {
        schedule.push_back({1.0f, 0.0f});
        return schedule;
    }

    for (int i = 0; i < n_steps; ++i) {
        float t = 1.0f - static_cast<float>(i) / static_cast<float>(n_steps);
        float s = 1.0f - static_cast<float>(i + 1) / static_cast<float>(n_steps);
        schedule.push_back({t, s});
    }
    return schedule;
}

void DiffusionSampler::ddim_step(
    const float* z_t,
    const float* x_reconst,
    float gamma_t,
    float gamma_s,
    float score_temp,
    int length,
    int embed_dim,
    float* z_s_out
) const {
    float alpha_t_sq = sigmoid(-gamma_t);
    float sigma_t_sq = sigmoid(gamma_t);
    float alpha_t = std::sqrt(alpha_t_sq);
    float sigma_t = std::sqrt(sigma_t_sq);

    float alpha_s_sq = sigmoid(-gamma_s);
    float sigma_s_sq = sigmoid(gamma_s);
    float alpha_s = std::sqrt(alpha_s_sq);
    float sigma_s = std::sqrt(sigma_s_sq);

    float temp = (score_temp > 0.0f) ? score_temp : 1.0f;
    int total_elements = length * embed_dim;

    for (int idx = 0; idx < total_elements; ++idx) {
        float z_val = z_t[idx];
        float x_val = x_reconst[idx];

        // 1. Predict noise epsilon
        float eps = (z_val - alpha_t * x_val) / (sigma_t + 1e-8f);

        // 2. Scale by score temperature
        eps /= temp;

        // 3. Extrapolate clean x0
        float x0 = (z_val - sigma_t * eps) / (alpha_t + 1e-8f);

        // 4. Step forward along trajectory to z_s
        z_s_out[idx] = alpha_s * x0 + sigma_s * eps;
    }
}

void DiffusionSampler::compute_x_reconst_from_logits(
    const float* logits,
    const float* embedding_matrix,
    int length,
    int vocab_size,
    int embed_dim,
    float* x_reconst_out,
    int start_pos,
    int end_pos
) const {
    int end = (end_pos >= 0 && end_pos <= length) ? end_pos : length;
    int start = (start_pos >= 0 && start_pos < end) ? start_pos : 0;

    // Zero out outside the target range if non-default
    if (start > 0) {
        std::fill(x_reconst_out, x_reconst_out + (start * embed_dim), 0.0f);
    }
    if (end < length) {
        std::fill(x_reconst_out + (end * embed_dim), x_reconst_out + (length * embed_dim), 0.0f);
    }

    #pragma omp parallel
    {
        std::vector<float> thread_probs(vocab_size);
        #pragma omp for schedule(static)
        for (int i = start; i < end; ++i) {
            const float* row = logits + (static_cast<size_t>(i) * vocab_size);
            float max_val = -1e30f;
            for (int v = 0; v < vocab_size; ++v) {
                if (row[v] > max_val) max_val = row[v];
            }

            float sum = 0.0f;
            for (int v = 0; v < vocab_size; ++v) {
                float exp_val = std::exp(row[v] - max_val);
                thread_probs[v] = exp_val;
                sum += exp_val;
            }

            float inv_sum = 1.0f / (sum + 1e-8f);
            for (int v = 0; v < vocab_size; ++v) {
                thread_probs[v] *= inv_sum;
            }

            // GEMV: x_reconst = probs @ E (contiguous row accumulation)
            float* out_row = x_reconst_out + (i * embed_dim);
            std::fill(out_row, out_row + embed_dim, 0.0f);
            for (int v = 0; v < vocab_size; ++v) {
                float p = thread_probs[v];
                const float* emb_v = embedding_matrix + (static_cast<size_t>(v) * embed_dim);
                for (int d = 0; d < embed_dim; ++d) {
                    out_row[d] += p * emb_v[d];
                }
            }
        }
    }
}

std::vector<int32_t> DiffusionSampler::sample_tokens_greedy(
    const float* logits,
    int length,
    int vocab_size
) const {
    std::vector<int32_t> tokens(length);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; ++i) {
        const float* row = logits + (static_cast<size_t>(i) * vocab_size);
        int32_t best_idx = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > best_val) {
                best_val = row[v];
                best_idx = v;
            }
        }
        tokens[i] = best_idx;
    }
    return tokens;
}

std::vector<int32_t> DiffusionSampler::sample_tokens_top_p(
    const float* logits,
    int length,
    int vocab_size,
    float temperature,
    float top_p,
    uint64_t seed
) const {
    if (temperature <= 0.01f) {
        return sample_tokens_greedy(logits, length, vocab_size);
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::vector<int32_t> tokens(length);

    struct Pair {
        float prob;
        int32_t idx;
    };
    std::vector<Pair> candidates(vocab_size);

    for (int i = 0; i < length; ++i) {
        const float* row = logits + (i * vocab_size);
        float max_val = -1e30f;
        for (int v = 0; v < vocab_size; ++v) {
            if (row[v] > max_val) max_val = row[v];
        }

        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            float p = std::exp((row[v] - max_val) / temperature);
            candidates[v] = {p, v};
            sum += p;
        }

        // Sort descending
        std::sort(candidates.begin(), candidates.end(), [](const Pair& a, const Pair& b) {
            return a.prob > b.prob;
        });

        float cumsum = 0.0f;
        float cutoff = top_p * sum;
        int cutoff_len = 0;
        for (int v = 0; v < vocab_size; ++v) {
            cumsum += candidates[v].prob;
            cutoff_len++;
            if (cumsum >= cutoff) break;
        }

        float draw = uniform_dist(rng) * cumsum;
        float acc = 0.0f;
        int32_t chosen = candidates[0].idx;
        for (int v = 0; v < cutoff_len; ++v) {
            acc += candidates[v].prob;
            if (acc >= draw) {
                chosen = candidates[v].idx;
                break;
            }
        }
        tokens[i] = chosen;
    }

    return tokens;
}

} // namespace tab_completion
