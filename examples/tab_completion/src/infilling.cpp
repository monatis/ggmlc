#include "infilling.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace tab_completion {

InfillingManager::InfillingManager(
    int canvas_len,
    int embed_dim,
    int eos_token_id,
    int pad_token_id
) : canvas_len_(canvas_len),
    embed_dim_(embed_dim),
    eos_token_id_(eos_token_id),
    pad_token_id_(pad_token_id),
    latent_scale_(std::sqrt(static_cast<float>(embed_dim))) {}

InfillingContext InfillingManager::prepare_context(
    const std::string& prefix,
    const std::string& suffix,
    int max_new_tokens,
    const ggmlc::pipeline::BPETokenizer& tokenizer
) const {
    InfillingContext ctx;
    ctx.prefix_text = prefix;
    ctx.suffix_text = suffix;
    ctx.canvas_len = canvas_len_;
    ctx.embed_dim = embed_dim_;

    ctx.prefix_tokens = tokenizer.encode(prefix, 0, false, false);
    ctx.suffix_tokens = tokenizer.encode(suffix, 0, false, false);

    int hole_budget = max_new_tokens;
    if (hole_budget <= 0 || hole_budget >= canvas_len_) {
        hole_budget = std::min(64, canvas_len_ / 2);
    }

    int remaining_budget = canvas_len_ - hole_budget;
    int p_len = static_cast<int>(ctx.prefix_tokens.size());
    int s_len = static_cast<int>(ctx.suffix_tokens.size());

    // Truncate if total prompt exceeds available canvas budget
    if (p_len + s_len > remaining_budget) {
        // Allocate up to 75% to prefix, 25% to suffix
        int p_budget = (remaining_budget * 3) / 4;
        int s_budget = remaining_budget - p_budget;

        if (p_len < p_budget) {
            s_budget = remaining_budget - p_len;
        } else if (s_len < s_budget) {
            p_budget = remaining_budget - s_len;
        }

        if (p_len > p_budget) {
            // Keep the most recent prefix tokens immediately before cursor
            ctx.prefix_tokens.erase(
                ctx.prefix_tokens.begin(),
                ctx.prefix_tokens.begin() + (p_len - p_budget)
            );
            p_len = p_budget;
        }
        if (s_len > s_budget) {
            // Keep the earliest suffix tokens immediately after cursor
            ctx.suffix_tokens.resize(s_budget);
            s_len = s_budget;
        }
    }

    ctx.prefix_len = p_len;
    ctx.hole_len = hole_budget;
    ctx.suffix_len = s_len;

    ctx.canvas_tokens.assign(canvas_len_, 0);
    ctx.is_clean.assign(canvas_len_, false);

    // 1. Prefix tokens at [0 : p_len]
    for (int i = 0; i < p_len; ++i) {
        ctx.canvas_tokens[i] = ctx.prefix_tokens[i];
        ctx.is_clean[i] = true;
    }

    // 2. Suffix tokens at tail [canvas_len_ - s_len : canvas_len_]
    for (int i = 0; i < s_len; ++i) {
        ctx.canvas_tokens[canvas_len_ - s_len + i] = ctx.suffix_tokens[i];
        ctx.is_clean[canvas_len_ - s_len + i] = true;
    }

    // Positions [p_len : canvas_len_ - s_len] are the generation region (is_clean = false)

    return ctx;
}

void InfillingManager::init_canvas_latents(
    const InfillingContext& ctx,
    const float* embedding_matrix,
    int vocab_size,
    float gamma_val,
    float* z_out,
    uint64_t seed
) const {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    float alpha_sq = 1.0f / (1.0f + std::exp(gamma_val));
    float sigma_sq = 1.0f / (1.0f + std::exp(-gamma_val));
    float z_var = (alpha_sq / embed_dim_) + sigma_sq;
    float z_scale = std::sqrt(z_var);

    for (int i = 0; i < canvas_len_; ++i) {
        if (ctx.is_clean[i]) {
            int32_t tok = ctx.canvas_tokens[i];
            if (tok < 0 || tok >= vocab_size) tok = 0;
            const float* emb_row = embedding_matrix + (tok * embed_dim_);
            for (int d = 0; d < embed_dim_; ++d) {
                z_out[i * embed_dim_ + d] = emb_row[d] * latent_scale_ * z_scale;
            }
        } else {
            // Hole and tail generation positions initialized with standard Gaussian noise
            for (int d = 0; d < embed_dim_; ++d) {
                z_out[i * embed_dim_ + d] = norm_dist(rng) * z_scale;
            }
        }
    }
}

void InfillingManager::pin_clean_latents(
    const InfillingContext& ctx,
    const float* embedding_matrix,
    int vocab_size,
    float gamma_val,
    float* z_latents
) const {
    float alpha_sq = 1.0f / (1.0f + std::exp(gamma_val));
    float sigma_sq = 1.0f / (1.0f + std::exp(-gamma_val));
    float z_var = (alpha_sq / embed_dim_) + sigma_sq;
    float z_scale = std::sqrt(z_var);

    for (int i = 0; i < canvas_len_; ++i) {
        if (ctx.is_clean[i]) {
            int32_t tok = ctx.canvas_tokens[i];
            if (tok < 0 || tok >= vocab_size) tok = 0;
            const float* emb_row = embedding_matrix + (tok * embed_dim_);
            for (int d = 0; d < embed_dim_; ++d) {
                z_latents[i * embed_dim_ + d] = emb_row[d] * latent_scale_ * z_scale;
            }
        }
    }
}

std::string InfillingManager::decode_completion(
    const InfillingContext& ctx,
    const std::vector<int32_t>& generated_tokens,
    const ggmlc::pipeline::BPETokenizer& tokenizer,
    bool stop_at_eos
) const {
    if (generated_tokens.empty()) return "";

    std::vector<int32_t> hole_tokens;
    int start_idx = ctx.prefix_len;
    int end_idx = ctx.prefix_len + ctx.hole_len;

    for (int i = start_idx; i < end_idx && i < static_cast<int>(generated_tokens.size()); ++i) {
        int32_t tok = generated_tokens[i];
        if (stop_at_eos) {
            if (tok == eos_token_id_ || tok == pad_token_id_ || tokenizer.is_special_token(tok)) {
                break;
            }
        }
        hole_tokens.push_back(tok);
    }

    return tokenizer.decode(hole_tokens, true);
}

} // namespace tab_completion
