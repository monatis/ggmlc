#pragma once

#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <cmath>
#include "ggmlc/pipeline/tokenizer.h"

namespace tab_completion {

struct InfillingContext {
    std::string prefix_text;
    std::string suffix_text;
    std::vector<int32_t> prefix_tokens;
    std::vector<int32_t> suffix_tokens;
    std::vector<int32_t> canvas_tokens; // [L]
    std::vector<bool> is_clean;          // [L]: true for prefix/suffix, false for hole
    int prefix_len = 0;
    int hole_len = 0;
    int suffix_len = 0;
    int canvas_len = 0;
    int embed_dim = 16;
};

class InfillingManager {
public:
    InfillingManager(
        int canvas_len = 256,
        int embed_dim = 16,
        int eos_token_id = 151643,
        int pad_token_id = 151643
    );

    // Prepare infilling context from prefix and suffix strings
    InfillingContext prepare_context(
        const std::string& prefix,
        const std::string& suffix,
        int max_new_tokens,
        const ggmlc::pipeline::BPETokenizer& tokenizer
    ) const;

    // Initialize canvas latent z [L x D] with clean prefix/suffix and Gaussian noise in the hole
    void init_canvas_latents(
        const InfillingContext& ctx,
        const float* embedding_matrix, // [V x D]
        int vocab_size,
        float gamma_val,
        float* z_out,                  // [L x D]
        uint64_t seed = 42
    ) const;

    // Pin clean prefix and suffix embeddings into z_latents [L x D]
    void pin_clean_latents(
        const InfillingContext& ctx,
        const float* embedding_matrix,
        int vocab_size,
        float gamma_val,
        float* z_latents
    ) const;

    // Extract generated tokens from the hole and decode to string
    std::string decode_completion(
        const InfillingContext& ctx,
        const std::vector<int32_t>& generated_tokens,
        const ggmlc::pipeline::BPETokenizer& tokenizer,
        bool stop_at_eos = true
    ) const;

private:
    int canvas_len_;
    int embed_dim_;
    int eos_token_id_;
    int pad_token_id_;
    float latent_scale_; // sqrt(embed_dim)
};

} // namespace tab_completion
