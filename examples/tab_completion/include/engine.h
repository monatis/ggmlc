#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>

#include "infilling.h"
#include "sampler.h"
#include "ggmlc/executor.h"
#include "ggmlc/loader.h"
#include "ggmlc/pipeline/tokenizer.h"

namespace tab_completion {

struct CompletionOptions {
    int max_new_tokens = 32;
    int sampling_steps = 1;      // 1-step distilled default for PlaidQ
    float score_temp = 0.99f;    // DDIM score temperature
    float sample_temp = 0.0f;    // 0.0 = greedy argmax, >0.0 = stochastic
    float top_p = 0.9f;
    uint64_t seed = 42;
    int n_threads = 4;
    bool stop_at_eos = true;
};

struct CompletionResult {
    std::string completion_text;
    std::vector<int32_t> generated_tokens;
    double total_time_ms = 0.0;
    double ddim_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double tokens_per_second = 0.0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

class TabCompletionEngine {
public:
    TabCompletionEngine();
    ~TabCompletionEngine();

    // Load compiled GGUF PlaidQ model from file
    bool load_model(
        const std::string& gguf_path,
        const std::string& device = "cpu",
        int canvas_len = 256
    );

    // Run tab completion given code before (prefix) and after (suffix) cursor
    CompletionResult complete(
        const std::string& prefix,
        const std::string& suffix,
        const CompletionOptions& options = CompletionOptions()
    );

    // Benchmarking helper: measure latency and throughput over multiple iterations
    void benchmark(
        int runs = 5,
        int warmup = 2,
        const CompletionOptions& options = CompletionOptions()
    );

    // Accessors
    bool is_loaded() const { return model_loaded_; }
    const std::string& device() const { return device_; }
    int canvas_len() const { return canvas_len_; }
    int vocab_size() const { return vocab_size_; }
    int embed_dim() const { return embed_dim_; }
    const ggmlc::pipeline::BPETokenizer& tokenizer() const { return tokenizer_; }

private:
    bool model_loaded_ = false;
    std::string device_ = "cpu";
    std::string model_path_;
    int canvas_len_ = 256;
    int vocab_size_ = 151936;
    int embed_dim_ = 16;
    int eos_token_id_ = 151643;
    int pad_token_id_ = 151643;
    float gamma_0_ = -3.0f;
    float gamma_1_ = 6.0f;

    std::vector<float> embedding_matrix_; // [V x D]
    std::unique_ptr<ggmlc::ModelExecutor> executor_;
    ggmlc::pipeline::BPETokenizer tokenizer_;
    std::unique_ptr<InfillingManager> infilling_mgr_;
    std::unique_ptr<DiffusionSampler> sampler_;

    // Internal model forward pass: binds z, gamma, x_selfcond and returns logits pointer
    const float* forward_model(
        const float* z_latents,    // [L x D]
        float gamma_val,
        const float* x_selfcond,  // [L x D]
        int n_threads
    );
};

} // namespace tab_completion
