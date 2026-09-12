#include "engine.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>

namespace tab_completion {

TabCompletionEngine::TabCompletionEngine() = default;
TabCompletionEngine::~TabCompletionEngine() = default;

bool TabCompletionEngine::load_model(
    const std::string& gguf_path,
    const std::string& device,
    int canvas_len
) {
    model_path_ = gguf_path;
    device_ = device;
    canvas_len_ = canvas_len;

    std::cerr << "[TabCompletion] Loading PlaidQ model from: " << gguf_path << " (device: " << device << ")..." << std::endl;

    // 1. Read metadata and codebook from GGUF file
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx) {
        std::cerr << "[TabCompletion] ERROR: Failed to open GGUF file: " << gguf_path << std::endl;
        return false;
    }

    auto safe_get_int = [](const struct gguf_context* c, int64_t kid, int def) -> int {
        if (kid < 0) return def;
        enum gguf_type type = gguf_get_kv_type(c, kid);
        if (type == GGUF_TYPE_INT32) return static_cast<int>(gguf_get_val_i32(c, kid));
        if (type == GGUF_TYPE_UINT32) return static_cast<int>(gguf_get_val_u32(c, kid));
        if (type == GGUF_TYPE_INT64) return static_cast<int>(gguf_get_val_i64(c, kid));
        if (type == GGUF_TYPE_UINT64) return static_cast<int>(gguf_get_val_u64(c, kid));
        return def;
    };

    // Read architectural metadata
    int64_t k_vocab = gguf_find_key(gguf_ctx, "plaidq.vocab_size");
    vocab_size_ = safe_get_int(gguf_ctx, k_vocab, vocab_size_);

    int64_t k_embed = gguf_find_key(gguf_ctx, "plaidq.embed_dim");
    embed_dim_ = safe_get_int(gguf_ctx, k_embed, embed_dim_);

    int64_t k_g0 = gguf_find_key(gguf_ctx, "plaidq.gamma_0");
    if (k_g0 >= 0) gamma_0_ = gguf_get_val_f32(gguf_ctx, k_g0);

    int64_t k_g1 = gguf_find_key(gguf_ctx, "plaidq.gamma_1");
    if (k_g1 >= 0) gamma_1_ = gguf_get_val_f32(gguf_ctx, k_g1);

    int64_t k_eos = gguf_find_key(gguf_ctx, "tokenizer.ggml.eos_token_id");
    eos_token_id_ = safe_get_int(gguf_ctx, k_eos, eos_token_id_);

    int64_t k_pad = gguf_find_key(gguf_ctx, "tokenizer.ggml.padding_token_id");
    pad_token_id_ = safe_get_int(gguf_ctx, k_pad, pad_token_id_);

    std::cerr << "[TabCompletion] Model metadata: vocab=" << vocab_size_
              << ", embed_dim=" << embed_dim_
              << ", gamma_0=" << gamma_0_
              << ", gamma_1=" << gamma_1_
              << ", canvas_len=" << canvas_len_ << std::endl;

    // 2. Initialize BPE Tokenizer from GGUF context
    if (!tokenizer_.init_from_gguf_ctx(gguf_ctx)) {
        std::cerr << "[TabCompletion] Note: GGUF has no embedded tokenizer; falling back to default tokenizer vocab." << std::endl;
    }

    // 3. Read embedding_matrix codebook
    int64_t tid = gguf_find_tensor(gguf_ctx, "embedding_matrix");
    if (tid < 0) tid = gguf_find_tensor(gguf_ctx, "emb");

    size_t total_codebook_elements = static_cast<size_t>(vocab_size_) * static_cast<size_t>(embed_dim_);
    embedding_matrix_.resize(total_codebook_elements, 0.0f);

    if (tid >= 0) {
        size_t data_offset = gguf_get_data_offset(gguf_ctx);
        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, tid);
        size_t abs_offset = data_offset + tensor_offset;
        enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, tid);

        std::ifstream f(gguf_path, std::ios::binary);
        if (f.is_open()) {
            f.seekg(abs_offset);
            if (tensor_type == GGML_TYPE_F32) {
                f.read(reinterpret_cast<char*>(embedding_matrix_.data()), total_codebook_elements * sizeof(float));
            } else if (tensor_type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(total_codebook_elements);
                f.read(reinterpret_cast<char*>(f16_buf.data()), total_codebook_elements * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < total_codebook_elements; ++i) {
                    embedding_matrix_[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }
            std::cerr << "[TabCompletion] Loaded embedding codebook (" << total_codebook_elements << " elements) from GGUF." << std::endl;
        }
    } else {
        std::cerr << "[TabCompletion] Warning: embedding_matrix not found in GGUF; initialized to zeros." << std::endl;
    }

    gguf_free(gguf_ctx);

    // 4. Load graph and initialize ModelExecutor
    try {
        ggmlc::SerializedModelGraph graph = ggmlc::ModelLoader::load_from_file(gguf_path);
        executor_ = std::make_unique<ggmlc::ModelExecutor>(graph, device);
        if (device == "cuda" || device.rfind("cuda", 0) == 0) {
            executor_->set_enable_cuda_graph(true);
        }
        executor_->prepare({{"s", canvas_len_}, {"seq_len", canvas_len_}});
        if (!graph.outputs.empty()) {
            auto out_shape = executor_->get_tensor_shape(graph.outputs[0]);
            if (out_shape[0] > 0) {
                vocab_size_ = static_cast<int>(out_shape[0]);
                std::cerr << "[TabCompletion] Inferred vocab size from output tensor: " << vocab_size_ << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[TabCompletion] Failed to initialize ModelExecutor: " << e.what() << std::endl;
        return false;
    }

    if (embedding_matrix_.size() < static_cast<size_t>(vocab_size_ * embed_dim_)) {
        embedding_matrix_.resize(vocab_size_ * embed_dim_, 0.0f);
    }

    if (eos_token_id_ >= vocab_size_) eos_token_id_ = vocab_size_ - 1;
    if (pad_token_id_ >= vocab_size_) pad_token_id_ = vocab_size_ - 1;

    // 5. Initialize Managers
    infilling_mgr_ = std::make_unique<InfillingManager>(canvas_len_, embed_dim_, eos_token_id_, pad_token_id_);
    sampler_ = std::make_unique<DiffusionSampler>(gamma_0_, gamma_1_);

    model_loaded_ = true;
    std::cerr << "[TabCompletion] PlaidQ Tab Completion Engine ready!" << std::endl;
    return true;
}

const float* TabCompletionEngine::forward_model(
    const float* z_latents,
    float gamma_val,
    const float* x_selfcond,
    int n_threads
) {
    if (!executor_) return nullptr;

    size_t canvas_bytes = static_cast<size_t>(canvas_len_) * static_cast<size_t>(embed_dim_) * sizeof(float);

    executor_->set_input_by_name("z", z_latents, canvas_bytes);
    executor_->set_input_by_name("gamma", &gamma_val, sizeof(float));
    executor_->set_input_by_name("x_selfcond", x_selfcond, canvas_bytes);

    executor_->run(n_threads);

    const auto& graph = executor_->model_graph();
    if (graph.outputs.empty()) return nullptr;

    uint32_t out_id = graph.outputs[0];
    return static_cast<const float*>(executor_->get_output_data(out_id));
}

CompletionResult TabCompletionEngine::complete(
    const std::string& prefix,
    const std::string& suffix,
    const CompletionOptions& options
) {
    CompletionResult result;
    if (!model_loaded_) {
        std::cerr << "[TabCompletion] Error: Model is not loaded." << std::endl;
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Prepare infilling context
    InfillingContext ctx = infilling_mgr_->prepare_context(
        prefix, suffix, options.max_new_tokens, tokenizer_
    );
    result.prompt_tokens = ctx.prefix_len + ctx.suffix_len;

    // 2. Allocate canvas latent buffers
    size_t total_elements = static_cast<size_t>(canvas_len_) * static_cast<size_t>(embed_dim_);
    std::vector<float> z_t(total_elements, 0.0f);
    std::vector<float> z_next(total_elements, 0.0f);
    std::vector<float> x_selfcond(total_elements, 0.0f);
    std::vector<float> x_reconst(total_elements, 0.0f);

    // 3. Initialize canvas latents: clean prefix/suffix, Gaussian noise in hole
    float start_gamma = (options.sampling_steps <= 1) ? gamma_1_ : sampler_->gamma_from_t(1.0f);
    infilling_mgr_->init_canvas_latents(
        ctx, embedding_matrix_.data(), vocab_size_, start_gamma, z_t.data(), options.seed
    );

    const float* final_logits = nullptr;

    if (options.sampling_steps <= 1) {
        // Fast path for 1-step distilled continuous diffusion (e.g. PlaidQ 1-step):
        // Directly evaluate at gamma_1 with zero self-conditioning.
        auto ddim_start = std::chrono::high_resolution_clock::now();
        final_logits = forward_model(z_t.data(), gamma_1_, x_selfcond.data(), options.n_threads);
        auto ddim_end = std::chrono::high_resolution_clock::now();
        result.ddim_time_ms = std::chrono::duration<double, std::milli>(ddim_end - ddim_start).count();
        result.decode_time_ms = 0.0;
    } else {
        // Multi-step DDIM diffusion sampling trajectory
        auto ddim_start = std::chrono::high_resolution_clock::now();
        auto schedule = sampler_->get_time_schedule(options.sampling_steps);

        for (const auto& step : schedule) {
            float t = step.first;
            float s = step.second;
            float gamma_t = sampler_->gamma_from_t(t);
            float gamma_s = sampler_->gamma_from_t(s);

            // Run forward pass at (z_t, gamma_t)
            const float* logits = forward_model(z_t.data(), gamma_t, x_selfcond.data(), options.n_threads);
            if (!logits) break;

            // Compute x_reconst = softmax(logits) @ E only for active hole positions
            sampler_->compute_x_reconst_from_logits(
                logits, embedding_matrix_.data(), canvas_len_, vocab_size_, embed_dim_, x_reconst.data(),
                ctx.prefix_len, ctx.prefix_len + ctx.hole_len
            );

            // Pin clean prefix and suffix before DDIM proposal
            infilling_mgr_->pin_clean_latents(ctx, embedding_matrix_.data(), vocab_size_, gamma_t, z_t.data());

            // Reverse DDIM step: z_t -> z_s
            sampler_->ddim_step(
                z_t.data(), x_reconst.data(), gamma_t, gamma_s,
                options.score_temp, canvas_len_, embed_dim_, z_next.data()
            );

            // Pin clean prefix and suffix on the proposal
            infilling_mgr_->pin_clean_latents(ctx, embedding_matrix_.data(), vocab_size_, gamma_s, z_next.data());

            z_t = z_next;
            x_selfcond = x_reconst;
            for (int i = 0; i < canvas_len_; ++i) {
                if (ctx.is_clean[i]) {
                    for (int d = 0; d < embed_dim_; ++d) {
                        x_selfcond[i * embed_dim_ + d] = 0.0f;
                    }
                }
            }
        }
        auto ddim_end = std::chrono::high_resolution_clock::now();
        result.ddim_time_ms = std::chrono::duration<double, std::milli>(ddim_end - ddim_start).count();

        // Final decode forward at gamma_0 (t=0)
        final_logits = forward_model(z_t.data(), gamma_0_, x_selfcond.data(), options.n_threads);
    }

    auto decode_start = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> pred_tokens;
    if (final_logits) {
        if (options.sample_temp <= 0.01f) {
            pred_tokens = sampler_->sample_tokens_greedy(final_logits, canvas_len_, vocab_size_);
        } else {
            pred_tokens = sampler_->sample_tokens_top_p(
                final_logits, canvas_len_, vocab_size_,
                options.sample_temp, options.top_p, options.seed
            );
        }
        std::cerr << "[TabCompletion] Prefix len: " << ctx.prefix_len << ", hole len: " << ctx.hole_len << std::endl;
        std::cerr << "[TabCompletion] Pred tokens (first 32): [";
        for (int i = 0; i < std::min(canvas_len_, 32); ++i) {
            std::cerr << pred_tokens[i] << " ";
        }
        std::cerr << "]" << std::endl;
        int hole_start = ctx.prefix_len;
        const float* hole_logits = final_logits + (hole_start * vocab_size_);
        std::cerr << "[TabCompletion] Logits at hole start (" << hole_start << "): [";
        for (int v = 0; v < 5; ++v) {
            std::cerr << hole_logits[v] << " ";
        }
        std::cerr << "... max=" << hole_logits[pred_tokens[hole_start]] << " at token=" << pred_tokens[hole_start] << "]" << std::endl;
    }
    auto decode_end = std::chrono::high_resolution_clock::now();
    result.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

    // 6. Decode completion text from hole positions
    result.completion_text = infilling_mgr_->decode_completion(
        ctx, pred_tokens, tokenizer_, options.stop_at_eos
    );
    result.generated_tokens = pred_tokens;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    result.completion_tokens = ctx.hole_len;
    if (result.total_time_ms > 0.0) {
        result.tokens_per_second = (static_cast<double>(result.completion_tokens) / result.total_time_ms) * 1000.0;
    }

    return result;
}

void TabCompletionEngine::benchmark(
    int runs,
    int warmup,
    const CompletionOptions& options
) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "  PlaidQ Offline Tab Completion Benchmark" << std::endl;
    std::cout << "  Device: " << device_ << " | Threads: " << options.n_threads << std::endl;
    std::cout << "  Canvas Length: " << canvas_len_ << " | Infill Tokens: " << options.max_new_tokens << std::endl;
    std::cout << "  Sampling Steps: " << options.sampling_steps << std::endl;
    std::cout << "=======================================================" << std::endl;

    std::string prefix = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    ";
    std::string suffix = "\n    return quicksort(left) + [pivot] + quicksort(right)\n";

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        complete(prefix, suffix, options);
    }

    double total_ms = 0.0;
    double total_ddim_ms = 0.0;
    double total_decode_ms = 0.0;

    for (int i = 0; i < runs; ++i) {
        auto res = complete(prefix, suffix, options);
        total_ms += res.total_time_ms;
        total_ddim_ms += res.ddim_time_ms;
        total_decode_ms += res.decode_time_ms;
        std::cout << "  Run " << (i + 1) << "/" << runs
                  << ": Total = " << std::fixed << std::setprecision(2) << res.total_time_ms << " ms"
                  << " (DDIM: " << res.ddim_time_ms << " ms, Decode: " << res.decode_time_ms << " ms)"
                  << " | Rate = " << res.tokens_per_second << " tok/s" << std::endl;
    }

    double avg_ms = total_ms / runs;
    double avg_ddim = total_ddim_ms / runs;
    double avg_decode = total_decode_ms / runs;
    double avg_tok_sec = (static_cast<double>(options.max_new_tokens) / avg_ms) * 1000.0;

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "  Average Latency: " << avg_ms << " ms" << std::endl;
    std::cout << "  - DDIM Step(s):  " << avg_ddim << " ms" << std::endl;
    std::cout << "  - t=0 Decode:    " << avg_decode << " ms" << std::endl;
    std::cout << "  Throughput:      " << avg_tok_sec << " tokens/sec" << std::endl;
    std::cout << "=======================================================\n" << std::endl;
}

} // namespace tab_completion
