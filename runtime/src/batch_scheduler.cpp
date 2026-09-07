#include "ggmlc/batch_scheduler.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

namespace ggmlc {

ContinuousBatchScheduler::ContinuousBatchScheduler(ModelExecutor& executor, size_t max_batch_size, int eos_token_id)
    : executor_(executor), max_batch_size_(max_batch_size), default_eos_token_id_(eos_token_id) {
    active_slots_.resize(max_batch_size_, nullptr);
    if (!executor_.is_paged_kv_cache_enabled()) {
        executor_.init_paged_kv_cache(max_batch_size_, 2048);
    }
    executor_.set_enable_cuda_graph_buckets(false);
    radix_tree_ = std::make_unique<PagedRadixTree>(executor_.get_tokens_per_page());
}

ContinuousBatchScheduler::~ContinuousBatchScheduler() {
    for (size_t i = 0; i < active_slots_.size(); ++i) {
        if (active_slots_[i]) {
            if (radix_tree_) {
                for (auto& node : active_slots_[i]->matched_radix_nodes) {
                    radix_tree_->release_node(node);
                }
                active_slots_[i]->matched_radix_nodes.clear();
            }
            executor_.paged_kv_free_slot(static_cast<int>(i));
            active_slots_[i] = nullptr;
        }
    }
}

uint64_t ContinuousBatchScheduler::add_request(const std::vector<int32_t>& prompt_tokens, int max_new_tokens, float temperature, int eos_token_id) {
    auto req = std::make_shared<GenerationRequest>();
    req->request_id = next_request_id_++;
    req->prompt_tokens = prompt_tokens;
    req->max_new_tokens = max_new_tokens;
    req->temperature = temperature;
    req->eos_token_id = (eos_token_id >= 0) ? eos_token_id : default_eos_token_id_;
    req->current_pos = 0;
    req->finished = false;

    pending_queue_.push_back(req);
    all_requests_[req->request_id] = req;
    return req->request_id;
}

bool ContinuousBatchScheduler::has_work() const {
    if (!pending_queue_.empty()) return true;
    for (const auto& slot : active_slots_) {
        if (slot != nullptr) return true;
    }
    return false;
}

size_t ContinuousBatchScheduler::active_count() const {
    size_t count = 0;
    for (const auto& slot : active_slots_) {
        if (slot != nullptr) count++;
    }
    return count;
}

size_t ContinuousBatchScheduler::pending_count() const {
    return pending_queue_.size();
}

std::shared_ptr<GenerationRequest> ContinuousBatchScheduler::get_request(uint64_t request_id) const {
    auto it = all_requests_.find(request_id);
    if (it != all_requests_.end()) return it->second;
    return nullptr;
}

int ContinuousBatchScheduler::find_available_slot() const {
    for (size_t i = 0; i < active_slots_.size(); ++i) {
        if (active_slots_[i] == nullptr) return static_cast<int>(i);
    }
    return -1;
}

int ContinuousBatchScheduler::select_batch_bucket(size_t max_active_slot) const {
    if (!executor_.is_cuda_graph_buckets_enabled()) {
        return static_cast<int>(max_active_slot + 1);
    }
    static const std::vector<int> buckets = {1, 2, 4, 8, 16};
    for (int b : buckets) {
        if (static_cast<size_t>(b) > max_active_slot) return b;
    }
    return static_cast<int>(buckets.back());
}

int32_t ContinuousBatchScheduler::sample_next_token(const float* logits, size_t vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        // Greedy argmax
        int32_t best_idx = 0;
        float best_val = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = static_cast<int32_t>(i);
            }
        }
        return best_idx;
    }

    // Temperature-scaled softmax sampling
    std::vector<float> probs(vocab_size);
    float max_l = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float accum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        accum += probs[i];
        if (accum >= r) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(vocab_size - 1);
}

StepResult ContinuousBatchScheduler::step() {
    StepResult res;

    // 1. Admit pending requests into free slots
    // 1. Admission of pending requests with Phase 1 Prefill
    bool admitted_any = false;
    while (!pending_queue_.empty()) {
        int free_slot = find_available_slot();
        if (free_slot < 0) break;

        auto req = pending_queue_.front();
        pending_queue_.pop_front();

        req->slot_id = free_slot;
        active_slots_[free_slot] = req;

        executor_.paged_kv_alloc_slot(free_slot);

        size_t matched_tokens = 0;
        if (prefix_caching_enabled_ && radix_tree_) {
            auto match = radix_tree_->match_prefix(req->prompt_tokens);
            if (match.matched_pages > 0) {
                matched_tokens = match.matched_tokens;
                req->prefix_tokens_matched = matched_tokens;
                req->matched_radix_nodes = match.matched_nodes;
                for (auto& node : match.matched_nodes) {
                    radix_tree_->retain_node(node);
                }
                // Map the pre-existing physical pages directly into slot's VA window
                executor_.paged_kv_map_existing_pages(free_slot, 0, match.page_k_handles, match.page_v_handles);

                total_prefix_cache_hits_++;
                total_prefix_tokens_saved_ += matched_tokens;
            }
        }

        int64_t prompt_len = static_cast<int64_t>(req->prompt_tokens.size());
        int64_t total_tokens_needed = prompt_len + req->max_new_tokens;
        executor_.paged_kv_ensure_tokens(free_slot, total_tokens_needed);

        const auto& mg = executor_.model_graph();
        uint32_t in_tid = mg.inputs[0];
        uint32_t out_tid = mg.outputs[0];

        int64_t prefill_len = prompt_len - matched_tokens;
        const int32_t* prefill_input_ptr = nullptr;
        int64_t prefill_pos = 0;
        int64_t effective_s = 0;

        if (prefill_len > 0) {
            prefill_input_ptr = req->prompt_tokens.data() + matched_tokens;
            prefill_pos = matched_tokens;
            effective_s = prefill_len;
        } else {
            // Full prefix cache hit: evaluate last prompt token to sample first decode token
            prefill_input_ptr = req->prompt_tokens.data() + prompt_len - 1;
            prefill_pos = prompt_len - 1;
            effective_s = 1;
        }

        std::unordered_map<std::string, int64_t> p_env;
        p_env["b"] = 1;
        p_env["s"] = effective_s;
        p_env["pos"] = prefill_pos;
        p_env["slot"] = free_slot;

        const auto& in_t = mg.tensors.at(in_tid);
        if (in_t.ne.size() > 0 && in_t.ne[0] && in_t.ne[0]->type == DimType::SYMBOL) {
            int64_t sym_idx = in_t.ne[0]->val;
            if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(mg.symbol_table.size())) {
                p_env[mg.symbol_table[sym_idx]] = effective_s;
            }
        }
        if (in_t.ne.size() > 1 && in_t.ne[1] && in_t.ne[1]->type == DimType::SYMBOL) {
            int64_t sym_idx = in_t.ne[1]->val;
            if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(mg.symbol_table.size())) {
                p_env[mg.symbol_table[sym_idx]] = 1;
            }
        }

        for (const auto& sym : mg.symbol_table) {
            if (p_env.find(sym) == p_env.end()) {
                if (sym.find("b") != std::string::npos || sym.find("batch") != std::string::npos) {
                    p_env[sym] = 1;
                } else {
                    p_env[sym] = effective_s;
                }
            }
        }

        executor_.prepare(p_env, true);
        executor_.set_input(in_tid, prefill_input_ptr, effective_s * sizeof(int32_t));
        executor_.run(1);

        // Register newly computed full pages into RadixTree
        if (prefix_caching_enabled_ && radix_tree_ && prefill_len > 0) {
            size_t tokens_per_page = radix_tree_->tokens_per_page();
            size_t start_page = matched_tokens / tokens_per_page;
            size_t total_full_pages = prompt_len / tokens_per_page;
            if (total_full_pages > start_page) {
                size_t num_new_pages = total_full_pages - start_page;
                std::vector<std::unordered_map<uint32_t, uint64_t>> page_k(num_new_pages);
                std::vector<std::unordered_map<uint32_t, uint64_t>> page_v(num_new_pages);
                for (size_t p = 0; p < num_new_pages; ++p) {
                    executor_.paged_kv_extract_page_handles(free_slot, start_page + p, page_k[p], page_v[p]);
                }
                radix_tree_->insert_prefix(req->prompt_tokens, start_page, page_k, page_v);
            }
        }

        // Sample first token from prefill logits
        const float* logits_data = static_cast<const float*>(executor_.get_output_data(out_tid));
        size_t total_elements = executor_.get_tensor_size_bytes(out_tid) / sizeof(float);
        size_t vocab_size = total_elements / effective_s;
        const float* last_logits = logits_data + (effective_s - 1) * vocab_size;
        int32_t first_tok = sample_next_token(last_logits, vocab_size, req->temperature);

        req->generated_tokens.push_back(first_tok);
        req->current_pos = prompt_len;
        res.new_tokens.push_back({req->request_id, first_tok});

        bool hit_eos = (req->eos_token_id >= 0 && first_tok == req->eos_token_id);
        bool hit_max = (static_cast<int>(req->generated_tokens.size()) >= req->max_new_tokens);
        if (hit_eos || hit_max) {
            req->finished = true;
            req->finish_reason = hit_eos ? "stop" : "length";
            res.completed_request_ids.push_back(req->request_id);
            if (radix_tree_) {
                for (auto& node : req->matched_radix_nodes) {
                    radix_tree_->release_node(node);
                }
                req->matched_radix_nodes.clear();
            }
            executor_.paged_kv_free_slot(free_slot);
            active_slots_[free_slot] = nullptr;
        }

        admitted_any = true;
    }

    if (admitted_any) {
        return res;
    }

    // 2. Select bucket based on highest active slot index
    int max_slot = -1;
    for (int i = 0; i < static_cast<int>(active_slots_.size()); ++i) {
        if (active_slots_[i] != nullptr) {
            max_slot = std::max(max_slot, i);
        }
    }
    if (max_slot < 0) return res;

    int bucket_b = select_batch_bucket(static_cast<size_t>(max_slot));
    bucket_b = std::min<int>(bucket_b, static_cast<int>(max_batch_size_));

    // 3. Prepare batch inputs (decode step, s = 1)
    std::vector<int32_t> batch_tokens(bucket_b, 0);
    int64_t current_pos = 0;
    for (int b = 0; b < bucket_b; ++b) {
        auto req = active_slots_[b];
        if (req != nullptr && !req->generated_tokens.empty()) {
            batch_tokens[b] = req->generated_tokens.back();
            current_pos = std::max(current_pos, req->current_pos);
        }
    }

    for (int b = 0; b < bucket_b; ++b) {
        if (!executor_.is_paged_slot_allocated(b)) {
            executor_.paged_kv_alloc_slot(b);
        }
        executor_.paged_kv_ensure_tokens(b, current_pos + 1);
    }

    // 4. Configure executor symbol environment and inputs
    const auto& mg = executor_.model_graph();
    uint32_t in_tid = mg.inputs[0];
    uint32_t out_tid = mg.outputs[0];

    std::unordered_map<std::string, int64_t> symbol_env;
    symbol_env["b"] = bucket_b;
    symbol_env["s"] = 1;
    symbol_env["pos"] = current_pos;

    // Check input tensor dynamic dimension symbols
    const auto& in_t = mg.tensors.at(in_tid);
    if (in_t.ne.size() > 0 && in_t.ne[0] && in_t.ne[0]->type == DimType::SYMBOL) {
        int64_t sym_idx = in_t.ne[0]->val;
        if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(mg.symbol_table.size())) {
            symbol_env[mg.symbol_table[sym_idx]] = 1;
        }
    }
    if (in_t.ne.size() > 1 && in_t.ne[1] && in_t.ne[1]->type == DimType::SYMBOL) {
        int64_t sym_idx = in_t.ne[1]->val;
        if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(mg.symbol_table.size())) {
            symbol_env[mg.symbol_table[sym_idx]] = bucket_b;
        }
    }

    // Populate any remaining symbols in symbol_table
    for (const auto& sym : mg.symbol_table) {
        if (symbol_env.find(sym) == symbol_env.end()) {
            if (sym.find("b") != std::string::npos || sym.find("batch") != std::string::npos) {
                symbol_env[sym] = bucket_b;
            } else {
                symbol_env[sym] = 1;
            }
        }
    }

    executor_.prepare(symbol_env, true);
    executor_.set_input(in_tid, batch_tokens.data(), batch_tokens.size() * sizeof(int32_t));

    // 5. Run forward step
    executor_.run(1);

    // 6. Process outputs per active slot
    const float* logits_base = static_cast<const float*>(executor_.get_output_data(out_tid));
    size_t total_elements = executor_.get_tensor_size_bytes(out_tid) / sizeof(float);
    size_t vocab_size = total_elements / bucket_b;

    for (int b = 0; b < bucket_b; ++b) {
        auto req = active_slots_[b];
        if (req == nullptr) continue;

        const float* row_logits = logits_base + b * vocab_size;
        int32_t next_tok = sample_next_token(row_logits, vocab_size, req->temperature);
        req->generated_tokens.push_back(next_tok);
        req->current_pos++;

        res.new_tokens.push_back({req->request_id, next_tok});

        // Check completion
        bool hit_eos = (req->eos_token_id >= 0 && next_tok == req->eos_token_id);
        bool hit_max = (static_cast<int>(req->generated_tokens.size()) >= req->max_new_tokens);

        if (hit_eos || hit_max) {
            req->finished = true;
            req->finish_reason = hit_eos ? "stop" : "length";
            res.completed_request_ids.push_back(req->request_id);

            // Release matched radix tree nodes
            if (radix_tree_) {
                for (auto& node : req->matched_radix_nodes) {
                    radix_tree_->release_node(node);
                }
                req->matched_radix_nodes.clear();
            }

            // Free slot physical memory immediately! (unmapped and returned to warm pool)
            executor_.paged_kv_free_slot(b);
            active_slots_[b] = nullptr;
        }
    }

    if (active_count() == 0) {
        for (size_t i = 0; i < active_slots_.size(); ++i) {
            if (executor_.is_paged_slot_allocated(static_cast<int>(i))) {
                executor_.paged_kv_free_slot(static_cast<int>(i));
            }
        }
    }

    return res;
}

} // namespace ggmlc
