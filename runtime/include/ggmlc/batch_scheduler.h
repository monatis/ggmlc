#pragma once

#include "ggmlc/executor.h"
#include <vector>
#include <deque>
#include <memory>
#include <string>
#include <cstdint>

namespace ggmlc {

struct GenerationRequest {
    uint64_t request_id = 0;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int slot_id = -1;
    int64_t current_pos = 0;
    int max_new_tokens = 32;
    float temperature = 0.0f;
    int eos_token_id = -1;
    bool finished = false;
    std::string finish_reason;
};

struct StepResult {
    std::vector<std::pair<uint64_t, int32_t>> new_tokens; // (request_id, token_id)
    std::vector<uint64_t> completed_request_ids;
};

class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(ModelExecutor& executor, size_t max_batch_size = 8, int eos_token_id = 0);
    ~ContinuousBatchScheduler();

    // Add a new request to the queue
    uint64_t add_request(const std::vector<int32_t>& prompt_tokens, int max_new_tokens = 32, float temperature = 0.0f, int eos_token_id = -1);

    // Run a single iteration-level step across all active requests
    StepResult step();

    // Query status
    bool has_work() const;
    size_t active_count() const;
    size_t pending_count() const;
    size_t max_batch_size() const { return max_batch_size_; }

    // Retrieve request information
    std::shared_ptr<GenerationRequest> get_request(uint64_t request_id) const;

private:
    ModelExecutor& executor_;
    size_t max_batch_size_;
    int default_eos_token_id_;
    uint64_t next_request_id_ = 1;

    std::deque<std::shared_ptr<GenerationRequest>> pending_queue_;
    std::vector<std::shared_ptr<GenerationRequest>> active_slots_;
    std::unordered_map<uint64_t, std::shared_ptr<GenerationRequest>> all_requests_;

    int find_available_slot() const;
    int select_batch_bucket(size_t max_active_slot) const;
    int32_t sample_next_token(const float* logits, size_t vocab_size, float temperature);
};

} // namespace ggmlc
