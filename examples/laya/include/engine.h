#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "questions.h"
#include "sequence.h"
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "ggmlc/pipeline/tokenizer.h"

namespace laya {

struct EngineOptions {
    std::string device = "auto";
    int n_threads = 4;
    bool cuda_graph = false;
    int max_batch = 0;  // 0 = GGUF metadata / default 8
};

class DecisionEngine {
public:
    DecisionEngine() = default;
    ~DecisionEngine() = default;

    bool load_model(const std::string& gguf_path, const EngineOptions& opt);
    bool is_loaded() const { return loaded_; }
    const std::string& device() const { return device_; }
    const std::string& family() const { return family_; }
    const std::string& model_name() const { return model_name_; }
    const SequenceConfig& seq_config() const { return seq_; }
    const ggmlc::SerializedModelGraph& graph() const { return graph_; }
    const ggmlc::pipeline::BPETokenizer& tokenizer() const { return tokenizer_; }

    DecideResult decide(const JsonValue& state, const std::vector<Question>& questions);
    DecideResult decide_one(const JsonValue& state, const Question& q);

    void print_info() const;
    void benchmark(const JsonValue& state, const std::vector<Question>& questions, int runs, int warmup);

private:
    bool loaded_ = false;
    std::string model_path_;
    std::string device_ = "auto";
    std::string family_ = "english";
    std::string model_name_ = "laya";
    int n_threads_ = 4;
    bool cuda_graph_ = false;

    SequenceConfig seq_;
    std::vector<int> length_buckets_ = {64, 128, 256, 512};
    int min_seq_ = 64;
    int max_batch_ = 8;
    bool dynamic_ = false;
    std::vector<float> temperature_ = {1.6369f, 1.25143f, 1.9834f};
    std::unordered_map<std::string, float> temperature_by_options_;

    ggmlc::SerializedModelGraph graph_;
    std::unique_ptr<ggmlc::ModelExecutor> executor_;
    ggmlc::pipeline::BPETokenizer tokenizer_;

    uint32_t in_ids_ = 0, in_att_ = 0, in_mpos_ = 0, in_mmask_ = 0, in_qtype_ = 0;
    uint32_t out_logits_ = 0, out_act_ = 0;

    int length_bucket(int n) const;
    int clamp_seq(int n) const;
    int batch_cap_for_seq(int seq_len) const;
    void fill_symbol_env(int batch, int seq_len, std::unordered_map<std::string, int64_t>& env) const;
    bool prepare_shape(int batch, int seq_len);
    void bind_and_run_batch(
        int batch,
        int seq_len,
        const std::vector<int32_t>& ids,
        const std::vector<float>& att,
        const std::vector<int32_t>& mpos,
        const std::vector<float>& mmask,
        const std::vector<int32_t>& qtype,
        std::vector<float>& logits,
        std::vector<float>& act
    );
    Answer decode_answer(const Question& q, const float* logits, int k, const float* act) const;
};

}  // namespace laya
