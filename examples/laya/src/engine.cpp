#include "engine.h"
#include "language.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace laya {

static int64_t meta_int(const ggmlc::SerializedModelGraph& g, const std::string& key, int64_t def) {
    auto it = g.metadata_int.find(key);
    return it == g.metadata_int.end() ? def : it->second;
}

static std::string meta_str(const ggmlc::SerializedModelGraph& g, const std::string& key, const std::string& def = "") {
    auto it = g.metadata_str.find(key);
    return it == g.metadata_str.end() ? def : it->second;
}

static uint32_t find_input(const ggmlc::SerializedModelGraph& g, const std::vector<std::string>& needles, uint32_t fallback) {
    for (uint32_t id : g.inputs) {
        auto it = g.tensors.find(id);
        if (it == g.tensors.end()) continue;
        const std::string& name = it->second.name;
        for (const auto& n : needles) {
            if (name.find(n) != std::string::npos) return id;
        }
    }
    return fallback;
}

bool DecisionEngine::load_model(const std::string& gguf_path, const EngineOptions& opt) {
    model_path_ = gguf_path;
    device_ = opt.device;
    n_threads_ = opt.n_threads;
    cuda_graph_ = opt.cuda_graph;
    loaded_ = false;

    std::cerr << "[laya] loading " << gguf_path << " device=" << device_ << std::endl;
    try {
        graph_ = ggmlc::ModelLoader::load_from_file(gguf_path);
        try {
            executor_ = std::make_unique<ggmlc::ModelExecutor>(graph_, device_);
        } catch (const std::exception& e) {
            if (device_ == "auto") {
                std::cerr << "[laya] auto device failed (" << e.what() << "), falling back to cpu\n";
                executor_ = std::make_unique<ggmlc::ModelExecutor>(graph_, "cpu");
            } else {
                throw;
            }
        }
        device_ = executor_->device();
    } catch (const std::exception& e) {
        std::cerr << "[laya] load failed: " << e.what() << std::endl;
        return false;
    }

    if (graph_.inputs.size() < 5 || graph_.outputs.size() < 1) {
        std::cerr << "[laya] unexpected graph arity: inputs=" << graph_.inputs.size()
                  << " outputs=" << graph_.outputs.size() << std::endl;
        return false;
    }

    in_ids_ = find_input(graph_, {"input_ids", "ids"}, graph_.inputs[0]);
    in_att_ = find_input(graph_, {"attention_mask", "attn"}, graph_.inputs.size() > 1 ? graph_.inputs[1] : graph_.inputs[0]);
    in_mpos_ = find_input(graph_, {"marker_pos"}, graph_.inputs.size() > 2 ? graph_.inputs[2] : graph_.inputs[0]);
    in_mmask_ = find_input(graph_, {"marker_mask"}, graph_.inputs.size() > 3 ? graph_.inputs[3] : graph_.inputs[0]);
    in_qtype_ = find_input(graph_, {"qtype"}, graph_.inputs.size() > 4 ? graph_.inputs[4] : graph_.inputs[0]);
    out_logits_ = graph_.outputs[0];
    out_act_ = graph_.outputs.size() > 1 ? graph_.outputs[1] : graph_.outputs[0];

    seq_.max_len = static_cast<int>(meta_int(graph_, "laya.max_len", 512));
    seq_.head_max_len = static_cast<int>(meta_int(graph_, "laya.head_max_len", 192));
    seq_.max_opts = static_cast<int>(meta_int(graph_, "laya.max_opts", 16));
    seq_.mask_id = static_cast<int32_t>(meta_int(graph_, "laya.mask_token_id", 50284));
    seq_.cls_id = static_cast<int32_t>(meta_int(graph_, "laya.cls_token_id", 50281));
    seq_.sep_id = static_cast<int32_t>(meta_int(graph_, "laya.sep_token_id", 50282));
    seq_.pad_id = static_cast<int32_t>(meta_int(graph_, "laya.pad_token_id", 50283));
    max_batch_ = static_cast<int>(meta_int(graph_, "laya.max_batch", 8));
    if (opt.max_batch > 0) max_batch_ = opt.max_batch;
    if (max_batch_ < 1) max_batch_ = 1;
    min_seq_ = static_cast<int>(meta_int(graph_, "laya.min_seq", 64));
    if (min_seq_ < 1) min_seq_ = 1;
    if (min_seq_ > seq_.max_len) min_seq_ = seq_.max_len;

    dynamic_ = !graph_.symbol_table.empty();
    std::string bj = meta_str(graph_, "laya.length_buckets");
    if (!bj.empty()) {
        try {
            JsonValue arr = JsonParser::parse_string(bj);
            if (arr.is_array()) {
                length_buckets_.clear();
                for (const auto& v : arr.arr) {
                    if (v.is_number()) length_buckets_.push_back(static_cast<int>(v.n));
                }
            }
        } catch (...) {
        }
    }
    if (length_buckets_.empty()) length_buckets_ = {64, 128, 256, 512};
    if (!dynamic_) {
        length_buckets_ = {seq_.max_len};
        min_seq_ = seq_.max_len;
        max_batch_ = 1;
    }

    std::string tjson = meta_str(graph_, "laya.temperature");
    if (!tjson.empty()) {
        try {
            JsonValue t = JsonParser::parse_string(tjson);
            if (t.is_array()) {
                temperature_.resize(3, 1.0f);
                for (size_t i = 0; i < t.arr.size() && i < 3; ++i) {
                    if (t.arr[i].is_number()) temperature_[i] = static_cast<float>(t.arr[i].n);
                }
            }
        } catch (...) {
        }
    }
    std::string bjson = meta_str(graph_, "laya.temperature_by_options");
    if (!bjson.empty()) {
        try {
            JsonValue b = JsonParser::parse_string(bjson);
            if (b.is_object()) {
                temperature_by_options_.clear();
                for (const auto& kv : b.obj) {
                    if (kv.second.is_number()) {
                        temperature_by_options_[kv.first] = static_cast<float>(kv.second.n);
                    }
                }
            }
        } catch (...) {
        }
    }

    if (!tokenizer_.init_from_gguf_file(gguf_path)) {
        std::cerr << "[laya] warning: GGUF has no tokenizer metadata; sequence encoding will fail." << std::endl;
    }

    model_name_ = meta_str(graph_, "laya.model_name", "laya");
    family_ = meta_str(graph_, "laya.family", "");
    if (family_.empty()) {
        family_ = infer_family(gguf_path, model_name_, meta_str(graph_, "laya.checkpoint"));
    }

    // Defer prepare until the first real (B, S). A load-time [1, 512] graph
    // pins the max activation footprint and OOMs the short buckets on 6 GB.
    if (cuda_graph_) executor_->set_enable_cuda_graph(true);

    loaded_ = true;
    std::cerr << "[laya] ready  max_len=" << seq_.max_len
              << " max_opts=" << seq_.max_opts
              << " max_batch=" << max_batch_
              << " dynamic=" << (dynamic_ ? "b,s" : "static")
              << " vocab=" << tokenizer_.vocab_size() << std::endl;
    return true;
}

int DecisionEngine::length_bucket(int n) const {
    int cap = seq_.max_len;
    int chosen = cap;
    for (int b : length_buckets_) {
        if (b >= n && b <= cap) {
            chosen = b;
            break;
        }
        if (b <= cap) chosen = b;
    }
    if (n > chosen) chosen = cap;
    return chosen;
}

int DecisionEngine::clamp_seq(int n) const {
    int s = std::max(n, min_seq_);
    if (s > seq_.max_len) s = seq_.max_len;
    return s;
}

int DecisionEngine::batch_cap_for_seq(int seq_len) const {
    if (!dynamic_ || seq_len <= 0) return 1;
    // gallocr reuses activations; keep a laptop VRAM ceiling with OOM-halve fallback.
    const int budget = (device_ == "cpu") ? 2048 : 1024;
    int cap = std::max(1, budget / seq_len);
    return std::min(cap, max_batch_);
}

static void bind_dim_value(
    const std::shared_ptr<ggmlc::DimExpr>& d,
    int64_t val,
    const std::vector<std::string>& table,
    std::unordered_map<std::string, int64_t>& env
) {
    if (!d) return;
    if (d->type == ggmlc::DimType::SYMBOL) {
        if (d->val >= 0 && d->val < static_cast<int64_t>(table.size())) {
            env[table[static_cast<size_t>(d->val)]] = val;
        }
        return;
    }
    bind_dim_value(d->left, val, table, env);
    bind_dim_value(d->right, val, table, env);
}

void DecisionEngine::fill_symbol_env(
    int batch, int seq_len, std::unordered_map<std::string, int64_t>& env
) const {
    env.clear();
    env["b"] = batch;
    env["s"] = seq_len;
    auto it = graph_.tensors.find(in_ids_);
    if (it != graph_.tensors.end()) {
        // PyTorch [B, S] -> GGML ne[0]=S, ne[1]=B
        bind_dim_value(it->second.ne[0], seq_len, graph_.symbol_table, env);
        bind_dim_value(it->second.ne[1], batch, graph_.symbol_table, env);
    }
    for (const auto& sym : graph_.symbol_table) {
        if (env.find(sym) == env.end()) env[sym] = seq_len;
    }
}

bool DecisionEngine::prepare_shape(int batch, int seq_len) {
    std::unordered_map<std::string, int64_t> env;
    if (dynamic_) fill_symbol_env(batch, seq_len, env);
    try {
        executor_->prepare(env, true);
        if (cuda_graph_) executor_->set_enable_cuda_graph(true);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[laya] prepare failed B=" << batch << " S=" << seq_len
                  << " : " << e.what() << std::endl;
        return false;
    }
}

void DecisionEngine::bind_and_run_batch(
    int batch,
    int seq_len,
    const std::vector<int32_t>& ids,
    const std::vector<float>& att,
    const std::vector<int32_t>& mpos,
    const std::vector<float>& mmask,
    const std::vector<int32_t>& qtype,
    std::vector<float>& logits,
    std::vector<float>& act
) {
    if (!prepare_shape(batch, seq_len)) {
        throw std::runtime_error("laya prepare_shape failed");
    }
    executor_->set_input(in_ids_, ids.data(), ids.size() * sizeof(int32_t));
    executor_->set_input(in_att_, att.data(), att.size() * sizeof(float));
    executor_->set_input(in_mpos_, mpos.data(), mpos.size() * sizeof(int32_t));
    executor_->set_input(in_mmask_, mmask.data(), mmask.size() * sizeof(float));
    executor_->set_input(in_qtype_, qtype.data(), qtype.size() * sizeof(int32_t));
    executor_->run(n_threads_);

    const int opts = seq_.max_opts;
    logits.assign(static_cast<size_t>(batch) * opts, -1.0e4f);
    const float* lp = static_cast<const float*>(executor_->get_output_data(out_logits_));
    if (lp) {
        size_t n = executor_->get_tensor_size_bytes(out_logits_) / sizeof(float);
        size_t want = static_cast<size_t>(batch) * opts;
        if (n > want) n = want;
        std::memcpy(logits.data(), lp, n * sizeof(float));
    }
    act.assign(static_cast<size_t>(batch) * 2, 0.0f);
    const float* ap = static_cast<const float*>(executor_->get_output_data(out_act_));
    if (ap && out_act_ != out_logits_) {
        size_t n = executor_->get_tensor_size_bytes(out_act_) / sizeof(float);
        size_t want = static_cast<size_t>(batch) * 2;
        if (n > want) n = want;
        std::memcpy(act.data(), ap, n * sizeof(float));
    }
}

Answer DecisionEngine::decode_answer(
    const Question& q, const float* logits, int k, const float* act
) const {
    std::vector<float> z(logits, logits + std::max(0, k));
    std::string bucket = temp_bucket(q.type, k);
    float tscale = temperature_[static_cast<int>(q.type) % 3];
    auto it = temperature_by_options_.find(bucket);
    if (it != temperature_by_options_.end()) tscale = it->second;
    std::vector<float> p = softmax_temp(z, tscale);

    float act_p = 0.0f;
    if (act) {
        float m = std::max(act[0], act[1]);
        float e0 = std::exp(act[0] - m);
        float e1 = std::exp(act[1] - m);
        act_p = e0 / (e0 + e1);
    }

    Answer a;
    a.id = q.id;
    a.type = q.type;
    a.act_probability = act_p;
    const auto keys = q.criteria;
    if (q.type == QType::Choice) {
        int arg = 0;
        for (int i = 1; i < k; ++i) if (p[i] > p[arg]) arg = i;
        if (arg < static_cast<int>(keys.size())) a.choice = keys[arg].first;
        for (int i = 0; i < k && i < static_cast<int>(keys.size()); ++i) {
            a.probabilities.emplace_back(keys[i].first, p[i]);
        }
        a.confidence = confidence_from_probs(p);
    } else if (q.type == QType::Score) {
        float expv = 0.0f;
        for (int i = 0; i < k; ++i) {
            expv += static_cast<float>(i) * p[i];
            a.probabilities.emplace_back(std::to_string(i), p[i]);
            if (i < static_cast<int>(keys.size())) a.legend.push_back(keys[i].second);
        }
        a.score = expv;
        a.confidence = confidence_from_probs(p);
    } else {
        float pt = (k >= 2) ? p[1] : 0.0f;
        a.noul = pt;
        a.confidence = std::max(pt, 1.0f - pt);
        if (k >= 2) {
            a.probabilities.emplace_back("false", p[0]);
            a.probabilities.emplace_back("true", p[1]);
        }
    }
    return a;
}

DecideResult DecisionEngine::decide_one(const JsonValue& state, const Question& q) {
    return decide(state, std::vector<Question>{q});
}

DecideResult DecisionEngine::decide(const JsonValue& state, const std::vector<Question>& questions) {
    DecideResult result;
    result.model = meta_str(graph_, "laya.model_name", "laya");
    if (!loaded_ || !executor_) return result;

    const std::string state_text = serialize_state(state);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<EncodedQuestion> encs;
    encs.reserve(questions.size());
    int tokens = 0;
    int max_live = 0;
    for (const auto& q : questions) {
        EncodedQuestion enc = build_sequence(tokenizer_, seq_, state_text, q);
        tokens += static_cast<int>(enc.ids.size());
        max_live = std::max(max_live, static_cast<int>(enc.ids.size()));
        encs.push_back(std::move(enc));
    }

    // Group by coarse length so a 400-token question does not pad 80-token
    // neighbours. Within a group, pad like Python collate_items: S = max(len_i),
    // attention_mask zeros the pads. That is not concat packing — FlashAttention
    // stays rectangular B×S, not one S=sum sequence with a block-diagonal mask.
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < static_cast<int>(encs.size()); ++i) {
        groups[length_bucket(static_cast<int>(encs[i].ids.size()))].push_back(i);
    }

    result.answers.assign(questions.size(), Answer{});
    result.input_tokens = tokens;
    result.seq_bucket = max_live > 0 ? clamp_seq(max_live) : 0;

    for (auto& g : groups) {
        auto& idxs = g.second;
        int S = min_seq_;
        for (int qi : idxs) {
            S = std::max(S, clamp_seq(static_cast<int>(encs[qi].ids.size())));
        }
        const int cap = batch_cap_for_seq(S);
        int graph_B = 0;
        size_t cursor = 0;
        while (cursor < idxs.size()) {
            const int remaining = static_cast<int>(idxs.size() - cursor);
            int live = std::min(remaining, cap);
            int batch = live;
            // Hold one (B,S) for the group. Dummy-pad later chunks so prepare
            // and CUDA graphs stay warm across the preset's forwards.
            if (graph_B < 1) graph_B = live;
            batch = graph_B;
            live = std::min(live, remaining);
            if (live < 1) live = 1;
            if (batch < live) batch = live;

            bool ok = false;
            for (int try_b = batch; try_b >= 1; try_b = (try_b > 1) ? (try_b / 2) : 0) {
                const int take = std::min(try_b, remaining);
                std::vector<EncodedQuestion> slice;
                slice.reserve(static_cast<size_t>(take));
                for (int j = 0; j < take; ++j) slice.push_back(encs[idxs[cursor + j]]);
                std::vector<int32_t> ids, mpos, qt;
                std::vector<float> att, mmask;
                pad_encoded_batch(slice, seq_, try_b, S, ids, att, mpos, mmask, qt);
                try {
                    std::vector<float> logits, act;
                    bind_and_run_batch(try_b, S, ids, att, mpos, mmask, qt, logits, act);
                    const int opts = seq_.max_opts;
                    for (int j = 0; j < take; ++j) {
                        const int qi = idxs[cursor + j];
                        const int k = static_cast<int>(encs[qi].markers.size());
                        const float* lp = logits.data() + static_cast<size_t>(j) * opts;
                        const float* ap = (act.size() >= static_cast<size_t>(j + 1) * 2)
                            ? act.data() + static_cast<size_t>(j) * 2
                            : nullptr;
                        result.answers[qi] = decode_answer(questions[qi], lp, k, ap);
                    }
                    result.n_forwards += 1;
                    result.seq_bucket = std::max(result.seq_bucket, S);
                    result.batch_bucket = std::max(result.batch_bucket, try_b);
                    cursor += static_cast<size_t>(take);
                    graph_B = try_b;
                    ok = true;
                    break;
                } catch (const std::exception& e) {
                    std::cerr << "[laya] forward failed B=" << try_b << " S=" << S
                              << " : " << e.what() << std::endl;
                }
            }
            if (!ok) throw std::runtime_error("laya forward failed");
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

void DecisionEngine::print_info() const {
    std::cout << "model: " << meta_str(graph_, "general.name", graph_.name) << "\n"
              << "family: " << family_ << "\n"
              << "checkpoint: " << meta_str(graph_, "laya.checkpoint", "") << "\n"
              << "device: " << device_ << "\n"
              << "max_len: " << seq_.max_len << "  min_seq: " << min_seq_
              << "  head_max_len: " << seq_.head_max_len
              << "  max_opts: " << seq_.max_opts << "  max_batch: " << max_batch_ << "\n"
              << "dynamic: " << (dynamic_ ? "b,s" : "static")
              << "  pad: max-in-batch  groups: [";
    for (size_t i = 0; i < length_buckets_.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << length_buckets_[i];
    }
    std::cout << "]\n"
              << "symbols:";
    for (const auto& s : graph_.symbol_table) std::cout << " " << s;
    std::cout << "\n"
              << "special tokens  cls=" << seq_.cls_id << " sep=" << seq_.sep_id
              << " pad=" << seq_.pad_id << " mask=" << seq_.mask_id << "\n"
              << "vocab: " << tokenizer_.vocab_size() << "\n"
              << "inputs: " << graph_.inputs.size() << "  outputs: " << graph_.outputs.size()
              << "  ops: " << graph_.ops.size() << "\n"
              << "temperature: [";
    for (size_t i = 0; i < temperature_.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << temperature_[i];
    }
    std::cout << "]\n";
    if (!temperature_by_options_.empty()) {
        std::cout << "temperature_by_options:\n";
        for (const auto& kv : temperature_by_options_) {
            std::cout << "  " << kv.first << " = " << kv.second << "\n";
        }
    }
}

void DecisionEngine::benchmark(const JsonValue& state, const std::vector<Question>& questions, int runs, int warmup) {
    const int nq = static_cast<int>(questions.size());
    std::cout << "\n=======================================================\n"
              << " Laya System 1 benchmark\n"
              << " device=" << device_
              << "  cuda_graph=" << (cuda_graph_ ? "on" : "off")
              << "  threads=" << n_threads_ << "\n"
              << " questions=" << nq
              << "  dynamic=" << (dynamic_ ? "b,s" : "static")
              << "  max_batch=" << max_batch_ << "\n"
              << " warmup=" << warmup << "  runs=" << runs << "\n"
              << "=======================================================\n";

    for (int i = 0; i < warmup; ++i) decide(state, questions);

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        double idx = p * static_cast<double>(v.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, v.size() - 1);
        double t = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - t) + v[hi] * t;
    };

    std::vector<double> wall;
    int tokens = 0, seq_b = 0, batch_b = 0, nfwd = 0;
    for (int i = 0; i < runs; ++i) {
        auto r = decide(state, questions);
        wall.push_back(r.latency_ms);
        tokens = r.input_tokens;
        seq_b = r.seq_bucket;
        batch_b = r.batch_bucket;
        nfwd = r.n_forwards;
        std::cout << "  run " << (i + 1) << "/" << runs
                  << ": wall " << std::fixed << std::setprecision(1) << r.latency_ms << " ms"
                  << "  (" << nq << " q, " << tokens << " tok"
                  << "  S=" << seq_b << " B=" << batch_b
                  << "  forwards=" << nfwd << ")\n";
    }

    const double mean_wall = std::accumulate(wall.begin(), wall.end(), 0.0) / std::max(1, runs);
    const double per_q = nq ? mean_wall / nq : 0.0;
    const double qps = (mean_wall > 0.0) ? (1000.0 * nq / mean_wall) : 0.0;
    std::cout << "-------------------------------------------------------\n"
              << std::fixed << std::setprecision(1)
              << "  Wall clock  mean " << mean_wall
              << "  p50 " << percentile(wall, 0.50)
              << "  best " << *std::min_element(wall.begin(), wall.end()) << " ms\n"
              << "  Equivalent  " << per_q << " ms/q\n"
              << std::setprecision(2)
              << "  Throughput  " << qps << " q/s  (wall, " << nq << "-question preset)\n"
              << "=======================================================\n";
}

}  // namespace laya
