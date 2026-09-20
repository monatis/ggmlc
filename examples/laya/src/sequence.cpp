#include "sequence.h"

#include <algorithm>

namespace laya {

static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static std::string mask_token_str(const ggmlc::pipeline::BPETokenizer& tok, int32_t mask_id) {
    std::string t = tok.decode({mask_id}, false);
    if (t.empty()) t = "[MASK]";
    return t;
}

EncodedQuestion build_sequence(
    const ggmlc::pipeline::BPETokenizer& tok,
    const SequenceConfig& cfg,
    const std::string& state_text,
    const Question& q
) {
    EncodedQuestion enc;
    enc.qtype = q.type;
    const std::string mask_tok = mask_token_str(tok, cfg.mask_id);
    const std::vector<std::string> opts = render_options(q);

    std::string ins = replace_all(q.instructions, mask_tok, " ");
    const std::string head_text = std::string(qtype_name(q.type)) + " question: " + ins;
    std::vector<int32_t> head_ids = tok.encode(head_text, 0, false, false);

    std::vector<std::vector<int32_t>> opt_ids;
    opt_ids.reserve(opts.size());
    for (const auto& opt : opts) {
        std::string ot = " " + replace_all(opt, mask_tok, " ");
        std::vector<int32_t> o = tok.encode(ot, 0, false, false);
        if (static_cast<int>(o.size()) > 48) o.resize(48);
        std::vector<int32_t> full;
        full.push_back(cfg.mask_id);
        full.insert(full.end(), o.begin(), o.end());
        opt_ids.push_back(std::move(full));
    }

    int opt_sum = 0;
    for (const auto& o : opt_ids) opt_sum += static_cast<int>(o.size());
    int opt_budget = cfg.head_max_len - opt_sum;
    if (opt_budget < 16) {
        int n = std::max(1, static_cast<int>(opt_ids.size()));
        int per = std::max(4, (cfg.head_max_len - 16) / n);
        opt_sum = 0;
        for (auto& o : opt_ids) {
            if (static_cast<int>(o.size()) > per) o.resize(per);
            opt_sum += static_cast<int>(o.size());
        }
        opt_budget = cfg.head_max_len - opt_sum;
    }
    int head_keep = std::max(8, opt_budget);
    if (static_cast<int>(head_ids.size()) > head_keep) head_ids.resize(head_keep);

    std::vector<int32_t> ids;
    ids.push_back(cfg.cls_id);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(cfg.sep_id);

    std::vector<int32_t> markers;
    for (const auto& o : opt_ids) {
        markers.push_back(static_cast<int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(cfg.sep_id);

    int room = std::max(0, cfg.max_len - static_cast<int>(ids.size()) - 1);
    std::string st_text = replace_all(state_text, mask_tok, " ");
    std::vector<int32_t> st = tok.encode(st_text, 0, false, false);
    if (static_cast<int>(st.size()) > room) st.resize(room);
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(cfg.sep_id);

    if (static_cast<int>(ids.size()) > cfg.max_len) ids.resize(cfg.max_len);
    for (int32_t m : markers) {
        if (m < cfg.max_len) enc.markers.push_back(m);
    }
    enc.ids = std::move(ids);
    return enc;
}

void pad_encoded(
    const EncodedQuestion& enc,
    const SequenceConfig& cfg,
    std::vector<int32_t>& input_ids,
    std::vector<float>& attention_mask,
    std::vector<int32_t>& marker_pos,
    std::vector<float>& marker_mask,
    int32_t& qtype,
    int seq_len
) {
    if (seq_len <= 0) seq_len = cfg.max_len;
    input_ids.assign(seq_len, cfg.pad_id);
    attention_mask.assign(seq_len, 0.0f);
    marker_pos.assign(cfg.max_opts, 0);
    marker_mask.assign(cfg.max_opts, 0.0f);
    const int n = std::min(static_cast<int>(enc.ids.size()), seq_len);
    for (int i = 0; i < n; ++i) {
        input_ids[i] = enc.ids[i];
        attention_mask[i] = 1.0f;
    }
    const int k = std::min(static_cast<int>(enc.markers.size()), cfg.max_opts);
    for (int i = 0; i < k; ++i) {
        marker_pos[i] = enc.markers[i];
        marker_mask[i] = 1.0f;
    }
    qtype = static_cast<int32_t>(enc.qtype);
}

void pad_encoded_batch(
    const std::vector<EncodedQuestion>& encs,
    const SequenceConfig& cfg,
    int batch,
    int seq_len,
    std::vector<int32_t>& input_ids,
    std::vector<float>& attention_mask,
    std::vector<int32_t>& marker_pos,
    std::vector<float>& marker_mask,
    std::vector<int32_t>& qtype
) {
    if (batch < 1) batch = 1;
    if (seq_len <= 0) seq_len = cfg.max_len;
    const int opts = cfg.max_opts;
    input_ids.assign(static_cast<size_t>(batch) * seq_len, cfg.pad_id);
    attention_mask.assign(static_cast<size_t>(batch) * seq_len, 0.0f);
    marker_pos.assign(static_cast<size_t>(batch) * opts, 0);
    marker_mask.assign(static_cast<size_t>(batch) * opts, 0.0f);
    qtype.assign(static_cast<size_t>(batch), 0);
    for (int b = 0; b < batch; ++b) {
        if (b >= static_cast<int>(encs.size())) continue;
        const EncodedQuestion& enc = encs[b];
        const int n = std::min(static_cast<int>(enc.ids.size()), seq_len);
        for (int i = 0; i < n; ++i) {
            input_ids[b * seq_len + i] = enc.ids[i];
            attention_mask[b * seq_len + i] = 1.0f;
        }
        const int k = std::min(static_cast<int>(enc.markers.size()), opts);
        for (int i = 0; i < k; ++i) {
            marker_pos[b * opts + i] = b * seq_len + enc.markers[i];
            marker_mask[b * opts + i] = 1.0f;
        }
        qtype[b] = static_cast<int32_t>(enc.qtype);
    }
}

}  // namespace laya
