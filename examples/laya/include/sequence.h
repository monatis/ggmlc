#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "ggmlc/pipeline/tokenizer.h"
#include "questions.h"

namespace laya {

struct EncodedQuestion {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
    QType qtype = QType::Choice;
};

struct SequenceConfig {
    int max_len = 512;
    int head_max_len = 192;
    int max_opts = 16;
    int32_t cls_id = 50281;
    int32_t sep_id = 50282;
    int32_t pad_id = 50283;
    int32_t mask_id = 50284;
};

EncodedQuestion build_sequence(
    const ggmlc::pipeline::BPETokenizer& tok,
    const SequenceConfig& cfg,
    const std::string& state_text,
    const Question& q
);

void pad_encoded(
    const EncodedQuestion& enc,
    const SequenceConfig& cfg,
    std::vector<int32_t>& input_ids,
    std::vector<float>& attention_mask,
    std::vector<int32_t>& marker_pos,
    std::vector<float>& marker_mask,
    int32_t& qtype,
    int seq_len = -1
);

// Pack up to B encoded questions into row-major [B, S] / [B, max_opts].
// marker_pos is flattened: slot (b,k) = b * S + token_index (host-side, no graph I32 ADD).
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
);

}  // namespace laya
