#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>

namespace ggmlc {
namespace pipeline {

struct TokenizerOutput {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
};

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load vocabulary, merges, and special tokens from raw lists (or GGUF metadata)
    void init(
        const std::vector<std::string>& tokens,
        const std::vector<std::string>& merges,
        int32_t bos_id = 49406,
        int32_t eos_id = 49407,
        int32_t pad_id = 49407,
        int32_t unk_id = 0,
        const std::string& pre_tokenizer = "clip"
    );

    // Tokenize text into sequence of token IDs
    std::vector<int32_t> encode(
        const std::string& text,
        int max_length = 77,
        bool add_special_tokens = true,
        bool pad_to_max = true
    ) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const;

    // Direct accessors
    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    size_t vocab_size() const { return encoder_.size(); }

private:
    // Pair hash for merges
    struct PairHash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<int32_t, std::string> decoder_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;
    
    // Byte-level mapping (bytes_to_unicode)
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int32_t bos_id_ = 49406;
    int32_t eos_id_ = 49407;
    int32_t pad_id_ = 49407;
    int32_t unk_id_ = 0;
    std::string pre_tokenizer_ = "clip";

    void init_byte_encoder();
    std::vector<std::string> bpe(const std::string& token) const;
    std::vector<std::string> bpe_clip_word(const std::string& word) const;
};

} // namespace pipeline
} // namespace ggmlc
