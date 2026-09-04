#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>

struct gguf_context;

namespace ggmlc {
namespace pipeline {

struct TokenizerOutput {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
};

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load vocabulary, merges, and special tokens from raw lists
    void init(
        const std::vector<std::string>& tokens,
        const std::vector<std::string>& merges,
        int32_t bos_id = 49406,
        int32_t eos_id = 49407,
        int32_t pad_id = 49407,
        int32_t unk_id = 0,
        const std::string& pre_tokenizer = "clip",
        const std::string& chat_template = ""
    );

    // Initialize directly from GGUF metadata file
    bool init_from_gguf_file(const std::string& filepath);

    // Initialize directly from an existing gguf_context
    bool init_from_gguf_ctx(const struct gguf_context* ctx);

    // Tokenize text into sequence of token IDs
    std::vector<int32_t> encode(
        const std::string& text,
        int max_length = 0,
        bool add_special_tokens = true,
        bool pad_to_max = false
    ) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const;

    // Decode a single token ID (useful for streaming generation)
    std::string decode_token(int32_t id, bool skip_special_tokens = true) const;

    // Check if a token ID is a special or control token
    bool is_special_token(int32_t id) const;

    // Apply chat template formatting for instruction-tuned models
    std::string apply_chat_template(
        const std::string& user_msg,
        const std::string& system_msg = "",
        bool add_generation_prompt = true
    ) const;

    // Direct accessors
    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t unk_token_id() const { return unk_id_; }
    const std::string& pre_tokenizer() const { return pre_tokenizer_; }
    const std::string& chat_template() const { return chat_template_; }
    void set_chat_template(const std::string& templ) { chat_template_ = templ; }
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
    std::string chat_template_;

    std::vector<std::string> special_tokens_;

    void init_byte_encoder();
    void encode_normal_text(const std::string& subtext, std::vector<int32_t>& out_tokens) const;
    std::vector<std::string> bpe(const std::string& token) const;
    std::vector<std::string> bpe_clip_word(const std::string& word) const;
};

} // namespace pipeline
} // namespace ggmlc
