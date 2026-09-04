#include "ggmlc/pipeline/tokenizer.h"
#include "gguf.h"

#include <regex>
#include <sstream>
#include <algorithm>
#include <set>
#include <iostream>

namespace ggmlc {
namespace pipeline {

// UTF-8 helper to encode unicode codepoint to std::string
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | ((cp >> 6) & 0x1f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | ((cp >> 12) & 0x0f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

void BPETokenizer::init_byte_encoder() {
    byte_to_unicode_.clear();
    unicode_to_byte_.clear();

    std::vector<int> bs;
    for (int i = L'!'; i <= L'~'; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        std::string utf8_str = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
        byte_to_unicode_[byte_val] = utf8_str;
        unicode_to_byte_[utf8_str] = byte_val;
    }
}

void BPETokenizer::init(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& merges,
    int32_t bos_id,
    int32_t eos_id,
    int32_t pad_id,
    int32_t unk_id,
    const std::string& pre_tokenizer,
    const std::string& chat_template
) {
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    pad_id_ = pad_id;
    unk_id_ = unk_id;
    pre_tokenizer_ = pre_tokenizer;
    chat_template_ = chat_template;

    encoder_.clear();
    decoder_.clear();
    for (size_t i = 0; i < tokens.size(); ++i) {
        encoder_[tokens[i]] = static_cast<int32_t>(i);
        decoder_[static_cast<int32_t>(i)] = tokens[i];
    }

    bpe_ranks_.clear();
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& line = merges[rank];
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            bpe_ranks_[{first, second}] = static_cast<int>(rank);
        }
    }

    init_byte_encoder();

    special_tokens_.clear();
    for (const auto& pair : encoder_) {
        const std::string& s = pair.first;
        if (s.size() >= 3 && s.front() == '<' && s.back() == '>') {
            special_tokens_.push_back(s);
        } else if (s == "<s>" || s == "</s>") {
            special_tokens_.push_back(s);
        }
    }
    for (int32_t tid : {bos_id_, eos_id_, pad_id_, unk_id_}) {
        if (tid >= 0) {
            auto it = decoder_.find(tid);
            if (it != decoder_.end() && std::find(special_tokens_.begin(), special_tokens_.end(), it->second) == special_tokens_.end()) {
                special_tokens_.push_back(it->second);
            }
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](const std::string& a, const std::string& b) {
        return a.size() > b.size();
    });
}

bool BPETokenizer::init_from_gguf_ctx(const struct gguf_context* ctx) {
    if (!ctx) return false;

    int64_t key_tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (key_tokens < 0) return false;

    size_t n_tokens = gguf_get_arr_n(ctx, key_tokens);
    std::vector<std::string> tokens;
    tokens.reserve(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        tokens.push_back(gguf_get_arr_str(ctx, key_tokens, i));
    }

    int64_t key_merges = gguf_find_key(ctx, "tokenizer.ggml.merges");
    std::vector<std::string> merges;
    if (key_merges >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, key_merges);
        merges.reserve(n_merges);
        for (size_t i = 0; i < n_merges; ++i) {
            merges.push_back(gguf_get_arr_str(ctx, key_merges, i));
        }
    }

    int64_t key_pre = gguf_find_key(ctx, "tokenizer.ggml.pre");
    std::string pre = (key_pre >= 0) ? gguf_get_val_str(ctx, key_pre) : "gpt2";

    int64_t key_bos = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    int64_t key_eos = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    int64_t key_pad = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    int64_t key_unk = gguf_find_key(ctx, "tokenizer.ggml.unknown_token_id");

    int32_t bos = (key_bos >= 0) ? gguf_get_val_i32(ctx, key_bos) : -1;
    int32_t eos = (key_eos >= 0) ? gguf_get_val_i32(ctx, key_eos) : -1;
    int32_t pad = (key_pad >= 0) ? gguf_get_val_i32(ctx, key_pad) : -1;
    int32_t unk = (key_unk >= 0) ? gguf_get_val_i32(ctx, key_unk) : 0;

    int64_t key_chat = gguf_find_key(ctx, "tokenizer.chat_template");
    std::string chat = (key_chat >= 0) ? gguf_get_val_str(ctx, key_chat) : "";

    init(tokens, merges, bos, eos, pad, unk, pre, chat);
    return true;
}

bool BPETokenizer::init_from_gguf_file(const std::string& filepath) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context* ctx = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx) return false;

    bool ok = init_from_gguf_ctx(ctx);
    gguf_free(ctx);
    return ok;
}

std::string BPETokenizer::apply_chat_template(
    const std::string& user_msg,
    const std::string& system_msg,
    bool add_generation_prompt
) const {
    bool is_chatml = (
        pre_tokenizer_ == "llama" || pre_tokenizer_ == "smol" ||
        chat_template_.find("<|im_start|>") != std::string::npos ||
        encoder_.find("<|im_start|>") != encoder_.end()
    );
    bool is_gemma = (
        pre_tokenizer_ == "gemma" ||
        chat_template_.find("<start_of_turn>") != std::string::npos ||
        encoder_.find("<start_of_turn>") != encoder_.end()
    );
    bool is_llama3 = (
        chat_template_.find("<|start_header_id|>") != std::string::npos ||
        encoder_.find("<|start_header_id|>") != encoder_.end()
    );

    std::string formatted;
    if (is_chatml) {
        if (!system_msg.empty()) {
            formatted += "<|im_start|>system\n" + system_msg + "<|im_end|>\n";
        }
        formatted += "<|im_start|>user\n" + user_msg + "<|im_end|>\n";
        if (add_generation_prompt) {
            formatted += "<|im_start|>assistant\n";
        }
    } else if (is_gemma) {
        if (!system_msg.empty()) {
            formatted += "<start_of_turn>system\n" + system_msg + "<end_of_turn>\n";
        }
        formatted += "<start_of_turn>user\n" + user_msg + "<end_of_turn>\n";
        if (add_generation_prompt) {
            formatted += "<start_of_turn>model\n";
        }
    } else if (is_llama3) {
        formatted += "<|begin_of_text|>";
        if (!system_msg.empty()) {
            formatted += "<|start_header_id|>system<|end_header_id|>\n\n" + system_msg + "<|eot_id|>";
        }
        formatted += "<|start_header_id|>user<|end_header_id|>\n\n" + user_msg + "<|eot_id|>";
        if (add_generation_prompt) {
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else {
        if (!system_msg.empty()) {
            formatted += "System: " + system_msg + "\n\n";
        }
        formatted += "User: " + user_msg + "\n\n";
        if (add_generation_prompt) {
            formatted += "Assistant: ";
        }
    }
    return formatted;
}

std::vector<std::string> BPETokenizer::bpe_clip_word(const std::string& word) const {
    if (word.empty()) return {};

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = word[i];
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        symbols.push_back(word.substr(i, len));
        i += len;
    }

    if (symbols.size() == 1) {
        symbols[0] += "</w>";
        return symbols;
    }

    symbols.back() += "</w>";

    while (symbols.size() > 1) {
        int min_rank = 100000000;
        int best_idx = -1;

        for (size_t j = 0; j < symbols.size() - 1; ++j) {
            auto it = bpe_ranks_.find({symbols[j], symbols[j + 1]});
            if (it != bpe_ranks_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(j);
                }
            }
        }

        if (best_idx == -1) break;

        std::vector<std::string> new_symbols;
        for (size_t j = 0; j < symbols.size(); ++j) {
            if (static_cast<int>(j) == best_idx) {
                new_symbols.push_back(symbols[j] + symbols[j + 1]);
                j++;
            } else {
                new_symbols.push_back(symbols[j]);
            }
        }
        symbols = std::move(new_symbols);
    }

    return symbols;
}

std::vector<std::string> BPETokenizer::bpe(const std::string& token) const {
    if (token.empty()) return {};

    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        unsigned char c = token[i];
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        word.push_back(token.substr(i, len));
        i += len;
    }

    if (word.size() <= 1) return word;

    while (word.size() > 1) {
        int min_rank = 100000000;
        int best_idx = -1;

        for (size_t j = 0; j < word.size() - 1; ++j) {
            auto it = bpe_ranks_.find({word[j], word[j + 1]});
            if (it != bpe_ranks_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(j);
                }
            }
        }

        if (best_idx == -1) break;

        std::vector<std::string> new_word;
        for (size_t j = 0; j < word.size(); ++j) {
            if (static_cast<int>(j) == best_idx) {
                new_word.push_back(word[j] + word[j + 1]);
                j++;
            } else {
                new_word.push_back(word[j]);
            }
        }
        word = std::move(new_word);
    }

    return word;
}

void BPETokenizer::encode_normal_text(const std::string& subtext, std::vector<int32_t>& out_tokens) const {
    if (subtext.empty()) return;

    if (pre_tokenizer_ == "clip") {
        std::string lower_text = subtext;
        for (char& c : lower_text) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        static const std::regex clip_pattern(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z0-9]+|[^\s\a-zA-Z0-9]+)");
        auto words_begin = std::sregex_iterator(lower_text.begin(), lower_text.end(), clip_pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::string match = it->str();
            if (match == "<|startoftext|>" || match == "<|endoftext|>") {
                continue;
            }
            std::vector<std::string> pieces = bpe_clip_word(match);
            for (const auto& piece : pieces) {
                auto enc_it = encoder_.find(piece);
                if (enc_it != encoder_.end()) {
                    out_tokens.push_back(enc_it->second);
                } else {
                    out_tokens.push_back(unk_id_);
                }
            }
        }
    } else {
        // Standard GPT-2 / Llama byte encoder
        static const std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\a-zA-Z0-9]+|\s+(?!\S)|\s+)");
        auto words_begin = std::sregex_iterator(subtext.begin(), subtext.end(), pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::string token_str = it->str();
            if (token_str.empty()) continue;

            // Byte encode token string
            std::string byte_encoded;
            for (unsigned char ch : token_str) {
                auto b_it = byte_to_unicode_.find(ch);
                if (b_it != byte_to_unicode_.end()) {
                    byte_encoded += b_it->second;
                } else {
                    byte_encoded += static_cast<char>(ch);
                }
            }

            // Apply BPE merges
            std::vector<std::string> bpe_pieces = bpe(byte_encoded);
            for (const auto& piece : bpe_pieces) {
                auto enc_it = encoder_.find(piece);
                if (enc_it != encoder_.end()) {
                    out_tokens.push_back(enc_it->second);
                } else {
                    out_tokens.push_back(unk_id_);
                }
            }
        }
    }
}

std::vector<int32_t> BPETokenizer::encode(
    const std::string& text,
    int max_length,
    bool add_special_tokens,
    bool pad_to_max
) const {
    std::vector<int32_t> bpe_tokens;

    if (add_special_tokens && bos_id_ >= 0) {
        bpe_tokens.push_back(bos_id_);
    }

    // Split text by special tokens if any exist
    if (!special_tokens_.empty()) {
        size_t pos = 0;
        size_t n = text.size();
        std::string normal_buf;

        while (pos < n) {
            bool matched_special = false;
            for (const auto& sp : special_tokens_) {
                if (pos + sp.size() <= n && text.compare(pos, sp.size(), sp) == 0) {
                    // Flush accumulated normal text
                    if (!normal_buf.empty()) {
                        encode_normal_text(normal_buf, bpe_tokens);
                        normal_buf.clear();
                    }
                    // Emit special token
                    auto it = encoder_.find(sp);
                    if (it != encoder_.end()) {
                        bpe_tokens.push_back(it->second);
                    }
                    pos += sp.size();
                    matched_special = true;
                    break;
                }
            }
            if (!matched_special) {
                normal_buf.push_back(text[pos]);
                pos++;
            }
        }
        if (!normal_buf.empty()) {
            encode_normal_text(normal_buf, bpe_tokens);
        }
    } else {
        encode_normal_text(text, bpe_tokens);
    }

    // Only append EOS for CLIP or BERT sequence tasks, not for Causal LM continuation
    if (add_special_tokens && eos_id_ >= 0) {
        if (pre_tokenizer_ == "clip" || pre_tokenizer_ == "bert") {
            bpe_tokens.push_back(eos_id_);
        }
    }

    // Truncate or pad if max_length is explicitly positive
    if (max_length > 0) {
        if (static_cast<int>(bpe_tokens.size()) > max_length) {
            bpe_tokens.resize(max_length);
            if (add_special_tokens && eos_id_ >= 0 && (pre_tokenizer_ == "clip" || pre_tokenizer_ == "bert")) {
                bpe_tokens[max_length - 1] = eos_id_;
            }
        } else if (pad_to_max) {
            while (static_cast<int>(bpe_tokens.size()) < max_length) {
                bpe_tokens.push_back(pad_id_);
            }
        }
    }

    return bpe_tokens;
}

bool BPETokenizer::is_special_token(int32_t id) const {
    if (id < 0) return true;
    if (id == bos_id_ || id == eos_id_ || id == pad_id_) return true;
    auto it = decoder_.find(id);
    if (it != decoder_.end()) {
        const std::string& s = it->second;
        if (s.rfind("<|", 0) == 0 && s.length() >= 4 && s.compare(s.length() - 2, 2, "|>") == 0) {
            return true;
        }
        if (s == "<start_of_turn>" || s == "<end_of_turn>") {
            return true;
        }
        if (s.rfind("<|start_header_id|>", 0) == 0 || s.rfind("<|end_header_id|>", 0) == 0 || s.rfind("<|eot_id|>", 0) == 0) {
            return true;
        }
    }
    return false;
}

std::string BPETokenizer::decode_token(int32_t id, bool skip_special_tokens) const {
    if (skip_special_tokens && is_special_token(id)) {
        return "";
    }
    return decode({id}, skip_special_tokens);
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string text;
    for (int32_t id : ids) {
        if (skip_special_tokens && is_special_token(id)) {
            continue;
        }
        auto it = decoder_.find(id);
        if (it != decoder_.end()) {
            text += it->second;
        }
    }

    // Convert unicode byte characters back to standard bytes
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        std::string utf8_char = text.substr(i, len);
        auto u_it = unicode_to_byte_.find(utf8_char);
        if (u_it != unicode_to_byte_.end()) {
            result.push_back(static_cast<char>(u_it->second));
        } else {
            result += utf8_char;
        }
        i += len;
    }

    // Replace SentencePiece space characters (U+2581: \xe2\x96\x81) with standard spaces
    std::string sp_marker = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = result.find(sp_marker, pos)) != std::string::npos) {
        result.replace(pos, sp_marker.length(), " ");
        pos += 1;
    }

    return result;
}

} // namespace pipeline
} // namespace ggmlc
