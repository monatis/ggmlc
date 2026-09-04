#include "ggmlc/pipeline/tokenizer.h"

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
    const std::string& pre_tokenizer
) {
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    pad_id_ = pad_id;
    unk_id_ = unk_id;
    pre_tokenizer_ = pre_tokenizer;

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
    if (!symbols.empty()) {
        symbols.back() += "</w>";
    }

    if (symbols.size() <= 1) return symbols;

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

    // Break token into individual unicode characters
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

        // Merge best pair
        std::vector<std::string> new_word;
        for (size_t j = 0; j < word.size(); ++j) {
            if (static_cast<int>(j) == best_idx) {
                new_word.push_back(word[j] + word[j + 1]);
                j++; // Skip next
            } else {
                new_word.push_back(word[j]);
            }
        }
        word = std::move(new_word);
    }

    return word;
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

    if (pre_tokenizer_ == "clip") {
        std::string lower_text = text;
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
                    bpe_tokens.push_back(enc_it->second);
                } else {
                    bpe_tokens.push_back(unk_id_);
                }
            }
        }
    } else {
        // Standard GPT-2 regex splitter and byte encoder
        static const std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\a-zA-Z0-9]+|\s+)");
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::string token_str = it->str();
            if (token_str.empty() || std::all_of(token_str.begin(), token_str.end(), ::isspace)) {
                continue;
            }

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
                    bpe_tokens.push_back(enc_it->second);
                } else {
                    bpe_tokens.push_back(unk_id_);
                }
            }
        }
    }

    if (add_special_tokens && eos_id_ >= 0) {
        bpe_tokens.push_back(eos_id_);
    }

    // Truncate or pad
    if (max_length > 0) {
        if (static_cast<int>(bpe_tokens.size()) > max_length) {
            bpe_tokens.resize(max_length);
            if (add_special_tokens && eos_id_ >= 0) {
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

std::string BPETokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string text;
    for (int32_t id : ids) {
        if (skip_special_tokens && (id == bos_id_ || id == eos_id_ || id == pad_id_)) {
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

    return result;
}

} // namespace pipeline
} // namespace ggmlc
