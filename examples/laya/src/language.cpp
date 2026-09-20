#include "language.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace laya {

static uint32_t next_cp(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        ++i;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        const uint32_t cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
                            (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        const uint32_t cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
                            ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                            (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        const uint32_t cp = (static_cast<uint32_t>(c & 0x07) << 18) |
                            ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
                            ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
                            (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4;
        return cp;
    }
    ++i;
    return c;
}

static bool in_range(uint32_t cp, uint32_t lo, uint32_t hi) { return cp >= lo && cp <= hi; }

static const char* script_of(uint32_t cp) {
    if (cp < 0x0250 || in_range(cp, 0x1E00, 0x1EFF)) return "latin";
    if (in_range(cp, 0x0370, 0x03FF) || in_range(cp, 0x1F00, 0x1FFF)) return "greek";
    if (in_range(cp, 0x0400, 0x052F) || in_range(cp, 0x2DE0, 0x2DFF) || in_range(cp, 0xA640, 0xA69F))
        return "cyrillic";
    if (in_range(cp, 0x0590, 0x05FF)) return "hebrew";
    if (in_range(cp, 0x0600, 0x06FF) || in_range(cp, 0x0750, 0x077F) || in_range(cp, 0x08A0, 0x08FF) ||
        in_range(cp, 0xFB50, 0xFDFF) || in_range(cp, 0xFE70, 0xFEFF))
        return "arabic";
    if (in_range(cp, 0x0900, 0x097F) || in_range(cp, 0xA8E0, 0xA8FF)) return "devanagari";
    if (in_range(cp, 0x0980, 0x09FF)) return "bengali";
    if (in_range(cp, 0x0A00, 0x0A7F)) return "gurmukhi";
    if (in_range(cp, 0x0A80, 0x0AFF)) return "gujarati";
    if (in_range(cp, 0x0B00, 0x0B7F)) return "oriya";
    if (in_range(cp, 0x0B80, 0x0BFF)) return "tamil";
    if (in_range(cp, 0x0C00, 0x0C7F)) return "telugu";
    if (in_range(cp, 0x0C80, 0x0CFF)) return "kannada";
    if (in_range(cp, 0x0D00, 0x0D7F)) return "malayalam";
    if (in_range(cp, 0x0D80, 0x0DFF)) return "sinhala";
    if (in_range(cp, 0x0E00, 0x0E7F)) return "thai";
    if (in_range(cp, 0x0E80, 0x0EFF)) return "lao";
    if (in_range(cp, 0x0F00, 0x0FFF)) return "tibetan";
    if (in_range(cp, 0x1000, 0x109F)) return "myanmar";
    if (in_range(cp, 0x10A0, 0x10FF)) return "georgian";
    if (in_range(cp, 0x1200, 0x137F)) return "ethiopic";
    if (in_range(cp, 0x1780, 0x17FF)) return "khmer";
    if (in_range(cp, 0x1100, 0x11FF) || in_range(cp, 0x3130, 0x318F) || in_range(cp, 0xAC00, 0xD7AF))
        return "hangul";
    if (in_range(cp, 0x3040, 0x309F) || in_range(cp, 0x30A0, 0x30FF) || in_range(cp, 0x31F0, 0x31FF))
        return "kana";
    if (in_range(cp, 0x3400, 0x4DBF) || in_range(cp, 0x4E00, 0x9FFF) || in_range(cp, 0xF900, 0xFAFF))
        return "han";
    return nullptr;
}

static bool is_alpha_cp(uint32_t cp) {
    if (cp < 128) return std::isalpha(static_cast<int>(cp)) != 0;
    return script_of(cp) != nullptr || in_range(cp, 0x00C0, 0x024F);
}

static const std::unordered_set<std::string>& english_words() {
    static const std::unordered_set<std::string> w = {
        "the",    "and",    "is",     "are",    "was",    "were",   "to",     "of",     "in",
        "for",    "with",   "that",   "this",   "it",     "you",    "have",   "has",    "not",
        "but",    "on",     "at",     "be",     "as",     "from",   "will",   "can",    "would",
        "there",  "their",  "what",   "which",  "please", "we",     "i",      "a",      "an",
        "or",     "if",     "my",     "your",   "our",    "they",   "them",   "he",     "she",
        "his",    "her",    "me",     "us",     "do",     "does",   "did",    "done",   "been",
        "being",  "am",     "so",     "than",   "then",   "when",   "who",    "how",    "why",
        "about",  "into",   "over",   "after",  "before", "because","could",  "should", "might",
        "must",   "need",   "want",   "just",   "also",   "only",   "more",   "most",   "some",
        "any",    "no",     "yes",    "here",   "now",    "today",  "please", "thanks", "thank",
        "hello",  "hi",     "please", "refund", "invoice","account","charged","please",
    };
    return w;
}

static void collect_strings(const JsonValue& v, std::vector<std::string>& out, int depth) {
    if (depth > 6) return;
    if (v.is_string()) {
        out.push_back(v.s);
        return;
    }
    if (v.is_object()) {
        for (const auto& kv : v.obj) collect_strings(kv.second, out, depth + 1);
        return;
    }
    if (v.is_array()) {
        for (const auto& item : v.arr) collect_strings(item, out, depth + 1);
    }
}

std::string collect_state_text(const JsonValue& state, int max_chars) {
    std::vector<std::string> parts;
    collect_strings(state, parts, 0);
    std::string out;
    for (const auto& p : parts) {
        if (!out.empty()) out.push_back(' ');
        out += p;
        if (static_cast<int>(out.size()) >= max_chars) break;
    }
    if (static_cast<int>(out.size()) > max_chars) out.resize(static_cast<size_t>(max_chars));
    return out;
}

LangGuess guess_language(const std::string& text) {
    LangGuess g;
    std::unordered_map<std::string, int> scripts;
    int latin = 0;
    size_t i = 0;
    while (i < text.size()) {
        const uint32_t cp = next_cp(text, i);
        if (!is_alpha_cp(cp)) continue;
        const char* sc = script_of(cp);
        if (!sc || std::string(sc) == "latin") {
            ++latin;
        } else {
            scripts[sc] += 1;
        }
    }
    scripts["latin"] = latin;
    int total = 0;
    for (const auto& kv : scripts) total += kv.second;
    if (total == 0) {
        g.script = "unknown";
        g.is_english = true;
        g.reason = "no letters; default english";
        return g;
    }
    auto best = std::max_element(scripts.begin(), scripts.end(),
                                 [](const auto& a, const auto& b) { return a.second < b.second; });
    g.script = best->first;

    if (g.script != "latin") {
        g.is_english = false;
        g.reason = "dominant script is " + g.script;
        return g;
    }

    std::string word;
    const auto& en = english_words();
    auto flush = [&]() {
        if (word.empty()) return;
        ++g.words;
        if (en.count(word)) ++g.english_hits;
        word.clear();
    };
    i = 0;
    while (i < text.size()) {
        const uint32_t cp = next_cp(text, i);
        if (cp < 128 && std::isalpha(static_cast<int>(cp))) {
            word.push_back(static_cast<char>(std::tolower(static_cast<int>(cp))));
        } else {
            flush();
        }
    }
    flush();

    g.english_ratio = g.words ? (static_cast<double>(g.english_hits) / g.words) : 0.0;
    // Short Latin snippets (ticket ids, names) stay on the English checkpoint.
    if (g.words < 4) {
        g.is_english = true;
        g.reason = "short latin text; default english";
        return g;
    }
    // Confident English: several function-word hits, or a healthy ratio of them.
    if (g.english_hits >= 3 || g.english_ratio >= 0.12) {
        g.is_english = true;
        std::ostringstream oss;
        oss << "english words " << g.english_hits << "/" << g.words;
        g.reason = oss.str();
        return g;
    }
    g.is_english = false;
    std::ostringstream oss;
    oss << "latin but weak english " << g.english_hits << "/" << g.words;
    g.reason = oss.str();
    return g;
}

LangGuess guess_language(const JsonValue& state) { return guess_language(collect_state_text(state)); }

static std::string lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string infer_family(const std::string& path, const std::string& model_name, const std::string& checkpoint) {
    const std::string blob = lower_copy(path + " " + model_name + " " + checkpoint);
    if (blob.find("typed") != std::string::npos) return "typed-decisions";
    if (blob.find("multilingual") != std::string::npos || blob.find("mmbert") != std::string::npos)
        return "multilingual";
    if (blob.find("english") != std::string::npos) return "english";
    if (model_name == "laya" || checkpoint.find("convaiinnovations/laya") != std::string::npos) {
        if (checkpoint.find("multilingual") != std::string::npos) return "multilingual";
        if (checkpoint.find("typed") != std::string::npos) return "typed-decisions";
        return "english";
    }
    return "english";
}

int quant_rank(const std::string& path) {
    const std::string p = lower_copy(path);
    if (p.find("f32") != std::string::npos) return 40;
    if (p.find("f16") != std::string::npos || p.find("fp16") != std::string::npos) return 30;
    if (p.find("q8") != std::string::npos) return 20;
    if (p.find("q4") != std::string::npos || p.find("ud_q4") != std::string::npos) return 10;
    return 15;
}

}  // namespace laya
