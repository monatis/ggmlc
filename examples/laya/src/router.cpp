#include "router.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace laya {
namespace fs = std::filesystem;

static const char* kTypedIds[][5] = {
    {"action", "needs_review", "outcome", "risk", "urgency"},
    {"action", "category", "churn_risk", "needs_human", "urgency"},
    {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
};

static bool matches_typed_workflow(const std::vector<Question>& questions) {
    if (questions.size() != 5) return false;
    std::vector<std::string> ids;
    ids.reserve(questions.size());
    for (const auto& q : questions) ids.push_back(q.id);
    std::sort(ids.begin(), ids.end());
    for (const auto& sig : kTypedIds) {
        std::vector<std::string> want(sig, sig + 5);
        std::sort(want.begin(), want.end());
        if (ids == want) return true;
    }
    return false;
}

static std::string family_from_graph(const ggmlc::SerializedModelGraph& g, const std::string& path) {
    auto meta = [&](const std::string& key) -> std::string {
        auto it = g.metadata_str.find(key);
        return it == g.metadata_str.end() ? std::string() : it->second;
    };
    std::string fam = meta("laya.family");
    if (!fam.empty()) return fam;
    return infer_family(path, meta("laya.model_name"), meta("laya.checkpoint"));
}

bool DecisionRouter::consider_gguf(const std::string& path) {
    try {
        auto graph = ggmlc::ModelLoader::load_from_file(path);
        auto meta = [&](const std::string& key) -> std::string {
            auto it = graph.metadata_str.find(key);
            return it == graph.metadata_str.end() ? std::string() : it->second;
        };
        const std::string tagged = meta("laya.family") + meta("laya.model_name") + meta("laya.checkpoint");
        const std::string lower_path = path;
        const bool name_laya = lower_path.find("laya") != std::string::npos ||
                               lower_path.find("Laya") != std::string::npos;
        if (tagged.empty() && !name_laya) {
            return false;
        }
        const std::string fam = family_from_graph(graph, path);
        auto it = paths_.find(fam);
        if (it == paths_.end() || quant_rank(path) > quant_rank(it->second)) {
            paths_[fam] = path;
            std::cerr << "[laya] catalog " << fam << " <- " << path << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[laya] skip " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool DecisionRouter::load_file(const std::string& gguf_path) {
    if (!consider_gguf(gguf_path)) return false;
    if (forced_family_ == "auto" && paths_.size() == 1) {
        forced_family_ = paths_.begin()->first;
    }
    return true;
}

bool DecisionRouter::load_dir(const std::string& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        std::cerr << "[laya] models dir not found: " << dir << std::endl;
        return false;
    }
    for (const auto& ent : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!ent.is_regular_file()) continue;
        auto p = ent.path();
        auto ext = p.extension().string();
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext != ".gguf") continue;
        consider_gguf(p.string());
    }
    if (paths_.empty()) {
        std::cerr << "[laya] no GGUF files in " << dir << std::endl;
        return false;
    }
    return true;
}

std::vector<std::string> DecisionRouter::discovered_families() const {
    std::vector<std::string> out;
    for (const auto& kv : paths_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::string DecisionRouter::device() const {
    if (!engines_.empty()) return engines_.begin()->second->device();
    return opt_.device;
}

DecisionEngine& DecisionRouter::ensure(const std::string& family) {
    auto eit = engines_.find(family);
    if (eit != engines_.end()) return *eit->second;
    auto pit = paths_.find(family);
    if (pit == paths_.end()) {
        throw std::runtime_error("no GGUF catalogued for family '" + family + "'");
    }
    auto eng = std::make_unique<DecisionEngine>();
    if (!eng->load_model(pit->second, opt_)) {
        throw std::runtime_error("failed to load " + pit->second);
    }
    DecisionEngine& ref = *eng;
    engines_[family] = std::move(eng);
    return ref;
}

DecisionEngine& DecisionRouter::engine() {
    if (!engines_.empty()) return *engines_.begin()->second;
    if (paths_.empty()) throw std::runtime_error("no models loaded");
    return ensure(paths_.begin()->first);
}

std::string DecisionRouter::choose_family(const JsonValue& state, const std::vector<Question>& questions) const {
    std::string want = forced_family_;
    for (char& c : want) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (want == "en") want = "english";
    if (want == "multi" || want == "ml") want = "multilingual";
    if (want == "typed" || want == "typed_decisions") want = "typed-decisions";

    if (want != "auto" && !want.empty()) return want;

    if (paths_.count("typed-decisions") && matches_typed_workflow(questions)) {
        return "typed-decisions";
    }
    LangGuess g = guess_language(state);
    if (g.is_english) {
        if (paths_.count("english")) return "english";
        if (paths_.count("typed-decisions")) return "typed-decisions";
        if (paths_.count("multilingual")) return "multilingual";
    } else {
        if (paths_.count("multilingual")) return "multilingual";
        if (paths_.count("english")) return "english";
        if (paths_.count("typed-decisions")) return "typed-decisions";
    }
    return paths_.begin()->first;
}

DecisionEngine& DecisionRouter::pick(const JsonValue& state, const std::vector<Question>& questions) {
    const std::string fam = choose_family(state, questions);
    LangGuess g = guess_language(state);
    last_.family = fam;
    last_.path = paths_.count(fam) ? paths_.at(fam) : "";
    last_.reason = (forced_family_ != "auto" && !forced_family_.empty())
                       ? ("forced " + fam)
                       : g.reason;
    return ensure(fam);
}

DecideResult DecisionRouter::decide(const JsonValue& state, const std::vector<Question>& questions) {
    DecisionEngine& eng = pick(state, questions);
    DecideResult r = eng.decide(state, questions);
    r.route_family = last_.family;
    r.route_reason = last_.reason;
    if (r.model.empty()) r.model = last_.family;
    return r;
}

void DecisionRouter::print_info() {
    std::cout << "router families:\n";
    for (const auto& kv : paths_) {
        const bool loaded = engines_.count(kv.first) > 0;
        std::cout << "  " << kv.first << (loaded ? "  [loaded]  " : "            ") << kv.second << "\n";
    }
    std::string fam = forced_family_;
    for (char& c : fam) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (fam.empty() || fam == "auto") fam = paths_.begin()->first;
    if (fam == "en") fam = "english";
    if (fam == "multi" || fam == "ml") fam = "multilingual";
    try {
        ensure(fam).print_info();
    } catch (const std::exception& e) {
        std::cerr << "[laya] info load failed: " << e.what() << std::endl;
    }
}

JsonValue DecisionRouter::health_json() const {
    JsonValue o = JsonValue::object();
    o.set("status", JsonValue::string("ok"));
    o.set("device", JsonValue::string(device()));
    JsonValue fams = JsonValue::array();
    for (const auto& f : discovered_families()) fams.arr.push_back(JsonValue::string(f));
    o.set("families", fams);
    o.set("family", JsonValue::string(forced_family_));
    if (!last_.family.empty()) {
        o.set("last_route", JsonValue::string(last_.family));
        o.set("last_reason", JsonValue::string(last_.reason));
    }
    return o;
}

}  // namespace laya
