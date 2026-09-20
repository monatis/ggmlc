#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "engine.h"
#include "language.h"

namespace laya {

struct RouteInfo {
    std::string family;
    std::string reason;
    std::string path;
};

class DecisionRouter {
public:
    void set_options(const EngineOptions& opt) { opt_ = opt; }
    void set_forced_family(const std::string& family) { forced_family_ = family; }

    bool load_file(const std::string& gguf_path);
    bool load_dir(const std::string& dir);

    bool empty() const { return paths_.empty(); }
    std::vector<std::string> discovered_families() const;
    std::string device() const;

    DecisionEngine& engine();
    DecisionEngine& pick(const JsonValue& state, const std::vector<Question>& questions);
    DecideResult decide(const JsonValue& state, const std::vector<Question>& questions);

    RouteInfo last_route() const { return last_; }
    LangGuess detect(const JsonValue& state) const { return guess_language(state); }

    void print_info();
    JsonValue health_json() const;

private:
    EngineOptions opt_;
    std::string forced_family_ = "auto";
    std::unordered_map<std::string, std::string> paths_;
    std::unordered_map<std::string, std::unique_ptr<DecisionEngine>> engines_;
    RouteInfo last_;

    bool consider_gguf(const std::string& path);
    DecisionEngine& ensure(const std::string& family);
    std::string choose_family(const JsonValue& state, const std::vector<Question>& questions) const;
};

}  // namespace laya
