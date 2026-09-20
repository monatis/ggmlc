#pragma once

#include <string>
#include <vector>
#include <utility>
#include "questions.h"

namespace laya {

struct Preset {
    std::string name;
    std::string title;
    std::string blurb;
    std::string state_key;  // primary freeform field: body / prompt / post / request / message
    JsonValue default_state;
    std::vector<Question> questions;
};

const std::vector<Preset>& all_presets();
const Preset* find_preset(const std::string& name);
std::vector<std::string> preset_names();

// Inject --text into the preset's primary state field (or wrap as a string state).
JsonValue apply_text_to_state(const Preset* preset, const JsonValue& state, const std::string& text);

}  // namespace laya
