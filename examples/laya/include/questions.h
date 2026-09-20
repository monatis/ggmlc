#pragma once

#include <string>
#include <vector>
#include <utility>
#include "json_util.h"

namespace laya {

enum class QType { Choice = 0, Score = 1, Noul = 2 };

inline const char* qtype_name(QType t) {
    switch (t) {
        case QType::Choice: return "choice";
        case QType::Score: return "score";
        case QType::Noul: return "noul";
    }
    return "choice";
}

inline QType qtype_from_name(const std::string& s) {
    if (s == "score") return QType::Score;
    if (s == "noul") return QType::Noul;
    return QType::Choice;
}

struct Question {
    std::string id;
    QType type = QType::Choice;
    std::string instructions;
    // Choice: insertion-ordered (label, description). Empty desc => label-only option text.
    // Score: labels are "0","1",... descriptions are level text.
    // Noul: optional "false"/"true" descriptions.
    std::vector<std::pair<std::string, std::string>> criteria;
};

struct Answer {
    std::string id;
    QType type = QType::Choice;
    std::vector<std::pair<std::string, float>> probabilities;
    std::string choice;     // choice winner
    float score = 0.0f;     // expected score
    float noul = 0.0f;      // p(true)
    float confidence = 0.0f;
    float act_probability = 0.0f;
    std::vector<std::string> legend;  // score levels
};

struct DecideResult {
    std::string model = "laya";
    std::string route_family;
    std::string route_reason;
    std::vector<Answer> answers;
    int input_tokens = 0;
    double latency_ms = 0.0;
    int seq_bucket = 0;
    int batch_bucket = 0;
    int n_forwards = 0;
};

std::string serialize_state(const JsonValue& state);
std::string render_criterion(const std::string& value);
std::vector<std::string> render_options(const Question& q);

Question question_from_json(const std::string& id, const JsonValue& qdef);
std::vector<Question> questions_from_json(const JsonValue& obj);
JsonValue questions_to_json(const std::vector<Question>& qs);

std::string format_answer_json(const DecideResult& result, bool pretty = true);
std::string format_answer_cli(const DecideResult& result);

float confidence_from_probs(const std::vector<float>& p);
std::string temp_bucket(QType t, int k);
std::vector<float> softmax_temp(const std::vector<float>& logits, float temperature);

}  // namespace laya
