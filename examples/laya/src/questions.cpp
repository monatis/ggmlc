#include "questions.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace laya {

std::string serialize_state(const JsonValue& state) {
    if (state.is_string()) return state.s;
    if (state.is_null()) return "";
    return json_dumps(state);
}

std::string render_criterion(const std::string& value) { return value; }

std::vector<std::string> render_options(const Question& q) {
    std::vector<std::string> opts;
    if (q.type == QType::Choice) {
        for (const auto& kv : q.criteria) {
            if (kv.second.empty()) opts.push_back(kv.first);
            else opts.push_back(kv.first + ": " + kv.second);
        }
        return opts;
    }
    if (q.type == QType::Score) {
        for (size_t i = 0; i < q.criteria.size(); ++i) {
            opts.push_back("level " + std::to_string(i) + ": " + q.criteria[i].second);
        }
        return opts;
    }
    std::string false_crit, true_crit;
    for (const auto& kv : q.criteria) {
        if (kv.first == "false") false_crit = kv.second;
        if (kv.first == "true") true_crit = kv.second;
    }
    opts.push_back(
        "false: " + (false_crit.empty() ? std::string("no, the statement does not hold") : false_crit)
    );
    opts.push_back(
        "true: " + (true_crit.empty() ? std::string("yes, the statement holds") : true_crit)
    );
    return opts;
}

Question question_from_json(const std::string& id, const JsonValue& qdef) {
    Question q;
    q.id = id;
    const JsonValue* t = qdef.get("type");
    if (t && t->is_string()) q.type = qtype_from_name(t->s);
    const JsonValue* ins = qdef.get("instructions");
    if (ins) {
        if (ins->is_string()) q.instructions = ins->s;
        else q.instructions = json_dumps(*ins);
    }
    const JsonValue* crit = qdef.get("criteria");
    if (crit && crit->is_object()) {
        for (const auto& kv : crit->obj) {
            if (kv.second.is_null()) q.criteria.emplace_back(kv.first, "");
            else if (kv.second.is_string()) q.criteria.emplace_back(kv.first, kv.second.s);
            else q.criteria.emplace_back(kv.first, json_dumps(kv.second));
        }
    } else if (crit && crit->is_array()) {
        if (q.type == QType::Choice) {
            for (const auto& item : crit->arr) {
                if (item.is_string()) q.criteria.emplace_back(item.s, "");
                else q.criteria.emplace_back(json_dumps(item), "");
            }
        } else {
            for (size_t i = 0; i < crit->arr.size(); ++i) {
                std::string lab = std::to_string(i);
                if (crit->arr[i].is_string()) q.criteria.emplace_back(lab, crit->arr[i].s);
                else if (crit->arr[i].is_null()) q.criteria.emplace_back(lab, "");
                else q.criteria.emplace_back(lab, json_dumps(crit->arr[i]));
            }
        }
    }
    return q;
}

std::vector<Question> questions_from_json(const JsonValue& obj) {
    std::vector<Question> out;
    if (obj.is_object()) {
        for (const auto& kv : obj.obj) {
            if (kv.second.is_object()) out.push_back(question_from_json(kv.first, kv.second));
        }
    }
    return out;
}

JsonValue questions_to_json(const std::vector<Question>& qs) {
    JsonValue o = JsonValue::object();
    for (const auto& q : qs) {
        JsonValue qd = JsonValue::object();
        qd.set("type", JsonValue::string(qtype_name(q.type)));
        qd.set("instructions", JsonValue::string(q.instructions));
        if (q.type == QType::Score) {
            JsonValue arr = JsonValue::array();
            for (const auto& kv : q.criteria) arr.arr.push_back(JsonValue::string(kv.second));
            qd.set("criteria", arr);
        } else if (q.type == QType::Choice) {
            JsonValue crit = JsonValue::object();
            for (const auto& kv : q.criteria) {
                if (kv.second.empty()) crit.set(kv.first, JsonValue::null());
                else crit.set(kv.first, JsonValue::string(kv.second));
            }
            qd.set("criteria", crit);
        } else if (!q.criteria.empty()) {
            JsonValue crit = JsonValue::object();
            for (const auto& kv : q.criteria) crit.set(kv.first, JsonValue::string(kv.second));
            qd.set("criteria", crit);
        }
        o.set(q.id, qd);
    }
    return o;
}

float confidence_from_probs(const std::vector<float>& p) {
    const int k = static_cast<int>(p.size());
    if (k < 2) return 1.0f;
    double ent = 0.0;
    for (float v : p) {
        double x = std::max(static_cast<double>(v), 1e-12);
        ent -= x * std::log(x);
    }
    double c = 1.0 - ent / std::log(static_cast<double>(k));
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    return static_cast<float>(c);
}

std::string temp_bucket(QType t, int k) {
    std::string size = (k <= 2) ? "2" : (k <= 5) ? "3-5" : (k <= 10) ? "6-10" : "11+";
    return std::string(qtype_name(t)) + ":" + size;
}

std::vector<float> softmax_temp(const std::vector<float>& logits, float temperature) {
    const float t = std::max(temperature, 1e-3f);
    std::vector<float> z = logits;
    float m = z.empty() ? 0.0f : z[0];
    for (float v : z) m = std::max(m, v);
    double sum = 0.0;
    for (float& v : z) {
        v = static_cast<float>(std::exp((v - m) / t));
        sum += v;
    }
    if (sum <= 0.0) sum = 1.0;
    for (float& v : z) v = static_cast<float>(v / sum);
    return z;
}

static float round4(float v) {
    return std::round(v * 10000.0f) / 10000.0f;
}

JsonValue answer_to_json(const Answer& a) {
    JsonValue o = JsonValue::object();
    o.set("type", JsonValue::string(qtype_name(a.type)));
    JsonValue probs = JsonValue::object();
    for (const auto& kv : a.probabilities) {
        probs.set(kv.first, JsonValue::number(round4(kv.second)));
    }
    JsonValue action = JsonValue::object();
    action.set("act_probability", JsonValue::number(round4(a.act_probability)));
    o.set("action", action);
    o.set("confidence", JsonValue::number(round4(a.confidence)));
    if (a.type == QType::Choice) {
        o.set("choice", JsonValue::string(a.choice));
        o.set("probabilities", probs);
    } else if (a.type == QType::Score) {
        o.set("score", JsonValue::number(round4(a.score)));
        JsonValue legend = JsonValue::object();
        for (size_t i = 0; i < a.legend.size(); ++i) {
            legend.set(std::to_string(i), JsonValue::string(a.legend[i]));
        }
        o.set("legend", legend);
        o.set("probabilities", probs);
    } else {
        o.set("noul", JsonValue::number(round4(a.noul)));
    }
    return o;
}

std::string format_answer_json(const DecideResult& result, bool pretty) {
    JsonValue root = JsonValue::object();
    root.set("model", JsonValue::string(result.model));
    JsonValue answers = JsonValue::object();
    for (const auto& a : result.answers) answers.set(a.id, answer_to_json(a));
    root.set("answers", answers);
    JsonValue usage = JsonValue::object();
    usage.set("input_tokens", JsonValue::number(static_cast<double>(result.input_tokens)));
    usage.set("output_tokens", JsonValue::number(0));
    usage.set("latency_ms", JsonValue::number(result.latency_ms));
    root.set("usage", usage);
    return pretty ? json_dumps_pretty(root) : json_dumps(root);
}

static std::string bar(float p, int width = 24) {
    int n = static_cast<int>(std::round(std::max(0.0f, std::min(1.0f, p)) * width));
    std::string s;
    for (int i = 0; i < width; ++i) s.push_back(i < n ? '#' : '.');
    return s;
}

std::string format_answer_cli(const DecideResult& result) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "model: " << result.model
        << "   tokens: " << result.input_tokens
        << "   latency: " << std::setprecision(1) << result.latency_ms << " ms\n";
    for (const auto& a : result.answers) {
        oss << "\n[" << a.id << "] " << qtype_name(a.type);
        oss << "   conf=" << std::setprecision(4) << a.confidence
            << "   act=" << a.act_probability << "\n";
        if (a.type == QType::Choice) {
            oss << "  choice: " << a.choice << "\n";
            for (const auto& kv : a.probabilities) {
                oss << "    " << bar(kv.second) << "  " << std::setw(6) << kv.second
                    << "  " << kv.first << "\n";
            }
        } else if (a.type == QType::Score) {
            oss << "  expected score: " << a.score << "\n";
            for (size_t i = 0; i < a.probabilities.size(); ++i) {
                std::string lab = a.probabilities[i].first;
                float p = a.probabilities[i].second;
                std::string desc = (i < a.legend.size()) ? a.legend[i] : "";
                oss << "    " << bar(p) << "  " << std::setw(6) << p
                    << "  level " << lab;
                if (!desc.empty()) oss << " - " << desc;
                oss << "\n";
            }
        } else {
            oss << "  noul (P(true)): " << a.noul << "\n";
            oss << "    " << bar(1.0f - a.noul) << "  false\n";
            oss << "    " << bar(a.noul) << "  true\n";
        }
    }
    return oss.str();
}

}  // namespace laya
