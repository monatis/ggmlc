#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace laya {

enum class JsonKind { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonKind kind = JsonKind::Null;
    bool b = false;
    double n = 0.0;
    std::string s;
    std::vector<JsonValue> arr;
    // Insertion-ordered object (Python json.dumps key order).
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue null() { return {}; }
    static JsonValue boolean(bool v) {
        JsonValue j;
        j.kind = JsonKind::Bool;
        j.b = v;
        return j;
    }
    static JsonValue number(double v) {
        JsonValue j;
        j.kind = JsonKind::Number;
        j.n = v;
        return j;
    }
    static JsonValue string(std::string v) {
        JsonValue j;
        j.kind = JsonKind::String;
        j.s = std::move(v);
        return j;
    }
    static JsonValue array(std::vector<JsonValue> v = {}) {
        JsonValue j;
        j.kind = JsonKind::Array;
        j.arr = std::move(v);
        return j;
    }
    static JsonValue object(std::vector<std::pair<std::string, JsonValue>> v = {}) {
        JsonValue j;
        j.kind = JsonKind::Object;
        j.obj = std::move(v);
        return j;
    }

    bool is_null() const { return kind == JsonKind::Null; }
    bool is_bool() const { return kind == JsonKind::Bool; }
    bool is_number() const { return kind == JsonKind::Number; }
    bool is_string() const { return kind == JsonKind::String; }
    bool is_array() const { return kind == JsonKind::Array; }
    bool is_object() const { return kind == JsonKind::Object; }

    const JsonValue* get(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    JsonValue* get_mut(const std::string& key) {
        for (auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    void set(const std::string& key, JsonValue v) {
        for (auto& kv : obj) {
            if (kv.first == key) {
                kv.second = std::move(v);
                return;
            }
        }
        obj.emplace_back(key, std::move(v));
        kind = JsonKind::Object;
    }
};

inline std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

inline std::string json_dumps(const JsonValue& v) {
    switch (v.kind) {
        case JsonKind::Null: return "null";
        case JsonKind::Bool: return v.b ? "true" : "false";
        case JsonKind::Number: {
            if (std::isfinite(v.n) && std::floor(v.n) == v.n && std::fabs(v.n) < 1e15) {
                return std::to_string(static_cast<long long>(v.n));
            }
            std::ostringstream oss;
            oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
            oss.precision(6);
            oss << v.n;
            return oss.str();
        }
        case JsonKind::String: return "\"" + json_escape(v.s) + "\"";
        case JsonKind::Array: {
            std::string out = "[";
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out += ", ";
                out += json_dumps(v.arr[i]);
            }
            out += "]";
            return out;
        }
        case JsonKind::Object: {
            std::string out = "{";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out += ", ";
                out += "\"" + json_escape(v.obj[i].first) + "\": " + json_dumps(v.obj[i].second);
            }
            out += "}";
            return out;
        }
    }
    return "null";
}

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : s_(src), i_(0) {}

    JsonValue parse() {
        skip();
        JsonValue v = parse_value();
        skip();
        return v;
    }

    static JsonValue parse_string(const std::string& src) {
        JsonParser p(src);
        return p.parse();
    }

private:
    const std::string& s_;
    size_t i_;

    void skip() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool eat(char c) {
        skip();
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }
    [[noreturn]] void fail(const char* msg) const {
        throw std::runtime_error(std::string("JSON parse error: ") + msg + " at " + std::to_string(i_));
    }

    JsonValue parse_value() {
        skip();
        if (i_ >= s_.size()) fail("unexpected end");
        char c = s_[i_];
        if (c == 'n') return parse_lit("null", JsonValue::null());
        if (c == 't') return parse_lit("true", JsonValue::boolean(true));
        if (c == 'f') return parse_lit("false", JsonValue::boolean(false));
        if (c == '"') return JsonValue::string(parse_str());
        if (c == '[') return parse_arr();
        if (c == '{') return parse_obj();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_num();
        fail("unexpected token");
    }

    JsonValue parse_lit(const char* lit, JsonValue v) {
        size_t n = std::char_traits<char>::length(lit);
        if (s_.compare(i_, n, lit) != 0) fail("bad literal");
        i_ += n;
        return v;
    }

    std::string parse_str() {
        if (!eat('"')) fail("expected string");
        std::string out;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i_ >= s_.size()) fail("bad escape");
                char e = s_[i_++];
                switch (e) {
                    case '"':
                    case '\\':
                    case '/': out.push_back(e); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) fail("bad unicode");
                        unsigned cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = s_[i_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp += h - '0';
                            else if (h >= 'a' && h <= 'f') cp += 10 + h - 'a';
                            else if (h >= 'A' && h <= 'F') cp += 10 + h - 'A';
                            else fail("bad hex");
                        }
                        if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
                        else if (cp <= 0x7FF) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_num() {
        size_t start = i_;
        if (s_[i_] == '-') ++i_;
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        return JsonValue::number(std::strtod(s_.substr(start, i_ - start).c_str(), nullptr));
    }

    JsonValue parse_arr() {
        if (!eat('[')) fail("expected [");
        JsonValue a = JsonValue::array();
        skip();
        if (eat(']')) return a;
        while (true) {
            a.arr.push_back(parse_value());
            skip();
            if (eat(']')) return a;
            if (!eat(',')) fail("expected , or ]");
        }
    }

    JsonValue parse_obj() {
        if (!eat('{')) fail("expected {");
        JsonValue o = JsonValue::object();
        skip();
        if (eat('}')) return o;
        while (true) {
            skip();
            std::string key = parse_str();
            if (!eat(':')) fail("expected :");
            o.obj.emplace_back(std::move(key), parse_value());
            skip();
            if (eat('}')) return o;
            if (!eat(',')) fail("expected , or }");
        }
    }
};

inline std::string json_dumps_pretty(const JsonValue& v, int indent = 0) {
    const std::string pad(indent, ' ');
    const std::string pad2(indent + 2, ' ');
    switch (v.kind) {
        case JsonKind::Array: {
            if (v.arr.empty()) return "[]";
            std::string out = "[\n";
            for (size_t i = 0; i < v.arr.size(); ++i) {
                out += pad2 + json_dumps_pretty(v.arr[i], indent + 2);
                if (i + 1 < v.arr.size()) out += ",";
                out += "\n";
            }
            out += pad + "]";
            return out;
        }
        case JsonKind::Object: {
            if (v.obj.empty()) return "{}";
            std::string out = "{\n";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                out += pad2 + "\"" + json_escape(v.obj[i].first) + "\": " +
                       json_dumps_pretty(v.obj[i].second, indent + 2);
                if (i + 1 < v.obj.size()) out += ",";
                out += "\n";
            }
            out += pad + "}";
            return out;
        }
        default:
            return json_dumps(v);
    }
}

}  // namespace laya
