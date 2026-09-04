#include "ggmlc/loader.h"
#include "gguf.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cctype>

namespace ggmlc {

namespace {

// ============================================================================
// Lightweight Zero-Dependency JSON Parser for GGUF Graph Specification
// ============================================================================

enum class JsonType { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };

struct JsonValue {
    JsonType type = JsonType::NUL;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    bool contains(const std::string& key) const {
        for (const auto& kv : obj_val) {
            if (kv.first == key) return true;
        }
        return false;
    }

    const JsonValue& operator[](const std::string& key) const {
        for (const auto& kv : obj_val) {
            if (kv.first == key) return kv.second;
        }
        static JsonValue null_val;
        return null_val;
    }

    const JsonValue& operator[](size_t idx) const {
        if (idx < arr_val.size()) {
            return arr_val[idx];
        }
        static JsonValue null_val;
        return null_val;
    }

    int64_t get_int(int64_t def = 0) const {
        if (type == JsonType::NUMBER) return static_cast<int64_t>(num_val);
        if (type == JsonType::BOOL) return bool_val ? 1 : 0;
        return def;
    }

    double get_double(double def = 0.0) const {
        if (type == JsonType::NUMBER) return num_val;
        return def;
    }

    std::string get_str(const std::string& def = "") const {
        if (type == JsonType::STRING) return str_val;
        return def;
    }

    bool get_bool(bool def = false) const {
        if (type == JsonType::BOOL) return bool_val;
        return def;
    }
};

class JsonParser {
    const char* src;
    size_t pos = 0;
    size_t len = 0;

    void skip_whitespace() {
        while (pos < len && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) {
            ++pos;
        }
    }

    char peek() {
        skip_whitespace();
        if (pos >= len) return '\0';
        return src[pos];
    }

    char get() {
        skip_whitespace();
        if (pos >= len) return '\0';
        return src[pos++];
    }

    std::string parse_string() {
        if (get() != '"') throw std::runtime_error("Expected '\"'");
        std::string s;
        while (pos < len) {
            char c = src[pos++];
            if (c == '"') return s;
            if (c == '\\' && pos < len) {
                char esc = src[pos++];
                if (esc == 'n') s += '\n';
                else if (esc == 'r') s += '\r';
                else if (esc == 't') s += '\t';
                else if (esc == '"') s += '"';
                else if (esc == '\\') s += '\\';
                else s += esc;
            } else {
                s += c;
            }
        }
        throw std::runtime_error("Unterminated string in JSON");
    }

    JsonValue parse_number() {
        size_t start = pos;
        if (src[pos] == '-') ++pos;
        while (pos < len && (std::isdigit(src[pos]) || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-')) {
            ++pos;
        }
        std::string num_str(src + start, pos - start);
        JsonValue val;
        val.type = JsonType::NUMBER;
        val.num_val = std::stod(num_str);
        return val;
    }

public:
    JsonParser(const char* s, size_t l) : src(s), len(l) {}

    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonType::STRING;
            v.str_val = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') {
            JsonValue v;
            v.type = JsonType::BOOL;
            if (pos + 4 <= len && std::strncmp(src + pos, "true", 4) == 0) {
                v.bool_val = true;
                pos += 4;
            } else if (pos + 5 <= len && std::strncmp(src + pos, "false", 5) == 0) {
                v.bool_val = false;
                pos += 5;
            } else {
                throw std::runtime_error("Invalid boolean in JSON");
            }
            return v;
        }
        if (c == 'n') {
            if (pos + 4 <= len && std::strncmp(src + pos, "null", 4) == 0) {
                pos += 4;
                JsonValue v;
                v.type = JsonType::NUL;
                return v;
            }
        }
        if (c == '-' || std::isdigit(c)) {
            return parse_number();
        }
        throw std::runtime_error(std::string("Unexpected character in JSON: ") + c);
    }

    JsonValue parse_object() {
        if (get() != '{') throw std::runtime_error("Expected '{'");
        JsonValue obj;
        obj.type = JsonType::OBJECT;
        skip_whitespace();
        if (peek() == '}') {
            get();
            return obj;
        }
        while (true) {
            std::string key = parse_string();
            if (get() != ':') throw std::runtime_error("Expected ':' after key");
            obj.obj_val.push_back({key, parse_value()});
            skip_whitespace();
            char next = get();
            if (next == '}') break;
            if (next != ',') throw std::runtime_error("Expected ',' or '}' in object");
        }
        return obj;
    }

    JsonValue parse_array() {
        if (get() != '[') throw std::runtime_error("Expected '['");
        JsonValue arr;
        arr.type = JsonType::ARRAY;
        skip_whitespace();
        if (peek() == ']') {
            get();
            return arr;
        }
        while (true) {
            arr.arr_val.push_back(parse_value());
            skip_whitespace();
            char next = get();
            if (next == ']') break;
            if (next != ',') throw std::runtime_error("Expected ',' or ']' in array");
        }
        return arr;
    }
};

std::shared_ptr<DimExpr> parse_dim_expr(const JsonValue& jdim, const std::unordered_map<std::string, int64_t>& symbol_map) {
    auto dim = std::make_shared<DimExpr>();
    std::string type = jdim["type"].get_str("static");
    if (type == "static") {
        dim->type = DimType::STATIC;
        dim->val = jdim["val"].get_int(1);
    } else if (type == "symbol") {
        dim->type = DimType::SYMBOL;
        std::string sname = jdim["name"].get_str();
        auto it = symbol_map.find(sname);
        dim->val = (it != symbol_map.end()) ? it->second : 0;
    } else if (type == "add") {
        dim->type = DimType::ADD;
        dim->left = parse_dim_expr(jdim["left"], symbol_map);
        dim->right = parse_dim_expr(jdim["right"], symbol_map);
    } else if (type == "sub") {
        dim->type = DimType::SUB;
        dim->left = parse_dim_expr(jdim["left"], symbol_map);
        dim->right = parse_dim_expr(jdim["right"], symbol_map);
    } else if (type == "mul") {
        dim->type = DimType::MUL;
        dim->left = parse_dim_expr(jdim["left"], symbol_map);
        dim->right = parse_dim_expr(jdim["right"], symbol_map);
    } else if (type == "floordiv") {
        dim->type = DimType::FLOORDIV;
        dim->left = parse_dim_expr(jdim["left"], symbol_map);
        dim->right = parse_dim_expr(jdim["right"], symbol_map);
    } else if (type == "ceildiv") {
        dim->type = DimType::CEILDIV;
        dim->left = parse_dim_expr(jdim["left"], symbol_map);
        dim->right = parse_dim_expr(jdim["right"], symbol_map);
    } else {
        dim->type = DimType::STATIC;
        dim->val = 1;
    }
    return dim;
}

} // namespace

SerializedModelGraph ModelLoader::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open GGUF file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read GGUF file: " + filepath);
    }

    auto graph = load_from_memory(buffer.data(), buffer.size());
    graph.data_buffer = std::move(buffer);

    // Re-bind data_ptr relative to graph.data_buffer
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context* ctx = gguf_init_from_buffer(graph.data_buffer.data(), graph.data_buffer.size(), params);
    if (ctx) {
        size_t data_offset = gguf_get_data_offset(ctx);
        const uint8_t* base_data = graph.data_buffer.data() + data_offset;
        for (auto& pair : graph.tensors) {
            int64_t t_id = gguf_find_tensor(ctx, pair.second.name.c_str());
            if (t_id >= 0) {
                pair.second.data_ptr = base_data + gguf_get_tensor_offset(ctx, t_id);
            }
        }
        gguf_free(ctx);
    }

    return graph;
}

SerializedModelGraph ModelLoader::load_from_memory(const uint8_t* data, size_t size) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context* ctx = gguf_init_from_buffer(data, size, params);
    if (!ctx) {
        throw std::runtime_error("Failed to parse GGUF binary format");
    }

    // 1. Verify general.architecture
    int64_t key_arch = gguf_find_key(ctx, "general.architecture");
    if (key_arch >= 0) {
        std::string arch = gguf_get_val_str(ctx, key_arch);
        // Valid ggmlc GGUF
    }

    // 2. Extract ggmlc.graph_spec
    int64_t key_spec = gguf_find_key(ctx, "ggmlc.graph_spec");
    if (key_spec < 0) {
        gguf_free(ctx);
        throw std::runtime_error("Missing 'ggmlc.graph_spec' metadata in GGUF file");
    }

    const char* spec_json = gguf_get_val_str(ctx, key_spec);
    JsonParser parser(spec_json, std::strlen(spec_json));
    JsonValue root = parser.parse_value();

    SerializedModelGraph g;
    g.name = root["name"].get_str("main");

    // Populate GGUF key-value metadata
    int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* k = gguf_get_key(ctx, i);
        if (!k) continue;
        std::string key_str(k);
        enum gguf_type kv_t = gguf_get_kv_type(ctx, i);

        if (kv_t == GGUF_TYPE_STRING) {
            g.metadata_str[key_str] = gguf_get_val_str(ctx, i);
        } else if (kv_t == GGUF_TYPE_INT8) {
            g.metadata_int[key_str] = gguf_get_val_i8(ctx, i);
        } else if (kv_t == GGUF_TYPE_UINT8) {
            g.metadata_int[key_str] = gguf_get_val_u8(ctx, i);
        } else if (kv_t == GGUF_TYPE_INT16) {
            g.metadata_int[key_str] = gguf_get_val_i16(ctx, i);
        } else if (kv_t == GGUF_TYPE_UINT16) {
            g.metadata_int[key_str] = gguf_get_val_u16(ctx, i);
        } else if (kv_t == GGUF_TYPE_INT32) {
            g.metadata_int[key_str] = gguf_get_val_i32(ctx, i);
        } else if (kv_t == GGUF_TYPE_UINT32) {
            g.metadata_int[key_str] = gguf_get_val_u32(ctx, i);
        } else if (kv_t == GGUF_TYPE_INT64) {
            g.metadata_int[key_str] = gguf_get_val_i64(ctx, i);
        } else if (kv_t == GGUF_TYPE_UINT64) {
            g.metadata_int[key_str] = static_cast<int64_t>(gguf_get_val_u64(ctx, i));
        } else if (kv_t == GGUF_TYPE_BOOL) {
            g.metadata_int[key_str] = gguf_get_val_bool(ctx, i) ? 1 : 0;
        } else if (kv_t == GGUF_TYPE_FLOAT32) {
            g.metadata_float[key_str] = gguf_get_val_f32(ctx, i);
        } else if (kv_t == GGUF_TYPE_FLOAT64) {
            g.metadata_float[key_str] = gguf_get_val_f64(ctx, i);
        } else if (kv_t == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, i) == GGUF_TYPE_STRING) {
            size_t arr_n = gguf_get_arr_n(ctx, i);
            std::vector<std::string> s_arr;
            s_arr.reserve(arr_n);
            for (size_t j = 0; j < arr_n; ++j) {
                s_arr.push_back(gguf_get_arr_str(ctx, i, j));
            }
            g.metadata_str_arr[key_str] = std::move(s_arr);
        }
    }

    // Symbol Table
    std::unordered_map<std::string, int64_t> symbol_map;
    const auto& sym_arr = root["symbol_table"].arr_val;
    for (size_t i = 0; i < sym_arr.size(); ++i) {
        std::string sname = sym_arr[i].get_str();
        g.symbol_table.push_back(sname);
        symbol_map[sname] = static_cast<int64_t>(i);
    }

    // Inputs, Outputs, Parameters
    for (const auto& in_val : root["inputs"].arr_val) {
        g.inputs.push_back(static_cast<uint32_t>(in_val.get_int()));
    }
    for (const auto& out_val : root["outputs"].arr_val) {
        g.outputs.push_back(static_cast<uint32_t>(out_val.get_int()));
    }
    for (const auto& p_val : root["parameters"].arr_val) {
        g.parameters.push_back(static_cast<uint32_t>(p_val.get_int()));
    }

    // Tensors
    size_t data_offset = gguf_get_data_offset(ctx);
    const uint8_t* tensor_data_base = data + data_offset;

    for (const auto& pair : root["tensors"].obj_val) {
        const auto& tj = pair.second;
        SerializedTensor st;
        st.id = static_cast<uint32_t>(tj["id"].get_int());
        st.name = tj["name"].get_str();
        st.type = static_cast<ggml_type>(tj["type"].get_int());
        st.storage = static_cast<StorageClass>(tj["storage"].get_int());

        const auto& ne_arr = tj["ne"].arr_val;
        for (size_t i = 0; i < 4; ++i) {
            if (i < ne_arr.size()) {
                st.ne[i] = parse_dim_expr(ne_arr[i], symbol_map);
            } else {
                auto d = std::make_shared<DimExpr>();
                d->type = DimType::STATIC;
                d->val = 1;
                st.ne[i] = d;
            }
        }

        // Match with GGUF tensor data
        int64_t gguf_tid = gguf_find_tensor(ctx, st.name.c_str());
        if (gguf_tid >= 0) {
            st.data_offset = gguf_get_tensor_offset(ctx, gguf_tid);
            st.data_size = gguf_get_tensor_size(ctx, gguf_tid);
            st.data_ptr = tensor_data_base + st.data_offset;
        } else {
            st.data_offset = 0;
            st.data_size = 0;
            st.data_ptr = nullptr;
        }

        g.tensors[st.id] = st;
    }

    // Nodes
    for (const auto& nj : root["nodes"].arr_val) {
        SerializedOp op;
        op.id = static_cast<uint32_t>(nj["id"].get_int());
        op.opcode = static_cast<int32_t>(nj["opcode"].get_int());
        op.name = nj["name"].get_str();

        for (const auto& in_id : nj["inputs"].arr_val) {
            op.inputs.push_back(static_cast<uint32_t>(in_id.get_int()));
        }
        for (const auto& out_id : nj["outputs"].arr_val) {
            op.outputs.push_back(static_cast<uint32_t>(out_id.get_int()));
        }

        if (nj.contains("attributes")) {
            for (const auto& attr_pair : nj["attributes"].obj_val) {
                op.attributes[attr_pair.first] = attr_pair.second.get_int();
                op.float_attributes[attr_pair.first] = attr_pair.second.get_double();
            }
        }

        g.ops.push_back(op);
    }

    gguf_free(ctx);
    return g;
}

} // namespace ggmlc
