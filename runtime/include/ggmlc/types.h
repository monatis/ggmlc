#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include "ggml.h"

namespace ggmlc {

enum class DimType : uint8_t {
    STATIC = 0,
    SYMBOL = 1,
    ADD = 2,
    SUB = 3,
    MUL = 4,
    FLOORDIV = 5,
    CEILDIV = 6
};

struct DimExpr {
    DimType type;
    int64_t val = 0; // static value or symbol index
    std::shared_ptr<DimExpr> left = nullptr;
    std::shared_ptr<DimExpr> right = nullptr;

    int64_t evaluate(const std::unordered_map<std::string, int64_t>& env,
                     const std::vector<std::string>& symbol_table) const {
        switch (type) {
            case DimType::STATIC:
                return val;
            case DimType::SYMBOL: {
                if (val < 0 || val >= static_cast<int64_t>(symbol_table.size())) {
                    throw std::runtime_error("Symbol index out of bounds");
                }
                const std::string& name = symbol_table[val];
                auto it = env.find(name);
                if (it == env.end()) {
                    throw std::runtime_error("Symbol not found in environment: " + name);
                }
                return it->second;
            }
            case DimType::ADD:
                return left->evaluate(env, symbol_table) + right->evaluate(env, symbol_table);
            case DimType::SUB:
                return left->evaluate(env, symbol_table) - right->evaluate(env, symbol_table);
            case DimType::MUL:
                return left->evaluate(env, symbol_table) * right->evaluate(env, symbol_table);
            case DimType::FLOORDIV: {
                int64_t r = right->evaluate(env, symbol_table);
                if (r == 0) throw std::runtime_error("Division by zero in DimExpr");
                return left->evaluate(env, symbol_table) / r;
            }
            case DimType::CEILDIV: {
                int64_t r = right->evaluate(env, symbol_table);
                if (r == 0) throw std::runtime_error("Division by zero in DimExpr");
                int64_t l = left->evaluate(env, symbol_table);
                return (l + r - 1) / r;
            }
            default:
                return 1;
        }
    }
};

enum class StorageClass : int32_t {
    INPUT = 0,
    PARAMETER = 1,
    CONSTANT = 2,
    ACTIVATION = 3,
    STATE = 4,
    OUTPUT = 5
};

struct SerializedTensor {
    uint32_t id;
    std::string name;
    ggml_type type;
    std::array<std::shared_ptr<DimExpr>, 4> ne;
    StorageClass storage;
    uint64_t data_offset;
    uint64_t data_size;
    const uint8_t* data_ptr = nullptr;
};

struct SerializedOp {
    uint32_t id;
    int32_t opcode; // ggml_op
    std::string name;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::unordered_map<std::string, int64_t> attributes;
    std::unordered_map<std::string, double> float_attributes;
};

struct SerializedModelGraph {
    std::string name;
    std::vector<std::string> symbol_table;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<uint32_t> parameters;
    std::unordered_map<uint32_t, SerializedTensor> tensors;
    std::vector<SerializedOp> ops;
    std::vector<uint8_t> data_buffer;

    // Metadata key-values
    std::unordered_map<std::string, std::string> metadata_str;
    std::unordered_map<std::string, int64_t> metadata_int;
    std::unordered_map<std::string, double> metadata_float;
    std::unordered_map<std::string, std::vector<std::string>> metadata_str_arr;

    bool has_tokenizer() const {
        return metadata_str_arr.count("tokenizer.ggml.tokens") > 0 ||
               metadata_str.count("tokenizer.ggml.model") > 0;
    }

    bool has_chat_template() const {
        auto it = metadata_str.find("tokenizer.chat_template");
        return it != metadata_str.end() && !it->second.empty();
    }

    bool has_vision() const {
        return metadata_int.count("clip.vision.image_size") > 0 ||
               metadata_str.count("preprocessor.image.crop_mode") > 0 ||
               metadata_str.count("preprocessor.image.channel_format") > 0;
    }

    std::vector<std::string> get_tasks() const {
        auto it = metadata_str_arr.find("ggmlc.tasks");
        if (it != metadata_str_arr.end()) {
            return it->second;
        }
        auto s_it = metadata_str.find("ggmlc.task");
        if (s_it != metadata_str.end() && !s_it->second.empty()) {
            return {s_it->second};
        }
        return {};
    }

    bool has_task(const std::string& task) const {
        for (const auto& t : get_tasks()) {
            if (t == task) return true;
        }
        return false;
    }
};

} // namespace ggmlc

