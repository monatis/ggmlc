#include "ggmlc/executor.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include "ggml.h"
#include "ggml-cpu.h"

namespace ggmlc {

ModelExecutor::ModelExecutor(const SerializedModelGraph& graph)
    : model_graph_(graph), ctx_(nullptr), cgraph_(nullptr) {}

ModelExecutor::~ModelExecutor() {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

void ModelExecutor::prepare(const std::unordered_map<std::string, int64_t>& symbol_env) {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    ggml_tensors_.clear();
    concrete_shapes_.clear();

    // 1. Evaluate concrete shapes and compute required memory
    size_t total_tensor_bytes = 0;
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        std::array<int64_t, 4> ne;
        int64_t numel = 1;
        for (int d = 0; d < 4; ++d) {
            ne[d] = t.ne[d]->evaluate(symbol_env, model_graph_.symbol_table);
            numel *= ne[d];
        }
        concrete_shapes_[tid] = ne;

        size_t type_size = ggml_type_size(t.type);
        total_tensor_bytes += numel * type_size + ggml_tensor_overhead();
    }

    // Allocate ggml context
    size_t ctx_size = total_tensor_bytes * 2 + 1024 * 1024 * 16;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize ggml_context");
    }

    // 2. Instantiate ggml_tensors
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        const auto& ne = concrete_shapes_[tid];

        struct ggml_tensor* g_t = ggml_new_tensor_4d(ctx_, t.type, ne[0], ne[1], ne[2], ne[3]);
        if (!g_t) {
            throw std::runtime_error("Failed to allocate ggml_tensor for: " + t.name);
        }
        ggml_set_name(g_t, t.name.c_str());

        // If parameter/constant with initial data, copy bytes
        if (t.data_ptr && t.data_size > 0) {
            std::memcpy(g_t->data, t.data_ptr, std::min<size_t>(t.data_size, ggml_nbytes(g_t)));
        }

        ggml_tensors_[tid] = g_t;
    }

    // 3. Build computation graph
    cgraph_ = ggml_new_graph(ctx_);

    for (const auto& op : model_graph_.ops) {
        if (op.outputs.empty()) continue;
        uint32_t out_id = op.outputs[0];

        struct ggml_tensor* in0 = op.inputs.size() > 0 ? ggml_tensors_[op.inputs[0]] : nullptr;
        struct ggml_tensor* in1 = op.inputs.size() > 1 ? ggml_tensors_[op.inputs[1]] : nullptr;
        struct ggml_tensor* result = nullptr;

        auto match_broadcast = [&](struct ggml_tensor* a, struct ggml_tensor* b) -> std::pair<struct ggml_tensor*, struct ggml_tensor*> {
            if (!a || !b) return {a, b};
            if (ggml_can_repeat(b, a)) {
                b = ggml_repeat(ctx_, b, a);
            } else if (ggml_can_repeat(a, b)) {
                a = ggml_repeat(ctx_, a, b);
            }
            return {a, b};
        };

        switch (op.opcode) {
            case GGML_OP_ADD: {
                auto p = match_broadcast(in0, in1);
                result = ggml_add(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_SUB: {
                auto p = match_broadcast(in0, in1);
                result = ggml_sub(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_MUL: {
                auto p = match_broadcast(in0, in1);
                result = ggml_mul(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_DIV: {
                auto p = match_broadcast(in0, in1);
                result = ggml_div(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_SQRT:
                result = ggml_sqrt(ctx_, in0);
                break;
            case GGML_OP_LOG:
                result = ggml_log(ctx_, in0);
                break;
            case GGML_OP_UNARY: {
                auto it = op.attributes.find("unary_op");
                int u = (it != op.attributes.end()) ? static_cast<int>(it->second) : 6; // default RELU
                if (u == 6) { // RELU
                    result = ggml_relu(ctx_, in0);
                } else if (u == 8) { // GELU
                    result = ggml_gelu(ctx_, in0);
                } else if (u == 10) { // SILU
                    result = ggml_silu(ctx_, in0);
                } else {
                    result = ggml_relu(ctx_, in0);
                }
                break;
            }
            case GGML_OP_RMS_NORM:
                result = ggml_rms_norm(ctx_, in0, 1e-5f);
                if (in1) {
                    result = ggml_mul(ctx_, result, in1);
                }
                break;
            case GGML_OP_SOFT_MAX:
                result = ggml_soft_max(ctx_, in0);
                break;
            case GGML_OP_MUL_MAT: {
                if (in0 && in1 && in0->ne[0] != in1->ne[0]) {
                    in0 = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                }
                result = ggml_mul_mat(ctx_, in0, in1);
                if (op.inputs.size() > 2) {
                    struct ggml_tensor* bias = ggml_tensors_[op.inputs[2]];
                    if (bias) {
                        if (ggml_can_repeat(bias, result)) {
                            bias = ggml_repeat(ctx_, bias, result);
                        }
                        result = ggml_add(ctx_, result, bias);
                    }
                }
                break;
            }
            case GGML_OP_RESHAPE: {
                const auto& ne = concrete_shapes_[out_id];
                result = ggml_reshape_4d(ctx_, in0, ne[0], ne[1], ne[2], ne[3]);
                break;
            }
            case GGML_OP_PERMUTE:
                result = ggml_permute(ctx_, in0, 1, 0, 2, 3);
                break;
            case GGML_OP_TRANSPOSE:
                result = ggml_transpose(ctx_, in0);
                break;
            default:
                // Fallback copy or identity
                result = ggml_dup(ctx_, in0);
                break;
        }

        if (result) {
            ggml_set_name(result, model_graph_.tensors[out_id].name.c_str());
            ggml_tensors_[out_id] = result;
            ggml_build_forward_expand(cgraph_, result);
        }
    }
}

void ModelExecutor::set_input(uint32_t tensor_id, const void* data, size_t size_bytes) {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    struct ggml_tensor* t = it->second;
    size_t expected_size = ggml_nbytes(t);
    if (size_bytes != expected_size) {
        throw std::runtime_error("Input size mismatch for tensor " + std::to_string(tensor_id) +
                                 ": got " + std::to_string(size_bytes) +
                                 ", expected " + std::to_string(expected_size));
    }
    std::memcpy(t->data, data, size_bytes);
}

void ModelExecutor::set_input_by_name(const std::string& name, const void* data, size_t size_bytes) {
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.name == name) {
            set_input(pair.first, data, size_bytes);
            return;
        }
    }
    throw std::runtime_error("Tensor name not found in model: " + name);
}

void ModelExecutor::run(int n_threads) {
    if (!ctx_ || !cgraph_) {
        throw std::runtime_error("Executor not prepared. Call prepare() first.");
    }
    ggml_graph_compute_with_ctx(ctx_, cgraph_, n_threads);
}

const void* ModelExecutor::get_output_data(uint32_t tensor_id) const {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    return it->second->data;
}

std::array<int64_t, 4> ModelExecutor::get_tensor_shape(uint32_t tensor_id) const {
    auto it = concrete_shapes_.find(tensor_id);
    if (it == concrete_shapes_.end()) {
        throw std::runtime_error("Tensor ID not found in concrete shapes: " + std::to_string(tensor_id));
    }
    return it->second;
}

size_t ModelExecutor::get_tensor_size_bytes(uint32_t tensor_id) const {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    return ggml_nbytes(it->second);
}

} // namespace ggmlc
