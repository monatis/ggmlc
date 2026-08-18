#include "ggmlc/executor.h"
#include <iostream>
#include <cstring>
#include <cmath>
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
            if (ggml_are_same_shape(a, b)) return {a, b};
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
            case GGML_OP_SQRT: {
                result = ggml_sqrt(ctx_, in0);
                if (op.attributes.count("is_rsqrt") && op.attributes.at("is_rsqrt")) {
                    struct ggml_tensor* one = ggml_new_f32(ctx_, 1.0f);
                    if (ggml_can_repeat(one, result)) {
                        one = ggml_repeat(ctx_, one, result);
                    }
                    result = ggml_div(ctx_, one, result);
                }
                break;
            }
            case GGML_OP_SQR:
                result = ggml_sqr(ctx_, in0);
                break;
            case GGML_OP_MEAN:
                result = ggml_mean(ctx_, in0);
                break;
            case GGML_OP_CONT:
                result = ggml_cont(ctx_, in0);
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
            case GGML_OP_NORM: {
                float eps = op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;
                result = ggml_norm(ctx_, in0, eps);
                if (op.inputs.size() > 1) {
                    struct ggml_tensor* w = ggml_tensors_[op.inputs[1]];
                    if (w) {
                        if (ggml_can_repeat(w, result)) {
                            w = ggml_repeat(ctx_, w, result);
                        }
                        result = ggml_mul(ctx_, result, w);
                    }
                }
                if (op.inputs.size() > 2) {
                    struct ggml_tensor* b = ggml_tensors_[op.inputs[2]];
                    if (b) {
                        if (ggml_can_repeat(b, result)) {
                            b = ggml_repeat(ctx_, b, result);
                        }
                        result = ggml_add(ctx_, result, b);
                    }
                }
                break;
            }
            case GGML_OP_RMS_NORM: {
                float eps = op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;
                result = ggml_rms_norm(ctx_, in0, eps);
                if (op.inputs.size() > 1) {
                    struct ggml_tensor* w = ggml_tensors_[op.inputs[1]];
                    if (w) {
                        if (ggml_can_repeat(w, result)) {
                            w = ggml_repeat(ctx_, w, result);
                        }
                        result = ggml_mul(ctx_, result, w);
                    }
                }
                break;
            }
            case GGML_OP_GET_ROWS: {
                if (in1 && in1->type != GGML_TYPE_I32) {
                    in1 = ggml_cast(ctx_, in1, GGML_TYPE_I32);
                }
                int64_t total_indices = in1->ne[0] * in1->ne[1] * in1->ne[2] * in1->ne[3];
                struct ggml_tensor* in1_flat = ggml_reshape_1d(ctx_, in1, total_indices);
                struct ggml_tensor* raw_rows = ggml_get_rows(ctx_, in0, in1_flat);
                const auto& out_ne = concrete_shapes_[out_id];
                result = ggml_reshape_4d(ctx_, raw_rows, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_ROPE: {
                int n_dims = op.attributes.count("n_dims") ? static_cast<int>(op.attributes.at("n_dims")) : in0->ne[0];
                int mode = op.attributes.count("mode") ? static_cast<int>(op.attributes.at("mode")) : 0;
                result = ggml_rope(ctx_, in0, in1, n_dims, mode);
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                struct ggml_tensor* q = in0;
                struct ggml_tensor* k = in1;
                struct ggml_tensor* v = op.inputs.size() > 2 ? ggml_tensors_[op.inputs[2]] : nullptr;
                struct ggml_tensor* mask = op.inputs.size() > 3 ? ggml_tensors_[op.inputs[3]] : nullptr;
                float scale = op.attributes.count("scale") ? static_cast<float>(op.attributes.at("scale")) : 1.0f / sqrtf((float)q->ne[0]);
                result = ggml_flash_attn_ext(ctx_, q, k, v, mask, scale, 0.0f, 0.0f);
                break;
            }
            case GGML_OP_GLU:
                result = ggml_swiglu(ctx_, in0);
                break;
            case GGML_OP_SOFT_MAX:
                result = ggml_soft_max(ctx_, in0);
                break;
            case GGML_OP_MUL_MAT: {
                bool transpose_in0 = op.attributes.count("transpose_in0") && op.attributes.at("transpose_in0") != 0;
                if (in0 && (transpose_in0 || (in1 && in0->ne[0] != in1->ne[0] && in0->ne[1] == in1->ne[0]))) {
                    in0 = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                }
                if (in0 && !ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
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
                if (!ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                result = ggml_reshape_4d(ctx_, in0, ne[0], ne[1], ne[2], ne[3]);
                break;
            }
            case GGML_OP_PERMUTE: {
                int ax0 = op.attributes.count("axis0") ? static_cast<int>(op.attributes.at("axis0")) : 1;
                int ax1 = op.attributes.count("axis1") ? static_cast<int>(op.attributes.at("axis1")) : 0;
                int ax2 = op.attributes.count("axis2") ? static_cast<int>(op.attributes.at("axis2")) : 2;
                int ax3 = op.attributes.count("axis3") ? static_cast<int>(op.attributes.at("axis3")) : 3;
                result = ggml_cont(ctx_, ggml_permute(ctx_, in0, ax0, ax1, ax2, ax3));
                break;
            }
            case GGML_OP_TRANSPOSE:
                result = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                break;
            case GGML_OP_VIEW: {
                const auto& out_ne = concrete_shapes_[out_id];
                int g_dim = op.attributes.count("ggml_dim") ? static_cast<int>(op.attributes.at("ggml_dim")) : 0;
                int64_t start = op.attributes.count("start") ? op.attributes.at("start") : 0;
                int64_t step = op.attributes.count("step") ? op.attributes.at("step") : 1;
                if (!ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                size_t offset = start * in0->nb[g_dim];
                size_t nb1 = in0->nb[1] * (g_dim == 1 ? step : 1);
                size_t nb2 = in0->nb[2] * (g_dim == 2 ? step : 1);
                size_t nb3 = in0->nb[3] * (g_dim == 3 ? step : 1);
                result = ggml_cont(ctx_, ggml_view_4d(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3], nb1, nb2, nb3, offset));
                break;
            }
            case GGML_OP_CONCAT: {
                int g_dim = op.attributes.count("ggml_dim") ? static_cast<int>(op.attributes.at("ggml_dim")) : 0;
                if (op.inputs.size() >= 2) {
                    result = ggml_concat(ctx_, in0, in1, g_dim);
                    for (size_t k = 2; k < op.inputs.size(); ++k) {
                        struct ggml_tensor* next_in = ggml_tensors_[op.inputs[k]];
                        result = ggml_concat(ctx_, result, next_in, g_dim);
                    }
                } else {
                    result = in0;
                }
                result = ggml_cont(ctx_, result);
                break;
            }
            case GGML_OP_REPEAT: {
                const auto& ne = concrete_shapes_[out_id];
                result = ggml_repeat_4d(ctx_, in0, ne[0], ne[1], ne[2], ne[3]);
                break;
            }
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
