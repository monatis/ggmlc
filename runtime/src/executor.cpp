#include "ggmlc/executor.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggmlc/stdlib_kernels.h"

namespace ggmlc {

ModelExecutor::ModelExecutor(const SerializedModelGraph& graph)
    : model_graph_(graph), ctx_(nullptr), cgraph_(nullptr) {}

ModelExecutor::~ModelExecutor() {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

void ModelExecutor::prepare(const std::unordered_map<std::string, int64_t>& symbol_env, bool enable_arena_reuse) {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    ggml_tensors_.clear();
    concrete_shapes_.clear();

    // 1. Evaluate concrete shapes and compute required memory
    size_t param_bytes = 0;
    size_t state_bytes = 0;
    size_t total_act_bytes = 0;
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
        size_t blck_size = ggml_blck_size(t.type);
        if (blck_size == 0) blck_size = 1;
        size_t sz = (numel / blck_size) * type_size;
        total_tensor_bytes += sz + ggml_tensor_overhead();

        if (t.storage == StorageClass::PARAMETER || t.storage == StorageClass::CONSTANT) {
            param_bytes += sz;
        } else if (t.storage == StorageClass::STATE) {
            state_bytes += sz;
        } else {
            total_act_bytes += sz;
        }
    }

    size_t effective_act_bytes = total_act_bytes;
    if (enable_arena_reuse) {
        // Liveness analysis over ops for peak activation memory planning
        int n_ops = static_cast<int>(model_graph_.ops.size());
        std::unordered_map<uint32_t, std::pair<int, int>> liveness;
        for (const auto& pair : model_graph_.tensors) {
            uint32_t tid = pair.first;
            bool is_persistent = (pair.second.storage == StorageClass::PARAMETER || pair.second.storage == StorageClass::STATE);
            liveness[tid] = {is_persistent ? 0 : n_ops, is_persistent ? n_ops : 0};
        }
        for (int op_idx = 0; op_idx < n_ops; ++op_idx) {
            const auto& op = model_graph_.ops[op_idx];
            for (uint32_t out_id : op.outputs) {
                if (liveness.count(out_id)) {
                    liveness[out_id].first = std::min(liveness[out_id].first, op_idx);
                    liveness[out_id].second = std::max(liveness[out_id].second, op_idx);
                }
            }
            for (uint32_t in_id : op.inputs) {
                if (liveness.count(in_id)) {
                    liveness[in_id].second = std::max(liveness[in_id].second, op_idx);
                }
            }
        }
        for (uint32_t out_id : model_graph_.outputs) {
            if (liveness.count(out_id)) liveness[out_id].second = n_ops;
        }
        for (uint32_t inp_id : model_graph_.inputs) {
            if (liveness.count(inp_id)) liveness[inp_id].first = 0;
        }

        size_t peak_act_bytes = 0;
        for (int op_idx = 0; op_idx < n_ops; ++op_idx) {
            size_t step_bytes = 0;
            for (const auto& pair : model_graph_.tensors) {
                uint32_t tid = pair.first;
                if (pair.second.storage == StorageClass::ACTIVATION || pair.second.storage == StorageClass::INPUT || pair.second.storage == StorageClass::OUTPUT) {
                    if (liveness[tid].first <= op_idx && liveness[tid].second >= op_idx) {
                        const auto& ne = concrete_shapes_[tid];
                        size_t numel = ne[0] * ne[1] * ne[2] * ne[3];
                        size_t t_size = ggml_type_size(pair.second.type);
                        size_t b_size = ggml_blck_size(pair.second.type);
                        if (b_size == 0) b_size = 1;
                        step_bytes += (numel / b_size) * t_size;
                    }
                }
            }
            peak_act_bytes = std::max(peak_act_bytes, step_bytes);
        }
        if (peak_act_bytes > 0) {
            effective_act_bytes = peak_act_bytes;
        }
    }

    // Allocate robust ggml context with safe workspace headroom for Conv2D / Attention intermediate objects
    size_t ctx_size = (param_bytes + state_bytes + effective_act_bytes) * 6 + 1024 * 1024 * 128 + model_graph_.ops.size() * 65536;
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
        } else if (t.storage == StorageClass::STATE) {
            auto s_it = persistent_states_.find(tid);
            if (s_it != persistent_states_.end() && s_it->second.size() == ggml_nbytes(g_t)) {
                std::memcpy(g_t->data, s_it->second.data(), s_it->second.size());
            } else {
                persistent_states_[tid].assign(ggml_nbytes(g_t), 0);
                std::memset(g_t->data, 0, ggml_nbytes(g_t));
            }
        }

        ggml_tensors_[tid] = g_t;
    }

    // 3. Build computation graph
    size_t graph_nodes = std::max<size_t>(32768, model_graph_.ops.size() * 16);
    cgraph_ = ggml_new_graph_custom(ctx_, graph_nodes, false);

    for (const auto& op : model_graph_.ops) {
        if (op.outputs.empty()) continue;
        uint32_t out_id = op.outputs[0];

        struct ggml_tensor* in0 = op.inputs.size() > 0 ? ggml_tensors_[op.inputs[0]] : nullptr;
        struct ggml_tensor* in1 = op.inputs.size() > 1 ? ggml_tensors_[op.inputs[1]] : nullptr;
        struct ggml_tensor* result = nullptr;

        auto match_broadcast = [&](struct ggml_tensor* a, struct ggml_tensor* b) -> std::pair<struct ggml_tensor*, struct ggml_tensor*> {
            if (!a || !b) return {a, b};
            if (ggml_are_same_shape(a, b)) return {a, b};

            int64_t target_ne[4];
            bool need_repeat_a = false;
            bool need_repeat_b = false;
            for (int d = 0; d < 4; ++d) {
                target_ne[d] = std::max(a->ne[d], b->ne[d]);
                if (a->ne[d] != target_ne[d]) need_repeat_a = true;
                if (b->ne[d] != target_ne[d]) need_repeat_b = true;
            }
            if (need_repeat_a) {
                if (!ggml_is_contiguous(a)) a = ggml_cont(ctx_, a);
                struct ggml_tensor* target_a = ggml_new_tensor_4d(ctx_, a->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
                a = ggml_repeat(ctx_, a, target_a);
            }
            if (need_repeat_b) {
                if (!ggml_is_contiguous(b)) b = ggml_cont(ctx_, b);
                struct ggml_tensor* target_b = ggml_new_tensor_4d(ctx_, b->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
                b = ggml_repeat(ctx_, b, target_b);
            }
            return {a, b};
        };

        switch (op.opcode) {
            case GGML_OP_REPEAT: {
                const auto& out_ne = concrete_shapes_[out_id];
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                int64_t in_elements = in0->ne[0] * in0->ne[1] * in0->ne[2] * in0->ne[3];
                int64_t out_elements = out_ne[0] * out_ne[1] * out_ne[2] * out_ne[3];
                if (in_elements == out_elements) {
                    result = ggml_reshape_4d(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                } else {
                    struct ggml_tensor* target = ggml_new_tensor_4d(ctx_, in0->type, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                    result = ggml_repeat(ctx_, in0, target);
                }
                break;
            }
            case GGML_OP_CPY: {
                const auto& out_ne = concrete_shapes_[out_id];
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                struct ggml_tensor* dst = ggml_new_tensor_4d(ctx_, in0->type, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                result = ggml_cpy(ctx_, in0, dst);
                break;
            }
            case GGML_OP_ADD: {
                auto p = match_broadcast(in0, in1);
                if (!ggml_can_repeat(p.second, p.first)) {
                    fprintf(stderr, "[OP_ADD FAIL] in0 id=%u shape=[%lld,%lld,%lld,%lld] in1 id=%u shape=[%lld,%lld,%lld,%lld]\n",
                        op.inputs[0], (long long)p.first->ne[0], (long long)p.first->ne[1], (long long)p.first->ne[2], (long long)p.first->ne[3],
                        op.inputs[1], (long long)p.second->ne[0], (long long)p.second->ne[1], (long long)p.second->ne[2], (long long)p.second->ne[3]);
                }
                result = ggml_add(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_SUB: {
                auto p = match_broadcast(in0, in1);
                if (!ggml_can_repeat(p.second, p.first)) {
                    fprintf(stderr, "[OP_SUB FAIL] in0 id=%u shape=[%lld,%lld,%lld,%lld] in1 id=%u shape=[%lld,%lld,%lld,%lld]\n",
                        op.inputs[0], (long long)p.first->ne[0], (long long)p.first->ne[1], (long long)p.first->ne[2], (long long)p.first->ne[3],
                        op.inputs[1], (long long)p.second->ne[0], (long long)p.second->ne[1], (long long)p.second->ne[2], (long long)p.second->ne[3]);
                }
                result = ggml_sub(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_MUL: {
                auto p = match_broadcast(in0, in1);
                if (!ggml_can_repeat(p.second, p.first)) {
                    fprintf(stderr, "[OP_MUL FAIL] in0 id=%u shape=[%lld,%lld,%lld,%lld] in1 id=%u shape=[%lld,%lld,%lld,%lld]\n",
                        op.inputs[0], (long long)p.first->ne[0], (long long)p.first->ne[1], (long long)p.first->ne[2], (long long)p.first->ne[3],
                        op.inputs[1], (long long)p.second->ne[0], (long long)p.second->ne[1], (long long)p.second->ne[2], (long long)p.second->ne[3]);
                }
                result = ggml_mul(ctx_, p.first, p.second);
                break;
            }
            case GGML_OP_DIV: {
                auto p = match_broadcast(in0, in1);
                if (!ggml_can_repeat(p.second, p.first)) {
                    fprintf(stderr, "[OP_DIV FAIL] in0 id=%u shape=[%lld,%lld,%lld,%lld] in1 id=%u shape=[%lld,%lld,%lld,%lld]\n",
                        op.inputs[0], (long long)p.first->ne[0], (long long)p.first->ne[1], (long long)p.first->ne[2], (long long)p.first->ne[3],
                        op.inputs[1], (long long)p.second->ne[0], (long long)p.second->ne[1], (long long)p.second->ne[2], (long long)p.second->ne[3]);
                }
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
            case GGML_OP_SQR: {
                float exp = 2.0f;
                if (op.attributes.count("exponent")) {
                    exp = static_cast<float>(op.attributes.at("exponent"));
                } else if (op.attributes.count("y")) {
                    exp = static_cast<float>(op.attributes.at("y"));
                }
                if (exp == 3.0f) {
                    result = ggml_mul(ctx_, in0, ggml_sqr(ctx_, in0));
                } else {
                    result = ggml_sqr(ctx_, in0);
                }
                break;
            }
            case GGML_OP_MEAN: {
                int g_dim = op.attributes.count("ggml_dim") ? static_cast<int>(op.attributes.at("ggml_dim")) : 0;
                if (g_dim == 1) {
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                    t = ggml_mean(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_mean(ctx_, in0);
                }
                const auto& out_ne = concrete_shapes_[out_id];
                result = ggml_reshape_4d(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_SUM:
            case GGML_OP_SUM_ROWS: {
                int g_dim = op.attributes.count("ggml_dim") ? static_cast<int>(op.attributes.at("ggml_dim")) : 0;
                if (g_dim == 1) {
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                    t = ggml_sum_rows(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_sum_rows(ctx_, in0);
                }
                const auto& out_ne = concrete_shapes_[out_id];
                result = ggml_reshape_4d(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_CONT:
                result = ggml_cont(ctx_, in0);
                break;
            case GGML_OP_SIN:
                result = ggml_sin(ctx_, in0);
                break;
            case GGML_OP_COS:
                result = ggml_cos(ctx_, in0);
                break;
            case GGML_OP_LOG:
                result = ggml_log(ctx_, in0);
                break;
            case GGML_OP_UNARY: {
                struct ggml_tensor* act_in = in0;
                if (act_in && ggml_nelements(act_in) == 1 && in1 && ggml_nelements(in1) > 1) {
                    act_in = in1;
                }
                auto it = op.attributes.find("unary_op");
                int u = (it != op.attributes.end()) ? static_cast<int>(it->second) : 6; // default RELU
                if (u == 6) { // RELU
                    result = ggml_relu(ctx_, act_in);
                } else if (u == 8) { // GELU
                    result = ggml_gelu(ctx_, act_in);
                } else if (u == 10) { // SILU
                    result = ggml_silu(ctx_, act_in);
                } else if (u == 4) { // TANH
                    result = ggml_tanh(ctx_, act_in);
                } else if (u == 2) { // NEG
                    result = ggml_neg(ctx_, act_in);
                } else {
                    result = ggml_relu(ctx_, act_in);
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
                if (!ggml_is_contiguous(q)) q = ggml_cont(ctx_, q);
                if (!ggml_is_contiguous(k)) k = ggml_cont(ctx_, k);
                if (v && !ggml_is_contiguous(v)) v = ggml_cont(ctx_, v);
                if (mask && !ggml_is_contiguous(mask)) mask = ggml_cont(ctx_, mask);

                float scale = 1.0f / sqrtf((float)q->ne[0]);
                bool is_causal = op.attributes.count("is_causal") && op.attributes.at("is_causal") != 0;

                // 1. Q_scaled = Q * scale
                struct ggml_tensor* q_scaled = ggml_scale(ctx_, q, scale);
                // 2. scores = mul_mat(K, Q_scaled) -> ne = [S_k, S_q, H, B]
                struct ggml_tensor* scores = ggml_mul_mat(ctx_, k, q_scaled);
                // 3. causal mask or explicit mask
                if (is_causal) {
                    scores = ggml_diag_mask_inf(ctx_, scores, 0);
                } else if (mask) {
                    if (ggml_can_repeat(mask, scores)) {
                        mask = ggml_repeat(ctx_, mask, scores);
                    }
                    scores = ggml_add(ctx_, scores, mask);
                }
                // 4. softmax over S_k (dim 0)
                struct ggml_tensor* probs = ggml_soft_max(ctx_, scores);
                // 5. context = mul_mat(transpose(V), probs) -> ne = [D, S_q, H, B]
                struct ggml_tensor* v_t = ggml_cont(ctx_, ggml_transpose(ctx_, v));
                result = ggml_mul_mat(ctx_, v_t, probs);

                // Print graph node names for SDPA
                std::fprintf(stderr, "[SDPA PREPARE] q: %p, k: %p, v: %p, scores: %p, probs: %p, v_t: %p, result: %p\n",
                    (void*)q, (void*)k, (void*)v, (void*)scores, (void*)probs, (void*)v_t, (void*)result);
                break;
            }
            case GGML_OP_GLU:
                result = ggml_swiglu(ctx_, in0);
                break;
            case GGML_OP_SOFT_MAX:
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                result = ggml_soft_max(ctx_, in0);
                break;
            case GGML_OP_MUL_MAT: {
                bool is_q = (in0->type != GGML_TYPE_F32 && in0->type != GGML_TYPE_F16);
                bool transpose_in0 = op.attributes.count("transpose_in0") && op.attributes.at("transpose_in0") != 0;
                if (!is_q && in0 && !ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                if (in1 && !ggml_is_contiguous(in1)) {
                    in1 = ggml_cont(ctx_, in1);
                }
                if (!is_q && in0 && (transpose_in0 || (in1 && in0->ne[0] != in1->ne[0] && in0->ne[1] == in1->ne[0]))) {
                    in0 = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                }
                result = ggml_mul_mat(ctx_, in0, in1);
                if (op.inputs.size() > 2) {
                    struct ggml_tensor* bias = ggml_tensors_[op.inputs[2]];
                    if (bias) {
                        if (!ggml_is_contiguous(bias)) bias = ggml_cont(ctx_, bias);
                        if (bias->ne[0] != result->ne[0] && bias->ne[1] == result->ne[0] && bias->ne[0] == 1) {
                            bias = ggml_reshape_1d(ctx_, bias, result->ne[0]);
                        }
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
                if (in0 && !ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                result = ggml_reshape_4d(ctx_, in0, ne[0], ne[1], ne[2], ne[3]);
                result = ggml_cont(ctx_, result);
                break;
            }
            case GGML_OP_PERMUTE: {
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                int ax0 = op.attributes.count("axis0") ? static_cast<int>(op.attributes.at("axis0")) : 1;
                int ax1 = op.attributes.count("axis1") ? static_cast<int>(op.attributes.at("axis1")) : 0;
                int ax2 = op.attributes.count("axis2") ? static_cast<int>(op.attributes.at("axis2")) : 2;
                int ax3 = op.attributes.count("axis3") ? static_cast<int>(op.attributes.at("axis3")) : 3;
                result = ggml_cont(ctx_, ggml_permute(ctx_, in0, ax0, ax1, ax2, ax3));
                break;
            }
            case GGML_OP_TRANSPOSE:
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
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
                result = in0;
                for (size_t k = 1; k < op.inputs.size(); ++k) {
                    struct ggml_tensor* next_in = ggml_tensors_[op.inputs[k]];
                    if (!next_in) continue;
                    if (!result || result->ne[0] == 0 || result->ne[1] == 0 || result->ne[2] == 0 || result->ne[3] == 0) {
                        result = next_in;
                        continue;
                    }
                    if (next_in->ne[0] == 0 || next_in->ne[1] == 0 || next_in->ne[2] == 0 || next_in->ne[3] == 0) {
                        continue;
                    }
                    if (!ggml_is_contiguous(result)) result = ggml_cont(ctx_, result);
                    if (!ggml_is_contiguous(next_in)) next_in = ggml_cont(ctx_, next_in);
                    result = ggml_concat(ctx_, result, next_in, g_dim);
                }
                if (result) result = ggml_cont(ctx_, result);
                break;
            }
            case GGML_OP_CONV_2D: {
                // in0: weight [KW, KH, IC, OC], in1: x [W, H, C, N]
                int s0 = op.attributes.count("stride_w") ? static_cast<int>(op.attributes.at("stride_w")) : 1;
                int s1 = op.attributes.count("stride_h") ? static_cast<int>(op.attributes.at("stride_h")) : 1;
                int p0 = op.attributes.count("pad_w") ? static_cast<int>(op.attributes.at("pad_w")) : 0;
                int p1 = op.attributes.count("pad_h") ? static_cast<int>(op.attributes.at("pad_h")) : 0;
                int d0 = op.attributes.count("dilation_w") ? static_cast<int>(op.attributes.at("dilation_w")) : 1;
                int d1 = op.attributes.count("dilation_h") ? static_cast<int>(op.attributes.at("dilation_h")) : 1;
                result = ggml_conv_2d(ctx_, in0, in1, s0, s1, p0, p1, d0, d1);
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
            case GGML_OP_POOL_2D: {
                enum ggml_op_pool pool_type = op.attributes.count("is_max") && op.attributes.at("is_max") != 0
                                              ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
                int k0, k1, s0, s1, p0, p1;
                if (op.attributes.count("is_adaptive") && op.attributes.at("is_adaptive") != 0) {
                    // Global adaptive pooling over whole feature map
                    k0 = static_cast<int>(in0->ne[0]);
                    k1 = static_cast<int>(in0->ne[1]);
                    s0 = k0;
                    s1 = k1;
                    p0 = 0;
                    p1 = 0;
                } else {
                    k0 = op.attributes.count("ksize_w") ? static_cast<int>(op.attributes.at("ksize_w")) : 2;
                    k1 = op.attributes.count("ksize_h") ? static_cast<int>(op.attributes.at("ksize_h")) : 2;
                    s0 = op.attributes.count("stride_w") ? static_cast<int>(op.attributes.at("stride_w")) : k0;
                    s1 = op.attributes.count("stride_h") ? static_cast<int>(op.attributes.at("stride_h")) : k1;
                    p0 = op.attributes.count("pad_w") ? static_cast<int>(op.attributes.at("pad_w")) : 0;
                    p1 = op.attributes.count("pad_h") ? static_cast<int>(op.attributes.at("pad_h")) : 0;
                }
                result = ggml_pool_2d(ctx_, in0, pool_type, k0, k1, s0, s1, static_cast<float>(p0), static_cast<float>(p1));
                break;
            }
            case 200: { // GGML_OP_CUSTOM_BIAS_GELU: in0=x, in1=bias
                if (!in0 || !in1) {
                    throw std::runtime_error("GGML_OP_CUSTOM_BIAS_GELU requires 2 inputs");
                }
                result = ggml_map_custom2(ctx_, in0, in1, ggmlc_compute_forward_bias_gelu, GGML_N_TASKS_MAX, nullptr);
                break;
            }
            case 201: { // GGML_OP_CUSTOM_LAYER_NORM: in0=x, in1=weight, in2=bias (optional)
                struct ggml_tensor* w = in1;
                struct ggml_tensor* b = op.inputs.size() > 2 ? ggml_tensors_[op.inputs[2]] : nullptr;
                float eps = op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;

                custom_params_storage_.emplace_back(sizeof(struct ggmlc_norm_params));
                struct ggmlc_norm_params* params = reinterpret_cast<struct ggmlc_norm_params*>(custom_params_storage_.back().data());
                params->eps = eps;

                result = ggml_map_custom3(ctx_, in0, w, b, ggmlc_compute_forward_layer_norm, GGML_N_TASKS_MAX, params);
                break;
            }
            case 202: { // GGML_OP_CUSTOM_RMS_NORM: in0=x, in1=weight
                struct ggml_tensor* w = in1;
                float eps = op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;

                custom_params_storage_.emplace_back(sizeof(struct ggmlc_norm_params));
                struct ggmlc_norm_params* params = reinterpret_cast<struct ggmlc_norm_params*>(custom_params_storage_.back().data());
                params->eps = eps;

                result = ggml_map_custom2(ctx_, in0, w, ggmlc_compute_forward_rms_norm, GGML_N_TASKS_MAX, params);
                break;
            }
            case 203: { // GGML_OP_CUSTOM_SWIGLU: in0=gate, in1=up
                if (!in0 || !in1) {
                    throw std::runtime_error("GGML_OP_CUSTOM_SWIGLU requires 2 inputs (gate, up)");
                }
                result = ggml_map_custom2(ctx_, in0, in1, ggmlc_compute_forward_swiglu, GGML_N_TASKS_MAX, nullptr);
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
        } else {
            fprintf(stderr, "[OP BUILD FAIL] op %d (opcode %d) out_id %u produced NULL result!\n", op.id, op.opcode, out_id);
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

    // Save persistent states
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        if (pair.second.storage == StorageClass::STATE) {
            auto it = ggml_tensors_.find(tid);
            if (it != ggml_tensors_.end()) {
                size_t sz = ggml_nbytes(it->second);
                persistent_states_[tid].resize(sz);
                std::memcpy(persistent_states_[tid].data(), it->second->data, sz);
            }
        }
    }
}

void ModelExecutor::set_state(uint32_t tensor_id, const void* data, size_t size_bytes) {
    persistent_states_[tensor_id].resize(size_bytes);
    std::memcpy(persistent_states_[tensor_id].data(), data, size_bytes);
    auto it = ggml_tensors_.find(tensor_id);
    if (it != ggml_tensors_.end() && ggml_nbytes(it->second) == size_bytes) {
        std::memcpy(it->second->data, data, size_bytes);
    }
}

void ModelExecutor::set_state_by_name(const std::string& name, const void* data, size_t size_bytes) {
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.name == name) {
            set_state(pair.first, data, size_bytes);
            return;
        }
    }
    throw std::runtime_error("State tensor name not found in model: " + name);
}

const void* ModelExecutor::get_state_data(uint32_t tensor_id) const {
    auto it = persistent_states_.find(tensor_id);
    if (it != persistent_states_.end() && !it->second.empty()) {
        return it->second.data();
    }
    auto t_it = ggml_tensors_.find(tensor_id);
    if (t_it != ggml_tensors_.end()) {
        return t_it->second->data;
    }
    throw std::runtime_error("State tensor ID not found in executor: " + std::to_string(tensor_id));
}

const void* ModelExecutor::get_state_data_by_name(const std::string& name) const {
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.name == name) {
            return get_state_data(pair.first);
        }
    }
    throw std::runtime_error("State tensor name not found in model: " + name);
}

void ModelExecutor::reset_state() {
    for (auto& pair : persistent_states_) {
        std::fill(pair.second.begin(), pair.second.end(), 0);
    }
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.storage == StorageClass::STATE) {
            auto it = ggml_tensors_.find(pair.first);
            if (it != ggml_tensors_.end()) {
                std::memset(it->second->data, 0, ggml_nbytes(it->second));
            }
        }
    }
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
    size_t sz = ggml_nbytes(it->second);
    if (sz == 0) {
        fprintf(stderr, "[DEBUG] tensor %u ne=[%lld,%lld,%lld,%lld] type=%d blck_size=%zu type_size=%zu\n",
            tensor_id, (long long)it->second->ne[0], (long long)it->second->ne[1], (long long)it->second->ne[2], (long long)it->second->ne[3],
            it->second->type, ggml_blck_size(it->second->type), ggml_type_size(it->second->type));
    }
    return sz;
}

} // namespace ggmlc
