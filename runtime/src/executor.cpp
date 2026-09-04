#include "ggmlc/executor.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "ggmlc/stdlib_kernels.h"

namespace ggmlc {

std::vector<std::string> ModelExecutor::get_available_devices() {
    std::vector<std::string> devices = {"cpu"};
#if defined(GGML_USE_CUDA)
    int n_cuda = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < n_cuda; ++i) {
        devices.push_back("cuda:" + std::to_string(i));
    }
    if (n_cuda > 0) {
        devices.push_back("cuda");
    }
#endif
    return devices;
}

ModelExecutor::ModelExecutor(const SerializedModelGraph& graph, const std::string& device)
    : model_graph_(graph), device_(device), backend_(nullptr), buffer_(nullptr), ctx_(nullptr), cgraph_(nullptr) {
    std::string dev_lower = device_;
    for (auto& c : dev_lower) c = std::tolower(static_cast<unsigned char>(c));

    if (dev_lower == "auto") {
#if defined(GGML_USE_CUDA)
        if (ggml_backend_cuda_get_device_count() > 0) {
            dev_lower = "cuda:0";
            device_ = "cuda:0";
        } else {
            dev_lower = "cpu";
            device_ = "cpu";
        }
#else
        dev_lower = "cpu";
        device_ = "cpu";
#endif
    }

    if (dev_lower.rfind("cuda", 0) == 0) {
#if defined(GGML_USE_CUDA)
        int device_idx = 0;
        if (dev_lower.size() > 5 && dev_lower[4] == ':') {
            device_idx = std::stoi(dev_lower.substr(5));
        }
        backend_ = ggml_backend_cuda_init(device_idx);
        if (!backend_) {
            throw std::runtime_error("Failed to initialize GGML CUDA backend on device " + std::to_string(device_idx));
        }
        device_ = "cuda:" + std::to_string(device_idx);
        is_cuda_ = true;
#else
        throw std::runtime_error("CUDA backend was requested ('" + device + "'), but ggmlc was compiled without CUDA support.");
#endif
    } else {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            throw std::runtime_error("Failed to initialize GGML CPU backend.");
        }
        device_ = "cpu";
        is_cuda_ = false;
    }
}

ModelExecutor::~ModelExecutor() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

void ModelExecutor::prepare(const std::unordered_map<std::string, int64_t>& symbol_env, bool enable_arena_reuse) {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    ggml_tensors_.clear();
    concrete_shapes_.clear();
    custom_params_storage_.clear();
    output_host_buffers_.clear();
    state_host_buffers_.clear();

    // 1. Evaluate concrete shapes and metadata
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        std::array<int64_t, 4> ne;
        for (int d = 0; d < 4; ++d) {
            ne[d] = t.ne[d]->evaluate(symbol_env, model_graph_.symbol_table);
        }
        concrete_shapes_[tid] = ne;
    }

    // Allocate ggml context with no_alloc=true so the backend buffer allocates memory
    size_t ctx_meta_size = (model_graph_.tensors.size() + model_graph_.ops.size() * 16 + 256) * ggml_tensor_overhead() + 8 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize ggml_context");
    }

    // 2. Instantiate ggml_tensors (metadata only)
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        const auto& ne = concrete_shapes_[tid];

        struct ggml_tensor* g_t = ggml_new_tensor_4d(ctx_, t.type, ne[0], ne[1], ne[2], ne[3]);
        if (!g_t) {
            throw std::runtime_error("Failed to allocate ggml_tensor for: " + t.name);
        }
        ggml_set_name(g_t, t.name.c_str());
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

            // If a or b is a 1D channel vector matching the other's channel dim (GGML dim 2), reshape to [1, 1, C, 1]
            if (a->ne[1] == 1 && a->ne[2] == 1 && a->ne[3] == 1 && a->ne[0] == b->ne[2]) {
                a = ggml_reshape_4d(ctx_, a, 1, 1, a->ne[0], 1);
            }
            if (b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && b->ne[0] == a->ne[2]) {
                b = ggml_reshape_4d(ctx_, b, 1, 1, b->ne[0], 1);
            }

            if (ggml_can_repeat(b, a)) return {a, b};
            if (ggml_can_repeat(a, b)) {
                if (!ggml_is_contiguous(a)) a = ggml_cont(ctx_, a);
                struct ggml_tensor* target_a = ggml_new_tensor_4d(ctx_, a->type, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
                struct ggml_tensor* rep_a = ggml_repeat(ctx_, a, target_a);
                a = ggml_cpy(ctx_, rep_a, target_a);
                return {a, b};
            }

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
                if (ggml_can_repeat(a, target_a)) {
                    struct ggml_tensor* rep_a = ggml_repeat(ctx_, a, target_a);
                    a = ggml_cpy(ctx_, rep_a, target_a);
                }
            }
            if (need_repeat_b) {
                if (!ggml_is_contiguous(b)) b = ggml_cont(ctx_, b);
                struct ggml_tensor* target_b = ggml_new_tensor_4d(ctx_, b->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
                if (ggml_can_repeat(b, target_b)) {
                    b = ggml_repeat(ctx_, b, target_b);
                }
            }
            if (a->type != b->type) {
                if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32) {
                    if (!ggml_is_contiguous(a)) a = ggml_cont(ctx_, a);
                    struct ggml_tensor* target_a = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
                    a = ggml_cpy(ctx_, a, target_a);
                } else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32) {
                    if (!ggml_is_contiguous(b)) b = ggml_cont(ctx_, b);
                    struct ggml_tensor* target_b = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
                    b = ggml_cpy(ctx_, b, target_b);
                }
            }
            return {a, b};
        };

        if (getenv("GGMLC_DEBUG_OPS")) {
            fprintf(stderr, "[PREPARING OP %d/%zu: %s (opcode=%d)]\n",
                op.id, model_graph_.ops.size(), op.name.c_str(), (int)op.opcode);
        }

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
                auto out_type = model_graph_.tensors.at(out_id).type;
                struct ggml_tensor* dst = ggml_new_tensor_4d(ctx_, out_type, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
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
                if (op.attributes.count("is_rsqrt") && op.attributes.at("is_rsqrt")) {
                    struct ggml_tensor* sqrt_x = ggml_sqrt(ctx_, in0);
                    result = ggml_div(ctx_, sqrt_x, in0);
                } else {
                    result = ggml_sqrt(ctx_, in0);
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
                const auto& out_ne = concrete_shapes_[out_id];
                if (g_dim == 1) {
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                    t = ggml_mean(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else if (g_dim >= 2) {
                    // Spatial reduction over dims 1 and 2 (e.g. NHWC global pool: [C, W, H, B] -> [C, 1, 1, B])
                    struct ggml_tensor* flat_hw = ggml_reshape_4d(ctx_, in0, in0->ne[0], in0->ne[1] * in0->ne[2], 1, in0->ne[3]);
                    flat_hw = ggml_cont(ctx_, flat_hw);
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, flat_hw));
                    t = ggml_mean(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_mean(ctx_, in0);
                }
                result = ggml_reshape_4d(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_SUM:
            case GGML_OP_SUM_ROWS: {
                int g_dim = op.attributes.count("ggml_dim") ? static_cast<int>(op.attributes.at("ggml_dim")) : 0;
                const auto& out_ne = concrete_shapes_[out_id];
                if (g_dim == 1) {
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                    t = ggml_sum_rows(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else if (g_dim >= 2) {
                    // Spatial reduction over dims 1 and 2 (e.g. NHWC global pool: [C, W, H, B] -> [C, 1, 1, B])
                    struct ggml_tensor* flat_hw = ggml_reshape_4d(ctx_, in0, in0->ne[0], in0->ne[1] * in0->ne[2], 1, in0->ne[3]);
                    flat_hw = ggml_cont(ctx_, flat_hw);
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, flat_hw));
                    t = ggml_sum_rows(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_sum_rows(ctx_, in0);
                }
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
                if (op.inputs.size() > 1) {
                    uint32_t in0_id = op.inputs[0];
                    uint32_t in1_id = op.inputs[1];
                    auto t0_it = model_graph_.tensors.find(in0_id);
                    auto t1_it = model_graph_.tensors.find(in1_id);
                    if (t0_it != model_graph_.tensors.end() && t1_it != model_graph_.tensors.end()) {
                        if (t0_it->second.storage == StorageClass::CONSTANT && t1_it->second.storage != StorageClass::CONSTANT) {
                            act_in = in1;
                        } else if (t1_it->second.storage == StorageClass::CONSTANT && t0_it->second.storage != StorageClass::CONSTANT) {
                            act_in = in0;
                        } else if (ggml_nelements(in0) == 1 && ggml_nelements(in1) > 1) {
                            act_in = in1;
                        }
                    }
                }
                auto it = op.attributes.find("unary_op");
                int u = (it != op.attributes.end()) ? static_cast<int>(it->second) : 6; // default RELU
                if (u == 6) { // RELU
                    result = ggml_relu(ctx_, act_in);
                } else if (u == 7) { // SIGMOID
                    result = ggml_sigmoid(ctx_, act_in);
                } else if (u == 8) { // GELU
                    result = ggml_gelu(ctx_, act_in);
                } else if (u == 10) { // SILU
                    result = ggml_silu(ctx_, act_in);
                } else if (u == 11) { // HARDSWISH
                    result = ggml_hardswish(ctx_, act_in);
                } else if (u == 12) { // HARDSIGMOID
                    result = ggml_hardsigmoid(ctx_, act_in);
                } else if (u == 4) { // TANH
                    result = ggml_tanh(ctx_, act_in);
                } else if (u == 2) { // NEG
                    result = ggml_neg(ctx_, act_in);
                } else if (u == 0) { // ABS
                    result = ggml_abs(ctx_, act_in);
                } else if (u == 13) { // EXP
                    result = ggml_exp(ctx_, act_in);
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
                bool cast_in0_back = false;
                if (in0 && in0->type == GGML_TYPE_I64) {
                    in0 = ggml_cast(ctx_, in0, GGML_TYPE_I32);
                    cast_in0_back = true;
                }
                if (in1 && in1->type != GGML_TYPE_I32) {
                    in1 = ggml_cast(ctx_, in1, GGML_TYPE_I32);
                }
                int64_t total_indices = in1->ne[0] * in1->ne[1] * in1->ne[2] * in1->ne[3];
                struct ggml_tensor* in1_flat = ggml_reshape_1d(ctx_, in1, total_indices);
                struct ggml_tensor* raw_rows = ggml_get_rows(ctx_, in0, in1_flat);
                if (cast_in0_back) {
                    raw_rows = ggml_cast(ctx_, raw_rows, GGML_TYPE_I64);
                }
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
                if (op.float_attributes.count("scale")) {
                    scale = static_cast<float>(op.float_attributes.at("scale"));
                } else if (op.attributes.count("scale")) {
                    scale = static_cast<float>(op.attributes.at("scale"));
                }
                bool is_causal = op.attributes.count("is_causal") && op.attributes.at("is_causal") != 0;

                // 1. Q_scaled = Q * scale
                struct ggml_tensor* q_scaled = (std::abs(scale - 1.0f) < 1e-6f) ? q : ggml_scale(ctx_, q, scale);
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
                bool explicit_transpose = op.attributes.count("transpose_in0") > 0;
                bool transpose_in0 = explicit_transpose ? (op.attributes.at("transpose_in0") != 0) : (!is_q && in1 && in0->ne[0] != in1->ne[0] && in0->ne[1] == in1->ne[0]);
                if (!is_q && in0 && !ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                if (in1 && !ggml_is_contiguous(in1)) {
                    in1 = ggml_cont(ctx_, in1);
                }
                if (!is_q && in0 && transpose_in0) {
                    in0 = ggml_cont(ctx_, ggml_transpose(ctx_, in0));
                }
                if (in0->ne[0] != in1->ne[0]) {
                    fprintf(stderr, "[MUL_MAT CANNOT COMPUTE!] op=%s explicit_trans=%d trans_in0=%d\n  in0 name=%s ne=[%lld,%lld,%lld,%lld]\n  in1 name=%s ne=[%lld,%lld,%lld,%lld]\n",
                        op.name.c_str(), (int)explicit_transpose, (int)transpose_in0,
                        in0->name, (long long)in0->ne[0], (long long)in0->ne[1], (long long)in0->ne[2], (long long)in0->ne[3],
                        in1->name, (long long)in1->ne[0], (long long)in1->ne[1], (long long)in1->ne[2], (long long)in1->ne[3]);
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
                if (in0 && (!ggml_is_contiguous(in0) || in0->view_src != nullptr)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                size_t offset = start * in0->nb[g_dim];
                size_t nb1 = in0->nb[1] * (g_dim == 1 ? step : 1);
                size_t nb2 = in0->nb[2] * (g_dim == 2 ? step : 1);
                size_t nb3 = in0->nb[3] * (g_dim == 3 ? step : 1);
                result = ggml_cont(ctx_, ggml_view_4d(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3], nb1, nb2, nb3, offset));
                break;
            }
            case GGML_OP_ARGMAX: {
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                if (in0 && in0->type != GGML_TYPE_F32) {
                    struct ggml_tensor* target_f32 = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, in0->ne[0], in0->ne[1], in0->ne[2], in0->ne[3]);
                    in0 = ggml_cpy(ctx_, in0, target_f32);
                }
                result = ggml_argmax(ctx_, in0);
                const auto& out_ne = concrete_shapes_[out_id];
                result = ggml_reshape_4d(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
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
                bool is_1d = op.attributes.count("is_1d") && op.attributes.at("is_1d") != 0;
                if (is_1d) {
                    if (in0->ne[3] == 1) {
                        in0 = ggml_reshape_4d(ctx_, in0, in0->ne[0], 1, in0->ne[1], in0->ne[2]);
                    }
                    if (in1->ne[3] == 1) {
                        in1 = ggml_reshape_4d(ctx_, in1, in1->ne[0], 1, in1->ne[1], in1->ne[2]);
                    }
                }
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
                        if (!ggml_is_contiguous(bias)) bias = ggml_cont(ctx_, bias);
                        if (bias->ne[0] == result->ne[2] && bias->ne[1] == 1 && bias->ne[2] == 1) {
                            bias = ggml_reshape_4d(ctx_, bias, 1, 1, result->ne[2], 1);
                        }
                        if (ggml_can_repeat(bias, result)) {
                            bias = ggml_repeat(ctx_, bias, result);
                        }
                        result = ggml_add(ctx_, result, bias);
                    }
                }
                if (op.attributes.count("fused_relu") && op.attributes.at("fused_relu") != 0) {
                    result = ggml_relu(ctx_, result);
                }
                if (is_1d && result->ne[1] == 1) {
                    result = ggml_reshape_4d(ctx_, result, result->ne[0], result->ne[2], result->ne[3], 1);
                }
                break;
            }
            case GGML_OP_CONV_2D_DW: {
                // in0: weight [KW, KH, 1, C], in1: x [W, H, C, N]
                int s0 = op.attributes.count("stride_w") ? static_cast<int>(op.attributes.at("stride_w")) : 1;
                int s1 = op.attributes.count("stride_h") ? static_cast<int>(op.attributes.at("stride_h")) : 1;
                int p0 = op.attributes.count("pad_w") ? static_cast<int>(op.attributes.at("pad_w")) : 0;
                int p1 = op.attributes.count("pad_h") ? static_cast<int>(op.attributes.at("pad_h")) : 0;
                int d0 = op.attributes.count("dilation_w") ? static_cast<int>(op.attributes.at("dilation_w")) : 1;
                int d1 = op.attributes.count("dilation_h") ? static_cast<int>(op.attributes.at("dilation_h")) : 1;
                result = ggml_conv_2d_dw(ctx_, in0, in1, s0, s1, p0, p1, d0, d1);
                if (op.inputs.size() > 2) {
                    struct ggml_tensor* bias = ggml_tensors_[op.inputs[2]];
                    if (bias) {
                        if (!ggml_is_contiguous(bias)) bias = ggml_cont(ctx_, bias);
                        if (bias->ne[0] == result->ne[2] && bias->ne[1] == 1 && bias->ne[2] == 1) {
                            bias = ggml_reshape_4d(ctx_, bias, 1, 1, result->ne[2], 1);
                        }
                        if (ggml_can_repeat(bias, result)) {
                            bias = ggml_repeat(ctx_, bias, result);
                        }
                        result = ggml_add(ctx_, result, bias);
                    }
                }
                if (op.attributes.count("fused_relu") && op.attributes.at("fused_relu") != 0) {
                    result = ggml_relu(ctx_, result);
                }
                break;
            }
            case GGML_OP_CLAMP: {
                float min_val = op.float_attributes.count("min") ? static_cast<float>(op.float_attributes.at("min"))
                              : op.attributes.count("min") ? static_cast<float>(op.attributes.at("min")) : 0.0f;
                float max_val = op.float_attributes.count("max") ? static_cast<float>(op.float_attributes.at("max"))
                              : op.attributes.count("max") ? static_cast<float>(op.attributes.at("max")) : 6.0f;
                result = ggml_clamp(ctx_, in0, min_val, max_val);
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
            case GGML_OP_PAD: {
                int p0 = op.attributes.count("pad_w") ? static_cast<int>(op.attributes.at("pad_w")) : 0;
                int p1 = op.attributes.count("pad_h") ? static_cast<int>(op.attributes.at("pad_h")) : 0;
                int p2 = op.attributes.count("pad_c") ? static_cast<int>(op.attributes.at("pad_c")) : 0;
                int p3 = op.attributes.count("pad_n") ? static_cast<int>(op.attributes.at("pad_n")) : 0;
                result = ggml_pad(ctx_, in0, p0, p1, p2, p3);
                break;
            }
            case 200: { // GGML_OP_CUSTOM_BIAS_GELU: in0=x, in1=bias
                if (!in0 || !in1) {
                    throw std::runtime_error("GGML_OP_CUSTOM_BIAS_GELU requires 2 inputs");
                }
                if (is_cuda_) {
                    struct ggml_tensor* b = in1;
                    if (ggml_can_repeat(b, in0)) {
                        b = ggml_repeat(ctx_, b, in0);
                    }
                    result = ggml_gelu(ctx_, ggml_add(ctx_, in0, b));
                } else {
                    result = ggml_map_custom2(ctx_, in0, in1, ggmlc_compute_forward_bias_gelu, GGML_N_TASKS_MAX, nullptr);
                }
                break;
            }
            case 201: { // GGML_OP_CUSTOM_LAYER_NORM: in0=x, in1=weight, in2=bias (optional)
                struct ggml_tensor* w = in1;
                struct ggml_tensor* b = op.inputs.size() > 2 ? ggml_tensors_[op.inputs[2]] : nullptr;
                float eps = op.float_attributes.count("eps") ? static_cast<float>(op.float_attributes.at("eps"))
                          : op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;

                result = ggml_norm(ctx_, in0, eps);
                if (w) {
                    if (ggml_can_repeat(w, result)) {
                        w = ggml_repeat(ctx_, w, result);
                    }
                    result = ggml_mul(ctx_, result, w);
                }
                if (b) {
                    if (ggml_can_repeat(b, result)) {
                        b = ggml_repeat(ctx_, b, result);
                    }
                    result = ggml_add(ctx_, result, b);
                }
                break;
            }
            case 202: { // GGML_OP_CUSTOM_RMS_NORM: in0=x, in1=weight
                struct ggml_tensor* w = in1;
                float eps = op.float_attributes.count("eps") ? static_cast<float>(op.float_attributes.at("eps"))
                          : op.attributes.count("eps") ? static_cast<float>(op.attributes.at("eps")) : 1e-5f;

                result = ggml_rms_norm(ctx_, in0, eps);
                if (w) {
                    if (ggml_can_repeat(w, result)) {
                        w = ggml_repeat(ctx_, w, result);
                    }
                    result = ggml_mul(ctx_, result, w);
                }
                break;
            }
            case 203: { // GGML_OP_CUSTOM_SWIGLU: in0=gate, in1=up
                if (!in0 || !in1) {
                    throw std::runtime_error("GGML_OP_CUSTOM_SWIGLU requires 2 inputs (gate, up)");
                }
                if (is_cuda_) {
                    result = ggml_mul(ctx_, ggml_silu(ctx_, in0), in1);
                } else {
                    result = ggml_map_custom2(ctx_, in0, in1, ggmlc_compute_forward_swiglu, GGML_N_TASKS_MAX, nullptr);
                }
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

    // 4. Allocate tensor storage on the backend (CPU or CUDA)
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buffer_) {
        throw std::runtime_error("Failed to allocate tensors via GGML backend (" + device_ + ")");
    }

    // 5. Initialize parameters and constants on device buffer
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        if ((t.storage == StorageClass::PARAMETER || t.storage == StorageClass::CONSTANT) && t.data_ptr && t.data_size > 0) {
            auto it = ggml_tensors_.find(tid);
            if (it != ggml_tensors_.end()) {
                size_t sz = std::min<size_t>(t.data_size, ggml_nbytes(it->second));
                ggml_backend_tensor_set(it->second, t.data_ptr, 0, sz);
            }
        } else if (t.storage == StorageClass::STATE) {
            auto s_it = persistent_states_.find(tid);
            auto it = ggml_tensors_.find(tid);
            if (it != ggml_tensors_.end()) {
                size_t sz = ggml_nbytes(it->second);
                if (s_it != persistent_states_.end() && s_it->second.size() == sz) {
                    ggml_backend_tensor_set(it->second, s_it->second.data(), 0, sz);
                } else {
                    persistent_states_[tid].assign(sz, 0);
                    ggml_backend_tensor_memset(it->second, 0, 0, sz);
                }
            }
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
    ggml_backend_tensor_set(t, data, 0, size_bytes);
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
    if (!ctx_ || !cgraph_ || !backend_) {
        throw std::runtime_error("Executor not prepared. Call prepare() first.");
    }
    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    enum ggml_status status = ggml_backend_graph_compute(backend_, cgraph_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML backend graph compute failed with status: " + std::to_string(status));
    }

    // Execution completed successfully

    // Save persistent states from backend
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        if (pair.second.storage == StorageClass::STATE) {
            auto it = ggml_tensors_.find(tid);
            if (it != ggml_tensors_.end()) {
                size_t sz = ggml_nbytes(it->second);
                persistent_states_[tid].resize(sz);
                ggml_backend_tensor_get(it->second, persistent_states_[tid].data(), 0, sz);
            }
        }
    }
}

void ModelExecutor::set_state(uint32_t tensor_id, const void* data, size_t size_bytes) {
    persistent_states_[tensor_id].resize(size_bytes);
    std::memcpy(persistent_states_[tensor_id].data(), data, size_bytes);
    auto it = ggml_tensors_.find(tensor_id);
    if (it != ggml_tensors_.end() && ggml_nbytes(it->second) == size_bytes) {
        ggml_backend_tensor_set(it->second, data, 0, size_bytes);
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

const void* ModelExecutor::get_state_data(uint32_t tensor_id) {
    auto it = persistent_states_.find(tensor_id);
    if (it != persistent_states_.end() && !it->second.empty()) {
        return it->second.data();
    }
    auto t_it = ggml_tensors_.find(tensor_id);
    if (t_it != ggml_tensors_.end()) {
        size_t sz = ggml_nbytes(t_it->second);
        auto& host_buf = state_host_buffers_[tensor_id];
        host_buf.resize(sz);
        ggml_backend_tensor_get(t_it->second, host_buf.data(), 0, sz);
        return host_buf.data();
    }
    throw std::runtime_error("State tensor ID not found in executor: " + std::to_string(tensor_id));
}

const void* ModelExecutor::get_state_data_by_name(const std::string& name) {
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
                ggml_backend_tensor_memset(it->second, 0, 0, ggml_nbytes(it->second));
            }
        }
    }
}

const void* ModelExecutor::get_output_data(uint32_t tensor_id) {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    size_t sz = ggml_nbytes(it->second);
    auto& host_buf = output_host_buffers_[tensor_id];
    host_buf.resize(sz);
    ggml_backend_tensor_get(it->second, host_buf.data(), 0, sz);
    return host_buf.data();
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
