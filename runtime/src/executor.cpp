#include "ggmlc/executor.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <chrono>
#include <map>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <climits>
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#include "ggmlc/stdlib_kernels.h"

namespace ggmlc {

namespace {
struct ScopedTimer {
    double& acc;
    int* count;
    bool enabled;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(bool enable, double& acc_ms, int* n = nullptr)
        : acc(acc_ms), count(n), enabled(enable) {
        if (enabled) t0 = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        if (!enabled) return;
        acc += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (count) (*count)++;
    }
};

bool is_metadata_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

int64_t pad_kv_len(int64_t n_kv, int64_t max_ctx, int64_t n_pad) {
    if (n_kv < 1) n_kv = 1;
    if (n_pad <= 1) return std::min(n_kv, max_ctx);
    int64_t padded = std::max(n_pad, static_cast<int64_t>(GGML_PAD(n_kv, n_pad)));
    if (padded > max_ctx) padded = max_ctx;
    return padded;
}

bool env_flag_enabled(const char* name) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return false;
    return e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T';
}

// Reinterpret a row-strided fused-QKV (or gate_up) slice as a head-split layout
// without CONT. Only matches VIEW into a wider parent where the outer stride
// still carries the packed-parent gap (nb[1] > ne[0]*nb[0]).
struct ggml_tensor* try_reshape_strided_view(
    struct ggml_context* ctx,
    struct ggml_tensor* in0,
    const std::array<int64_t, 4>& ne
) {
    if (!in0 || !ctx) return nullptr;
    if (!in0->view_src) return nullptr;
    if (ggml_nelements(in0) != ne[0] * ne[1] * ne[2] * ne[3]) return nullptr;
    if (!ggml_is_contiguous_rows(in0)) return nullptr;
    // Must be a slice of a wider packed parent (fused QKV / gate_up).
    if (in0->view_src->ne[0] <= in0->ne[0]) return nullptr;
    if (in0->nb[1] <= in0->ne[0] * in0->nb[0]) return nullptr;

    // Split ne[0] -> (ne[0], ne[1]) while keeping trailing dims: [E, S, A, B] -> [D, H, S, A]
    if (in0->ne[0] == ne[0] * ne[1] &&
        in0->ne[1] == ne[2] &&
        in0->ne[2] == ne[3] &&
        in0->ne[3] == 1) {
        const size_t nb0 = in0->nb[0];
        const size_t nb1 = nb0 * static_cast<size_t>(ne[0]);
        const size_t nb2 = in0->nb[1];
        const size_t nb3 = nb2 * static_cast<size_t>(ne[2]);
        return ggml_view_4d(ctx, in0, ne[0], ne[1], ne[2], ne[3], nb1, nb2, nb3, 0);
    }

    // [E, S, 1, 1] -> [D, H, S, 1]
    if (in0->ne[0] == ne[0] * ne[1] &&
        in0->ne[1] == ne[2] &&
        ne[3] == 1 &&
        in0->ne[2] == 1 && in0->ne[3] == 1) {
        const size_t nb0 = in0->nb[0];
        const size_t nb1 = nb0 * static_cast<size_t>(ne[0]);
        const size_t nb2 = in0->nb[1];
        const size_t nb3 = nb2 * static_cast<size_t>(ne[2]);
        return ggml_view_4d(ctx, in0, ne[0], ne[1], ne[2], ne[3], nb1, nb2, nb3, 0);
    }

    return nullptr;
}

struct ggml_tensor* reshape4d_contig(
    struct ggml_context* ctx,
    struct ggml_tensor* t,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3
) {
    if (!t) return nullptr;
    if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
    return ggml_reshape_4d(ctx, t, ne0, ne1, ne2, ne3);
}

void fill_f16_causal_mask(struct ggml_tensor* mask, int64_t pos, int64_t s_q, int64_t s_kv) {
    if (!mask || mask->buffer == nullptr || s_q <= 0 || s_kv <= 0) return;
    std::vector<ggml_fp16_t> mask_data(static_cast<size_t>(s_kv * s_q));
    ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
    ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(ATTN_MASK_MIN_FP16);
    for (int64_t i = 0; i < s_q; ++i) {
        for (int64_t j = 0; j < s_kv; ++j) {
            mask_data[static_cast<size_t>(i * s_kv + j)] =
                (j <= pos + i) ? zero_f16 : neg_inf_f16;
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
}
} // namespace

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
#if defined(GGML_USE_METAL)
    devices.push_back("metal");
#endif
    return devices;
}

ModelExecutor::ModelExecutor(const SerializedModelGraph& graph, const std::string& device)
    : model_graph_(graph), device_(device), backend_(nullptr), buffer_(nullptr), galloc_(nullptr), ctx_(nullptr), cgraph_(nullptr) {
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
#elif defined(GGML_USE_METAL)
        dev_lower = "metal";
        device_ = "metal";
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
    }
#if defined(GGML_USE_METAL)
    else if (dev_lower == "metal") {
        backend_ = ggml_backend_metal_init();
        if (!backend_) {
            throw std::runtime_error("Failed to initialize GGML Metal backend.");
        }
        device_ = "metal";
        is_cuda_ = false;
    }
#endif
    else {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) {
            throw std::runtime_error("Failed to initialize GGML CPU backend.");
        }
        device_ = "cpu";
        is_cuda_ = false;
    }
}

ModelExecutor::~ModelExecutor() {
    if (cpu_threadpool_) {
        ggml_backend_cpu_set_threadpool(backend_, nullptr);
        ggml_threadpool_free(cpu_threadpool_);
        cpu_threadpool_ = nullptr;
    }
    clear_all_graph_buckets();
    if (galloc_) {
        ggml_gallocr_free(galloc_);
        galloc_ = nullptr;
    }
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    if (state_buffer_) {
        ggml_backend_buffer_free(state_buffer_);
        state_buffer_ = nullptr;
    }
    if (ctx_state_) {
        ggml_free(ctx_state_);
        ctx_state_ = nullptr;
    }
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (ctx_w_) {
        ggml_free(ctx_w_);
        ctx_w_ = nullptr;
    }
    if (kv_cache_buffer_) {
        ggml_backend_buffer_free(kv_cache_buffer_);
        kv_cache_buffer_ = nullptr;
    }
    if (ctx_kv_cache_) {
        ggml_free(ctx_kv_cache_);
        ctx_kv_cache_ = nullptr;
    }
    if (vmm_mgr_) {
        vmm_mgr_->reset();
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

bool ModelExecutor::has_state_tensors() const {
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.storage == StorageClass::STATE) return true;
    }
    return false;
}

void ModelExecutor::init_weights() {
    if (weights_loaded_) return;

    size_t n_weights = 0;
    for (const auto& pair : model_graph_.tensors) {
        const auto& t = pair.second;
        if ((t.storage == StorageClass::PARAMETER || t.storage == StorageClass::CONSTANT) && t.is_static()) {
            n_weights++;
        }
    }
    if (n_weights == 0) {
        weights_loaded_ = true;
        return;
    }

    size_t ctx_meta_size = (n_weights + 64) * ggml_tensor_overhead() + 4 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_w_ = ggml_init(params);
    if (!ctx_w_) {
        throw std::runtime_error("Failed to initialize ggml_context for weights");
    }

    std::unordered_map<std::string, int64_t> empty_env;
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        if ((t.storage == StorageClass::PARAMETER || t.storage == StorageClass::CONSTANT) && t.is_static()) {
            std::array<int64_t, 4> ne;
            for (int d = 0; d < 4; ++d) {
                ne[d] = t.ne[d]->evaluate(empty_env, model_graph_.symbol_table);
            }
            concrete_shapes_[tid] = ne;

            struct ggml_tensor* g_t = ggml_new_tensor_4d(ctx_w_, t.type, ne[0], ne[1], ne[2], ne[3]);
            if (!g_t) {
                throw std::runtime_error("Failed to allocate weight ggml_tensor for: " + t.name);
            }
            ggml_set_name(g_t, t.name.c_str());
            weight_tensors_[tid] = g_t;
        }
    }

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_w_, backend_);
    if (!weight_buffer_) {
        throw std::runtime_error("Failed to allocate weight buffer on backend (" + device_ + ")");
    }

    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        if ((t.storage == StorageClass::PARAMETER || t.storage == StorageClass::CONSTANT) && t.is_static() && t.data_ptr && t.data_size > 0) {
            auto it = weight_tensors_.find(tid);
            if (it != weight_tensors_.end()) {
                size_t sz = std::min<size_t>(t.data_size, ggml_nbytes(it->second));
                ggml_backend_tensor_set(it->second, t.data_ptr, 0, sz);
            }
        }
    }

    weights_loaded_ = true;
}

void ModelExecutor::init_states(const std::unordered_map<std::string, int64_t>& symbol_env) {
    if (states_allocated_) return;

    size_t n_states = 0;
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.storage == StorageClass::STATE) n_states++;
    }
    if (n_states == 0) {
        states_allocated_ = true;
        return;
    }

    size_t ctx_meta_size = (n_states + 32) * ggml_tensor_overhead() + 2 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_state_ = ggml_init(params);
    if (!ctx_state_) {
        throw std::runtime_error("Failed to initialize ggml_context for state tensors");
    }

    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        const auto& t = pair.second;
        if (t.storage == StorageClass::STATE) {
            std::array<int64_t, 4> ne;
            for (int d = 0; d < 4; ++d) {
                ne[d] = t.ne[d]->evaluate(symbol_env, model_graph_.symbol_table);
            }
            concrete_shapes_[tid] = ne;

            struct ggml_tensor* g_t = ggml_new_tensor_4d(ctx_state_, t.type, ne[0], ne[1], ne[2], ne[3]);
            if (!g_t) {
                throw std::runtime_error("Failed to allocate state ggml_tensor for: " + t.name);
            }
            ggml_set_name(g_t, t.name.c_str());
            state_tensors_[tid] = g_t;
        }
    }

    state_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_state_, backend_);
    if (!state_buffer_) {
        throw std::runtime_error("Failed to allocate state buffer on backend (" + device_ + ")");
    }

    for (const auto& pair : state_tensors_) {
        uint32_t tid = pair.first;
        struct ggml_tensor* g_t = pair.second;
        size_t sz = ggml_nbytes(g_t);
        auto s_it = persistent_states_.find(tid);
        if (s_it != persistent_states_.end() && s_it->second.size() == sz) {
            ggml_backend_tensor_set(g_t, s_it->second.data(), 0, sz);
        } else {
            persistent_states_[tid].assign(sz, 0);
            ggml_backend_tensor_memset(g_t, 0, 0, sz);
        }
    }

    states_allocated_ = true;
}

void ModelExecutor::init_kv_cache(int64_t max_ctx) {
    if (kv_cache_buffer_) return;
    kv_cache_max_ctx_ = max_ctx;
    if (const char* env_pad = std::getenv("GGMLC_KV_PAD")) {
        kv_n_pad_ = std::max<int64_t>(1, std::atoll(env_pad));
    }

    // Only enable KV cache if model supports positional offsets (RoPE, explicit pos symbol, or arange constant tensors).
    // Models with absolute position embeddings hardcoded to input length (e.g. GPT-2)
    // require full sequence evaluation during autoregressive decode.
    bool has_rope = false;
    for (const auto& op : model_graph_.ops) {
        if (op.opcode == GGML_OP_ROPE) {
            has_rope = true;
            break;
        }
    }
    bool has_pos_sym = false;
    for (const auto& sym : model_graph_.symbol_table) {
        if (sym == "pos") {
            has_pos_sym = true;
            break;
        }
    }
    bool has_pos_input = false;
    for (uint32_t in_id : model_graph_.inputs) {
        auto it = model_graph_.tensors.find(in_id);
        if (it != model_graph_.tensors.end()) {
            if (it->second.name == "position_ids" || it->second.name.find("pos") != std::string::npos) {
                has_pos_input = true;
                break;
            }
        }
    }
    bool has_arange_tensor = false;
    for (const auto& pair : model_graph_.tensors) {
        if (pair.second.name.find("arange") != std::string::npos && pair.second.data_ptr) {
            has_arange_tensor = true;
            break;
        }
    }
    if (!has_rope && !has_pos_sym && !has_arange_tensor && !has_pos_input) return;

    std::vector<const SerializedOp*> attn_ops;
    for (const auto& op : model_graph_.ops) {
        if (op.opcode == GGML_OP_FLASH_ATTN_EXT) {
            attn_ops.push_back(&op);
        }
    }
    if (attn_ops.empty()) return;

    size_t ctx_meta_size = (attn_ops.size() * 2 + 32) * ggml_tensor_overhead() + 2 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_kv_cache_ = ggml_init(params);
    if (!ctx_kv_cache_) {
        throw std::runtime_error("Failed to initialize ggml_context for KV cache");
    }

    std::unordered_map<std::string, int64_t> dummy_env;
    dummy_env["s"] = 1;
    for (const auto& sym : model_graph_.symbol_table) {
        dummy_env[sym] = 1;
    }

    for (const auto* op : attn_ops) {
        uint32_t k_id = op->inputs[1];
        uint32_t v_id = op->inputs.size() > 2 ? op->inputs[2] : k_id;
        const auto& k_t = model_graph_.tensors.at(k_id);
        const auto& v_t = model_graph_.tensors.at(v_id);

        int64_t head_dim = k_t.ne[0]->evaluate(dummy_env, model_graph_.symbol_table);
        int64_t num_kv_heads = k_t.ne[2]->evaluate(dummy_env, model_graph_.symbol_table);
        int64_t batch = k_t.ne[3]->evaluate(dummy_env, model_graph_.symbol_table);

        enum ggml_type kv_type = (k_t.type == GGML_TYPE_F32) ? GGML_TYPE_F16 : k_t.type;
        struct ggml_tensor* g_k = ggml_new_tensor_4d(ctx_kv_cache_, kv_type, head_dim, max_ctx, num_kv_heads, batch);
        struct ggml_tensor* g_v = ggml_new_tensor_4d(ctx_kv_cache_, kv_type, head_dim, max_ctx, num_kv_heads, batch);

        std::string k_name = "kv_cache_k_" + std::to_string(op->id);
        std::string v_name = "kv_cache_v_" + std::to_string(op->id);
        ggml_set_name(g_k, k_name.c_str());
        ggml_set_name(g_v, v_name.c_str());

        kv_cache_k_[op->id] = g_k;
        kv_cache_v_[op->id] = g_v;
    }

    int64_t max_chunk_s = 2048;
    shared_mask_tensor_ = ggml_new_tensor_4d(ctx_kv_cache_, GGML_TYPE_F16, max_ctx, max_chunk_s, 1, 1);
    ggml_set_name(shared_mask_tensor_, "shared_causal_mask");

    kv_cache_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_kv_cache_, backend_);
    if (!kv_cache_buffer_) {
        throw std::runtime_error("Failed to allocate KV cache buffer on backend (" + device_ + ")");
    }

    reset_kv_cache();
    kv_cache_enabled_ = true;
}

void ModelExecutor::reset_kv_cache() {
    if (paged_kv_enabled_) {
        for (int i = 0; i < static_cast<int>(paged_slots_.size()); ++i) {
            paged_kv_free_slot(i);
        }
        // Paged slot teardown invalidates view offsets into VMM windows.
        clear_all_graph_buckets();
        decode_graph_cached_ = false;
        decode_attn_views_.clear();
        decode_rope_arange_tensors_.clear();
        decode_cached_n_kv_ = -1;
        chunk_graph_cached_ = false;
        chunk_attn_views_.clear();
        chunk_cached_pos_ = -1;
        chunk_cached_s_ = -1;
        chunk_cached_n_kv_ = -1;
        return;
    }
    for (const auto& pair : kv_cache_k_) {
        if (pair.second) {
            ggml_backend_tensor_memset(pair.second, 0, 0, ggml_nbytes(pair.second));
        }
    }
    for (const auto& pair : kv_cache_v_) {
        if (pair.second) {
            ggml_backend_tensor_memset(pair.second, 0, 0, ggml_nbytes(pair.second));
        }
    }
    // Keep prepared (s, n_kv) graph buckets alive across resets — llama.cpp
    // likewise reuses can_reuse graphs after clearing KV contents.
}

void ModelExecutor::init_paged_kv_cache(size_t max_batch, size_t max_ctx) {
    if (paged_kv_enabled_) return;
    paged_max_batch_ = max_batch;
    paged_max_ctx_ = max_ctx;

    if (!is_cuda_) {
        init_kv_cache(max_ctx);
        return;
    }

    vmm_mgr_ = std::make_unique<VMMBlockManager>();
    if (!vmm_mgr_->init(0)) {
        fprintf(stderr, "[PAGED_KV] VMM initialization failed on device 0, falling back to standard KV cache\n");
        init_kv_cache(max_ctx);
        return;
    }

    if (!weights_loaded_) {
        init_weights();
    }

    std::vector<const SerializedOp*> attn_ops;
    for (const auto& op : model_graph_.ops) {
        if (op.opcode == GGML_OP_FLASH_ATTN_EXT) {
            attn_ops.push_back(&op);
        }
    }
    if (attn_ops.empty()) return;

    paged_slots_.resize(max_batch);
    for (size_t i = 0; i < max_batch; ++i) {
        paged_slots_[i].slot_id = static_cast<int>(i);
        paged_slots_[i].active = false;
        paged_slots_[i].current_tokens = 0;
    }

    size_t ctx_meta_size = (attn_ops.size() * 2 + 32) * ggml_tensor_overhead() + 2 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_kv_cache_ = ggml_init(params);
    if (!ctx_kv_cache_) {
        throw std::runtime_error("Failed to initialize ggml_context for Paged KV cache");
    }

    std::unordered_map<std::string, int64_t> dummy_env;
    dummy_env["s"] = 1;
    for (const auto& sym : model_graph_.symbol_table) {
        dummy_env[sym] = 1;
    }

    for (const auto* op : attn_ops) {
        uint32_t k_id = op->inputs[1];
        uint32_t v_id = op->inputs.size() > 2 ? op->inputs[2] : k_id;
        const auto& k_t = model_graph_.tensors.at(k_id);
        const auto& v_t = model_graph_.tensors.at(v_id);

        int64_t head_dim = k_t.ne[0]->evaluate(dummy_env, model_graph_.symbol_table);
        int64_t num_kv_heads = k_t.ne[2]->evaluate(dummy_env, model_graph_.symbol_table);
        enum ggml_type kv_type = (k_t.type == GGML_TYPE_F32) ? GGML_TYPE_F16 : k_t.type;
        size_t elem_sz = ggml_type_size(kv_type);

        size_t bytes_per_token = head_dim * num_kv_heads * elem_sz;
        size_t page_sz = vmm_mgr_->page_size();
        if (tokens_per_page_ == 0) {
            tokens_per_page_ = (bytes_per_token > 0) ? std::min<size_t>(1024, page_sz / bytes_per_token) : 1024;
        }

        size_t raw_slot_bytes = head_dim * max_ctx * num_kv_heads * elem_sz;
        size_t slot_bytes = ((raw_slot_bytes + page_sz - 1) / page_sz) * page_sz;
        size_t total_window = max_batch * slot_bytes;

        uint64_t va_k = vmm_mgr_->reserve_virtual_window(total_window);
        uint64_t va_v = vmm_mgr_->reserve_virtual_window(total_window);

        vmm_k_va_windows_[op->id] = va_k;
        vmm_v_va_windows_[op->id] = va_v;
        vmm_slot_bytes_[op->id] = slot_bytes;

        struct ggml_tensor* g_k = ggml_new_tensor_4d(ctx_kv_cache_, kv_type, head_dim, max_ctx, num_kv_heads, max_batch);
        struct ggml_tensor* g_v = ggml_new_tensor_4d(ctx_kv_cache_, kv_type, head_dim, max_ctx, num_kv_heads, max_batch);

        g_k->nb[3] = slot_bytes;
        g_v->nb[3] = slot_bytes;
        g_k->data = reinterpret_cast<void*>(va_k);
        g_v->data = reinterpret_cast<void*>(va_v);
        g_k->buffer = weight_buffer_;
        g_v->buffer = weight_buffer_;

        std::string k_name = "kv_cache_k_" + std::to_string(op->id);
        std::string v_name = "kv_cache_v_" + std::to_string(op->id);
        ggml_set_name(g_k, k_name.c_str());
        ggml_set_name(g_v, v_name.c_str());

        kv_cache_k_[op->id] = g_k;
        kv_cache_v_[op->id] = g_v;
    }

    paged_kv_enabled_ = true;
    kv_cache_enabled_ = true;
}

void ModelExecutor::paged_kv_alloc_slot(int slot_id, const std::string& prefix_hash) {
    if (!paged_kv_enabled_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) return;
    auto& slot = paged_slots_[slot_id];
    slot.active = true;
    slot.prefix_hash = prefix_hash;
    slot.current_tokens = 0;

    if (!prefix_hash.empty() && vmm_mgr_) {
        for (const auto& pair : vmm_slot_bytes_) {
            uint32_t op_id = pair.first;
            size_t slot_bytes = pair.second;
            std::string key_k = prefix_hash + "_k_op" + std::to_string(op_id);
            std::string key_v = prefix_hash + "_v_op" + std::to_string(op_id);
            uint64_t shared_k = vmm_mgr_->get_prefix_page(key_k);
            uint64_t shared_v = vmm_mgr_->get_prefix_page(key_v);
            if (shared_k && shared_v) {
                uint64_t va_k_base = vmm_k_va_windows_[op_id] + slot_id * slot_bytes;
                uint64_t va_v_base = vmm_v_va_windows_[op_id] + slot_id * slot_bytes;
                vmm_mgr_->retain_physical_page(shared_k);
                vmm_mgr_->retain_physical_page(shared_v);
                vmm_mgr_->map_page(va_k_base, shared_k);
                vmm_mgr_->map_page(va_v_base, shared_v);
                slot.mapped_pages_k[op_id].push_back(shared_k);
                slot.mapped_pages_v[op_id].push_back(shared_v);
            }
        }
    }
}

void ModelExecutor::paged_kv_free_slot(int slot_id) {
    if (!paged_kv_enabled_ || !vmm_mgr_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) return;
    auto& slot = paged_slots_[slot_id];
    if (!slot.active && slot.mapped_pages_k.empty()) return;

    size_t page_sz = vmm_mgr_->page_size();

    for (const auto& pair : vmm_slot_bytes_) {
        uint32_t op_id = pair.first;
        size_t slot_bytes = pair.second;
        uint64_t va_k_base = vmm_k_va_windows_[op_id] + slot_id * slot_bytes;
        uint64_t va_v_base = vmm_v_va_windows_[op_id] + slot_id * slot_bytes;

        auto it_k = slot.mapped_pages_k.find(op_id);
        if (it_k != slot.mapped_pages_k.end()) {
            for (size_t i = 0; i < it_k->second.size(); ++i) {
                vmm_mgr_->unmap_page(va_k_base + i * page_sz);
                vmm_mgr_->release_physical_page(it_k->second[i]);
            }
            it_k->second.clear();
        }

        auto it_v = slot.mapped_pages_v.find(op_id);
        if (it_v != slot.mapped_pages_v.end()) {
            for (size_t i = 0; i < it_v->second.size(); ++i) {
                vmm_mgr_->unmap_page(va_v_base + i * page_sz);
                vmm_mgr_->release_physical_page(it_v->second[i]);
            }
            it_v->second.clear();
        }
    }
    slot.mapped_pages_k.clear();
    slot.mapped_pages_v.clear();
    slot.current_tokens = 0;
    slot.active = false;
    slot.prefix_hash.clear();
}

bool ModelExecutor::is_paged_slot_allocated(int slot_id) const {
    if (!paged_kv_enabled_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) {
        return false;
    }
    return paged_slots_[slot_id].active;
}

void ModelExecutor::paged_kv_ensure_tokens(int slot_id, int64_t total_tokens) {
    if (!paged_kv_enabled_ || !vmm_mgr_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) return;
    auto& slot = paged_slots_[slot_id];

    size_t page_sz = vmm_mgr_->page_size();

    for (const auto& pair : vmm_slot_bytes_) {
        uint32_t op_id = pair.first;
        size_t slot_bytes = pair.second;
        uint64_t va_k_base = vmm_k_va_windows_[op_id] + slot_id * slot_bytes;
        uint64_t va_v_base = vmm_v_va_windows_[op_id] + slot_id * slot_bytes;

        auto& layer_pages_k = slot.mapped_pages_k[op_id];
        auto& layer_pages_v = slot.mapped_pages_v[op_id];
        size_t current_mapped = layer_pages_k.size() * page_sz;

        while (current_mapped < slot_bytes) {
            uint64_t h_k = vmm_mgr_->alloc_physical_page();
            uint64_t h_v = vmm_mgr_->alloc_physical_page();

            vmm_mgr_->map_page(va_k_base + current_mapped, h_k);
            vmm_mgr_->map_page(va_v_base + current_mapped, h_v);

            layer_pages_k.push_back(h_k);
            layer_pages_v.push_back(h_v);
            current_mapped += page_sz;
        }
    }
    slot.current_tokens = total_tokens;
}

size_t ModelExecutor::get_paged_active_vram_bytes() const {
    return vmm_mgr_ ? vmm_mgr_->total_mapped_physical_bytes() : 0;
}

void ModelExecutor::configure_vmm_pool(size_t max_warm_pages, size_t prealloc_pages) {
    if (vmm_mgr_) {
        vmm_mgr_->configure_pool(max_warm_pages, prealloc_pages);
    }
}

void ModelExecutor::paged_kv_map_existing_pages(
    int slot_id,
    size_t start_page_idx,
    const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_k_handles,
    const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_v_handles
) {
    if (!paged_kv_enabled_ || !vmm_mgr_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) return;
    auto& slot = paged_slots_[slot_id];
    size_t page_sz = vmm_mgr_->page_size();

    for (size_t p = 0; p < page_k_handles.size(); ++p) {
        size_t page_idx = start_page_idx + p;
        const auto& k_layer_map = page_k_handles[p];
        const auto& v_layer_map = page_v_handles[p];

        for (const auto& pair : vmm_slot_bytes_) {
            uint32_t op_id = pair.first;
            size_t slot_bytes = pair.second;
            uint64_t va_k_base = vmm_k_va_windows_[op_id] + slot_id * slot_bytes;
            uint64_t va_v_base = vmm_v_va_windows_[op_id] + slot_id * slot_bytes;

            auto it_k = k_layer_map.find(op_id);
            auto it_v = v_layer_map.find(op_id);
            if (it_k != k_layer_map.end() && it_v != v_layer_map.end()) {
                uint64_t h_k = it_k->second;
                uint64_t h_v = it_v->second;

                vmm_mgr_->retain_physical_page(h_k);
                vmm_mgr_->retain_physical_page(h_v);
                vmm_mgr_->map_page(va_k_base + page_idx * page_sz, h_k);
                vmm_mgr_->map_page(va_v_base + page_idx * page_sz, h_v);

                auto& layer_pages_k = slot.mapped_pages_k[op_id];
                auto& layer_pages_v = slot.mapped_pages_v[op_id];
                if (page_idx < layer_pages_k.size()) {
                    layer_pages_k[page_idx] = h_k;
                    layer_pages_v[page_idx] = h_v;
                } else {
                    layer_pages_k.push_back(h_k);
                    layer_pages_v.push_back(h_v);
                }
            }
        }
    }
}

void ModelExecutor::paged_kv_extract_page_handles(
    int slot_id,
    size_t page_idx,
    std::unordered_map<uint32_t, uint64_t>& out_k,
    std::unordered_map<uint32_t, uint64_t>& out_v
) const {
    out_k.clear();
    out_v.clear();
    if (!paged_kv_enabled_ || slot_id < 0 || static_cast<size_t>(slot_id) >= paged_slots_.size()) return;
    const auto& slot = paged_slots_[slot_id];

    for (const auto& pair : vmm_slot_bytes_) {
        uint32_t op_id = pair.first;
        auto it_k = slot.mapped_pages_k.find(op_id);
        auto it_v = slot.mapped_pages_v.find(op_id);
        if (it_k != slot.mapped_pages_k.end() && page_idx < it_k->second.size() &&
            it_v != slot.mapped_pages_v.end() && page_idx < it_v->second.size()) {
            out_k[op_id] = it_k->second[page_idx];
            out_v[op_id] = it_v->second[page_idx];
        }
    }
}

int64_t ModelExecutor::graph_bucket_key(int64_t s, int64_t n_kv) {
    return (s << 32) ^ n_kv;
}

void ModelExecutor::free_prepared_bucket(PreparedGraphBucket& bucket) {
    if (bucket.galloc) {
        ggml_gallocr_free(bucket.galloc);
        bucket.galloc = nullptr;
    }
    if (bucket.buffer) {
        ggml_backend_buffer_free(bucket.buffer);
        bucket.buffer = nullptr;
    }
    if (bucket.ctx) {
        ggml_free(bucket.ctx);
        bucket.ctx = nullptr;
    }
    bucket.cgraph = nullptr;
    bucket.compute_tensors.clear();
    bucket.attn_views.clear();
    bucket.kv_indices = nullptr;
    bucket.kv_work_mask = nullptr;
    bucket.dynamic_causal_masks.clear();
    bucket.rope_arange_tensors.clear();
    bucket.concrete_shapes.clear();
    bucket.custom_params.clear();
    bucket.s = -1;
    bucket.n_kv = -1;
    bucket.pos = -1;
}

void ModelExecutor::clear_decode_graph_buckets() {
    for (auto& pair : decode_graph_buckets_) {
        free_prepared_bucket(pair.second);
    }
    decode_graph_buckets_.clear();
}

void ModelExecutor::clear_chunk_graph_buckets() {
    for (auto& pair : chunk_graph_buckets_) {
        free_prepared_bucket(pair.second);
    }
    chunk_graph_buckets_.clear();
}

void ModelExecutor::clear_all_graph_buckets() {
    clear_decode_graph_buckets();
    clear_chunk_graph_buckets();
}

void ModelExecutor::relink_ggml_tensors_from_compute() {
    ggml_tensors_.clear();
    for (const auto& pair : weight_tensors_) {
        ggml_tensors_[pair.first] = pair.second;
    }
    for (const auto& pair : state_tensors_) {
        ggml_tensors_[pair.first] = pair.second;
    }
    for (const auto& pair : compute_tensors_) {
        ggml_tensors_[pair.first] = pair.second;
    }
}

void ModelExecutor::stash_active_decode_bucket() {
    if (!decode_graph_cached_ || decode_cached_n_kv_ < 0 || !ctx_) return;
    const int64_t key = decode_cached_n_kv_;
    PreparedGraphBucket& slot = decode_graph_buckets_[key];
    if (slot.ctx && slot.ctx != ctx_) {
        free_prepared_bucket(slot);
    }
    slot.s = 1;
    slot.n_kv = decode_cached_n_kv_;
    slot.pos = decode_cached_pos_;
    slot.buffer = buffer_;
    slot.galloc = galloc_;
    slot.ctx = ctx_;
    slot.cgraph = cgraph_;
    slot.compute_tensors = std::move(compute_tensors_);
    slot.attn_views = std::move(decode_attn_views_);
    slot.kv_indices = kv_indices_tensor_;
    slot.kv_work_mask = kv_work_mask_tensor_;
    slot.dynamic_causal_masks = std::move(dynamic_causal_masks_);
    slot.rope_arange_tensors = std::move(decode_rope_arange_tensors_);
    slot.concrete_shapes = concrete_shapes_;
    slot.custom_params = std::move(custom_params_storage_);

    buffer_ = nullptr;
    galloc_ = nullptr;
    ctx_ = nullptr;
    cgraph_ = nullptr;
    compute_tensors_.clear();
    decode_attn_views_.clear();
    kv_indices_tensor_ = nullptr;
    kv_work_mask_tensor_ = nullptr;
    dynamic_causal_masks_.clear();
    decode_rope_arange_tensors_.clear();
    custom_params_storage_.clear();
    decode_graph_cached_ = false;
    decode_cached_n_kv_ = -1;
    decode_cached_pos_ = -1;
    prepared_ = false;
}

void ModelExecutor::stash_active_chunk_bucket() {
    if (!chunk_graph_cached_ || chunk_cached_n_kv_ < 0 || chunk_cached_s_ < 0 || !ctx_) return;
    const int64_t key = graph_bucket_key(chunk_cached_s_, chunk_cached_n_kv_);
    PreparedGraphBucket& slot = chunk_graph_buckets_[key];
    if (slot.ctx && slot.ctx != ctx_) {
        free_prepared_bucket(slot);
    }
    slot.s = chunk_cached_s_;
    slot.n_kv = chunk_cached_n_kv_;
    slot.pos = chunk_cached_pos_;
    slot.buffer = buffer_;
    slot.galloc = galloc_;
    slot.ctx = ctx_;
    slot.cgraph = cgraph_;
    slot.compute_tensors = std::move(compute_tensors_);
    slot.attn_views = std::move(chunk_attn_views_);
    slot.kv_indices = kv_indices_tensor_;
    slot.kv_work_mask = kv_work_mask_tensor_;
    slot.dynamic_causal_masks = std::move(dynamic_causal_masks_);
    slot.rope_arange_tensors = std::move(decode_rope_arange_tensors_);
    slot.concrete_shapes = concrete_shapes_;
    slot.custom_params = std::move(custom_params_storage_);

    buffer_ = nullptr;
    galloc_ = nullptr;
    ctx_ = nullptr;
    cgraph_ = nullptr;
    compute_tensors_.clear();
    chunk_attn_views_.clear();
    kv_indices_tensor_ = nullptr;
    kv_work_mask_tensor_ = nullptr;
    dynamic_causal_masks_.clear();
    decode_rope_arange_tensors_.clear();
    custom_params_storage_.clear();
    chunk_graph_cached_ = false;
    chunk_cached_n_kv_ = -1;
    chunk_cached_pos_ = -1;
    chunk_cached_s_ = -1;
    prepared_ = false;
}

bool ModelExecutor::activate_decode_bucket(int64_t n_kv) {
    auto it = decode_graph_buckets_.find(n_kv);
    if (it == decode_graph_buckets_.end() || !it->second.ctx) return false;
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
    PreparedGraphBucket slot = std::move(it->second);
    decode_graph_buckets_.erase(it);

    stash_active_chunk_bucket();
    if (decode_graph_cached_ && decode_cached_n_kv_ >= 0 && decode_cached_n_kv_ != n_kv) {
        stash_active_decode_bucket();
    } else if (ctx_ && !(decode_graph_cached_ && decode_cached_n_kv_ == n_kv)) {
        if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
        if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
        if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
        cgraph_ = nullptr;
        compute_tensors_.clear();
        custom_params_storage_.clear();
        decode_attn_views_.clear();
        dynamic_causal_masks_.clear();
        decode_rope_arange_tensors_.clear();
        kv_indices_tensor_ = nullptr;
        kv_work_mask_tensor_ = nullptr;
    }

    buffer_ = slot.buffer;
    galloc_ = slot.galloc;
    ctx_ = slot.ctx;
    cgraph_ = slot.cgraph;
    compute_tensors_ = std::move(slot.compute_tensors);
    decode_attn_views_ = std::move(slot.attn_views);
    kv_indices_tensor_ = slot.kv_indices;
    kv_work_mask_tensor_ = slot.kv_work_mask;
    dynamic_causal_masks_ = std::move(slot.dynamic_causal_masks);
    decode_rope_arange_tensors_ = std::move(slot.rope_arange_tensors);
    concrete_shapes_ = std::move(slot.concrete_shapes);
    custom_params_storage_ = std::move(slot.custom_params);

    chunk_graph_cached_ = false;
    chunk_attn_views_.clear();
    chunk_cached_n_kv_ = -1;
    chunk_cached_s_ = -1;
    chunk_cached_pos_ = -1;

    decode_graph_cached_ = true;
    decode_cached_n_kv_ = n_kv;
    decode_cached_pos_ = slot.pos;
    relink_ggml_tensors_from_compute();
    prepared_ = true;
    if (cuda_graph_mgr_) {
        cuda_graph_mgr_->reset();
    }
    cuda_graph_needs_update_ = false;
    return true;
}

bool ModelExecutor::activate_chunk_bucket(int64_t s, int64_t n_kv) {
    const int64_t key = graph_bucket_key(s, n_kv);
    auto it = chunk_graph_buckets_.find(key);
    if (it == chunk_graph_buckets_.end() || !it->second.ctx) return false;
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
    PreparedGraphBucket slot = std::move(it->second);
    chunk_graph_buckets_.erase(it);

    stash_active_decode_bucket();
    if (chunk_graph_cached_ && (chunk_cached_s_ != s || chunk_cached_n_kv_ != n_kv)) {
        stash_active_chunk_bucket();
    } else if (ctx_ && !(chunk_graph_cached_ && chunk_cached_s_ == s && chunk_cached_n_kv_ == n_kv)) {
        if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = nullptr; }
        if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
        if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
        cgraph_ = nullptr;
        compute_tensors_.clear();
        custom_params_storage_.clear();
        chunk_attn_views_.clear();
        dynamic_causal_masks_.clear();
        decode_rope_arange_tensors_.clear();
        kv_indices_tensor_ = nullptr;
        kv_work_mask_tensor_ = nullptr;
    }

    buffer_ = slot.buffer;
    galloc_ = slot.galloc;
    ctx_ = slot.ctx;
    cgraph_ = slot.cgraph;
    compute_tensors_ = std::move(slot.compute_tensors);
    chunk_attn_views_ = std::move(slot.attn_views);
    kv_indices_tensor_ = slot.kv_indices;
    kv_work_mask_tensor_ = slot.kv_work_mask;
    dynamic_causal_masks_ = std::move(slot.dynamic_causal_masks);
    decode_rope_arange_tensors_ = std::move(slot.rope_arange_tensors);
    concrete_shapes_ = std::move(slot.concrete_shapes);
    custom_params_storage_ = std::move(slot.custom_params);

    decode_graph_cached_ = false;
    decode_attn_views_.clear();
    decode_cached_n_kv_ = -1;
    decode_cached_pos_ = -1;

    chunk_graph_cached_ = true;
    chunk_cached_s_ = s;
    chunk_cached_n_kv_ = n_kv;
    chunk_cached_pos_ = slot.pos;
    relink_ggml_tensors_from_compute();
    prepared_ = true;
    if (cuda_graph_mgr_) {
        cuda_graph_mgr_->reset();
    }
    cuda_graph_needs_update_ = false;
    return true;
}

void ModelExecutor::set_decode_pos(int64_t pos) {
    ScopedTimer timer(enable_profile_, profile_.set_decode_pos_ms, &profile_.n_set_decode_pos);
    if (!decode_graph_cached_) return;
    decode_cached_pos_ = pos;

    if (kv_indices_tensor_ && kv_indices_tensor_->buffer) {
        int64_t idx = pos;
        ggml_backend_tensor_set(kv_indices_tensor_, &idx, 0, sizeof(int64_t));
        if (kv_work_mask_tensor_ && kv_work_mask_tensor_->buffer) {
            ScopedTimer mask_timer(enable_profile_, profile_.mask_fill_ms);
            const int64_t mask_s_q = kv_work_mask_tensor_->ne[1];
            const int64_t mask_n_kv = decode_cached_n_kv_ > 0
                ? decode_cached_n_kv_
                : pad_kv_len(pos + 1, kv_cache_max_ctx_, kv_n_pad_);
            fill_f16_causal_mask(kv_work_mask_tensor_, pos, mask_s_q, mask_n_kv);
        }
    } else {
        // Legacy view+cpy path: mutate destination offsets and active KV length.
        for (auto& pair : decode_attn_views_) {
            uint32_t op_id = pair.first;
            auto& refs = pair.second;
            struct ggml_tensor* k_cache = kv_cache_k_[op_id];
            struct ggml_tensor* v_cache = kv_cache_v_[op_id];

            refs.k_slot->view_offs = refs.slot_base_offset_k + pos * k_cache->nb[1];
            refs.k_slot->data = static_cast<char*>(k_cache->data) + refs.k_slot->view_offs;

            refs.v_slot->view_offs = refs.slot_base_offset_v + pos * v_cache->nb[1];
            refs.v_slot->data = static_cast<char*>(v_cache->data) + refs.v_slot->view_offs;

            if (refs.k_cpy) {
                refs.k_cpy->view_offs = refs.k_slot->view_offs;
                refs.k_cpy->data = refs.k_slot->data;
            }
            if (refs.v_cpy) {
                refs.v_cpy->view_offs = refs.v_slot->view_offs;
                refs.v_cpy->data = refs.v_slot->data;
            }

            int64_t s_kv = pos + 1;
            refs.k_active->ne[1] = s_kv;
            refs.v_active->ne[1] = s_kv;
            if (refs.scores) refs.scores->ne[0] = s_kv;
            if (refs.probs) refs.probs->ne[0] = s_kv;
            if (refs.v_t) refs.v_t->ne[1] = s_kv;
        }
        if (enable_cuda_graph_ && cuda_graph_mgr_ && cuda_graph_mgr_->is_captured()) {
            cuda_graph_needs_update_ = true;
        }
    }

    // 2. Update RoPE arange position constants
    for (const auto& pair : decode_rope_arange_tensors_) {
        struct ggml_tensor* arange_tensor = pair.first;
        uint32_t tid = pair.second;
        const auto& t = model_graph_.tensors.at(tid);
        if (t.data_ptr && t.data_size > 0) {
            size_t elem_sz = ggml_type_size(t.type);
            size_t offset = static_cast<size_t>(pos) * elem_sz;
            if (offset < t.data_size) {
                size_t sz = std::min<size_t>(t.data_size - offset, ggml_nbytes(arange_tensor));
                ggml_backend_tensor_set(arange_tensor, static_cast<const uint8_t*>(t.data_ptr) + offset, 0, sz);
            }
        }
    }

    // 3. Update explicit position input tensors in-place
    for (uint32_t tid : pos_input_tids_) {
        auto it = ggml_tensors_.find(tid);
        if (it != ggml_tensors_.end() && it->second && it->second->buffer) {
            struct ggml_tensor* t = it->second;
            if (t->type == GGML_TYPE_I32) {
                int32_t p32 = static_cast<int32_t>(pos);
                ggml_backend_tensor_set(t, &p32, 0, sizeof(int32_t));
            } else if (t->type == GGML_TYPE_I64) {
                int64_t p64 = pos;
                ggml_backend_tensor_set(t, &p64, 0, sizeof(int64_t));
            }
        }
    }
}

void ModelExecutor::set_chunk_pos(int64_t pos, int64_t s_q) {
    ScopedTimer timer(enable_profile_, profile_.set_chunk_pos_ms, &profile_.n_set_chunk_pos);
    if (!chunk_graph_cached_) return;
    chunk_cached_pos_ = pos;
    chunk_cached_s_ = s_q;

    // SET_ROWS path: write via indices (no CPY). Graph shapes stay fixed for the
    // active (s_q, n_kv) pad bucket — matching llama.cpp can_reuse semantics.
    if (kv_indices_tensor_ && kv_indices_tensor_->buffer) {
        if (kv_indices_tensor_->ne[0] == s_q) {
            std::vector<int64_t> idxs(static_cast<size_t>(s_q));
            for (int64_t i = 0; i < s_q; ++i) idxs[static_cast<size_t>(i)] = pos + i;
            ggml_backend_tensor_set(kv_indices_tensor_, idxs.data(), 0, idxs.size() * sizeof(int64_t));
        }
        const int64_t n_kv = chunk_cached_n_kv_ > 0
            ? chunk_cached_n_kv_
            : pad_kv_len(pos + s_q, kv_cache_max_ctx_, kv_n_pad_);
        if (kv_work_mask_tensor_ && kv_work_mask_tensor_->buffer) {
            ScopedTimer mask_timer(enable_profile_, profile_.mask_fill_ms);
            // Fill the parent buffer; FA reads the active view prefix of width n_kv.
            fill_f16_causal_mask(kv_work_mask_tensor_, pos, s_q, n_kv);
        }
    } else {
        // Legacy view+cpy path: mutate destination offsets and active KV length.
        const int64_t s_kv = pos + s_q;
        for (auto& pair : chunk_attn_views_) {
            uint32_t op_id = pair.first;
            auto& refs = pair.second;
            struct ggml_tensor* k_cache = kv_cache_k_[op_id];
            struct ggml_tensor* v_cache = kv_cache_v_[op_id];

            refs.k_slot->view_offs = refs.slot_base_offset_k + pos * k_cache->nb[1];
            refs.k_slot->data = static_cast<char*>(k_cache->data) + refs.k_slot->view_offs;

            refs.v_slot->view_offs = refs.slot_base_offset_v + pos * v_cache->nb[1];
            refs.v_slot->data = static_cast<char*>(v_cache->data) + refs.v_slot->view_offs;

            if (refs.k_cpy) {
                refs.k_cpy->view_offs = refs.k_slot->view_offs;
                refs.k_cpy->data = refs.k_slot->data;
            }
            if (refs.v_cpy) {
                refs.v_cpy->view_offs = refs.v_slot->view_offs;
                refs.v_cpy->data = refs.v_slot->data;
            }

            refs.k_active->ne[1] = s_kv;
            refs.v_active->ne[1] = s_kv;

            if (refs.mask) {
                refs.mask->ne[0] = s_kv;
                refs.mask->ne[1] = s_q;
                refs.mask->nb[0] = sizeof(ggml_fp16_t);
                refs.mask->nb[1] = s_kv * sizeof(ggml_fp16_t);
                refs.mask->nb[2] = s_q * refs.mask->nb[1];
                refs.mask->nb[3] = refs.mask->nb[2];
            }
        }

        for (auto& minfo : dynamic_causal_masks_) {
            if (!minfo.mask_tensor || minfo.mask_tensor->buffer == nullptr) continue;
            minfo.mask_tensor->ne[0] = s_kv;
            minfo.mask_tensor->ne[1] = s_q;
            minfo.mask_tensor->nb[0] = sizeof(ggml_fp16_t);
            minfo.mask_tensor->nb[1] = s_kv * sizeof(ggml_fp16_t);
            minfo.mask_tensor->nb[2] = s_q * minfo.mask_tensor->nb[1];
            minfo.mask_tensor->nb[3] = minfo.mask_tensor->nb[2];
            minfo.pos = pos;
            minfo.s_kv = s_kv;
            minfo.s_q = s_q;
            size_t s_kv_sz = static_cast<size_t>(s_kv);
            size_t s_q_sz = static_cast<size_t>(s_q);
            std::vector<ggml_fp16_t> mask_data(s_kv_sz * s_q_sz);
            ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
            ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(ATTN_MASK_MIN_FP16);
            {
                ScopedTimer mask_timer(enable_profile_, profile_.mask_fill_ms);
                for (size_t i = 0; i < s_q_sz; ++i) {
                    for (size_t j = 0; j < s_kv_sz; ++j) {
                        mask_data[i * s_kv_sz + j] =
                            (j <= static_cast<size_t>(pos + i)) ? zero_f16 : neg_inf_f16;
                    }
                }
                ggml_backend_tensor_set(
                    minfo.mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
            }
        }

        if (enable_cuda_graph_ && cuda_graph_mgr_ && cuda_graph_mgr_->is_captured()) {
            cuda_graph_needs_update_ = true;
        }
    }

    // 3. Update RoPE arange position constants
    for (const auto& pair : decode_rope_arange_tensors_) {
        struct ggml_tensor* arange_tensor = pair.first;
        uint32_t tid = pair.second;
        const auto& t = model_graph_.tensors.at(tid);
        if (t.data_ptr && t.data_size > 0) {
            size_t elem_sz = ggml_type_size(t.type);
            size_t offset = static_cast<size_t>(pos) * elem_sz;
            if (offset < t.data_size) {
                size_t sz = std::min<size_t>(t.data_size - offset, ggml_nbytes(arange_tensor));
                ggml_backend_tensor_set(arange_tensor, static_cast<const uint8_t*>(t.data_ptr) + offset, 0, sz);
            }
        }
    }

    // 4. Update explicit position input tensors in-place
    for (uint32_t tid : pos_input_tids_) {
        auto it = ggml_tensors_.find(tid);
        if (it != ggml_tensors_.end() && it->second && it->second->buffer) {
            struct ggml_tensor* t = it->second;
            if (t->type == GGML_TYPE_I32) {
                std::vector<int32_t> p_vec(s_q);
                for (int64_t i = 0; i < s_q; ++i) p_vec[i] = static_cast<int32_t>(pos + i);
                size_t cpy_sz = std::min<size_t>(p_vec.size() * sizeof(int32_t), ggml_nbytes(t));
                ggml_backend_tensor_set(t, p_vec.data(), 0, cpy_sz);
            } else if (t->type == GGML_TYPE_I64) {
                std::vector<int64_t> p_vec(s_q);
                for (int64_t i = 0; i < s_q; ++i) p_vec[i] = pos + i;
                size_t cpy_sz = std::min<size_t>(p_vec.size() * sizeof(int64_t), ggml_nbytes(t));
                ggml_backend_tensor_set(t, p_vec.data(), 0, cpy_sz);
            }
        }
    }

    if (enable_cuda_graph_ && cuda_graph_mgr_ && cuda_graph_mgr_->is_captured()) {
        cuda_graph_needs_update_ = true;
    }
}

void ModelExecutor::prepare(const std::unordered_map<std::string, int64_t>& symbol_env, bool enable_arena_reuse) {
    ScopedTimer timer(enable_profile_, profile_.prepare_ms, &profile_.n_prepare);
    if (!weights_loaded_) {
        init_weights();
    }
    if (!states_allocated_ && has_state_tensors()) {
        init_states(symbol_env);
    }
    if (symbol_env.count("pos") > 0 && !kv_cache_enabled_) {
        init_kv_cache(kv_cache_max_ctx_);
    }

    bool is_single_token = false;
    if (symbol_env.count("s") > 0 && symbol_env.at("s") == 1) is_single_token = true;
    for (const auto& sym : model_graph_.symbol_table) {
        if (symbol_env.count(sym) > 0 && symbol_env.at(sym) == 1) {
            is_single_token = true;
            break;
        }
    }
    bool is_decode_step = kv_cache_enabled_ && symbol_env.count("pos") > 0 && is_single_token;

    pos_input_tids_.clear();
    for (uint32_t in_id : model_graph_.inputs) {
        auto it = model_graph_.tensors.find(in_id);
        if (it != model_graph_.tensors.end()) {
            if (it->second.name == "position_ids" || it->second.name.find("pos") != std::string::npos) {
                pos_input_tids_.push_back(in_id);
            }
        }
    }

    // Fast path 1: decode graph reuse (llama.cpp can_reuse analogue).
    // Same topology when all symbols except pos match and padded n_kv matches.
    bool can_use_cached_decode = is_decode_step && decode_graph_cached_ && !getenv("GGMLC_DISABLE_DECODE_CACHE");
    if (can_use_cached_decode) {
        for (const auto& pair : symbol_env) {
            if (pair.first == "pos") continue;
            auto it = last_symbol_env_.find(pair.first);
            if (it == last_symbol_env_.end() || it->second != pair.second) {
                can_use_cached_decode = false;
                break;
            }
        }
    }
    int64_t decode_n_kv = -1;
    if (is_decode_step && kv_cache_enabled_ && symbol_env.count("pos") > 0 && kv_n_pad_ > 1) {
        decode_n_kv = pad_kv_len(symbol_env.at("pos") + 1, kv_cache_max_ctx_, kv_n_pad_);
    }
    if (can_use_cached_decode && decode_n_kv > 0 && decode_n_kv != decode_cached_n_kv_) {
        can_use_cached_decode = false;
    }
    if (can_use_cached_decode) {
        set_decode_pos(symbol_env.at("pos"));
        last_symbol_env_ = symbol_env;
        return;
    }
    if (is_decode_step && decode_n_kv > 0 && !getenv("GGMLC_DISABLE_DECODE_CACHE")) {
        bool symbols_ok = true;
        for (const auto& pair : symbol_env) {
            if (pair.first == "pos") continue;
            auto it = last_symbol_env_.find(pair.first);
            if (it == last_symbol_env_.end() || it->second != pair.second) {
                symbols_ok = false;
                break;
            }
        }
        // Also accept first decode after a chunked prefill if only s/pos differ.
        if (!symbols_ok && last_symbol_env_.count("s") && last_symbol_env_.at("s") != 1) {
            symbols_ok = true;
            for (const auto& pair : symbol_env) {
                if (pair.first == "pos" || pair.first == "s") continue;
                auto it = last_symbol_env_.find(pair.first);
                if (it != last_symbol_env_.end() && it->second != pair.second) {
                    symbols_ok = false;
                    break;
                }
            }
        }
        if (symbols_ok && activate_decode_bucket(decode_n_kv)) {
            set_decode_pos(symbol_env.at("pos"));
            last_symbol_env_ = symbol_env;
            return;
        }
    }

    // Fast path 2: chunked prefill graph reuse / (s, n_kv) bucket swap.
    int64_t current_s = symbol_env.count("s") > 0 ? symbol_env.at("s") : -1;
    if (current_s < 0) {
        for (const auto& sym : model_graph_.symbol_table) {
            if (symbol_env.count(sym) > 0) {
                current_s = symbol_env.at(sym);
                break;
            }
        }
    }
    int64_t chunk_n_kv = -1;
    if (kv_cache_enabled_ && symbol_env.count("pos") > 0 && !is_single_token && current_s > 0) {
        chunk_n_kv = pad_kv_len(symbol_env.at("pos") + current_s, kv_cache_max_ctx_, kv_n_pad_);
    }
    bool can_use_cached_chunk = kv_cache_enabled_ && symbol_env.count("pos") > 0 && !is_single_token &&
                                chunk_graph_cached_ && chunk_cached_s_ == current_s && current_s > 0 &&
                                !getenv("GGMLC_DISABLE_CHUNK_CACHE");
    if (can_use_cached_chunk) {
        for (const auto& pair : symbol_env) {
            if (pair.first == "pos") continue;
            auto it = last_symbol_env_.find(pair.first);
            if (it == last_symbol_env_.end() || it->second != pair.second) {
                can_use_cached_chunk = false;
                break;
            }
        }
    }
    if (can_use_cached_chunk && kv_indices_tensor_) {
        // llama-style: reuse only when padded n_kv matches; otherwise swap buckets.
        if (kv_indices_tensor_->ne[0] != current_s ||
            (chunk_n_kv > 0 && chunk_cached_n_kv_ > 0 && chunk_n_kv != chunk_cached_n_kv_)) {
            can_use_cached_chunk = false;
        }
    } else if (can_use_cached_chunk && !kv_indices_tensor_) {
        // Legacy path still mutates s_kv in-place via set_chunk_pos.
    }
    if (can_use_cached_chunk) {
        set_chunk_pos(symbol_env.at("pos"), current_s);
        last_symbol_env_ = symbol_env;
        return;
    }
    if (kv_cache_enabled_ && symbol_env.count("pos") > 0 && !is_single_token &&
        current_s > 0 && chunk_n_kv > 0 && !getenv("GGMLC_DISABLE_CHUNK_CACHE")) {
        bool symbols_ok = true;
        for (const auto& pair : symbol_env) {
            if (pair.first == "pos") continue;
            auto it = last_symbol_env_.find(pair.first);
            if (it == last_symbol_env_.end() || it->second != pair.second) {
                symbols_ok = false;
                break;
            }
        }
        if (symbols_ok && activate_chunk_bucket(current_s, chunk_n_kv)) {
            set_chunk_pos(symbol_env.at("pos"), current_s);
            last_symbol_env_ = symbol_env;
            return;
        }
    }

    if (prepared_ && symbol_env == last_symbol_env_ && enable_arena_reuse == last_enable_arena_reuse_) {
        return;
    }

    // Rebuild path: stash the live bucket (so it can be swapped back later),
    // then free only the current compute context. Preserve sibling buckets.
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
    const bool keep_chunk_buckets =
        kv_cache_enabled_ && !is_single_token && current_s > 0 && chunk_n_kv > 0 &&
        !getenv("GGMLC_DISABLE_CHUNK_CACHE");
    const bool keep_decode_buckets =
        is_decode_step && decode_n_kv > 0 && !getenv("GGMLC_DISABLE_DECODE_CACHE");

    if (keep_chunk_buckets) {
        stash_active_chunk_bucket();
        stash_active_decode_bucket();
        // Drop chunk buckets with a different s_q — topology differs.
        for (auto it = chunk_graph_buckets_.begin(); it != chunk_graph_buckets_.end(); ) {
            if (it->second.s != current_s) {
                free_prepared_bucket(it->second);
                it = chunk_graph_buckets_.erase(it);
            } else {
                ++it;
            }
        }
    } else if (keep_decode_buckets) {
        stash_active_decode_bucket();
        stash_active_chunk_bucket();
    } else {
        clear_all_graph_buckets();
        decode_graph_cached_ = false;
        decode_cached_n_kv_ = -1;
        decode_attn_views_.clear();
        chunk_graph_cached_ = false;
        chunk_cached_n_kv_ = -1;
        chunk_cached_s_ = -1;
        chunk_attn_views_.clear();
    }

    decode_rope_arange_tensors_.clear();
    dynamic_causal_masks_.clear();
    kv_indices_tensor_ = nullptr;
    kv_work_mask_tensor_ = nullptr;
    if (cuda_graph_mgr_) {
        cuda_graph_mgr_->reset();
    }
    cuda_graph_needs_update_ = false;

    if (galloc_) {
        ggml_gallocr_free(galloc_);
        galloc_ = nullptr;
    }
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    compute_tensors_.clear();
    ggml_tensors_.clear();
    custom_params_storage_.clear();
    output_host_buffers_.clear();
    state_host_buffers_.clear();

    // Re-link pre-allocated static weights and persistent states
    for (const auto& pair : weight_tensors_) {
        ggml_tensors_[pair.first] = pair.second;
    }
    for (const auto& pair : state_tensors_) {
        ggml_tensors_[pair.first] = pair.second;
    }

    // 1. Evaluate concrete shapes and metadata for compute tensors (inputs, activations, outputs, and dynamic constants)
    size_t n_compute_tensors = 0;
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        if (weight_tensors_.find(tid) == weight_tensors_.end() && state_tensors_.find(tid) == state_tensors_.end()) {
            const auto& t = pair.second;
            std::array<int64_t, 4> ne;
            for (int d = 0; d < 4; ++d) {
                ne[d] = t.ne[d]->evaluate(symbol_env, model_graph_.symbol_table);
            }
            concrete_shapes_[tid] = ne;
            n_compute_tensors++;
        }
    }

    // Allocate ggml context with no_alloc=true so the backend buffer allocates memory
    size_t ctx_meta_size = (n_compute_tensors + model_graph_.ops.size() * 16 + 256) * ggml_tensor_overhead() + 8 * 1024 * 1024;
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize ggml_context");
    }

    // 2. Instantiate non-weight, non-state ggml_tensors (metadata only)
    for (const auto& pair : model_graph_.tensors) {
        uint32_t tid = pair.first;
        if (weight_tensors_.find(tid) == weight_tensors_.end() && state_tensors_.find(tid) == state_tensors_.end()) {
            const auto& t = pair.second;
            const auto& ne = concrete_shapes_[tid];
            struct ggml_tensor* g_t = ggml_new_tensor_4d(ctx_, t.type, ne[0], ne[1], ne[2], ne[3]);
            if (!g_t) {
                throw std::runtime_error("Failed to allocate ggml_tensor for: " + t.name);
            }
            ggml_set_name(g_t, t.name.c_str());
            if (t.data_ptr && t.data_size > 0) {
                ggml_set_output(g_t);
            }
            compute_tensors_[tid] = g_t;
            ggml_tensors_[tid] = g_t;
        }
    }

    // Graph Allocator Lifecycle Protection:
    // Mark all model graph inputs with both ggml_set_input and ggml_set_output so that:
    // 1. ggml_gallocr allocates them at non-overlapping addresses at the beginning of the arena.
    // 2. ggml_gallocr never frees or reuses their memory buffers for downstream operations,
    //    guaranteeing that input tensors remain intact across repeated inference runs (e.g. run_benchmark)
    //    and CUDA graph replays without redundant PCIe re-transfers.
    for (uint32_t inp_id : model_graph_.inputs) {
        if (compute_tensors_.count(inp_id)) {
            struct ggml_tensor* inp_t = compute_tensors_[inp_id];
            ggml_set_input(inp_t);
            ggml_set_output(inp_t);
        }
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
                a = reshape4d_contig(ctx_, a, 1, 1, a->ne[0], 1);
            }
            if (b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && b->ne[0] == a->ne[2]) {
                b = reshape4d_contig(ctx_, b, 1, 1, b->ne[0], 1);
            }

            if (ggml_can_repeat(b, a)) return {a, b};
            if (ggml_can_repeat(a, b)) {
                if (!ggml_is_contiguous(a)) a = ggml_cont(ctx_, a);
                a = ggml_repeat(ctx_, a, b);
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
                a = ggml_repeat_4d(ctx_, a, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
            }
            if (need_repeat_b) {
                if (!ggml_is_contiguous(b)) b = ggml_cont(ctx_, b);
                b = ggml_repeat_4d(ctx_, b, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
            }
            if (a->type != b->type) {
                if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32) {
                    if (!ggml_is_contiguous(a)) a = ggml_cont(ctx_, a);
                    a = ggml_cast(ctx_, a, GGML_TYPE_F32);
                } else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32) {
                    if (!ggml_is_contiguous(b)) b = ggml_cont(ctx_, b);
                    b = ggml_cast(ctx_, b, GGML_TYPE_F32);
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
                    result = reshape4d_contig(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                } else {
                    result = ggml_repeat_4d(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                }
                break;
            }
            case GGML_OP_CPY: {
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                auto out_type = model_graph_.tensors.at(out_id).type;
                if (in0->type == out_type) {
                    result = ggml_dup(ctx_, in0);
                } else {
                    result = ggml_cast(ctx_, in0, out_type);
                }
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
                // CUDA unary kernels require contiguous src0 (unary.cu).
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                if (op.attributes.count("is_rsqrt") && op.attributes.at("is_rsqrt")) {
                    struct ggml_tensor* sqrt_x = ggml_sqrt(ctx_, in0);
                    result = ggml_div(ctx_, sqrt_x, in0);
                } else {
                    result = ggml_sqrt(ctx_, in0);
                }
                break;
            }
            case GGML_OP_SQR: {
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
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
                    struct ggml_tensor* flat_hw = reshape4d_contig(ctx_, in0, in0->ne[0], in0->ne[1] * in0->ne[2], 1, in0->ne[3]);
                    flat_hw = ggml_cont(ctx_, flat_hw);
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, flat_hw));
                    t = ggml_mean(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_mean(ctx_, in0);
                }
                result = reshape4d_contig(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
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
                    struct ggml_tensor* flat_hw = reshape4d_contig(ctx_, in0, in0->ne[0], in0->ne[1] * in0->ne[2], 1, in0->ne[3]);
                    flat_hw = ggml_cont(ctx_, flat_hw);
                    struct ggml_tensor* t = ggml_cont(ctx_, ggml_transpose(ctx_, flat_hw));
                    t = ggml_sum_rows(ctx_, t);
                    result = ggml_cont(ctx_, ggml_transpose(ctx_, t));
                } else {
                    result = ggml_sum_rows(ctx_, in0);
                }
                result = reshape4d_contig(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_CONT:
                result = ggml_cont(ctx_, in0);
                break;
            case GGML_OP_SIN:
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                result = ggml_sin(ctx_, in0);
                break;
            case GGML_OP_COS:
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                result = ggml_cos(ctx_, in0);
                break;
            case GGML_OP_LOG:
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
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
                // Soft-cap / activations on fused-QKV views (e.g. Gemma3 tanh) need CONT;
                // keep the VIEW itself non-contiguous for the RoPE→FA prefill path.
                if (act_in && !ggml_is_contiguous(act_in)) act_in = ggml_cont(ctx_, act_in);
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
                        result = ggml_mul(ctx_, result, w);
                    }
                }
                if (op.inputs.size() > 2) {
                    struct ggml_tensor* b = ggml_tensors_[op.inputs[2]];
                    if (b) {
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
                result = reshape4d_contig(ctx_, raw_rows, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
                break;
            }
            case GGML_OP_ROPE: {
                int n_dims = op.attributes.count("n_dims") ? static_cast<int>(op.attributes.at("n_dims")) : in0->ne[0];
                int mode = op.attributes.count("mode") ? static_cast<int>(op.attributes.at("mode")) : 2; // Default to GGML_ROPE_TYPE_NEOX
                float freq_base = 10000.0f;
                if (op.float_attributes.count("freq_base")) {
                    freq_base = static_cast<float>(op.float_attributes.at("freq_base"));
                } else if (op.attributes.count("freq_base")) {
                    freq_base = static_cast<float>(op.attributes.at("freq_base"));
                }
                float freq_scale = 1.0f;
                if (op.float_attributes.count("freq_scale")) {
                    freq_scale = static_cast<float>(op.float_attributes.at("freq_scale"));
                } else if (op.attributes.count("freq_scale")) {
                    freq_scale = static_cast<float>(op.attributes.at("freq_scale"));
                }
                int n_ctx_orig = op.attributes.count("n_ctx_orig") ? static_cast<int>(op.attributes.at("n_ctx_orig")) : 0;
                float ext_factor = op.float_attributes.count("ext_factor") ? static_cast<float>(op.float_attributes.at("ext_factor")) : 0.0f;
                float attn_factor = op.float_attributes.count("attn_factor") ? static_cast<float>(op.float_attributes.at("attn_factor")) : 1.0f;
                float beta_fast = op.float_attributes.count("beta_fast") ? static_cast<float>(op.float_attributes.at("beta_fast")) : 0.0f;
                float beta_slow = op.float_attributes.count("beta_slow") ? static_cast<float>(op.float_attributes.at("beta_slow")) : 0.0f;

                struct ggml_tensor* freq_factors = op.inputs.size() > 2 ? ggml_tensors_[op.inputs[2]] : nullptr;

                struct ggml_tensor* rope_in = in0;
                bool permuted = false;
                if (in1 && in0->ne[2] != in1->ne[0] && in0->ne[1] == in1->ne[0]) {
                    rope_in = ggml_permute(ctx_, in0, 0, 2, 1, 3);
                    permuted = true;
                }

                struct ggml_tensor* rope_res = ggml_rope_ext(
                    ctx_, rope_in, in1, freq_factors, n_dims, mode, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                result = permuted ? ggml_permute(ctx_, rope_res, 0, 2, 1, 3) : rope_res;
                break;
            }
            case GGML_OP_FLASH_ATTN_EXT: {
                struct ggml_tensor* q = in0;
                struct ggml_tensor* k = in1;
                struct ggml_tensor* v = op.inputs.size() > 2 ? ggml_tensors_[op.inputs[2]] : nullptr;
                struct ggml_tensor* mask = op.inputs.size() > 3 ? ggml_tensors_[op.inputs[3]] : nullptr;
                if (!kv_cache_enabled_) {
                    if (!ggml_is_contiguous(q)) q = ggml_cont(ctx_, q);
                    if (!ggml_is_contiguous(k)) k = ggml_cont(ctx_, k);
                    if (v && !ggml_is_contiguous(v)) v = ggml_cont(ctx_, v);
                }
                if (mask) {
                    mask = ggml_clamp(ctx_, mask, ATTN_MASK_MIN_FP16, 0.0f);
                    if (mask->type != GGML_TYPE_F16) mask = ggml_cast(ctx_, mask, GGML_TYPE_F16);
                    if (!ggml_is_contiguous(mask)) mask = ggml_cont(ctx_, mask);
                }

                float scale = 1.0f / sqrtf((float)q->ne[0]);
                if (op.float_attributes.count("scale")) {
                    scale = static_cast<float>(op.float_attributes.at("scale"));
                } else if (op.attributes.count("scale")) {
                    scale = static_cast<float>(op.attributes.at("scale"));
                }
                bool is_causal = op.attributes.count("is_causal") && op.attributes.at("is_causal") != 0;
                bool fused_transpose = op.attributes.count("fused_transpose") && op.attributes.at("fused_transpose") != 0;

                if (kv_cache_enabled_ && symbol_env.count("pos") > 0 && kv_cache_k_.count(op.id) > 0 && kv_cache_v_.count(op.id) > 0) {
                    int64_t pos = symbol_env.at("pos");
                    int64_t s_q = q->ne[1];
                    struct ggml_tensor* k_cache = kv_cache_k_[op.id];
                    struct ggml_tensor* v_cache = kv_cache_v_[op.id];
                    int64_t head_dim = k->ne[0];
                    int64_t num_kv_heads = k->ne[2];
                    int64_t batch = k->ne[3];

                    int64_t slot_idx = symbol_env.count("slot") > 0 ? symbol_env.at("slot") : 0;
                    size_t base_slot_offset_k = slot_idx * k_cache->nb[3];
                    size_t base_slot_offset_v = slot_idx * v_cache->nb[3];

                    // Decode (s_q==1) and chunked prefill (s_q>1) both use SET_ROWS into a
                    // padded n_kv view so topology stays CUDA-graph stable within a pad bucket.
                    // Crossing a pad boundary (e.g. 512→1024) rebuilds once — same as llama.cpp.
                    const bool use_set_rows = kv_n_pad_ > 1 && !paged_kv_enabled_ &&
                                              !env_flag_enabled("GGMLC_DISABLE_SET_ROWS");

                    if (use_set_rows) {
                        const int64_t n_kv = pad_kv_len(pos + s_q, kv_cache_max_ctx_, kv_n_pad_);
                        if (!kv_indices_tensor_) {
                            kv_indices_tensor_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I64, s_q);
                            ggml_set_input(kv_indices_tensor_);
                            ggml_set_output(kv_indices_tensor_);
                            ggml_set_name(kv_indices_tensor_, "kv_row_indices");
                        }
                        if (!kv_work_mask_tensor_) {
                            // Parent buffer sized for max ctx; each (s, n_kv) bucket
                            // views a fixed-width prefix (llama.cpp-style pad buckets).
                            kv_work_mask_tensor_ = ggml_new_tensor_4d(
                                ctx_, GGML_TYPE_F16, kv_cache_max_ctx_, s_q, 1, 1);
                            ggml_set_input(kv_work_mask_tensor_);
                            ggml_set_output(kv_work_mask_tensor_);
                            ggml_set_name(kv_work_mask_tensor_, "kv_work_mask");
                        }
                        struct ggml_tensor* mask_active = ggml_view_4d(
                            ctx_, kv_work_mask_tensor_, n_kv, s_q, 1, 1,
                            n_kv * sizeof(ggml_fp16_t),
                            s_q * n_kv * sizeof(ggml_fp16_t),
                            s_q * n_kv * sizeof(ggml_fp16_t),
                            0);

                        struct ggml_tensor* k_src = k;
                        struct ggml_tensor* v_src = v;
                        if (k_src && !ggml_is_contiguous_rows(k_src)) k_src = ggml_cont(ctx_, k_src);
                        if (v_src && !ggml_is_contiguous_rows(v_src)) v_src = ggml_cont(ctx_, v_src);
                        if (k_src && k_src->type != GGML_TYPE_F32 && k_src->type != GGML_TYPE_F16) {
                            k_src = ggml_cast(ctx_, k_src, GGML_TYPE_F16);
                        }
                        if (v_src && v_src->type != GGML_TYPE_F32 && v_src->type != GGML_TYPE_F16) {
                            v_src = ggml_cast(ctx_, v_src, GGML_TYPE_F16);
                        }

                        struct ggml_tensor* k_dst = k_cache;
                        struct ggml_tensor* v_dst = v_cache;
                        if (slot_idx != 0 || k_cache->ne[3] != batch) {
                            k_dst = ggml_view_4d(
                                ctx_, k_cache, head_dim, k_cache->ne[1], num_kv_heads, batch,
                                k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], base_slot_offset_k);
                            v_dst = ggml_view_4d(
                                ctx_, v_cache, head_dim, v_cache->ne[1], num_kv_heads, batch,
                                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], base_slot_offset_v);
                        }

                        struct ggml_tensor* k_set = ggml_set_rows(ctx_, k_dst, k_src, kv_indices_tensor_);
                        struct ggml_tensor* v_set = ggml_set_rows(ctx_, v_dst, v_src, kv_indices_tensor_);
                        ggml_build_forward_expand(cgraph_, k_set);
                        ggml_build_forward_expand(cgraph_, v_set);

                        struct ggml_tensor* k_active = ggml_view_4d(
                            ctx_, k_cache, head_dim, n_kv, num_kv_heads, batch,
                            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                            base_slot_offset_k);
                        struct ggml_tensor* v_active = ggml_view_4d(
                            ctx_, v_cache, head_dim, n_kv, num_kv_heads, batch,
                            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                            base_slot_offset_v);

                        struct ggml_tensor* fattn_out = ggml_flash_attn_ext(
                            ctx_, q, k_active, v_active, mask_active, scale, 0.0f, 0.0f);
                        result = fused_transpose ? fattn_out : ggml_permute(ctx_, fattn_out, 0, 2, 1, 3);

                        AttnViewRefs refs;
                        refs.k_active = k_active;
                        refs.v_active = v_active;
                        refs.mask = mask_active;
                        refs.indices = kv_indices_tensor_;
                        refs.slot_base_offset_k = base_slot_offset_k;
                        refs.slot_base_offset_v = base_slot_offset_v;
                        if (is_decode_step) {
                            decode_cached_n_kv_ = n_kv;
                            decode_attn_views_[op.id] = refs;
                        } else if (s_q > 1) {
                            chunk_cached_n_kv_ = n_kv;
                            chunk_attn_views_[op.id] = refs;
                        }
                    } else {
                    // View slot in cache for new tokens at offset pos
                    struct ggml_tensor* k_slot = ggml_view_4d(
                        ctx_, k_cache, head_dim, s_q, num_kv_heads, batch,
                        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                        base_slot_offset_k + pos * k_cache->nb[1]
                    );
                    struct ggml_tensor* v_slot = ggml_view_4d(
                        ctx_, v_cache, head_dim, s_q, num_kv_heads, batch,
                        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                        base_slot_offset_v + pos * v_cache->nb[1]
                    );

                    // Copy new k and v into cache
                    struct ggml_tensor* k_cpy = ggml_cpy(ctx_, k, k_slot);
                    struct ggml_tensor* v_cpy = ggml_cpy(ctx_, v, v_slot);
                    ggml_build_forward_expand(cgraph_, k_cpy);
                    ggml_build_forward_expand(cgraph_, v_cpy);

                    // Active keys and values up to pos + s_q
                    int64_t s_kv = pos + s_q;
                    struct ggml_tensor* k_active = ggml_view_4d(
                        ctx_, k_cache, head_dim, s_kv, num_kv_heads, batch,
                        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                        base_slot_offset_k
                    );
                    struct ggml_tensor* v_active = ggml_view_4d(
                        ctx_, v_cache, head_dim, s_kv, num_kv_heads, batch,
                        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                        base_slot_offset_v
                    );

                    if (s_q == 1) {
                        // Direct fused flash attention kernel without causal mask (since s_q == 1)
                        struct ggml_tensor* fattn_out = ggml_flash_attn_ext(
                            ctx_, q, k_active, v_active, nullptr, scale, 0.0f, 0.0f
                        );
                        result = fused_transpose ? fattn_out : ggml_permute(ctx_, fattn_out, 0, 2, 1, 3);

                        if (is_decode_step) {
                            AttnViewRefs refs;
                            refs.k_slot = k_slot;
                            refs.v_slot = v_slot;
                            refs.k_cpy = k_cpy;
                            refs.v_cpy = v_cpy;
                            refs.k_active = k_active;
                            refs.v_active = v_active;
                            refs.slot_base_offset_k = base_slot_offset_k;
                            refs.slot_base_offset_v = base_slot_offset_v;
                            decode_attn_views_[op.id] = refs;
                        }
                    } else {
                        // Prefill with s_q > 1 using causal mask
                        struct ggml_tensor* mask_t = nullptr;
                        if (is_causal) {
                            if (shared_mask_tensor_) {
                                mask_t = shared_mask_tensor_;
                                mask_t->ne[0] = s_kv;
                                mask_t->ne[1] = s_q;
                                mask_t->nb[0] = sizeof(ggml_fp16_t);
                                mask_t->nb[1] = s_kv * sizeof(ggml_fp16_t);
                                mask_t->nb[2] = s_q * mask_t->nb[1];
                                mask_t->nb[3] = mask_t->nb[2];
                                if (dynamic_causal_masks_.empty()) {
                                    dynamic_causal_masks_.push_back({mask_t, pos, s_q, s_kv});
                                }
                            } else {
                                mask_t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, s_kv, s_q, 1, 1);
                                ggml_set_input(mask_t);
                                ggml_set_output(mask_t);
                                dynamic_causal_masks_.push_back({mask_t, pos, s_q, s_kv});
                            }
                        } else {
                            mask_t = mask;
                        }
                        struct ggml_tensor* fattn_out = ggml_flash_attn_ext(
                            ctx_, q, k_active, v_active, mask_t, scale, 0.0f, 0.0f
                        );
                        result = fused_transpose ? fattn_out : ggml_permute(ctx_, fattn_out, 0, 2, 1, 3);

                        AttnViewRefs refs;
                        refs.k_slot = k_slot;
                        refs.v_slot = v_slot;
                        refs.k_cpy = k_cpy;
                        refs.v_cpy = v_cpy;
                        refs.k_active = k_active;
                        refs.v_active = v_active;
                        refs.mask = mask_t;
                        refs.slot_base_offset_k = base_slot_offset_k;
                        refs.slot_base_offset_v = base_slot_offset_v;
                        chunk_attn_views_[op.id] = refs;
                    }
                    }
                } else {
                    auto is_fattn_supported = [](int64_t hd) {
                        return (hd == 40 || hd == 64 || hd == 72 || hd == 80 ||
                                hd == 96 || hd == 112 || hd == 128 || hd == 192 || hd == 256);
                    };
                    if (!is_fattn_supported(q->ne[0])) {
                        struct ggml_tensor* kq = ggml_mul_mat(ctx_, k, q);
                        struct ggml_tensor* kq_scaled = ggml_scale(ctx_, kq, scale);
                        if (is_causal) {
                            kq_scaled = ggml_diag_mask_inf(ctx_, kq_scaled, 0);
                        }
                        if (mask) {
                            if (mask->type != kq_scaled->type) {
                                mask = ggml_cast(ctx_, mask, kq_scaled->type);
                            }
                            kq_scaled = ggml_add(ctx_, kq_scaled, mask);
                        }
                        struct ggml_tensor* kq_soft = ggml_soft_max(ctx_, kq_scaled);
                        struct ggml_tensor* v_t = ggml_cont(ctx_, ggml_transpose(ctx_, v));
                        result = ggml_mul_mat(ctx_, v_t, kq_soft);
                        if (fused_transpose) {
                            result = ggml_permute(ctx_, result, 0, 2, 1, 3);
                        }
                    } else {
                        int64_t s_q = q->ne[1];
                        int64_t s_k = k->ne[1];
                        struct ggml_tensor* mask_t = nullptr;
                        if (is_causal && s_q > 1) {
                            mask_t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, s_k, s_q, 1, 1);
                            ggml_set_input(mask_t);
                            ggml_set_output(mask_t);
                            dynamic_causal_masks_.push_back({mask_t, 0, s_q, s_k});
                        } else if (is_causal && s_q == 1) {
                            mask_t = nullptr;
                        } else {
                            mask_t = mask;
                        }

                        if (k->type != GGML_TYPE_F16) k = ggml_cast(ctx_, k, GGML_TYPE_F16);
                        if (v && v->type != GGML_TYPE_F16) v = ggml_cast(ctx_, v, GGML_TYPE_F16);

                        struct ggml_tensor* fattn_out = ggml_flash_attn_ext(
                            ctx_, q, k, v, mask_t, scale, 0.0f, 0.0f
                        );
                        result = fused_transpose ? fattn_out : ggml_permute(ctx_, fattn_out, 0, 2, 1, 3);
                    }
                }
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
                // Bypass permute/transpose on quantized weight matrices.
                // Quantized types (e.g. Q4_0) cannot be non-contiguously permuted,
                // so we use the original pre-transpose weight directly and let
                // ggml_mul_mat handle the implicit transpose.
                // This must NOT apply to F32/F16 activation tensors (e.g. K^T in attention).
                if (!op.inputs.empty()) {
                    for (const auto& other_op : model_graph_.ops) {
                        for (uint32_t out_id_check : other_op.outputs) {
                            if (out_id_check == op.inputs[0]) {
                                if ((other_op.opcode == GGML_OP_PERMUTE || other_op.opcode == GGML_OP_TRANSPOSE) && !other_op.inputs.empty()) {
                                    struct ggml_tensor* orig_w = ggml_tensors_[other_op.inputs[0]];
                                    bool orig_is_quantized = orig_w && (orig_w->type != GGML_TYPE_F32 && orig_w->type != GGML_TYPE_F16);
                                    if (orig_is_quantized && orig_w && in1 && orig_w->ne[0] == in1->ne[0]) {
                                        in0 = orig_w;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
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
                // llama.cpp n_outputs=1 analogue: graph-output MUL_MAT (lm_head / tied
                // embed) only needs the last token during prefill/decode sampling.
                if (logits_last_only_ && in1 && in1->ne[1] > 1) {
                    bool is_graph_output = false;
                    for (uint32_t oid : model_graph_.outputs) {
                        if (oid == out_id) { is_graph_output = true; break; }
                    }
                    if (is_graph_output) {
                        const int64_t last = in1->ne[1] - 1;
                        in1 = ggml_view_2d(ctx_, in1, in1->ne[0], 1, in1->nb[1], last * in1->nb[1]);
                        if (!ggml_is_contiguous(in1)) {
                            in1 = ggml_cont(ctx_, in1);
                        }
                    }
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
                        result = ggml_add(ctx_, result, bias);
                    }
                }
                break;
            }
            case GGML_OP_RESHAPE: {
                const auto& ne = concrete_shapes_[out_id];
                // Prefer a strided view over CONT for fused-QKV VIEW->RESHAPE->ROPE/FA.
                // GGMLC_RESHAPE_FORCE_CONT=1 restores eager materialization for A/B.
                if (in0 && !ggml_is_contiguous(in0) && !env_flag_enabled("GGMLC_RESHAPE_FORCE_CONT")) {
                    struct ggml_tensor* viewed = try_reshape_strided_view(ctx_, in0, ne);
                    if (viewed) {
                        result = viewed;
                        break;
                    }
                }
                if (in0 && !ggml_is_contiguous(in0)) {
                    in0 = ggml_cont(ctx_, in0);
                }
                result = reshape4d_contig(ctx_, in0, ne[0], ne[1], ne[2], ne[3]);
                break;
            }
            case GGML_OP_PERMUTE: {
                // Keep permutes as metadata views when every consumer is FlashAttention
                // (RoPE->[D,H,S]->PERMUTE->[D,S,H]->FA). Vision graphs that feed CONV/LINEAR
                // after a permute still need CONT. GGMLC_PERMUTE_FORCE_CONT=1 always materializes.
                int ax0 = op.attributes.count("axis0") ? static_cast<int>(op.attributes.at("axis0")) : 1;
                int ax1 = op.attributes.count("axis1") ? static_cast<int>(op.attributes.at("axis1")) : 0;
                int ax2 = op.attributes.count("axis2") ? static_cast<int>(op.attributes.at("axis2")) : 2;
                int ax3 = op.attributes.count("axis3") ? static_cast<int>(op.attributes.at("axis3")) : 3;
                result = ggml_permute(ctx_, in0, ax0, ax1, ax2, ax3);

                bool force_cont = env_flag_enabled("GGMLC_PERMUTE_FORCE_CONT");
                bool fa_only = !op.outputs.empty();
                if (fa_only && !force_cont) {
                    const uint32_t pout = op.outputs[0];
                    bool saw_consumer = false;
                    for (const auto& other : model_graph_.ops) {
                        bool uses = false;
                        for (uint32_t iid : other.inputs) {
                            if (iid == pout) { uses = true; break; }
                        }
                        if (!uses) continue;
                        saw_consumer = true;
                        if (other.opcode != GGML_OP_FLASH_ATTN_EXT) {
                            fa_only = false;
                            break;
                        }
                    }
                    if (!saw_consumer) fa_only = false;
                } else {
                    fa_only = false;
                }

                if (force_cont || !fa_only) {
                    if (in0 && !ggml_is_contiguous(in0) && force_cont) {
                        // already permuted from possibly non-contig src; materialize result
                    }
                    if (ggml_nelements(result) < 10000000) {
                        result = ggml_cont(ctx_, result);
                    }
                }
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
                // Default: keep QKV fusion slices as non-contiguous views (zero copy).
                // GGMLC_VIEW_FORCE_CONT=1 restores eager materialization for A/B.
                if (env_flag_enabled("GGMLC_VIEW_FORCE_CONT")) {
                    if (in0 && (!ggml_is_contiguous(in0) || in0->view_src != nullptr)) {
                        in0 = ggml_cont(ctx_, in0);
                    }
                } else if (in0 && in0->view_src != nullptr && !ggml_is_contiguous(in0)) {
                    // Break pathological view-of-view chains that break offset math.
                    in0 = ggml_cont(ctx_, in0);
                }
                size_t offset = start * in0->nb[g_dim];
                size_t nb1 = in0->nb[1] * (g_dim == 1 ? step : 1);
                size_t nb2 = in0->nb[2] * (g_dim == 2 ? step : 1);
                size_t nb3 = in0->nb[3] * (g_dim == 3 ? step : 1);
                struct ggml_tensor* v = ggml_view_4d(ctx_, in0, out_ne[0], out_ne[1], out_ne[2], out_ne[3], nb1, nb2, nb3, offset);
                // Graph outputs must be host-readable via ggml_backend_tensor_get, which
                // linearly copies ggml_nbytes (span of a strided view ≠ logical payload).
                // Intermediate QKV/gate_up views stay non-contiguous for FA zero-copy.
                const bool is_graph_output = std::find(
                    model_graph_.outputs.begin(), model_graph_.outputs.end(), out_id) != model_graph_.outputs.end();
                if ((is_graph_output || env_flag_enabled("GGMLC_VIEW_FORCE_CONT")) && !ggml_is_contiguous(v)) {
                    result = ggml_cont(ctx_, v);
                } else {
                    result = v;
                }
                break;
            }
            case GGML_OP_ARGMAX: {
                if (in0 && !ggml_is_contiguous(in0)) in0 = ggml_cont(ctx_, in0);
                if (in0 && in0->type != GGML_TYPE_F32) {
                    in0 = ggml_cast(ctx_, in0, GGML_TYPE_F32);
                }
                result = ggml_argmax(ctx_, in0);
                const auto& out_ne = concrete_shapes_[out_id];
                result = reshape4d_contig(ctx_, result, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
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
                        in0 = reshape4d_contig(ctx_, in0, in0->ne[0], 1, in0->ne[1], in0->ne[2]);
                    }
                    if (in1->ne[3] == 1) {
                        in1 = reshape4d_contig(ctx_, in1, in1->ne[0], 1, in1->ne[1], in1->ne[2]);
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
                            bias = reshape4d_contig(ctx_, bias, 1, 1, result->ne[2], 1);
                        }
                        if (!ggml_are_same_shape(bias, result) && ggml_can_repeat(bias, result)) {
                            bias = ggml_repeat(ctx_, bias, result);
                        }
                        result = ggml_add(ctx_, result, bias);
                    }
                }
                if (op.attributes.count("fused_relu") && op.attributes.at("fused_relu") != 0) {
                    result = ggml_relu(ctx_, result);
                }
                if (is_1d && result->ne[1] == 1) {
                    result = reshape4d_contig(ctx_, result, result->ne[0], result->ne[2], result->ne[3], 1);
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
                            bias = reshape4d_contig(ctx_, bias, 1, 1, result->ne[2], 1);
                        }
                        if (!ggml_are_same_shape(bias, result) && ggml_can_repeat(bias, result)) {
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
                    result = ggml_mul(ctx_, result, w);
                }
                if (b) {
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
                    result = ggml_mul(ctx_, result, w);
                }
                break;
            }
            case 203: { // GGML_OP_CUSTOM_SWIGLU: in0=gate, in1=up, or single concatenated input
                if (!in0) {
                    throw std::runtime_error("GGML_OP_CUSTOM_SWIGLU requires at least 1 input");
                }
                if (op.inputs.size() == 1 || in1 == nullptr) {
                    bool swapped = op.attributes.count("swapped") && op.attributes.at("swapped") != 0;
                    if (swapped) {
                        result = ggml_swiglu_swapped(ctx_, in0);
                    } else {
                        result = ggml_swiglu(ctx_, in0);
                    }
                } else {
                    result = ggml_swiglu_split(ctx_, in0, in1);
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

            // Ensure all output tensors are anchored at the end of cgraph_ so that
            // ggml_gallocr treats them as live outputs throughout execution and never
            // frees or reuses their memory buffers for subsequent intermediate operations.
            for (uint32_t out_id : model_graph_.outputs) {
                if (ggml_tensors_.count(out_id)) {
                    struct ggml_tensor* out_t = ggml_tensors_[out_id];
                    ggml_set_output(out_t);
                    struct ggml_tensor* anchor = reshape4d_contig(
                        ctx_, out_t, out_t->ne[0], out_t->ne[1], out_t->ne[2], out_t->ne[3]);
                    ggml_set_name(anchor, "output_anchor");
                    ggml_build_forward_expand(cgraph_, anchor);
                }
            }

    // 4. Allocate tensor storage for compute activations on backend (CPU or CUDA)
    if (enable_arena_reuse) {
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!galloc_) {
            throw std::runtime_error("Failed to create ggml_gallocr for backend (" + device_ + ")");
        }
        if (!ggml_gallocr_alloc_graph(galloc_, cgraph_)) {
            throw std::runtime_error("Failed to allocate graph tensors via ggml_gallocr on backend (" + device_ + ")");
        }
    } else {
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_) {
            throw std::runtime_error("Failed to allocate tensors via GGML backend (" + device_ + ")");
        }
    }

    // Initialize any compute-time constants that have static data
    for (const auto& pair : compute_tensors_) {
        uint32_t tid = pair.first;
        const auto& t = model_graph_.tensors.at(tid);
        if (t.data_ptr && t.data_size > 0) {
            if (pair.second->buffer == nullptr) {
                continue;
            }
            size_t sz = std::min<size_t>(t.data_size, ggml_nbytes(pair.second));
            size_t offset = 0;
            if (symbol_env.count("pos") > 0 && t.name.find("arange") != std::string::npos) {
                int64_t pos = symbol_env.at("pos");
                size_t elem_sz = ggml_type_size(t.type);
                if (static_cast<size_t>(pos) * elem_sz < t.data_size) {
                    offset = static_cast<size_t>(pos) * elem_sz;
                    sz = std::min<size_t>(t.data_size - offset, ggml_nbytes(pair.second));
                }
                if (is_decode_step || (kv_cache_enabled_ && symbol_env.count("pos") > 0)) {
                    decode_rope_arange_tensors_.push_back({pair.second, tid});
                }
            }
            ggml_backend_tensor_set(pair.second, static_cast<const uint8_t*>(t.data_ptr) + offset, 0, sz);
        }
    }

    // Initialize dynamic causal masks for Flash Attention
    {
        ScopedTimer mask_timer(enable_profile_, profile_.mask_fill_ms);
        for (const auto& minfo : dynamic_causal_masks_) {
            if (!minfo.mask_tensor || minfo.mask_tensor->buffer == nullptr) continue;
            size_t s_kv = static_cast<size_t>(minfo.s_kv);
            size_t s_q = static_cast<size_t>(minfo.s_q);
            int64_t pos = minfo.pos;
            std::vector<ggml_fp16_t> mask_data(s_kv * s_q);
            ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
            ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(ATTN_MASK_MIN_FP16);
            for (size_t i = 0; i < s_q; ++i) {
                for (size_t j = 0; j < s_kv; ++j) {
                    mask_data[i * s_kv + j] = (j <= static_cast<size_t>(pos + i)) ? zero_f16 : neg_inf_f16;
                }
            }
            ggml_backend_tensor_set(minfo.mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
        if (symbol_env.count("pos") > 0 && kv_indices_tensor_ && kv_indices_tensor_->buffer) {
            const int64_t pos = symbol_env.at("pos");
            const int64_t n_idx = kv_indices_tensor_->ne[0];
            if (n_idx == 1) {
                int64_t idx = pos;
                ggml_backend_tensor_set(kv_indices_tensor_, &idx, 0, sizeof(int64_t));
            } else if (n_idx > 1) {
                std::vector<int64_t> idxs(static_cast<size_t>(n_idx));
                for (int64_t i = 0; i < n_idx; ++i) idxs[static_cast<size_t>(i)] = pos + i;
                ggml_backend_tensor_set(kv_indices_tensor_, idxs.data(), 0, idxs.size() * sizeof(int64_t));
            }
        }
        if (symbol_env.count("pos") > 0 && kv_work_mask_tensor_ && kv_work_mask_tensor_->buffer) {
            const int64_t pos = symbol_env.at("pos");
            const int64_t s_q_mask = kv_indices_tensor_ ? kv_indices_tensor_->ne[0]
                                                        : kv_work_mask_tensor_->ne[1];
            int64_t n_kv_mask = is_decode_step ? decode_cached_n_kv_ : chunk_cached_n_kv_;
            if (n_kv_mask < 1) {
                n_kv_mask = pad_kv_len(pos + s_q_mask, kv_cache_max_ctx_, kv_n_pad_);
            }
            fill_f16_causal_mask(kv_work_mask_tensor_, pos, s_q_mask, n_kv_mask);
        }
    }

    if (is_decode_step) {
        decode_graph_cached_ = true;
        decode_cached_pos_ = symbol_env.at("pos");
    } else if (kv_cache_enabled_ && symbol_env.count("pos") > 0 && !is_single_token && current_s > 0) {
        chunk_graph_cached_ = true;
        chunk_cached_pos_ = symbol_env.at("pos");
        chunk_cached_s_ = current_s;
    }

    last_symbol_env_ = symbol_env;
    last_enable_arena_reuse_ = enable_arena_reuse;
    prepared_ = true;
}

void ModelExecutor::set_input(uint32_t tensor_id, const void* data, size_t size_bytes) {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    struct ggml_tensor* t = it->second;
    if (t->buffer == nullptr) {
        throw std::runtime_error("Cannot set input for tensor " + std::to_string(tensor_id) + " because its buffer is not allocated (not part of the active compute graph).");
    }
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

void ModelExecutor::set_enable_cuda_graph(bool enable) {
    enable_cuda_graph_ = enable;
    if (enable && is_cuda_ && !cuda_graph_mgr_) {
        cuda_graph_mgr_ = std::make_unique<CUDAGraphManager>();
        if (backend_) {
            cuda_graph_mgr_->init(backend_);
        }
    }
}

bool ModelExecutor::is_cuda_graph_captured() const {
    return cuda_graph_mgr_ && cuda_graph_mgr_->is_captured();
}

void ModelExecutor::set_enable_cuda_graph_buckets(bool enable) {
    enable_cuda_graph_buckets_ = enable;
    if (enable && is_cuda_ && !cuda_graph_mgr_) {
        cuda_graph_mgr_ = std::make_unique<CUDAGraphManager>();
        if (backend_) {
            cuda_graph_mgr_->init(backend_);
        }
    }
}

bool ModelExecutor::is_cuda_graph_bucket_captured(int batch_size) const {
    return cuda_graph_mgr_ && cuda_graph_mgr_->is_bucket_captured(batch_size);
}

void ModelExecutor::run(int n_threads) {
    ScopedTimer timer(enable_profile_, profile_.run_ms, &profile_.n_run);
    if (!ctx_ || !cgraph_ || !backend_) {
        throw std::runtime_error("Executor not prepared. Call prepare() first.");
    }
    if (ggml_backend_is_cpu(backend_)) {
        if (n_threads < 1 || n_threads > GGML_MAX_N_THREADS) {
            throw std::invalid_argument("CPU thread count must be between 1 and " + std::to_string(GGML_MAX_N_THREADS));
        }
        if (!cpu_threadpool_ || cpu_threadpool_n_threads_ != n_threads) {
            auto params = ggml_threadpool_params_default(n_threads);
            auto new_pool = ggml_threadpool_new(&params);
            if (!new_pool) {
                throw std::runtime_error("Failed to create CPU thread pool");
            }
            // Detach the previous pool before freeing it; the backend borrows it.
            ggml_backend_cpu_set_threadpool(backend_, new_pool);
            if (cpu_threadpool_) ggml_threadpool_free(cpu_threadpool_);
            cpu_threadpool_ = new_pool;
            cpu_threadpool_n_threads_ = n_threads;
        }
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }

    if (is_cuda_ && enable_cuda_graph_buckets_) {
        if (!cuda_graph_mgr_) {
            cuda_graph_mgr_ = std::make_unique<CUDAGraphManager>();
        }
        if (!cuda_graph_mgr_->is_initialized()) {
            cuda_graph_mgr_->init(backend_);
        }
        if (cuda_graph_mgr_->is_initialized()) {
            int current_batch = 1;
            auto sym_it = last_symbol_env_.find("b");
            if (sym_it != last_symbol_env_.end()) {
                current_batch = static_cast<int>(sym_it->second);
            } else if (!model_graph_.inputs.empty()) {
                uint32_t inp_id = model_graph_.inputs[0];
                auto it = ggml_tensors_.find(inp_id);
                if (it != ggml_tensors_.end() && it->second) {
                    current_batch = it->second->ne[1] > 1 ? static_cast<int>(it->second->ne[1]) : 1;
                }
            }

            if (cuda_graph_mgr_->is_bucket_captured(current_batch)) {
                if (cuda_graph_mgr_->launch_bucket(current_batch)) {
                    return;
                }
            } else {
                if (cuda_graph_mgr_->begin_capture_bucket(current_batch)) {
                    ggml_backend_graph_compute_async(backend_, cgraph_);
                    if (cuda_graph_mgr_->end_capture_and_instantiate_bucket(current_batch)) {
                        if (cuda_graph_mgr_->launch_bucket(current_batch)) {
                            return;
                        }
                    }
                }
            }
        }
    }

    // enable_cuda_graph_: rely on GGML's built-in CUDA graphs (keyed per cgraph).
    // Outer CUDAGraphManager capture nests with ggml_backend_cuda_graph_compute and
    // aborts on bucket switches; pad-stable SET_ROWS graphs make that wrapper unnecessary.

    enum ggml_status status = ggml_backend_graph_compute(backend_, cgraph_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML backend graph compute failed with status: " + std::to_string(status));
    }

    // Execution completed successfully
}

void ModelExecutor::synchronize() {
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
}

void ModelExecutor::set_state(uint32_t tensor_id, const void* data, size_t size_bytes) {
    struct ggml_tensor* g_t = nullptr;
    auto it = state_tensors_.find(tensor_id);
    if (it != state_tensors_.end()) {
        g_t = it->second;
    } else {
        auto g_it = ggml_tensors_.find(tensor_id);
        if (g_it != ggml_tensors_.end()) {
            g_t = g_it->second;
        }
    }
    if (g_t != nullptr && ggml_nbytes(g_t) == size_bytes) {
        if (g_t->buffer == nullptr) {
            throw std::runtime_error("Cannot set state for tensor " + std::to_string(tensor_id) + " because its buffer is not allocated.");
        }
        ggml_backend_tensor_set(g_t, data, 0, size_bytes);
    }
    persistent_states_[tensor_id].assign(reinterpret_cast<const uint8_t*>(data), reinterpret_cast<const uint8_t*>(data) + size_bytes);
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
    struct ggml_tensor* g_t = nullptr;
    auto t_it = state_tensors_.find(tensor_id);
    if (t_it != state_tensors_.end()) {
        g_t = t_it->second;
    } else {
        auto g_it = ggml_tensors_.find(tensor_id);
        if (g_it != ggml_tensors_.end()) {
            g_t = g_it->second;
        }
    }
    if (g_t != nullptr) {
        size_t sz = ggml_nbytes(g_t);
        auto& host_buf = state_host_buffers_[tensor_id];
        host_buf.resize(sz);
        ggml_backend_tensor_get(g_t, host_buf.data(), 0, sz);
        return host_buf.data();
    }
    auto it = persistent_states_.find(tensor_id);
    if (it != persistent_states_.end() && !it->second.empty()) {
        return it->second.data();
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
    for (const auto& pair : state_tensors_) {
        if (pair.second) {
            ggml_backend_tensor_memset(pair.second, 0, 0, ggml_nbytes(pair.second));
        }
    }
}

const void* ModelExecutor::get_output_data(uint32_t tensor_id) {
    auto it = ggml_tensors_.find(tensor_id);
    if (it == ggml_tensors_.end()) {
        throw std::runtime_error("Tensor ID not found in executor: " + std::to_string(tensor_id));
    }
    struct ggml_tensor* t = it->second;
    // Logical payload (nelements), not ggml_nbytes span of a strided non-contiguous view.
    const size_t type_size = ggml_type_size(t->type);
    const int64_t blck = std::max<int64_t>(1, ggml_blck_size(t->type));
    size_t sz = static_cast<size_t>(ggml_nelements(t) * type_size / blck);
    if (!ggml_is_contiguous(t)) {
        throw std::runtime_error(
            "get_output_data: tensor " + std::to_string(tensor_id) +
            " is a non-contiguous view; graph outputs must be materialized with ggml_cont");
    }
    auto& host_buf = output_host_buffers_[tensor_id];
    host_buf.resize(sz);
    ggml_backend_tensor_get(t, host_buf.data(), 0, sz);
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
    struct ggml_tensor* t = it->second;
    const size_t type_size = ggml_type_size(t->type);
    const int64_t blck = std::max<int64_t>(1, ggml_blck_size(t->type));
    size_t sz = static_cast<size_t>(ggml_nelements(t) * type_size / blck);
    if (sz == 0) {
        fprintf(stderr, "[DEBUG] tensor %u ne=[%lld,%lld,%lld,%lld] type=%d blck_size=%lld type_size=%zu\n",
            tensor_id, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
            t->type, (long long)ggml_blck_size(t->type), ggml_type_size(t->type));
    }
    return sz;
}

void ModelExecutor::set_enable_profile(bool enable) {
    enable_profile_ = enable;
}

void ModelExecutor::set_logits_last_only(bool enable) {
    if (logits_last_only_ == enable) return;
    logits_last_only_ = enable;
    // Force graph rebuild so lm_head gather is applied/removed.
    prepared_ = false;
    if (cgraph_) {
        // Drop prepared compute graph; next prepare() rebuilds.
        // Buckets similarly invalidated via prepared_ flag path.
    }
}

void ModelExecutor::reset_profile() {
    profile_ = ExecutorProfile{};
}

bool ModelExecutor::ggml_cuda_graphs_compiled() {
#ifdef GGML_CUDA_USE_GRAPHS
    return true;
#else
    return false;
#endif
}

std::string ModelExecutor::runtime_graph_summary() const {
    std::ostringstream oss;
    if (!cgraph_) {
        oss << "{\"n_nodes\":0,\"n_kernel_nodes\":0,\"ops\":{}}";
        return oss.str();
    }
    const int n = ggml_graph_n_nodes(cgraph_);
    std::map<std::string, int> hist;
    int n_kernel = 0;
    int n_cpy = 0;
    int n_cont = 0;
    int n_mul_mat = 0;
    int n_fa = 0;
    int n_set_rows = 0;

    // Vertical / CUDA-adjacency fusion discovery counters.
    // Stock ggml-cuda fuses only when ops are *consecutive* in cgraph->nodes.
    int adj_rms_mul = 0;
    int adj_rms_mul_add = 0;
    int adj_rms_mul_rope = 0;
    int adj_rms_mul_rope_set = 0;
    int adj_rope_view_set = 0;
    int adj_add_rms = 0;          // residual → norm (no stock CUDA fuse)
    int adj_add_rms_mul = 0;      // residual → rms → weight (Phase 3 candidate)
    int adj_mul_mul_mat = 0;      // scaled norm act → GEMM (already post RMS+MUL fuse)
    int adj_norm_mul = 0;         // LayerNorm scale
    int adj_add_norm = 0;
    int broken_rms_then_mul = 0;  // RMS_NORM present but next kernel ≠ MUL
    int rms_mul_edge_ok = 0;     // adjacent AND mul.src uses rms result
    int rms_mul_edge_bad = 0;
    std::map<std::string, int> kernel_bigrams;
    std::vector<std::string> kernel_seq;
    kernel_seq.reserve(static_cast<size_t>(n));

    auto op_at = [&](int i) -> ggml_op {
        struct ggml_tensor* t = ggml_graph_node(cgraph_, i);
        return t ? t->op : GGML_OP_NONE;
    };
    auto tensor_at = [&](int i) -> struct ggml_tensor* {
        return ggml_graph_node(cgraph_, i);
    };
    // Skip pure view/reshape metadata when scoring "next kernel" breaks.
    auto next_kernel = [&](int i) -> int {
        for (int j = i + 1; j < n; ++j) {
            struct ggml_tensor* t = tensor_at(j);
            if (!t) continue;
            if (!is_metadata_op(t->op)) return j;
        }
        return -1;
    };

    for (int i = 0; i < n; ++i) {
        struct ggml_tensor* node = ggml_graph_node(cgraph_, i);
        if (!node) continue;
        const char* name = ggml_op_name(node->op);
        hist[name]++;
        if (!is_metadata_op(node->op)) {
            n_kernel++;
            kernel_seq.emplace_back(name);
        }
        if (node->op == GGML_OP_CPY) n_cpy++;
        if (node->op == GGML_OP_CONT || node->op == GGML_OP_DUP) n_cont++;
        if (node->op == GGML_OP_MUL_MAT) n_mul_mat++;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) n_fa++;
        if (node->op == GGML_OP_SET_ROWS) n_set_rows++;

        // Strict consecutive adjacency (matches ggml_cuda_can_fuse indexing).
        if (i + 1 < n && op_at(i) == GGML_OP_RMS_NORM && op_at(i + 1) == GGML_OP_MUL) {
            adj_rms_mul++;
            struct ggml_tensor* rms = tensor_at(i);
            struct ggml_tensor* mul = tensor_at(i + 1);
            if (mul && rms && (mul->src[0] == rms || mul->src[1] == rms)) {
                rms_mul_edge_ok++;
            } else {
                rms_mul_edge_bad++;
            }
            if (i + 2 < n && op_at(i + 2) == GGML_OP_ADD) adj_rms_mul_add++;
            if (i + 2 < n && op_at(i + 2) == GGML_OP_ROPE) {
                adj_rms_mul_rope++;
                if (i + 4 < n && op_at(i + 3) == GGML_OP_VIEW && op_at(i + 4) == GGML_OP_SET_ROWS) {
                    adj_rms_mul_rope_set++;
                }
            }
            if (i + 2 < n && op_at(i + 2) == GGML_OP_MUL_MAT) adj_mul_mul_mat++;
        }
        if (i + 2 < n && op_at(i) == GGML_OP_ROPE && op_at(i + 1) == GGML_OP_VIEW &&
            op_at(i + 2) == GGML_OP_SET_ROWS) {
            adj_rope_view_set++;
        }
        if (i + 1 < n && op_at(i) == GGML_OP_ADD && op_at(i + 1) == GGML_OP_RMS_NORM) {
            adj_add_rms++;
            if (i + 2 < n && op_at(i + 2) == GGML_OP_MUL) adj_add_rms_mul++;
        }
        if (i + 1 < n && op_at(i) == GGML_OP_NORM && op_at(i + 1) == GGML_OP_MUL) adj_norm_mul++;
        if (i + 1 < n && op_at(i) == GGML_OP_ADD && op_at(i + 1) == GGML_OP_NORM) adj_add_norm++;

        if (node->op == GGML_OP_RMS_NORM) {
            if (!(i + 1 < n && op_at(i + 1) == GGML_OP_MUL)) {
                // Also check if MUL is next *kernel* after metadata (still not CUDA-fusible).
                const int nk = next_kernel(i);
                if (nk < 0 || op_at(nk) != GGML_OP_MUL) {
                    broken_rms_then_mul++;
                } else {
                    // MUL exists but not adjacent — CUDA fuse miss.
                    broken_rms_then_mul++;
                }
            }
        }
    }
    for (size_t ki = 0; ki + 1 < kernel_seq.size(); ++ki) {
        kernel_bigrams[kernel_seq[ki] + "->" + kernel_seq[ki + 1]]++;
    }
    // Keep top kernel bigrams for discovery (by count).
    std::vector<std::pair<std::string, int>> bigram_sorted(kernel_bigrams.begin(), kernel_bigrams.end());
    std::sort(bigram_sorted.begin(), bigram_sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const size_t bigram_keep = std::min<size_t>(24, bigram_sorted.size());

    oss << "{\"n_nodes\":" << n
        << ",\"n_kernel_nodes\":" << n_kernel
        << ",\"n_cpy\":" << n_cpy
        << ",\"n_cont_or_dup\":" << n_cont
        << ",\"n_mul_mat\":" << n_mul_mat
        << ",\"n_flash_attn\":" << n_fa
        << ",\"n_set_rows\":" << n_set_rows
        << ",\"kv_n_pad\":" << kv_n_pad_
        << ",\"decode_cached_n_kv\":" << decode_cached_n_kv_
        << ",\"chunk_cached_n_kv\":" << chunk_cached_n_kv_
        << ",\"set_rows_kv\":" << (kv_indices_tensor_ ? "true" : "false")
        << ",\"ggml_cuda_graphs_compiled\":" << (ggml_cuda_graphs_compiled() ? "true" : "false")
        << ",\"cuda_graph_manager_captured\":" << (is_cuda_graph_captured() ? "true" : "false")
        << ",\"vertical\":{"
        << "\"adj_rms_mul\":" << adj_rms_mul
        << ",\"rms_mul_edge_ok\":" << rms_mul_edge_ok
        << ",\"rms_mul_edge_bad\":" << rms_mul_edge_bad
        << ",\"adj_rms_mul_add\":" << adj_rms_mul_add
        << ",\"adj_rms_mul_rope\":" << adj_rms_mul_rope
        << ",\"adj_rms_mul_rope_set\":" << adj_rms_mul_rope_set
        << ",\"adj_rope_view_set\":" << adj_rope_view_set
        << ",\"adj_add_rms\":" << adj_add_rms
        << ",\"adj_add_rms_mul\":" << adj_add_rms_mul
        << ",\"adj_mul_mul_mat\":" << adj_mul_mul_mat
        << ",\"adj_norm_mul\":" << adj_norm_mul
        << ",\"adj_add_norm\":" << adj_add_norm
        << ",\"broken_rms_then_mul\":" << broken_rms_then_mul
        << "}"
        << ",\"kernel_bigrams\":{";
    for (size_t bi = 0; bi < bigram_keep; ++bi) {
        if (bi) oss << ",";
        oss << "\"" << bigram_sorted[bi].first << "\":" << bigram_sorted[bi].second;
    }
    oss << "}"
        << ",\"ops\":{";
    bool first = true;
    for (const auto& pair : hist) {
        if (!first) oss << ",";
        first = false;
        oss << "\"" << pair.first << "\":" << pair.second;
    }
    oss << "}}";
    return oss.str();
}

std::string ModelExecutor::runtime_mul_mat_shape_summary() const {
    std::ostringstream oss;
    if (!cgraph_) {
        return "{\"mul_mat\":[],\"flash_attn\":[]}";
    }
    // key -> count; key encodes type, ne, contiguity, mmq-fallback
    std::map<std::string, int> mm;
    std::map<std::string, int> fa;
    const int n = ggml_graph_n_nodes(cgraph_);
    for (int i = 0; i < n; ++i) {
        struct ggml_tensor* node = ggml_graph_node(cgraph_, i);
        if (!node) continue;
        if (node->op == GGML_OP_MUL_MAT) {
            const struct ggml_tensor* a = node->src[0];
            const struct ggml_tensor* b = node->src[1];
            if (!a || !b) continue;
            const bool fallback = (a->ne[1] % 128) != 0;
            const bool a_cont = ggml_is_contiguous(a);
            const bool b_cont = ggml_is_contiguous(b);
            // Predicted MMQ J tile for ampere/ada Q8 path: largest J|8 that
            // minimizes ceil(ncols_max / J); ncols_max ~= b->ne[1] (tokens).
            const int64_t ncols = b->ne[1];
            int j_best = 0;
            int ntiles_best = INT_MAX;
            for (int J = 8; J <= 128; J += 8) {
                // Ampere Q8_0 fast: J in {8,16,24,...,128}; fallback skips some.
                if (fallback && (J % 16 != 0) && J != 8) {
                    // rough: fallback configs are sparser; keep all multiples of 8 for dump
                }
                const int ntiles = static_cast<int>((ncols + J - 1) / J);
                if (ntiles < ntiles_best) {
                    ntiles_best = ntiles;
                    j_best = J;
                }
            }
            std::ostringstream key;
            key << ggml_type_name(a->type)
                << " w=[" << a->ne[0] << "," << a->ne[1] << "," << a->ne[2] << "," << a->ne[3] << "]"
                << " x=[" << b->ne[0] << "," << b->ne[1] << "," << b->ne[2] << "," << b->ne[3] << "]"
                << " dst=[" << node->ne[0] << "," << node->ne[1] << "]"
                << " a_cont=" << (a_cont ? 1 : 0)
                << " b_cont=" << (b_cont ? 1 : 0)
                << " nb0=[" << a->nb[0] << "," << a->nb[1] << "]"
                << " nb1=[" << b->nb[0] << "," << b->nb[1] << "]"
                << " fallback=" << (fallback ? 1 : 0)
                << " J~=" << j_best;
            mm[key.str()]++;
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            std::ostringstream key;
            key << "q=[";
            for (int d = 0; d < 4; ++d) {
                if (d) key << ",";
                key << (node->src[0] ? node->src[0]->ne[d] : 0);
            }
            key << "] k=[";
            for (int d = 0; d < 4; ++d) {
                if (d) key << ",";
                key << (node->src[1] ? node->src[1]->ne[d] : 0);
            }
            key << "] v=[";
            for (int d = 0; d < 4; ++d) {
                if (d) key << ",";
                key << (node->src[2] ? node->src[2]->ne[d] : 0);
            }
            key << "]";
            fa[key.str()]++;
        }
    }
    oss << "{\"mul_mat\":[";
    bool first = true;
    for (const auto& p : mm) {
        if (!first) oss << ",";
        first = false;
        // escape is unnecessary: keys have no quotes
        oss << "{\"n\":" << p.second << ",\"shape\":\"" << p.first << "\"}";
    }
    oss << "],\"flash_attn\":[";
    first = true;
    for (const auto& p : fa) {
        if (!first) oss << ",";
        first = false;
        oss << "{\"n\":" << p.second << ",\"shape\":\"" << p.first << "\"}";
    }
    oss << "]}";
    return oss.str();
}

} // namespace ggmlc
