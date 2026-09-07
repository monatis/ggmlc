#pragma once

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggmlc/types.h"
#include "ggmlc/cuda_graph.h"
#include "ggmlc/vmm_pool.h"

namespace ggmlc {

struct CustomOpParams {
    virtual ~CustomOpParams() = default;
};

struct LayerNormOpParams : public CustomOpParams {
    float eps = 1e-5f;
};

class ModelExecutor {
public:
    explicit ModelExecutor(const SerializedModelGraph& graph, const std::string& device = "cpu");
    ~ModelExecutor();

    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;
    ModelExecutor(ModelExecutor&&) noexcept = default;
    ModelExecutor& operator=(ModelExecutor&&) noexcept = default;

    // Query available hardware execution devices (e.g. ["cpu", "cuda:0"])
    static std::vector<std::string> get_available_devices();

    // Current configured execution device
    const std::string& device() const { return device_; }
    const SerializedModelGraph& model_graph() const { return model_graph_; }

    // Prepare execution context for given dynamic symbol values and optional memory arena reuse
    void prepare(const std::unordered_map<std::string, int64_t>& symbol_env = {}, bool enable_arena_reuse = true);

    // Set input tensor data
    void set_input(uint32_t tensor_id, const void* data, size_t size_bytes);
    void set_input_by_name(const std::string& name, const void* data, size_t size_bytes);

    // Run execution graph
    void run(int n_threads = 1);

    // State tensor access
    void set_state(uint32_t tensor_id, const void* data, size_t size_bytes);
    void set_state_by_name(const std::string& name, const void* data, size_t size_bytes);
    const void* get_state_data(uint32_t tensor_id);
    const void* get_state_data_by_name(const std::string& name);
    void reset_state();

    // Get output tensor data pointer and concrete shape
    const void* get_output_data(uint32_t tensor_id);
    std::array<int64_t, 4> get_tensor_shape(uint32_t tensor_id) const;
    size_t get_tensor_size_bytes(uint32_t tensor_id) const;

    // KV Cache management for autoregressive attention ops
    void init_kv_cache(int64_t max_ctx = 2048);
    void reset_kv_cache();
    bool has_kv_cache() const { return kv_cache_buffer_ != nullptr || paged_kv_enabled_; }
    void set_decode_pos(int64_t pos);

    // VMM Paged KV Cache management (Driver-level virtual memory mapping)
    void init_paged_kv_cache(size_t max_batch = 8, size_t max_ctx = 2048);
    bool is_paged_kv_cache_enabled() const { return paged_kv_enabled_; }
    void paged_kv_alloc_slot(int slot_id, const std::string& prefix_hash = "");
    void paged_kv_free_slot(int slot_id);
    bool is_paged_slot_allocated(int slot_id) const;
    void paged_kv_ensure_tokens(int slot_id, int64_t total_tokens);
    size_t get_paged_active_vram_bytes() const;
    void configure_vmm_pool(size_t max_warm_pages, size_t prealloc_pages = 0);
    size_t get_tokens_per_page() const { return tokens_per_page_; }
    void set_tokens_per_page(size_t tpp) { tokens_per_page_ = tpp; }
    void paged_kv_map_existing_pages(
        int slot_id,
        size_t start_page_idx,
        const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_k_handles,
        const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_v_handles
    );
    void paged_kv_extract_page_handles(
        int slot_id,
        size_t page_idx,
        std::unordered_map<uint32_t, uint64_t>& out_k,
        std::unordered_map<uint32_t, uint64_t>& out_v
    ) const;
    VMMBlockManager* vmm_block_manager() { return vmm_mgr_.get(); }

    // CUDA Graph execution
    void set_enable_cuda_graph(bool enable);
    bool is_cuda_graph_enabled() const { return enable_cuda_graph_; }
    bool is_cuda_graph_captured() const;

    // Multi-bucket CUDA Graph execution (B in {1, 2, 4, 8, 16})
    void set_enable_cuda_graph_buckets(bool enable);
    bool is_cuda_graph_buckets_enabled() const { return enable_cuda_graph_buckets_; }
    bool is_cuda_graph_bucket_captured(int batch_size) const;

private:
    void init_weights();
    void init_states(const std::unordered_map<std::string, int64_t>& symbol_env);
    bool has_state_tensors() const;

    SerializedModelGraph model_graph_;
    std::string device_ = "cpu";
    bool is_cuda_ = false;
    ggml_backend_t backend_ = nullptr;

    // Weights: static parameters & constants (allocated & initialized once)
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    struct ggml_context* ctx_w_ = nullptr;
    bool weights_loaded_ = false;
    std::unordered_map<uint32_t, struct ggml_tensor*> weight_tensors_;

    // States: persistent recurrent / KV cache tensors
    ggml_backend_buffer_t state_buffer_ = nullptr;
    struct ggml_context* ctx_state_ = nullptr;
    bool states_allocated_ = false;
    std::unordered_map<uint32_t, struct ggml_tensor*> state_tensors_;

    // Compute: dynamic activations, inputs, outputs
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    struct ggml_context* ctx_ = nullptr;
    struct ggml_cgraph* cgraph_ = nullptr;
    std::unordered_map<uint32_t, struct ggml_tensor*> compute_tensors_;

    // Global tensor map (points to weight_tensors_, state_tensors_, or compute_tensors_)
    std::unordered_map<uint32_t, struct ggml_tensor*> ggml_tensors_;
    std::unordered_map<uint32_t, std::array<int64_t, 4>> concrete_shapes_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> persistent_states_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> output_host_buffers_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> state_host_buffers_;
    std::vector<std::unique_ptr<CustomOpParams>> custom_params_storage_;

    // KV Cache: persistent memory for key and value attention caches
    bool kv_cache_enabled_ = false;
    int64_t kv_cache_max_ctx_ = 2048;
    ggml_backend_buffer_t kv_cache_buffer_ = nullptr;
    struct ggml_context* ctx_kv_cache_ = nullptr;
    std::unordered_map<uint32_t, struct ggml_tensor*> kv_cache_k_;
    std::unordered_map<uint32_t, struct ggml_tensor*> kv_cache_v_;

    // Decode Graph Cache: static execution graph and buffer for S=1 decode
    struct AttnViewRefs {
        struct ggml_tensor* k_slot = nullptr;
        struct ggml_tensor* v_slot = nullptr;
        struct ggml_tensor* k_active = nullptr;
        struct ggml_tensor* v_active = nullptr;
        struct ggml_tensor* scores = nullptr;
        struct ggml_tensor* probs = nullptr;
        struct ggml_tensor* v_t = nullptr;
        size_t slot_base_offset_k = 0;
        size_t slot_base_offset_v = 0;
    };
    bool decode_graph_cached_ = false;
    int64_t decode_cached_pos_ = -1;
    std::unordered_map<uint32_t, AttnViewRefs> decode_attn_views_;
    std::vector<std::pair<struct ggml_tensor*, uint32_t>> decode_rope_arange_tensors_;

    std::unordered_map<std::string, int64_t> last_symbol_env_;
    bool last_enable_arena_reuse_ = true;
    bool prepared_ = false;

    // CUDA Graph Management
    bool enable_cuda_graph_ = false;
    bool enable_cuda_graph_buckets_ = false;
    std::unique_ptr<CUDAGraphManager> cuda_graph_mgr_;
    bool cuda_graph_needs_update_ = false;

    // VMM Paged KV Cache Management
    bool paged_kv_enabled_ = false;
    size_t paged_max_batch_ = 1;
    size_t paged_max_ctx_ = 2048;
    size_t tokens_per_page_ = 0;
    std::unique_ptr<VMMBlockManager> vmm_mgr_;
    std::unordered_map<uint32_t, uint64_t> vmm_k_va_windows_; // op_id -> va_base
    std::unordered_map<uint32_t, uint64_t> vmm_v_va_windows_; // op_id -> va_base
    std::unordered_map<uint32_t, size_t> vmm_slot_bytes_;     // op_id -> bytes per slot

    struct PagedSlotState {
        int slot_id = -1;
        bool active = false;
        std::string prefix_hash;
        int64_t current_tokens = 0;
        std::unordered_map<uint32_t, std::vector<uint64_t>> mapped_pages_k; // op_id -> handles
        std::unordered_map<uint32_t, std::vector<uint64_t>> mapped_pages_v; // op_id -> handles
    };
    std::vector<PagedSlotState> paged_slots_;
};

} // namespace ggmlc
