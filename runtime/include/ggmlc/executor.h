#pragma once

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggmlc/types.h"

namespace ggmlc {

class ModelExecutor {
public:
    explicit ModelExecutor(const SerializedModelGraph& graph, const std::string& device = "cpu");
    ~ModelExecutor();

    // Query available hardware execution devices (e.g. ["cpu", "cuda:0"])
    static std::vector<std::string> get_available_devices();

    // Current configured execution device
    const std::string& device() const { return device_; }

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

private:
    SerializedModelGraph model_graph_;
    std::string device_ = "cpu";
    bool is_cuda_ = false;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    struct ggml_context* ctx_ = nullptr;
    struct ggml_cgraph* cgraph_ = nullptr;
    std::unordered_map<uint32_t, struct ggml_tensor*> ggml_tensors_;
    std::unordered_map<uint32_t, std::array<int64_t, 4>> concrete_shapes_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> persistent_states_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> output_host_buffers_;
    std::unordered_map<uint32_t, std::vector<uint8_t>> state_host_buffers_;
    std::vector<std::vector<uint8_t>> custom_params_storage_;
};

} // namespace ggmlc
