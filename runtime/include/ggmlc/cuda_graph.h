#pragma once

#include <memory>
#include "ggml.h"
#include "ggml-backend.h"

namespace ggmlc {

class CUDAGraphManager {
public:
    CUDAGraphManager();
    ~CUDAGraphManager();

    CUDAGraphManager(const CUDAGraphManager&) = delete;
    CUDAGraphManager& operator=(const CUDAGraphManager&) = delete;
    CUDAGraphManager(CUDAGraphManager&&) noexcept;
    CUDAGraphManager& operator=(CUDAGraphManager&&) noexcept;

    // Check device compute capability and CUDA graph support
    static bool is_supported_on_device(int device_id);
    static int get_device_cc(int device_id);

    // Initialize manager with active CUDA backend
    void init(ggml_backend_t backend);
    bool is_initialized() const;
    bool is_captured() const;

    // Stream capture lifecycle (single instance)
    bool begin_capture();
    bool end_capture_and_instantiate();

    // Multi-bucket execution for batch sizes B in {1, 2, 4, 8, 16}
    bool begin_capture_bucket(int batch_size);
    bool end_capture_and_instantiate_bucket(int batch_size);
    bool launch_bucket(int batch_size);
    bool is_bucket_captured(int batch_size) const;

    // Update executable graph with newly modified dynamic parameters / shapes
    bool update_executable(struct ggml_cgraph* cgraph, ggml_backend_t backend);

    // Launch instantiated executable CUDA graph
    bool launch();

    // Free captured graph resources
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ggmlc
