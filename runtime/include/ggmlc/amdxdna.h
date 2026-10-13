#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "ggml.h"
#include "ggml-backend.h"

namespace ggmlc {

// Synchronous XDNA2 GEMM offload with host GGML buffers. CPU graph segments
// share those buffers, so unsupported operators need no tensor relocation.
class AmdxdnaExecutor {
public:
    AmdxdnaExecutor();
    ~AmdxdnaExecutor();
    AmdxdnaExecutor(const AmdxdnaExecutor&) = delete;
    AmdxdnaExecutor& operator=(const AmdxdnaExecutor&) = delete;

    static bool is_available();
    static bool supports(const ggml_tensor* node);
    void run(ggml_cgraph* graph, ggml_backend_t cpu);
    std::string summary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ggmlc
