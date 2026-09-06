#include "ggmlc/cuda_graph.h"
#include <iostream>
#include <cuda_runtime.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"

namespace ggmlc {

struct CUDAGraphManager::Impl {
    ggml_backend_t backend = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    bool is_capturing = false;

    ~Impl() {
        reset();
    }

    void reset() {
        if (instance) {
            cudaGraphExecDestroy(instance);
            instance = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        is_capturing = false;
    }
};

CUDAGraphManager::CUDAGraphManager() : impl_(std::make_unique<Impl>()) {}
CUDAGraphManager::~CUDAGraphManager() = default;

CUDAGraphManager::CUDAGraphManager(CUDAGraphManager&&) noexcept = default;
CUDAGraphManager& CUDAGraphManager::operator=(CUDAGraphManager&&) noexcept = default;

bool CUDAGraphManager::is_supported_on_device(int device_id) {
    int cc = get_device_cc(device_id);
    // CUDA graphs are supported on Compute Capability >= 6.0 (Pascal and newer)
    return cc >= 60;
}

int CUDAGraphManager::get_device_cc(int device_id) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.major * 10 + prop.minor;
    }
    return 0;
}

void CUDAGraphManager::init(ggml_backend_t backend) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->reset();
    impl_->backend = backend;
    if (backend && ggml_backend_is_cuda(backend)) {
        ggml_backend_cuda_context* cuda_ctx = static_cast<ggml_backend_cuda_context*>(backend->context);
        if (cuda_ctx) {
            impl_->stream = cuda_ctx->stream();
        }
    } else {
        impl_->stream = nullptr;
    }
}

bool CUDAGraphManager::is_initialized() const {
    return impl_ && impl_->stream != nullptr;
}

bool CUDAGraphManager::is_captured() const {
    return impl_ && impl_->instance != nullptr;
}

bool CUDAGraphManager::begin_capture() {
    if (!impl_ || !impl_->stream) return false;
    impl_->reset();
    cudaError_t err = cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    impl_->is_capturing = true;
    return true;
}

bool CUDAGraphManager::end_capture_and_instantiate() {
    if (!impl_ || !impl_->is_capturing) return false;
    impl_->is_capturing = false;
    cudaError_t err = cudaStreamEndCapture(impl_->stream, &impl_->graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaGraphInstantiate(&impl_->instance, impl_->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] cudaGraphInstantiate failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(impl_->graph);
        impl_->graph = nullptr;
        return false;
    }
    return true;
}

bool CUDAGraphManager::update_executable(struct ggml_cgraph* cgraph, ggml_backend_t backend) {
    if (!impl_ || !impl_->instance || !impl_->stream) return false;

    cudaGraph_t step_graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] update_executable cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    ggml_backend_graph_compute_async(backend, cgraph);

    err = cudaStreamEndCapture(impl_->stream, &step_graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] update_executable cudaStreamEndCapture failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult update_result;
    err = cudaGraphExecUpdate(impl_->instance, step_graph, &errorNode, &update_result);
    if (err != cudaSuccess) {
        // In case update failed due to structural change, re-instantiate
        cudaGraphExecDestroy(impl_->instance);
        impl_->instance = nullptr;
        err = cudaGraphInstantiate(&impl_->instance, step_graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] cudaGraphInstantiate re-creation failed: %s\n", cudaGetErrorString(err));
            cudaGraphDestroy(step_graph);
            return false;
        }
    }

    if (impl_->graph) {
        cudaGraphDestroy(impl_->graph);
    }
    impl_->graph = step_graph;
    return true;
}

bool CUDAGraphManager::launch() {
    if (!impl_ || !impl_->instance || !impl_->stream) return false;
    cudaError_t err = cudaGraphLaunch(impl_->instance, impl_->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] cudaGraphLaunch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaStreamSynchronize(impl_->stream);
    return true;
}

void CUDAGraphManager::reset() {
    if (impl_) {
        impl_->reset();
    }
}

} // namespace ggmlc
