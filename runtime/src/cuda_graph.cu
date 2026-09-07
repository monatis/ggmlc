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
    std::unordered_map<int, cudaGraph_t> bucket_graphs;
    std::unordered_map<int, cudaGraphExec_t> bucket_instances;
    int capturing_bucket = -1;
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
        for (auto& pair : bucket_instances) {
            if (pair.second) cudaGraphExecDestroy(pair.second);
        }
        bucket_instances.clear();
        for (auto& pair : bucket_graphs) {
            if (pair.second) cudaGraphDestroy(pair.second);
        }
        bucket_graphs.clear();
        capturing_bucket = -1;
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

bool CUDAGraphManager::begin_capture_bucket(int batch_size) {
    if (!impl_ || !impl_->stream) return false;
    impl_->capturing_bucket = batch_size;
    cudaError_t err = cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] begin_capture_bucket(%d) failed: %s\n", batch_size, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CUDAGraphManager::end_capture_and_instantiate_bucket(int batch_size) {
    if (!impl_ || !impl_->stream || impl_->capturing_bucket != batch_size) return false;
    impl_->capturing_bucket = -1;

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(impl_->stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] end_capture_bucket(%d) failed: %s\n", batch_size, cudaGetErrorString(err));
        return false;
    }

    cudaGraphExec_t instance = nullptr;
    err = cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] instantiate_bucket(%d) failed: %s\n", batch_size, cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        return false;
    }

    // Clean up any existing instance for this bucket
    if (impl_->bucket_instances.count(batch_size) && impl_->bucket_instances[batch_size]) {
        cudaGraphExecDestroy(impl_->bucket_instances[batch_size]);
    }
    if (impl_->bucket_graphs.count(batch_size) && impl_->bucket_graphs[batch_size]) {
        cudaGraphDestroy(impl_->bucket_graphs[batch_size]);
    }

    impl_->bucket_graphs[batch_size] = graph;
    impl_->bucket_instances[batch_size] = instance;
    return true;
}

bool CUDAGraphManager::launch_bucket(int batch_size) {
    if (!impl_ || !impl_->stream) return false;
    auto it = impl_->bucket_instances.find(batch_size);
    if (it == impl_->bucket_instances.end() || !it->second) return false;

    cudaError_t err = cudaGraphLaunch(it->second, impl_->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA_GRAPH] launch_bucket(%d) failed: %s\n", batch_size, cudaGetErrorString(err));
        return false;
    }
    cudaStreamSynchronize(impl_->stream);
    return true;
}

bool CUDAGraphManager::is_bucket_captured(int batch_size) const {
    return impl_ && impl_->bucket_instances.find(batch_size) != impl_->bucket_instances.end();
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
