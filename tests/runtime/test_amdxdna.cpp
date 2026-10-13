#include "ggmlc/executor.h"
#include "ggmlc/amdxdna.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void check_type(ggml_type type, int width) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-8, 8);
    std::vector<float> weights(width * width);
    // Dyadic operands isolate layout/padding from vendor BF16 product rounding.
    // Full-model drift is measured separately by validate_amdxdna.py.
    for (auto& v : weights) v = float(dist(rng) % 2) / 32.f;
    std::vector<uint8_t> encoded(ggml_row_size(type, width) * width);
    ggml_quantize_chunk(type, weights.data(), encoded.data(), 0, width, width, nullptr);

    ggmlc::SerializedModelGraph graph;
    graph.name = "amdxdna_test";
    graph.symbol_table = {"rows", "batch"};
    graph.inputs = {0}; graph.parameters = {1}; graph.outputs = {2, 4};
    for (uint32_t id = 0; id < 5; ++id) {
        ggmlc::SerializedTensor t{};
        t.id = id; t.name = "tensor_" + std::to_string(id);
        t.type = id == 1 ? type : GGML_TYPE_F32;
        t.storage = id == 0 ? ggmlc::StorageClass::INPUT :
                    id == 1 ? ggmlc::StorageClass::PARAMETER : ggmlc::StorageClass::ACTIVATION;
        for (int d = 0; d < 4; ++d) {
            t.ne[d] = std::make_shared<ggmlc::DimExpr>();
            t.ne[d]->type = ggmlc::DimType::STATIC;
            t.ne[d]->val = d == 0 ? width : 1;
        }
        if (id == 1) {
            t.ne[1]->val = width;
            t.data_ptr = encoded.data(); t.data_size = encoded.size();
        } else {
            t.ne[1]->type = ggmlc::DimType::SYMBOL; t.ne[1]->val = 0;
            t.ne[2]->type = ggmlc::DimType::SYMBOL; t.ne[2]->val = 1;
        }
        graph.tensors[id] = t;
    }
    graph.ops = {
        {0, GGML_OP_MUL_MAT, "linear1", {1, 0}, {2}, {}, {}},
        {1, GGML_OP_UNARY, "relu", {2}, {3}, {{"unary_op", GGML_UNARY_OP_RELU}}, {}},
        {2, GGML_OP_MUL_MAT, "linear2", {1, 3}, {4}, {}, {}},
    };
    ggmlc::ModelExecutor cpu(graph, "cpu"), npu(graph, "amdxdna");
    require(npu.device() == "amdxdna", "Explicit NPU request was not honored");
    // Covers padding in M/K/N, batch planes, buffer growth/shrink, different
    // inputs at the same shape, and cached weights across graph rebuilds.
    for (const auto shape : {std::pair<int,int>{1,1}, {64,2}, {257,1}, {1,1}}) {
        const int rows = shape.first, batch = shape.second;
        std::vector<float> input(width * rows * batch);
        for (auto& v : input) v = float(dist(rng)) / 16.f;
        for (auto* executor : {&cpu, &npu}) {
            executor->prepare({{"rows", rows}, {"batch", batch}}, rows != 257);
            executor->set_input(0, input.data(), input.size() * sizeof(float));
            executor->run(4);
        }
        const auto* ref = static_cast<const float*>(cpu.get_output_data(4));
        const auto* got = static_cast<const float*>(npu.get_output_data(4));
        double err2 = 0, ref2 = 0; float maxerr = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            require(std::isfinite(got[i]), "NPU returned nonfinite output");
            const double err = got[i] - ref[i];
            err2 += err * err; ref2 += double(ref[i]) * ref[i];
            maxerr = std::max(maxerr, float(std::abs(err)));
        }
        const double relative_rmse = std::sqrt(err2 / std::max(ref2, 1e-20));
        std::cout << ggml_type_name(type) << " width=" << width << " rows=" << rows
                  << " batch=" << batch << " relative_rmse=" << relative_rmse
                  << " max_error=" << maxerr << ' ' << npu.amdxdna_summary() << '\n';
        require(relative_rmse < (type == GGML_TYPE_Q4_K ? 0.025 : 0.015), "NPU MLP exceeded BF16 error tolerance");
        require(maxerr < 0.025, "NPU MLP has a large individual error");
        require(npu.amdxdna_summary().find(width >= 512 ? "\"npu_matmuls\":2" : "\"npu_matmuls\":0") != std::string::npos,
                "Unexpected number of NPU GEMMs");
    }
}

static void check_cpu_fallback() {
    auto* ctx = ggml_init({1 << 20, nullptr, true});
    require(ctx != nullptr, "Failed to create GGML context");
    auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2);
    auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 128);
    auto* out = ggml_mul_mat(ctx, w, x);
    auto* cpu = ggml_backend_cpu_init();
    auto* buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    require(buf != nullptr, "Failed to allocate fallback test tensors");
    // No immutable weight buffer: a dynamic GEMM must not enter the cache.
    require(!ggmlc::AmdxdnaExecutor::supports(out), "Mutable weights were accepted for caching");
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    require(ggmlc::AmdxdnaExecutor::supports(out), "Supported host GEMM was rejected");
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    require(!ggmlc::AmdxdnaExecutor::supports(out), "An explicit F32 precision requirement was ignored");
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    if (!ggmlc::AmdxdnaExecutor::is_available()) {
        std::cout << "SKIP: XDNA2 NPU or XRT is unavailable\n";
        return 77;
    }
    try {
        check_cpu_fallback();
        check_type(GGML_TYPE_F32, 32);
        check_type(GGML_TYPE_F32, 544);
        check_type(GGML_TYPE_F16, 544);
        check_type(GGML_TYPE_BF16, 544);
        check_type(GGML_TYPE_Q8_0, 544);
        check_type(GGML_TYPE_Q4_K, 768);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
    return 0;
}
