#include "ggmlc/executor.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct MLP {
    ggmlc::SerializedModelGraph graph;
    std::vector<float> weights;
    int dim;

    explicit MLP(int width) : weights(width * width), dim(width) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = float(int(i % 31) - 15) / (31 * std::sqrt(float(dim)));
        }
        graph.name = "cpu_test_mlp";
        graph.symbol_table = {"rows"};
        graph.inputs = {0};
        graph.outputs = {4};
        graph.parameters = {1};
        for (uint32_t id = 0; id < 5; ++id) {
            ggmlc::SerializedTensor t{};
            t.id = id;
            t.name = "tensor_" + std::to_string(id);
            t.type = GGML_TYPE_F32;
            t.storage = id == 0 ? ggmlc::StorageClass::INPUT :
                        id == 1 ? ggmlc::StorageClass::PARAMETER : ggmlc::StorageClass::ACTIVATION;
            for (int d = 0; d < 4; ++d) {
                t.ne[d] = std::make_shared<ggmlc::DimExpr>();
                t.ne[d]->type = ggmlc::DimType::STATIC;
                t.ne[d]->val = d == 0 ? dim : 1;
            }
            if (id == 1) {
                t.ne[1]->val = dim;
                t.data_ptr = reinterpret_cast<const uint8_t*>(weights.data());
                t.data_size = weights.size() * sizeof(float);
            } else {
                t.ne[1]->type = ggmlc::DimType::SYMBOL;
                t.ne[1]->val = 0;
            }
            graph.tensors[id] = t;
        }
        graph.ops = {
            {0, GGML_OP_MUL_MAT, "linear1", {1, 0}, {2}, {}, {}},
            {1, GGML_OP_UNARY, "relu", {2}, {3}, {{"unary_op", GGML_UNARY_OP_RELU}}, {}},
            {2, GGML_OP_MUL_MAT, "linear2", {1, 3}, {4}, {}, {}},
        };
    }

    void check(const std::vector<float>& input, const float* output) const {
        std::vector<double> hidden(dim);
        for (size_t row = 0; row < input.size() / dim; ++row) {
            for (int j = 0; j < dim; ++j) {
                double sum = 0;
                for (int k = 0; k < dim; ++k) sum += weights[j * dim + k] * double(input[row * dim + k]);
                hidden[j] = std::max(0.0, sum);
            }
            for (int j = 0; j < dim; ++j) {
                double sum = 0;
                for (int k = 0; k < dim; ++k) sum += weights[j * dim + k] * hidden[k];
                require(std::abs(output[row * dim + j] - sum) < 1e-4, "MLP output mismatch");
            }
        }
    }
};

static std::vector<float> make_input(int rows, int dim, int seed) {
    std::vector<float> input(rows * dim);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i + seed) % 17) - 8) / 17;
    return input;
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    MLP model(benchmark ? 1024 : 32);
    for (int repeat = 0; repeat < (benchmark ? 1 : 3); ++repeat) {
        ggmlc::ModelExecutor executor(model.graph, "cpu");
        for (int rows : {1, 64, 1}) {
            for (int threads : {1, 4, 2, 1}) {
                auto input = make_input(rows, model.dim, threads + repeat);
                std::vector<double> times;
                for (int i = 0; i < (benchmark ? 60 : 3); ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    executor.prepare({{"rows", rows}});
                    executor.set_input(0, input.data(), input.size() * sizeof(float));
                    executor.run(threads);
                    const auto* output = static_cast<const float*>(executor.get_output_data(4));
                    const auto stop = std::chrono::steady_clock::now();
                    if (!benchmark || i == 0) model.check(input, output);
                    if (i >= 10) times.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
                }
                if (benchmark) {
                    std::sort(times.begin(), times.end());
                    std::cout << "rows=" << rows << " threads=" << threads
                              << " median_us=" << times[times.size() / 2] << '\n';
                }
            }
        }
        if (!benchmark) {
            for (int invalid_threads : {0, -1, GGML_MAX_N_THREADS + 1}) {
                bool rejected = false;
                try {
                    executor.run(invalid_threads);
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                require(rejected, "Invalid CPU thread count was accepted");
            }
            // A rejected call must leave the executor usable.
            auto input = make_input(1, model.dim, 1 + repeat);
            executor.set_input(0, input.data(), input.size() * sizeof(float));
            executor.run(2);
            model.check(input, static_cast<const float*>(executor.get_output_data(4)));
        }
    }
    std::cout << "CPU inference checks passed; AVX2=" << ggml_cpu_has_avx2()
              << " AVX512=" << ggml_cpu_has_avx512() << '\n';
}
