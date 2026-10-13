#include "ggmlc/amdxdna.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "modules/gemm.hpp"
#include "npu_utils/npu_utils_xrt.hpp"

namespace ggmlc {
namespace {
constexpr int64_t tile_m = 256; // four compute rows, 64 tokens each
constexpr int64_t tile_k = 512;
constexpr int64_t tile_n = 128;

std::string xclbin_path() {
    const char* env = std::getenv("GGMLC_AMDXDNA_XCLBIN");
    if (env && *env) return env;

    // Resolve from the executable rather than the working directory or the
    // configure-time prefix, so cmake --install --prefix and relocation work.
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) {
        const auto installed = (executable.parent_path() /
            GGMLC_AMDXDNA_INSTALLED_XCLBIN).lexically_normal();
        if (std::filesystem::is_regular_file(installed, error))
            return installed.string();
    }
    return GGMLC_AMDXDNA_DEFAULT_XCLBIN;
}

// The FastFlowLM binaries used here target Strix Point/Halo (XDNA2).
// Reject Phoenix and later generations before submitting incompatible code.
bool has_xdna2() {
    const std::filesystem::path root("/sys/bus/pci/drivers/amdxdna");
    if (!std::filesystem::exists(root)) return false;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        std::string vendor, device, revision;
        std::ifstream(entry.path() / "vendor") >> vendor;
        std::ifstream(entry.path() / "device") >> device;
        std::ifstream(entry.path() / "revision") >> revision;
        if (vendor == "0x1022" && device == "0x17f0" &&
            (revision == "0x10" || revision == "0x11")) return true;
    }
    return false;
}

int64_t pad(int64_t n, int64_t tile) { return ((n + tile - 1) / tile) * tile; }

size_t weight_index(int64_t k, int64_t n, int64_t padded_k) {
    // [N/128, K/512, N_tile/8, K_inner(8), K_tile/8, N_inner(8)]
    return static_cast<size_t>((n / 128) * (padded_k / 512) * 65536 +
        (k / 512) * 65536 + ((n % 128) / 8) * 4096 +
        (k % 8) * 512 + ((k % 512) / 8) * 8 + n % 8);
}

void read_row(const ggml_tensor* tensor, int64_t row, std::vector<float>& out) {
    out.resize(static_cast<size_t>(tensor->ne[0]));
    const auto* src = static_cast<const uint8_t*>(tensor->data) + row * tensor->nb[1];
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src, out.size() * sizeof(float));
    } else {
        const auto* traits = ggml_get_type_traits(tensor->type);
        if (!traits->to_float) throw std::runtime_error("amdxdna: unsupported tensor conversion");
        traits->to_float(src, out.data(), tensor->ne[0]);
    }
}

bool host_tensor(const ggml_tensor* t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

bool floating_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool trace_enabled() {
    const char* env = std::getenv("GGMLC_AMDXDNA_TRACE");
    return env && std::string(env) == "1";
}
} // namespace

struct AmdxdnaExecutor::Impl {
    // Member order keeps the device/context alive until apps and BOs are freed.
    xrt::device device;
    npu_xclbin_manager manager;
    npu_app_manager* app_manager;
    LM_Config config;
    Gemm gemm;
    struct Weight {
        buffer<bf16> bo;
        int64_t k, n;
    };
    using WeightKey = std::tuple<uintptr_t, int, int64_t, int64_t, size_t>;
    std::map<WeightKey, Weight> weights;
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::unique_ptr<npu_app>> apps;
    buffer<bf16> input, output;
    size_t input_capacity = 0, output_capacity = 0;
    uint64_t npu_matmuls = 0, cpu_nodes = 0, cached_weight_bytes = 0;
    double npu_ms = 0;
    bool reported = false;
    std::mutex mutex;

    Impl() : device(0), manager(device_npu2, &device),
             app_manager(manager.register_xclbin(xclbin_path())), gemm(config) {
        if (gemm.get_m() != 64 || gemm.get_k() != tile_k || gemm.get_n() != tile_n)
            throw std::runtime_error("amdxdna: FastFlowLM GEMM tile ABI mismatch");
    }

    Weight& weight(const ggml_tensor* tensor) {
        WeightKey key{reinterpret_cast<uintptr_t>(tensor->data), int(tensor->type), tensor->ne[0], tensor->ne[1], tensor->nb[1]};
        auto found = weights.find(key);
        if (found != weights.end()) return found->second;
        const int64_t k = pad(tensor->ne[0], tile_k), n = pad(tensor->ne[1], tile_n);
        auto bo = app_manager->create_bo_buffer<bf16>(static_cast<size_t>(k * n));
        std::fill(bo.data(), bo.data() + k * n, bf16(0.0f));
        std::vector<float> row;
        for (int64_t j = 0; j < tensor->ne[1]; ++j) {
            read_row(tensor, j, row);
            for (int64_t i = 0; i < tensor->ne[0]; ++i)
                bo.data()[weight_index(i, j, k)] = bf16(row[i]);
        }
        bo.sync_to_device();
        cached_weight_bytes += static_cast<uint64_t>(k * n * sizeof(bf16));
        return weights.emplace(key, Weight{std::move(bo), k, n}).first->second;
    }

    void compute(ggml_tensor* node) {
        const auto start = std::chrono::steady_clock::now();
        Weight& w = weight(node->src[0]);
        const auto* x = node->src[1];
        const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
        const int64_t m = pad(rows, tile_m);
        const size_t in_size = static_cast<size_t>(m * w.k), out_size = static_cast<size_t>(m * w.n);
        if (input_capacity < in_size) {
            input = app_manager->create_bo_buffer<bf16>(in_size);
            input_capacity = in_size;
        }
        if (output_capacity < out_size) {
            output = app_manager->create_bo_buffer<bf16>(out_size);
            output_capacity = out_size;
        }
        std::fill(input.data(), input.data() + in_size, bf16(0.0f));
        std::vector<float> row;
        int64_t flat = 0;
        for (int64_t b3 = 0; b3 < x->ne[3]; ++b3) {
            for (int64_t b2 = 0; b2 < x->ne[2]; ++b2) {
                ggml_tensor plane = *x;
                plane.data = static_cast<uint8_t*>(x->data) + b3 * x->nb[3] + b2 * x->nb[2];
                for (int64_t r = 0; r < x->ne[1]; ++r, ++flat) {
                    read_row(&plane, r, row);
                    for (int64_t k = 0; k < x->ne[0]; ++k)
                        input.data()[flat * w.k + k] = bf16(row[k]);
                }
            }
        }
        auto key = std::make_tuple(m, w.k, w.n);
        auto& app = apps[key];
        if (!app) {
            app = std::make_unique<npu_app>(app_manager->create_app());
            gemm.generate_seq(app->seq(), static_cast<uint32_t>(m),
                static_cast<uint32_t>(w.k), static_cast<uint32_t>(w.n),
                0, false, Gemm::NO_Activation, 0);
            app->seq()->cmds2seq();
            app->update_ctrl_seq();
        }
        input.sync_to_device();
        // FastFlowLM's patch indices: 0=output, 1=input, 2=packed weights.
        auto run = app->create_run(output, input, w.bo);
        run.start();
        const auto state = run.wait(std::chrono::milliseconds(30000));
        if (state == ERT_CMD_STATE_TIMEOUT) run.abort();
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("amdxdna: NPU GEMM failed, ERT state=" + std::to_string(int(state)));
        output.sync_from_device();
        auto* dest = static_cast<float*>(node->data);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t n = 0; n < node->ne[0]; ++n)
                dest[r * node->ne[0] + n] = float(output.data()[r * w.n + n]);
        ++npu_matmuls;
        npu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};

bool AmdxdnaExecutor::is_available() {
    try {
        if (!has_xdna2() || !std::filesystem::is_regular_file(xclbin_path())) return false;
        xrt::device device(0);
        return true;
    } catch (...) { return false; }
}

AmdxdnaExecutor::AmdxdnaExecutor() {
    if (!has_xdna2()) throw std::runtime_error("amdxdna: no supported XDNA2 device (PCI 1022:17f0, revision 10/11) bound to amdxdna");
    if (!std::filesystem::is_regular_file(xclbin_path()))
        throw std::runtime_error("amdxdna: GEMM xclbin missing: " + xclbin_path());
    try { impl_ = std::make_unique<Impl>(); }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("amdxdna initialization failed: ") + e.what() +
            ". Check xrt-smi examine, /dev/accel permissions and memlock limit.");
    }
}
AmdxdnaExecutor::~AmdxdnaExecutor() = default;

bool AmdxdnaExecutor::supports(const ggml_tensor* node) {
    if (!node || node->op != GGML_OP_MUL_MAT || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return false;
    if (ggml_get_op_params_i32(node, 0) == GGML_PREC_F32) return false;
    const auto* w = node->src[0];
    const auto* x = node->src[1];
    if (!host_tensor(w) || !host_tensor(x) || !host_tensor(node)) return false;
    // Cache only immutable model weights. Attention GEMMs and mutable inputs
    // remain on CPU, including weights derived by graph operators.
    if (ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        w->op != GGML_OP_NONE || w->ne[2] != 1 || w->ne[3] != 1 ||
        w->ne[0] < tile_k || w->ne[1] < tile_n ||
        !ggml_is_contiguous_rows(w) || !floating_type(x->type) ||
        x->nb[0] != ggml_type_size(x->type) || x->ne[0] != w->ne[0]) return false;
    if (w->type != GGML_TYPE_F32 && !ggml_get_type_traits(w->type)->to_float) return false;
    const auto k = pad(w->ne[0], tile_k), n = pad(w->ne[1], tile_n);
    const auto m = pad(x->ne[1] * x->ne[2] * x->ne[3], tile_m);
    // FastFlowLM DMA offsets are 32-bit byte offsets.
    constexpr int64_t max_elements = std::numeric_limits<uint32_t>::max() / 2;
    return k <= max_elements / n && m <= max_elements / k && m <= max_elements / n;
}

void AmdxdnaExecutor::run(ggml_cgraph* graph, ggml_backend_t cpu) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->npu_matmuls = impl_->cpu_nodes = 0;
    impl_->npu_ms = 0;
    const int count = ggml_graph_n_nodes(graph);
    int begin = 0;
    auto cpu_range = [&](int end) {
        if (end <= begin) return;
        auto view = ggml_graph_view(graph, begin, end);
        if (ggml_backend_graph_compute(cpu, &view) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("amdxdna: CPU graph segment failed");
        impl_->cpu_nodes += end - begin;
    };
    for (int i = 0; i < count; ++i) {
        auto* node = ggml_graph_node(graph, i);
        if (!supports(node)) continue;
        cpu_range(i);
        impl_->compute(node);
        begin = i + 1;
    }
    cpu_range(count);
    if (!impl_->reported || trace_enabled()) {
        std::cerr << "[amdxdna] " << summary() << '\n';
        impl_->reported = true;
    }
}

std::string AmdxdnaExecutor::summary() const {
    std::ostringstream out;
    out << "{\"npu_matmuls\":" << impl_->npu_matmuls << ",\"cpu_nodes\":" << impl_->cpu_nodes
        << ",\"npu_ms\":" << impl_->npu_ms << ",\"cached_weight_bytes\":" << impl_->cached_weight_bytes
        << ",\"precision\":\"bf16\"}";
    return out.str();
}
} // namespace ggmlc
