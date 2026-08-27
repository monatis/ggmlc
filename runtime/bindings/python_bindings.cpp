#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/array.h>

#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "gguf.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_runtime, m) {
    m.doc() = "Native C++ execution runtime for ggmlc models (nanobind)";

    // SerializedTensor
    nb::class_<ggmlc::SerializedTensor>(m, "SerializedTensor")
        .def_ro("id", &ggmlc::SerializedTensor::id)
        .def_ro("name", &ggmlc::SerializedTensor::name)
        .def_ro("data_size", &ggmlc::SerializedTensor::data_size)
        .def_prop_ro("storage", [](const ggmlc::SerializedTensor& t) {
            return static_cast<int32_t>(t.storage);
        })
        .def_prop_ro("type", [](const ggmlc::SerializedTensor& t) {
            return static_cast<int32_t>(t.type);
        });

    // SerializedModelGraph
    nb::class_<ggmlc::SerializedModelGraph>(m, "SerializedModelGraph")
        .def_ro("name", &ggmlc::SerializedModelGraph::name)
        .def_ro("symbol_table", &ggmlc::SerializedModelGraph::symbol_table)
        .def_ro("inputs", &ggmlc::SerializedModelGraph::inputs)
        .def_ro("outputs", &ggmlc::SerializedModelGraph::outputs)
        .def_ro("parameters", &ggmlc::SerializedModelGraph::parameters)
        .def_ro("tensors", &ggmlc::SerializedModelGraph::tensors);

    // ModelLoader
    nb::class_<ggmlc::ModelLoader>(m, "ModelLoader")
        .def_static("load_from_file", &ggmlc::ModelLoader::load_from_file, "filepath"_a)
        .def_static("load_from_bytes", [](nb::bytes bytes_obj) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(bytes_obj.c_str());
            size_t size = bytes_obj.size();
            auto graph = ggmlc::ModelLoader::load_from_memory(ptr, size);
            graph.data_buffer.assign(ptr, ptr + size);

            // Re-bind data_ptr relative to graph.data_buffer
            struct gguf_init_params params = { true, nullptr };
            struct gguf_context* ctx = gguf_init_from_buffer(graph.data_buffer.data(), graph.data_buffer.size(), params);
            if (ctx) {
                size_t data_offset = gguf_get_data_offset(ctx);
                const uint8_t* base_data = graph.data_buffer.data() + data_offset;
                for (auto& pair : graph.tensors) {
                    int64_t t_id = gguf_find_tensor(ctx, pair.second.name.c_str());
                    if (t_id >= 0) {
                        pair.second.data_ptr = base_data + gguf_get_tensor_offset(ctx, t_id);
                    }
                }
                gguf_free(ctx);
            }
            return graph;
        }, "data"_a);

    // Query available devices
    m.def("get_available_devices", &ggmlc::ModelExecutor::get_available_devices, "Query available execution devices");

    // ModelExecutor
    nb::class_<ggmlc::ModelExecutor>(m, "ModelExecutor")
        .def(nb::init<const ggmlc::SerializedModelGraph&, const std::string&>(), "graph"_a, "device"_a = "cpu")
        .def_prop_ro("device", &ggmlc::ModelExecutor::device)
        .def("prepare", [](ggmlc::ModelExecutor& self,
                           const std::unordered_map<std::string, int64_t>& symbols,
                           bool enable_arena_reuse) {
            self.prepare(symbols, enable_arena_reuse);
        }, "symbols"_a = std::unordered_map<std::string, int64_t>{}, "enable_arena_reuse"_a = true)
        .def("set_input_by_id", [](ggmlc::ModelExecutor& self, uint32_t tensor_id, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array) {
            self.set_input(tensor_id, array.data(), array.size() * array.itemsize());
        }, "tensor_id"_a, "array"_a)
        .def("set_input_by_name", [](ggmlc::ModelExecutor& self, const std::string& name, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array) {
            self.set_input_by_name(name, array.data(), array.size() * array.itemsize());
        }, "name"_a, "array"_a)
        .def("set_state_by_id", [](ggmlc::ModelExecutor& self, uint32_t tensor_id, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array) {
            self.set_state(tensor_id, array.data(), array.size() * array.itemsize());
        }, "tensor_id"_a, "array"_a)
        .def("set_state_by_name", [](ggmlc::ModelExecutor& self, const std::string& name, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array) {
            self.set_state_by_name(name, array.data(), array.size() * array.itemsize());
        }, "name"_a, "array"_a)
        .def("run", [](ggmlc::ModelExecutor& self, int n_threads) {
            nb::gil_scoped_release release;
            self.run(n_threads);
        }, "n_threads"_a = 1)
        .def("get_output_bytes", [](ggmlc::ModelExecutor& self, uint32_t tensor_id) -> nb::bytes {
            const void* ptr = self.get_output_data(tensor_id);
            size_t size_bytes = self.get_tensor_size_bytes(tensor_id);
            if (!ptr || size_bytes == 0) {
                return nb::bytes("", 0);
            }
            return nb::bytes(reinterpret_cast<const char*>(ptr), size_bytes);
        }, "tensor_id"_a)
        .def("get_state_bytes", [](ggmlc::ModelExecutor& self, uint32_t tensor_id) -> nb::bytes {
            const void* ptr = self.get_state_data(tensor_id);
            size_t size_bytes = self.get_tensor_size_bytes(tensor_id);
            if (!ptr || size_bytes == 0) {
                return nb::bytes("", 0);
            }
            return nb::bytes(reinterpret_cast<const char*>(ptr), size_bytes);
        }, "tensor_id"_a)
        .def("get_state_bytes_by_name", [](ggmlc::ModelExecutor& self, const std::string& name) -> nb::bytes {
            const void* ptr = self.get_state_data_by_name(name);
            if (!ptr) {
                return nb::bytes("", 0);
            }
            return nb::bytes(reinterpret_cast<const char*>(ptr), 0);
        }, "name"_a)
        .def("get_tensor_shape", [](const ggmlc::ModelExecutor& self, uint32_t tensor_id) -> std::vector<int64_t> {
            auto s = self.get_tensor_shape(tensor_id);
            return std::vector<int64_t>(s.begin(), s.end());
        }, "tensor_id"_a)
        .def("get_tensor_size_bytes", &ggmlc::ModelExecutor::get_tensor_size_bytes, "tensor_id"_a)
        .def("reset_state", &ggmlc::ModelExecutor::reset_state);
}
