#include "ggmlc/loader.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>

namespace ggmlc {

namespace {

class ByteReader {
    const uint8_t* ptr;
    size_t remaining;

public:
    ByteReader(const uint8_t* p, size_t sz) : ptr(p), remaining(sz) {}

    template <typename T>
    T read() {
        if (remaining < sizeof(T)) {
            throw std::runtime_error("Unexpected EOF while reading binary .ggmlc format");
        }
        T val;
        std::memcpy(&val, ptr, sizeof(T));
        ptr += sizeof(T);
        remaining -= sizeof(T);
        return val;
    }

    std::string read_str() {
        uint32_t len = read<uint32_t>();
        if (remaining < len) {
            throw std::runtime_error("Unexpected EOF reading string in .ggmlc format");
        }
        std::string s(reinterpret_cast<const char*>(ptr), len);
        ptr += len;
        remaining -= len;
        return s;
    }

    void read_bytes(void* dest, size_t n) {
        if (remaining < n) {
            throw std::runtime_error("Unexpected EOF reading raw bytes in .ggmlc format");
        }
        std::memcpy(dest, ptr, n);
        ptr += n;
        remaining -= n;
    }

    size_t get_remaining() const { return remaining; }
    const uint8_t* get_ptr() const { return ptr; }
};

std::shared_ptr<DimExpr> read_dim(ByteReader& r) {
    auto dim = std::make_shared<DimExpr>();
    dim->type = static_cast<DimType>(r.read<uint8_t>());
    if (dim->type == DimType::STATIC || dim->type == DimType::SYMBOL) {
        dim->val = r.read<int64_t>();
    } else {
        dim->left = read_dim(r);
        dim->right = read_dim(r);
    }
    return dim;
}

} // namespace

SerializedModelGraph ModelLoader::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filepath);
    }

    return load_from_memory(buffer.data(), buffer.size());
}

SerializedModelGraph ModelLoader::load_from_memory(const uint8_t* data, size_t size) {
    ByteReader r(data, size);

    // 1. Header
    char magic[8];
    r.read_bytes(magic, 8);
    if (std::memcmp(magic, "GGMLC\x01\x00\x00", 8) != 0) {
        throw std::runtime_error("Invalid magic header in .ggmlc file");
    }

    uint32_t version = r.read<uint32_t>();
    if (version != 1) {
        throw std::runtime_error("Unsupported .ggmlc version: " + std::to_string(version));
    }

    SerializedModelGraph g;
    g.name = r.read_str();

    // 2. Symbol Table
    uint32_t num_symbols = r.read<uint32_t>();
    g.symbol_table.reserve(num_symbols);
    for (uint32_t i = 0; i < num_symbols; ++i) {
        g.symbol_table.push_back(r.read_str());
    }

    // 3. Inputs, Outputs, Parameters
    uint32_t num_inputs = r.read<uint32_t>();
    g.inputs.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i) {
        g.inputs.push_back(r.read<uint32_t>());
    }

    uint32_t num_outputs = r.read<uint32_t>();
    g.outputs.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i) {
        g.outputs.push_back(r.read<uint32_t>());
    }

    uint32_t num_params = r.read<uint32_t>();
    g.parameters.reserve(num_params);
    for (uint32_t i = 0; i < num_params; ++i) {
        g.parameters.push_back(r.read<uint32_t>());
    }

    // 4. Tensors
    uint32_t num_tensors = r.read<uint32_t>();
    for (uint32_t i = 0; i < num_tensors; ++i) {
        SerializedTensor t;
        t.id = r.read<uint32_t>();
        t.name = r.read_str();
        t.type = static_cast<ggml_type>(r.read<int32_t>());
        for (int d = 0; d < 4; ++d) {
            t.ne[d] = read_dim(r);
        }
        t.storage = static_cast<StorageClass>(r.read<int32_t>());
        t.data_offset = r.read<uint64_t>();
        t.data_size = r.read<uint64_t>();
        g.tensors[t.id] = t;
    }

    // 5. Operations
    uint32_t num_ops = r.read<uint32_t>();
    g.ops.reserve(num_ops);
    for (uint32_t i = 0; i < num_ops; ++i) {
        SerializedOp op;
        op.id = r.read<uint32_t>();
        op.opcode = r.read<int32_t>();
        op.name = r.read_str();

        uint32_t num_in = r.read<uint32_t>();
        op.inputs.reserve(num_in);
        for (uint32_t j = 0; j < num_in; ++j) {
            op.inputs.push_back(r.read<uint32_t>());
        }

        uint32_t num_out = r.read<uint32_t>();
        op.outputs.reserve(num_out);
        for (uint32_t j = 0; j < num_out; ++j) {
            op.outputs.push_back(r.read<uint32_t>());
        }

        uint32_t num_attrs = r.read<uint32_t>();
        for (uint32_t k = 0; k < num_attrs; ++k) {
            std::string attr_name = r.read_str();
            int64_t attr_val = r.read<int64_t>();
            op.attributes[attr_name] = attr_val;
        }

        g.ops.push_back(op);
    }

    // 6. Data Buffer
    uint64_t data_buf_size = r.read<uint64_t>();
    if (data_buf_size > 0) {
        g.data_buffer.resize(data_buf_size);
        r.read_bytes(g.data_buffer.data(), data_buf_size);

        // Wire up tensor data pointers
        for (auto& pair : g.tensors) {
            auto& t = pair.second;
            if (t.data_size > 0 && t.data_offset + t.data_size <= data_buf_size) {
                t.data_ptr = g.data_buffer.data() + t.data_offset;
            }
        }
    }

    return g;
}

} // namespace ggmlc
