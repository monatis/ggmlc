#pragma once

#include <string>
#include <vector>
#include <memory>
#include "ggmlc/types.h"

namespace ggmlc {

class ModelLoader {
public:
    static SerializedModelGraph load_from_file(const std::string& filepath);
    static SerializedModelGraph load_from_memory(const uint8_t* data, size_t size);
};

} // namespace ggmlc
