#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace ggmlc {
namespace pipeline {

struct ImageTensor {
    std::vector<float> data; // NCHW layout: shape [1, C, H, W]
    int channels = 3;
    int height = 224;
    int width = 224;
};

class ImagePreprocessor {
public:
    ImagePreprocessor() = default;

    // Load image from disk and preprocess into normalized CHW float tensor matching torchvision/PIL
    static ImageTensor preprocess_file(
        const std::string& filepath,
        int target_width = 224,
        int target_height = 224,
        const std::vector<float>& mean = {0.48145466f, 0.4578275f, 0.40821073f},
        const std::vector<float>& std = {0.26862954f, 0.26130258f, 0.27577711f},
        bool do_center_crop = true
    );

    // Preprocess in-memory raw image bytes (JPEG, PNG, etc.)
    static ImageTensor preprocess_memory(
        const uint8_t* buffer,
        size_t buffer_len,
        int target_width = 224,
        int target_height = 224,
        const std::vector<float>& mean = {0.48145466f, 0.4578275f, 0.40821073f},
        const std::vector<float>& std = {0.26862954f, 0.26130258f, 0.27577711f},
        bool do_center_crop = true
    );

    // Bicubic interpolation kernel (Keys bicubic with a = -0.5 matching PIL / torchvision)
    static void bicubic_resize(
        const uint8_t* src, int src_w, int src_h, int channels,
        float* dst, int dst_w, int dst_h
    );

    // Center crop from HxW to crop_Hxcrop_W
    static void center_crop(
        const float* src, int src_w, int src_h, int channels,
        float* dst, int crop_w, int crop_h
    );
};

} // namespace pipeline
} // namespace ggmlc
