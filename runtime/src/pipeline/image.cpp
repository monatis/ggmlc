#include "ggmlc/pipeline/image.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../third_party/ggml/examples/stb_image.h"

namespace ggmlc {
namespace pipeline {

static inline float cubic_filter(float x) {
    x = std::fabs(x);
    if (x <= 1.0f) {
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

static inline int clamp_coord(int pos, int max_size) {
    return std::max(0, std::min(pos, max_size - 1));
}

void ImagePreprocessor::bicubic_resize(
    const uint8_t* src, int src_w, int src_h, int channels,
    float* dst, int dst_w, int dst_h
) {
    float x_scale = (float)src_w / (float)dst_w;
    float y_scale = (float)src_h / (float)dst_h;

    // Temporary horizontal pass buffer: dst_w x src_h x channels
    std::vector<float> temp_buf(dst_w * src_h * channels, 0.0f);

    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float src_x = (x + 0.5f) * x_scale - 0.5f;
            int x_int = (int)std::floor(src_x);
            float x_frac = src_x - x_int;

            float weights[4];
            float weight_sum = 0.0f;
            for (int k = -1; k <= 2; ++k) {
                weights[k + 1] = cubic_filter(x_frac - (float)k);
                weight_sum += weights[k + 1];
            }
            if (weight_sum != 0.0f) {
                for (int k = 0; k < 4; ++k) weights[k] /= weight_sum;
            }

            for (int c = 0; c < channels; ++c) {
                float val = 0.0f;
                for (int k = -1; k <= 2; ++k) {
                    int sx = clamp_coord(x_int + k, src_w);
                    val += (float)src[(y * src_w + sx) * channels + c] * weights[k + 1];
                }
                temp_buf[(y * dst_w + x) * channels + c] = val;
            }
        }
    }

    // Vertical pass: dst_w x dst_h x channels
    for (int y = 0; y < dst_h; ++y) {
        float src_y = (y + 0.5f) * y_scale - 0.5f;
        int y_int = (int)std::floor(src_y);
        float y_frac = src_y - y_int;

        float weights[4];
        float weight_sum = 0.0f;
        for (int k = -1; k <= 2; ++k) {
            weights[k + 1] = cubic_filter(y_frac - (float)k);
            weight_sum += weights[k + 1];
        }
        if (weight_sum != 0.0f) {
            for (int k = 0; k < 4; ++k) weights[k] /= weight_sum;
        }

        for (int x = 0; x < dst_w; ++x) {
            for (int c = 0; c < channels; ++c) {
                float val = 0.0f;
                for (int k = -1; k <= 2; ++k) {
                    int sy = clamp_coord(y_int + k, src_h);
                    val += temp_buf[(sy * dst_w + x) * channels + c] * weights[k + 1];
                }
                dst[(y * dst_w + x) * channels + c] = std::max(0.0f, std::min(255.0f, val));
            }
        }
    }
}

void ImagePreprocessor::center_crop(
    const float* src, int src_w, int src_h, int channels,
    float* dst, int crop_w, int crop_h
) {
    int start_x = std::max(0, (src_w - crop_w) / 2);
    int start_y = std::max(0, (src_h - crop_h) / 2);

    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            int src_idx = ((start_y + y) * src_w + (start_x + x)) * channels;
            int dst_idx = (y * crop_w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

ImageTensor ImagePreprocessor::preprocess_memory(
    const uint8_t* buffer,
    size_t buffer_len,
    int target_width,
    int target_height,
    const std::vector<float>& mean,
    const std::vector<float>& std_dev,
    bool do_center_crop
) {
    int w, h, c;
    uint8_t* img_data = stbi_load_from_memory(buffer, (int)buffer_len, &w, &h, &c, 3);
    if (!img_data) {
        throw std::runtime_error("Failed to decode image data via stbi_load_from_memory");
    }

    int channels = 3;
    int resize_w = target_width;
    int resize_h = target_height;

    if (do_center_crop) {
        // Shorter side resize matching torchvision/PIL:
        if (w < h) {
            resize_w = target_width;
            resize_h = (int)std::round((float)h * (float)target_width / (float)w);
        } else {
            resize_h = target_height;
            resize_w = (int)std::round((float)w * (float)target_height / (float)h);
        }
    }

    std::vector<float> resized(resize_w * resize_h * channels);
    bicubic_resize(img_data, w, h, channels, resized.data(), resize_w, resize_h);
    stbi_image_free(img_data);

    std::vector<float> cropped(target_width * target_height * channels);
    if (do_center_crop && (resize_w != target_width || resize_h != target_height)) {
        center_crop(resized.data(), resize_w, resize_h, channels, cropped.data(), target_width, target_height);
    } else {
        cropped = std::move(resized);
    }

    // Convert HWC [0, 255] to CHW normalized float [0, 1] -> (x - mean) / std
    ImageTensor result;
    result.channels = channels;
    result.width = target_width;
    result.height = target_height;
    result.data.resize(channels * target_height * target_width);

    float m[3] = {mean.size() > 0 ? mean[0] : 0.48145466f, mean.size() > 1 ? mean[1] : 0.4578275f, mean.size() > 2 ? mean[2] : 0.40821073f};
    float s[3] = {std_dev.size() > 0 ? std_dev[0] : 0.26862954f, std_dev.size() > 1 ? std_dev[1] : 0.26130258f, std_dev.size() > 2 ? std_dev[2] : 0.27577711f};

    int plane_size = target_width * target_height;
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int hwc_idx = (y * target_width + x) * channels;
            int pixel_pos = y * target_width + x;
            for (int ch = 0; ch < channels; ++ch) {
                float val = cropped[hwc_idx + ch] / 255.0f;
                float norm_val = (val - m[ch]) / s[ch];
                result.data[ch * plane_size + pixel_pos] = norm_val;
            }
        }
    }

    return result;
}

ImageTensor ImagePreprocessor::preprocess_file(
    const std::string& filepath,
    int target_width,
    int target_height,
    const std::vector<float>& mean,
    const std::vector<float>& std_dev,
    bool do_center_crop
) {
    int w, h, c;
    uint8_t* img_data = stbi_load(filepath.c_str(), &w, &h, &c, 3);
    if (!img_data) {
        throw std::runtime_error("Failed to load image file: " + filepath);
    }

    int channels = 3;
    int resize_w = target_width;
    int resize_h = target_height;

    if (do_center_crop) {
        if (w < h) {
            resize_w = target_width;
            resize_h = (int)std::round((float)h * (float)target_width / (float)w);
        } else {
            resize_h = target_height;
            resize_w = (int)std::round((float)w * (float)target_height / (float)h);
        }
    }

    std::vector<float> resized(resize_w * resize_h * channels);
    bicubic_resize(img_data, w, h, channels, resized.data(), resize_w, resize_h);
    stbi_image_free(img_data);

    std::vector<float> cropped(target_width * target_height * channels);
    if (do_center_crop && (resize_w != target_width || resize_h != target_height)) {
        center_crop(resized.data(), resize_w, resize_h, channels, cropped.data(), target_width, target_height);
    } else {
        cropped = std::move(resized);
    }

    ImageTensor result;
    result.channels = channels;
    result.width = target_width;
    result.height = target_height;
    result.data.resize(channels * target_height * target_width);

    float m[3] = {mean.size() > 0 ? mean[0] : 0.48145466f, mean.size() > 1 ? mean[1] : 0.4578275f, mean.size() > 2 ? mean[2] : 0.40821073f};
    float s[3] = {std_dev.size() > 0 ? std_dev[0] : 0.26862954f, std_dev.size() > 1 ? std_dev[1] : 0.26130258f, std_dev.size() > 2 ? std_dev[2] : 0.27577711f};

    int plane_size = target_width * target_height;
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int hwc_idx = (y * target_width + x) * channels;
            int pixel_pos = y * target_width + x;
            for (int ch = 0; ch < channels; ++ch) {
                float val = cropped[hwc_idx + ch] / 255.0f;
                float norm_val = (val - m[ch]) / s[ch];
                result.data[ch * plane_size + pixel_pos] = norm_val;
            }
        }
    }

    return result;
}

} // namespace pipeline
} // namespace ggmlc
