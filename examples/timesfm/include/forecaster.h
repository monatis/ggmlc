#pragma once

#include <string>
#include <vector>
#include <memory>
#include "export.h"
#include "revin.h"
#include "detrending.h"
#include "patching.h"
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"

namespace timesfm {

struct ForecastConfig {
    int64_t horizon = 128;
    bool normalize = true;
    bool detrend = true;
    bool sort_quantiles = true;
    bool make_positive = false;
    bool auto_positive = true;
    bool use_symmetric_averaging = false;
    float detrend_r2_threshold = 0.5f;
    std::string device = "cpu";
    int n_threads = 4;
};

class TimesFMForecaster {
public:
    TimesFMForecaster();
    ~TimesFMForecaster();

    bool load_model(const std::string& model_path, const std::string& device = "cpu", int n_threads = 4);

    ForecastResult forecast(
        const std::vector<float>& context_series,
        const ForecastConfig& config = {}
    );

    std::vector<ForecastResult> forecast_batch(
        const std::vector<std::vector<float>>& batch_series,
        const ForecastConfig& config = {}
    );

    const ggmlc::SerializedModelGraph& model_graph() const { return graph_; }

private:
    bool is_loaded_ = false;
    std::string model_path_;
    std::string device_ = "cpu";
    int n_threads_ = 4;

    ggmlc::SerializedModelGraph graph_;
    std::unique_ptr<ggmlc::ModelExecutor> executor_;

    uint32_t input_tensor_id_ = 0;
    uint32_t output_tensor_id_ = 0;
    std::vector<float> quantiles_;

    // Single 64-step model forecast step for a single series
    void forecast_single_chunk(
        const std::vector<float>& input_patches,
        int64_t num_patches,
        std::vector<float>& out_64x9
    );

    // Batched 64-step model forecast step for B series
    void forecast_chunk_batch(
        const std::vector<float>& batch_input_patches,
        int64_t batch_size,
        int64_t num_patches,
        std::vector<float>& out_batch_64x9
    );
};

} // namespace timesfm
