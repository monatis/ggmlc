#include "forecaster.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace timesfm {

TimesFMForecaster::TimesFMForecaster() {
    quantiles_ = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
}

TimesFMForecaster::~TimesFMForecaster() = default;

bool TimesFMForecaster::load_model(const std::string& model_path, const std::string& device, int n_threads) {
    try {
        model_path_ = model_path;
        device_ = device;
        n_threads_ = n_threads;

        graph_ = ggmlc::ModelLoader::load_from_file(model_path);
        executor_ = std::make_unique<ggmlc::ModelExecutor>(graph_, device);

        if (graph_.inputs.empty() || graph_.outputs.empty()) {
            std::cerr << "TimesFMForecaster error: model has no valid input/output tensors." << std::endl;
            return false;
        }

        input_tensor_id_ = graph_.inputs[0];
        output_tensor_id_ = graph_.outputs[0];
        is_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "TimesFMForecaster load error: " << e.what() << std::endl;
        is_loaded_ = false;
        return false;
    }
}

void TimesFMForecaster::forecast_chunk_batch(
    const std::vector<float>& batch_input_patches,
    int64_t batch_size,
    int64_t num_patches,
    std::vector<float>& out_batch_64x9
) {
    int64_t patch_out_size = OUTPUT_PATCH_LEN * DEFAULT_NUM_QUANTILES;
    out_batch_64x9.assign(batch_size * patch_out_size, 0.0f);
    if (!is_loaded_ || !executor_ || batch_size <= 0 || num_patches <= 0) return;

    std::unordered_map<std::string, int64_t> env;
    env["b"] = batch_size;
    env["n"] = num_patches;
    env["s"] = num_patches;

    // Auto-map free symbols in input tensor
    if (input_tensor_id_ < graph_.tensors.size()) {
        const auto& in_tensor = graph_.tensors[input_tensor_id_];
        if (in_tensor.ne.size() > 1 && in_tensor.ne[1] && in_tensor.ne[1]->type == ggmlc::DimType::SYMBOL) {
            int64_t sym_idx = in_tensor.ne[1]->val;
            if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(graph_.symbol_table.size())) {
                env[graph_.symbol_table[sym_idx]] = num_patches;
            }
        }
        if (in_tensor.ne.size() > 3 && in_tensor.ne[3] && in_tensor.ne[3]->type == ggmlc::DimType::SYMBOL) {
            int64_t sym_idx = in_tensor.ne[3]->val;
            if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(graph_.symbol_table.size())) {
                env[graph_.symbol_table[sym_idx]] = batch_size;
            }
        }
    }
    for (const auto& sym : graph_.symbol_table) {
        if (env.find(sym) == env.end()) {
            env[sym] = (sym == "b" || sym.find("batch") != std::string::npos) ? batch_size : num_patches;
        }
    }

    executor_->prepare(env, true);
    executor_->set_input(input_tensor_id_, batch_input_patches.data(), batch_input_patches.size() * sizeof(float));
    executor_->run(n_threads_);

    const float* out_ptr = static_cast<const float*>(executor_->get_output_data(output_tensor_id_));
    if (!out_ptr) return;

    // Output shape: [576, num_patches, 1, batch_size]
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t last_patch_offset = b * (num_patches * patch_out_size) + (num_patches - 1) * patch_out_size;
        const float* src = out_ptr + last_patch_offset;
        float* dst = out_batch_64x9.data() + b * patch_out_size;
        std::copy(src, src + patch_out_size, dst);
    }
}

void TimesFMForecaster::forecast_single_chunk(
    const std::vector<float>& input_patches,
    int64_t num_patches,
    std::vector<float>& out_64x9
) {
    forecast_chunk_batch(input_patches, 1, num_patches, out_64x9);
}

ForecastResult TimesFMForecaster::forecast(
    const std::vector<float>& context_series,
    const ForecastConfig& config
) {
    auto results = forecast_batch({context_series}, config);
    if (results.empty()) return {};
    return results[0];
}

std::vector<ForecastResult> TimesFMForecaster::forecast_batch(
    const std::vector<std::vector<float>>& batch_series,
    const ForecastConfig& config
) {
    if (config.use_symmetric_averaging) {
        ForecastConfig sub_cfg = config;
        sub_cfg.use_symmetric_averaging = false;

        // Run positive batch
        auto pos_results = forecast_batch(batch_series, sub_cfg);

        // Prepare negated batch
        std::vector<std::vector<float>> neg_series = batch_series;
        for (auto& s : neg_series) {
            for (auto& val : s) {
                if (std::isfinite(val)) val = -val;
            }
        }
        auto neg_results = forecast_batch(neg_series, sub_cfg);

        // Average symmetric predictions
        int64_t B = static_cast<int64_t>(batch_series.size());
        std::vector<ForecastResult> sym_results = pos_results;
        for (int64_t b = 0; b < B; ++b) {
            int64_t num_q = sym_results[b].num_quantiles;
            int64_t horiz = sym_results[b].horizon;
            for (int64_t q = 0; q < num_q; ++q) {
                int64_t opp_q = num_q - 1 - q;
                for (int64_t t = 0; t < horiz; ++t) {
                    float pos_v = pos_results[b].predictions[q * horiz + t];
                    float neg_v = neg_results[b].predictions[opp_q * horiz + t];
                    sym_results[b].predictions[q * horiz + t] = (pos_v - neg_v) * 0.5f;
                }
            }
            if (config.make_positive) {
                for (auto& val : sym_results[b].predictions) {
                    if (val < 0.0f) val = 0.0f;
                }
            }
        }
        return sym_results;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t B = static_cast<int64_t>(batch_series.size());
    std::vector<ForecastResult> results(B);
    if (B == 0 || !is_loaded_) return results;

    for (int64_t i = 0; i < B; ++i) {
        results[i].horizon = config.horizon;
        results[i].num_quantiles = static_cast<int64_t>(quantiles_.size());
        results[i].quantiles = quantiles_;
        results[i].predictions.assign(results[i].num_quantiles * results[i].horizon, 0.0f);
    }

    struct SeriesPreproc {
        std::vector<float> normed_ctx;
        int64_t original_len = 0;
        LinearTrend trend;
        RevINState revin;
        std::vector<float> norm_forecast;
        bool all_positive = true;
    };

    std::vector<SeriesPreproc> preprocs(B);

    for (int64_t b = 0; b < B; ++b) {
        const auto& raw = batch_series[b];
        if (raw.empty()) continue;

        std::vector<float> clean = raw;
        interpolate_nans(clean);
        preprocs[b].original_len = static_cast<int64_t>(clean.size());

        for (float val : clean) {
            if (std::isfinite(val) && val < 0.0f) {
                preprocs[b].all_positive = false;
                break;
            }
        }

        if (config.detrend) {
            std::vector<float> detrended(clean.size());
            preprocs[b].trend = fit_and_remove_linear_trend(
                clean.data(),
                preprocs[b].original_len,
                config.detrend_r2_threshold,
                detrended.data()
            );
            clean = std::move(detrended);
        }
        results[b].trend_slope = preprocs[b].trend.slope;
        results[b].trend_r2 = preprocs[b].trend.r_squared;
        results[b].detrend_applied = preprocs[b].trend.applied;

        if (config.normalize) {
            std::vector<float> normed(clean.size());
            revin_normalize(
                clean.data(),
                preprocs[b].original_len,
                nullptr,
                1e-5f,
                &preprocs[b].revin,
                normed.data()
            );
            preprocs[b].normed_ctx = std::move(normed);
        } else {
            preprocs[b].normed_ctx = std::move(clean);
        }
        results[b].mean = preprocs[b].revin.mean;
        results[b].std = preprocs[b].revin.std;
        preprocs[b].norm_forecast.assign(results[b].num_quantiles * results[b].horizon, 0.0f);
    }

    int64_t steps_generated = 0;
    int64_t q_num = static_cast<int64_t>(quantiles_.size());
    int64_t median_q = q_num / 2;

    while (steps_generated < config.horizon) {
        std::vector<std::vector<float>> all_patches(B);
        int64_t max_patches = 0;

        for (int64_t b = 0; b < B; ++b) {
            if (preprocs[b].normed_ctx.empty()) continue;
            int64_t np = create_model_patches(preprocs[b].normed_ctx, all_patches[b]);
            max_patches = std::max(max_patches, np);
        }

        if (max_patches == 0) break;

        std::vector<float> batch_input(B * max_patches * 192, 0.0f);
        for (int64_t b = 0; b < B; ++b) {
            if (all_patches[b].empty()) continue;
            int64_t cur_np = static_cast<int64_t>(all_patches[b].size()) / 192;
            int64_t pad_offset = (max_patches - cur_np) * 192;
            float* dst = batch_input.data() + b * (max_patches * 192) + pad_offset;
            std::copy(all_patches[b].begin(), all_patches[b].end(), dst);
        }

        std::vector<float> batch_out_64x9;
        forecast_chunk_batch(batch_input, B, max_patches, batch_out_64x9);

        int64_t chunk_steps = std::min<int64_t>(OUTPUT_PATCH_LEN, config.horizon - steps_generated);

        for (int64_t b = 0; b < B; ++b) {
            if (preprocs[b].normed_ctx.empty()) continue;
            const float* chunk_64x9 = batch_out_64x9.data() + b * (OUTPUT_PATCH_LEN * q_num);

            for (int64_t t = 0; t < chunk_steps; ++t) {
                int64_t global_t = steps_generated + t;
                for (int64_t q = 0; q < q_num; ++q) {
                    float val = chunk_64x9[t * q_num + q];
                    preprocs[b].norm_forecast[q * config.horizon + global_t] = val;
                }
            }

            for (int64_t t = 0; t < chunk_steps; ++t) {
                float median_val = chunk_64x9[t * q_num + median_q];
                preprocs[b].normed_ctx.push_back(median_val);
            }
        }

        steps_generated += chunk_steps;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double per_item_ms = B > 0 ? (elapsed_ms / B) : 0.0;

    for (int64_t b = 0; b < B; ++b) {
        if (preprocs[b].original_len == 0) continue;

        if (config.normalize && preprocs[b].revin.applied) {
            revin_denormalize(
                preprocs[b].norm_forecast.data(),
                results[b].horizon,
                results[b].num_quantiles,
                preprocs[b].revin,
                results[b].predictions.data()
            );
            cpm_iterative_revin_refine(
                results[b].predictions.data(),
                results[b].horizon,
                results[b].num_quantiles,
                preprocs[b].revin,
                0.95f
            );
        } else {
            results[b].predictions = preprocs[b].norm_forecast;
        }

        if (preprocs[b].trend.applied) {
            add_linear_trend_to_forecast(
                results[b].predictions.data(),
                results[b].horizon,
                results[b].num_quantiles,
                preprocs[b].original_len,
                preprocs[b].trend
            );
        }

        if (config.sort_quantiles) {
            for (int64_t t = 0; t < results[b].horizon; ++t) {
                std::vector<float> step_quantiles(results[b].num_quantiles);
                for (int64_t q = 0; q < results[b].num_quantiles; ++q) {
                    step_quantiles[q] = results[b].predictions[q * results[b].horizon + t];
                }
                std::sort(step_quantiles.begin(), step_quantiles.end());
                for (int64_t q = 0; q < results[b].num_quantiles; ++q) {
                    results[b].predictions[q * results[b].horizon + t] = step_quantiles[q];
                }
            }
        }

        bool should_clamp_pos = config.make_positive || (config.auto_positive && preprocs[b].all_positive);
        if (should_clamp_pos) {
            for (auto& val : results[b].predictions) {
                if (val < 0.0f) val = 0.0f;
            }
        }

        results[b].inference_time_ms = per_item_ms;
    }

    return results;
}

} // namespace timesfm
