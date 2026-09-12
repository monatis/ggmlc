#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include "forecaster.h"

int main(int argc, char** argv) {
    std::string model_path = "scratch/timesfm3_f16.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }
    std::string device = "cuda";
    if (argc > 2) {
        device = argv[2];
    }

    std::cout << "====================================================================\n"
              << " Google TimesFM 3.0 Native C++ Hardware Benchmark (CPU vs CUDA GPU)\n"
              << " Model: " << model_path << " | Device: " << device << "\n"
              << "====================================================================\n\n";

    timesfm::TimesFMForecaster forecaster;
    if (!forecaster.load_model(model_path, device, 4)) {
        std::cerr << "Failed to load model on " << device << std::endl;
        return 1;
    }

    std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16, 32};
    timesfm::ForecastConfig cfg;
    cfg.horizon = 64;
    cfg.device = device;
    cfg.n_threads = 4;

    std::cout << "| Batch Size | Total Latency (ms) | Per-Series (ms) | Throughput (series/s) | Points/sec (pts/s) |\n";
    std::cout << "| :--- | :--- | :--- | :--- | :--- |\n";

    for (int64_t B : batch_sizes) {
        // Generate B synthetic series of 128 points
        std::vector<std::vector<float>> batch(B);
        for (int64_t b = 0; b < B; ++b) {
            batch[b].resize(128);
            for (int i = 0; i < 128; ++i) {
                batch[b][i] = 100.0f + static_cast<float>(i + b) * 0.3f + 10.0f * std::sin((i / 12.0f) * 6.28f);
            }
        }

        // Warmup
        for (int w = 0; w < 2; ++w) {
            auto warmup_res = forecaster.forecast_batch(batch, cfg);
        }

        // Timed Runs
        int num_runs = 5;
        std::vector<double> latencies;
        for (int r = 0; r < num_runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = forecaster.forecast_batch(batch, cfg);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies.push_back(ms);
        }

        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double avg_ms = sum / num_runs;
        double per_series_ms = avg_ms / static_cast<double>(B);
        double series_per_sec = (static_cast<double>(B) / avg_ms) * 1000.0;
        double points_per_sec = series_per_sec * 64.0;

        std::cout << "| B=" << std::setw(2) << B
                  << " | " << std::fixed << std::setprecision(2) << std::setw(8) << avg_ms << " ms"
                  << " | " << std::setw(7) << per_series_ms << " ms"
                  << " | " << std::setw(6) << std::setprecision(1) << series_per_sec << " series/s"
                  << " | **" << std::setprecision(0) << std::setw(6) << points_per_sec << " pts/s** |\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
