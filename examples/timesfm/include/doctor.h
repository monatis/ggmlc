#pragma once

#include <string>
#include <vector>
#include "forecaster.h"
#include "data_loader.h"
#include "backtest.h"

namespace timesfm {

struct DiagnosticCheck {
    std::string name;
    std::string category;     // "Sanity", "Speed", "Accuracy"
    bool passed = false;
    std::string measured_value;
    std::string target_threshold;
    std::string details;
};

struct DoctorReport {
    std::string model_name;
    std::string device;
    int n_threads = 4;
    int total_checks = 0;
    int passed_checks = 0;
    bool all_passed = false;
    double total_runtime_ms = 0.0;
    double single_latency_ms = 0.0;
    double batch8_throughput_pts_sec = 0.0;
    std::vector<DiagnosticCheck> checks;
};

class DoctorEngine {
public:
    // Runs full automated diagnostic health, speed, and accuracy check suite.
    static DoctorReport run_diagnostics(
        TimesFMForecaster& forecaster,
        const std::string& device = "cpu",
        int n_threads = 4
    );

    // Prints formatted CLI summary report with tables and pass/fail indicators.
    static void print_report(const DoctorReport& report);
};

} // namespace timesfm
