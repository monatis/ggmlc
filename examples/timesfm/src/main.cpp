#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>

#include "forecaster.h"
#include "data_loader.h"
#include "export.h"
#include "visualizer.h"
#include "backtest.h"
#include "doctor.h"
#include "server.h"

static void print_help(const char* prog) {
    std::cout << "====================================================================\n"
              << " Google TimesFM 3.0 — High-Performance Foundation Forecasting Engine\n"
              << " Standalone Zero-Dependency C++ Implementation powered by ggmlc\n"
              << "====================================================================\n\n"
              << "USAGE:\n"
              << "  " << prog << " [model.gguf] [options]\n\n"
              << "INSPECTION, HEALTH & SERVING:\n"
              << "  -h, --help             Show this help menu and exit\n"
              << "  --info                 Inspect GGUF metadata, shapes, and capabilities\n"
              << "  --doctor               Run automated diagnostic health, speed & accuracy checks\n"
              << "  --serve                Start embedded Web Dashboard & REST API server\n"
              << "  --port <P>             HTTP server port (default: 8080)\n\n"
              << "DATA INGESTION & PRESETS:\n"
              << "  --input <file.csv>     Input time series CSV dataset\n"
              << "  --text \"<v1,v2...>\"   Direct freeform sequence of pasted numbers\n"
              << "  --preset <name>        Load built-in synthetic or benchmark dataset\n"
              << "  --list-presets         List all available built-in preset scenarios\n"
              << "  --column <name>        Target value column name (default: auto-detect)\n"
              << "  --horizon <H>          Forecast horizon length (default: 128)\n"
              << "  --no-norm              Disable RevIN normalization & CPM refinement\n"
              << "  --no-detrend           Disable linear detrending\n"
              << "  --no-sort              Disable quantile monotonicity sorting\n"
              << "  --non-negative         Enforce non-negativity constraint (clamp >= 0)\n"
              << "  --sym-avg              Enable symmetric flip-invariance averaging\n\n"
              << "BACKTESTING ENGINE:\n"
              << "  --backtest             Run rolling-window backtest on input series\n"
              << "  --context <C>          Backtest context window length (default: 128)\n"
              << "  --stride <S>           Stride between rolling windows (default: 64)\n"
              << "  --max-windows <N>      Maximum rolling evaluation windows (default: 32)\n"
              << "  --output-backtest <f>  Save backtest metrics and window forecasts to JSON\n\n"
              << "EXPORT & VISUALIZATION:\n"
              << "  --output-csv <file>    Export forecast quantiles to CSV format\n"
              << "  --output-json <file>   Export forecast quantiles & metrics to JSON\n"
              << "  --svg <file.svg>       Generate standalone dark vector SVG line chart\n"
              << "  --html <file.html>     Generate standalone interactive Chart.js HTML\n\n"
              << "HARDWARE EXECUTION:\n"
              << "  --device <cpu|cuda>    Execution target device (default: cpu)\n"
              << "  --threads <N>          Worker threads for CPU execution (default: 4)\n\n"
              << "EXAMPLES:\n"
              << "  1. Run automated system doctor diagnostics:\n"
              << "     " << prog << " timesfm3_f16.gguf --doctor\n\n"
              << "  2. Run 128-step forecast on freeform numbers and export SVG:\n"
              << "     " << prog << " timesfm3_f16.gguf --text \"12.1, 14.5, 18.2, 22.0, 25.1, 28.4, 31.0, 35.2\" --horizon 16 --svg forecast.svg\n\n"
              << "  3. Run forecast on built-in 'weekly_retail' preset:\n"
              << "     " << prog << " timesfm3_f16.gguf --preset weekly_retail --horizon 28 --svg retail.svg\n\n"
              << "  4. Run rolling backtest on classic 'airline_passengers' benchmark:\n"
              << "     " << prog << " timesfm3_f16.gguf --preset airline_passengers --backtest --stride 12 --horizon 12 --svg airline_bt.svg\n\n"
              << "  5. Launch Web Dashboard & REST API:\n"
              << "     " << prog << " timesfm3_f16.gguf --serve --port 8080\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help("timesfm.exe");
        return 1;
    }

    std::string model_path = argv[1];
    if (model_path == "-h" || model_path == "--help") {
        print_help(argv[0]);
        return 0;
    }
    if (model_path == "--list-presets") {
        std::cout << "Available Built-in Presets:\n";
        for (const auto& p : timesfm::DataLoader::get_preset_names()) {
            std::cout << "  - " << p << "\n";
        }
        return 0;
    }

    std::string input_csv;
    std::string input_text;
    std::string preset_name;
    std::string target_column;
    std::string output_csv;
    std::string output_json;
    std::string output_backtest;
    std::string svg_path;
    std::string html_path;

    timesfm::ForecastConfig config;
    timesfm::BacktestConfig bt_config;
    bool backtest_mode = false;
    bool serve_mode = false;
    bool info_mode = false;
    bool doctor_mode = false;
    int port = 8080;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--list-presets") {
            std::cout << "Available Built-in Presets:\n";
            for (const auto& p : timesfm::DataLoader::get_preset_names()) {
                std::cout << "  - " << p << "\n";
            }
            return 0;
        } else if (arg == "--info") {
            info_mode = true;
        } else if (arg == "--doctor") {
            doctor_mode = true;
        } else if (arg == "--serve" || arg == "--server") {
            serve_mode = true;
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--input" && i + 1 < argc) {
            input_csv = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            input_text = argv[++i];
        } else if (arg == "--preset" && i + 1 < argc) {
            preset_name = argv[++i];
        } else if (arg == "--column" && i + 1 < argc) {
            target_column = argv[++i];
        } else if (arg == "--horizon" && i + 1 < argc) {
            int64_t h = std::atoll(argv[++i]);
            config.horizon = h;
            bt_config.horizon = h;
        } else if (arg == "--backtest") {
            backtest_mode = true;
        } else if ((arg == "--context" || arg == "--context-len") && i + 1 < argc) {
            bt_config.context_len = std::atoll(argv[++i]);
        } else if (arg == "--stride" && i + 1 < argc) {
            bt_config.stride = std::atoll(argv[++i]);
        } else if (arg == "--max-windows" && i + 1 < argc) {
            bt_config.max_windows = std::atoll(argv[++i]);
        } else if (arg == "--output-backtest" && i + 1 < argc) {
            output_backtest = argv[++i];
        } else if (arg == "--no-norm") {
            config.normalize = false;
            bt_config.normalize = false;
        } else if (arg == "--no-detrend") {
            config.detrend = false;
            bt_config.detrend = false;
        } else if (arg == "--no-sort") {
            config.sort_quantiles = false;
            bt_config.sort_quantiles = false;
        } else if (arg == "--make-positive" || arg == "--non-negative") {
            config.make_positive = true;
            bt_config.make_positive = true;
        } else if (arg == "--sym-avg") {
            config.use_symmetric_averaging = true;
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--output-json" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "--svg" && i + 1 < argc) {
            svg_path = argv[++i];
        } else if (arg == "--html" && i + 1 < argc) {
            html_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device = argv[++i];
            bt_config.device = config.device;
        } else if (arg == "--threads" && i + 1 < argc) {
            config.n_threads = std::atoi(argv[++i]);
            bt_config.n_threads = config.n_threads;
        }
    }

    timesfm::TimesFMForecaster forecaster;
    std::cout << "[INFO] Loading TimesFM 3.0 model from: " << model_path << std::endl;
    if (!forecaster.load_model(model_path, config.device, config.n_threads)) {
        std::cerr << "[ERROR] Failed to load GGUF model: " << model_path << std::endl;
        return 1;
    }

    if (doctor_mode) {
        std::cout << "[INFO] Running TimesFM 3.0 Doctor Diagnostics on " << config.device << "..." << std::endl;
        timesfm::DoctorReport rep = timesfm::DoctorEngine::run_diagnostics(forecaster, config.device, config.n_threads);
        timesfm::DoctorEngine::print_report(rep);
        return rep.all_passed ? 0 : 1;
    }

    if (info_mode) {
        const auto& g = forecaster.model_graph();
        std::cout << "\n================ Model Metadata ================\n"
                  << " Model Name      : " << g.name << "\n"
                  << " Input Tensors   : " << g.inputs.size() << "\n"
                  << " Output Tensors  : " << g.outputs.size() << "\n"
                  << " Dynamic Symbols : ";
        for (const auto& sym : g.symbol_table) {
            std::cout << sym << " ";
        }
        std::cout << "\n Parameters Count: " << g.parameters.size() << "\n"
                  << " Total Ops       : " << g.ops.size() << "\n"
                  << " Total Tensors   : " << g.tensors.size() << "\n"
                  << "================================================\n" << std::endl;
        return 0;
    }

    if (serve_mode) {
        timesfm::Server server(forecaster, port);
        server.start();
        return 0;
    }

    // Load time series data
    timesfm::TimeSeriesData data;
    std::string err_msg;

    if (!input_csv.empty()) {
        std::cout << "[INFO] Ingesting CSV dataset: " << input_csv << std::endl;
        if (!timesfm::DataLoader::load_csv(input_csv, target_column, data, err_msg)) {
            std::cerr << "[ERROR] " << err_msg << std::endl;
            return 1;
        }
        std::cout << "[INFO] Parsed " << data.values.size() << " points from column '" << data.column_name << "'";
        if (data.missing_count > 0) {
            std::cout << " (" << data.missing_count << " missing values interpolated)";
        }
        std::cout << std::endl;
    } else if (!input_text.empty()) {
        std::cout << "[INFO] Ingesting freeform text sequence..." << std::endl;
        if (!timesfm::DataLoader::parse_series_text(input_text, data, err_msg)) {
            std::cerr << "[ERROR] " << err_msg << std::endl;
            return 1;
        }
        std::cout << "[INFO] Parsed " << data.values.size() << " points from text input." << std::endl;
    } else if (!preset_name.empty()) {
        std::cout << "[INFO] Loading preset scenario: " << preset_name << std::endl;
        if (!timesfm::DataLoader::load_preset(preset_name, data, 192)) {
            std::cerr << "[ERROR] Failed to load preset: " << preset_name << std::endl;
            return 1;
        }
        std::cout << "[INFO] Loaded preset '" << data.column_name << "' (" << data.values.size() << " observations)." << std::endl;
    } else {
        std::cout << "[INFO] No dataset specified. Loading default 'trend_seasonal' preset (192 observations)..." << std::endl;
        timesfm::DataLoader::load_preset("trend_seasonal", data, 192);
    }

    if (backtest_mode) {
        std::cout << "[INFO] Running TimesFM 3.0 Backtesting Engine across " << data.values.size()
                  << " steps (Context=" << bt_config.context_len
                  << ", Horizon=" << bt_config.horizon
                  << ", Stride=" << bt_config.stride
                  << ", Device=" << bt_config.device << ")..." << std::endl;

        timesfm::BacktestResult bt_res = timesfm::BacktestEngine::run_backtest(forecaster, data.values, bt_config);

        std::cout << "\n================ Backtest Evaluation Metrics ================\n"
                  << " Evaluated Windows          : " << bt_res.total_windows << "\n"
                  << " Mean Absolute Error (MAE)  : " << std::fixed << std::setprecision(4) << bt_res.avg_mae << "\n"
                  << " Root Mean Sq Error (RMSE)  : " << bt_res.avg_rmse << "\n"
                  << " Symmetric MAPE (sMAPE)     : " << std::setprecision(2) << bt_res.avg_smape << "%\n"
                  << " Continuous RPS (CRPS)      : " << std::setprecision(4) << bt_res.avg_crps << "\n"
                  << " 80% Central Coverage (q10-90): " << std::setprecision(2) << bt_res.avg_coverage_80 << "%\n"
                  << " 40% Central Coverage (q30-70): " << std::setprecision(2) << bt_res.avg_coverage_40 << "%\n"
                  << " Naive Persistence Baseline : " << std::setprecision(4) << bt_res.avg_naive_mae << "\n"
                  << " MAE vs Naive Skill Ratio   : " << std::setprecision(3) << bt_res.avg_mae_vs_naive_ratio
                  << (bt_res.avg_mae_vs_naive_ratio < 1.0f ? " (Outperforms Naive Baseline!)" : "") << "\n"
                  << " Total Engine Runtime       : " << std::setprecision(2) << bt_res.total_time_ms << " ms\n"
                  << "=============================================================\n" << std::endl;

        if (!output_backtest.empty()) {
            if (timesfm::BacktestEngine::save_json(output_backtest, bt_res, data.values)) {
                std::cout << "[SUCCESS] Exported backtest metrics JSON to: " << output_backtest << std::endl;
            }
        }

        if (!svg_path.empty()) {
            if (timesfm::BacktestEngine::generate_svg(svg_path, data.values, bt_res, "TimesFM 3.0 Backtesting: " + data.column_name)) {
                std::cout << "[SUCCESS] Generated backtest multi-window SVG chart: " << svg_path << std::endl;
            }
        }

        return 0;
    }

    std::cout << "[INFO] Running TimesFM 3.0 forecast (Horizon=" << config.horizon << " steps, Device=" << config.device << ")..." << std::endl;
    timesfm::ForecastResult result = forecaster.forecast(data.values, config);

    std::cout << "\n================ Forecast Results ================\n"
              << " Horizon Steps   : " << result.horizon << "\n"
              << " Quantiles (9)   : 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\n"
              << " RevIN Stats     : Mean=" << std::fixed << std::setprecision(2) << result.mean
              << ", Std=" << result.std << "\n"
              << " Detrending      : " << (result.detrend_applied ? "Applied" : "Skipped")
              << " (Slope=" << std::setprecision(4) << result.trend_slope
              << ", R2=" << std::setprecision(2) << result.trend_r2 << ")\n"
              << " Inference Time  : " << std::setprecision(2) << result.inference_time_ms << " ms\n"
              << " Median Preview  : ";

    int preview_steps = std::min<int>(6, static_cast<int>(result.horizon));
    int q_med = static_cast<int>(result.num_quantiles / 2);
    for (int t = 0; t < preview_steps; ++t) {
        std::cout << std::setprecision(2) << result.predictions[q_med * result.horizon + t] << " ";
    }
    if (result.horizon > preview_steps) {
        std::cout << "... " << result.predictions[q_med * result.horizon + result.horizon - 1];
    }
    std::cout << "\n==================================================\n" << std::endl;

    // Export CSV
    if (!output_csv.empty()) {
        if (timesfm::Exporter::save_csv(output_csv, result, data.column_name)) {
            std::cout << "[SUCCESS] Exported forecast CSV to: " << output_csv << std::endl;
        }
    }

    // Export JSON
    if (!output_json.empty()) {
        if (timesfm::Exporter::save_json(output_json, result, data.values, data.timestamps)) {
            std::cout << "[SUCCESS] Exported forecast JSON to: " << output_json << std::endl;
        }
    }

    // Export SVG
    if (!svg_path.empty()) {
        if (timesfm::Visualizer::generate_svg(svg_path, data.values, result, "TimesFM 3.0: " + data.column_name)) {
            std::cout << "[SUCCESS] Generated vector SVG chart: " << svg_path << std::endl;
        }
    }

    // Export HTML
    if (!html_path.empty()) {
        if (timesfm::Visualizer::generate_html_chart(html_path, data.values, result, "TimesFM 3.0: " + data.column_name)) {
            std::cout << "[SUCCESS] Generated interactive HTML chart: " << html_path << std::endl;
        }
    }

    return 0;
}
