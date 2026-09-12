#include "engine.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

static void print_usage(const char* bin_name) {
    std::cout << "PlaidQ Offline Instant Code Tab Completion Engine\n"
              << "Usage: " << bin_name << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>       Path to compiled PlaidQ GGUF model (required)\n"
              << "  --prefix <str>       Code preceding cursor\n"
              << "  --suffix <str>       Code following cursor (default: \"\")\n"
              << "  --max-tokens <N>     Maximum tokens to complete (default: 32)\n"
              << "  --steps <N>          Diffusion sampling steps (default: 1)\n"
              << "  --device <dev>       Hardware backend: 'cpu', 'cuda', 'metal' (default: cpu)\n"
              << "  --threads <N>        Worker threads for CPU backend (default: 4)\n"
              << "  --temp <T>           Sampling temperature (0.0 = greedy, default: 0.0)\n"
              << "  --score-temp <T>     DDIM score temperature (default: 0.99)\n"
              << "  --canvas-len <N>     Total canvas sequence length (default: 256)\n"
              << "  --bench              Run continuous performance benchmark\n"
              << "  --daemon             Run JSON-RPC newline server on stdin/stdout for IDEs\n"
              << "  -h, --help           Show this help message\n\n"
              << "Examples:\n"
              << "  " << bin_name << " --model plaidq.gguf --prefix \"def quicksort(arr):\\n    \"\n"
              << "  " << bin_name << " --model plaidq.gguf --daemon --device cuda\n"
              << "  " << bin_name << " --model plaidq.gguf --bench --device cpu --threads 4\n";
}

// Simple JSON string unescaper helper
static std::string json_unescape(const std::string& s) {
    std::string res;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            if (next == 'n') res += '\n';
            else if (next == 't') res += '\t';
            else if (next == 'r') res += '\r';
            else if (next == '"') res += '"';
            else if (next == '\\') res += '\\';
            else res += next;
            i++;
        } else {
            res += s[i];
        }
    }
    return res;
}

// Simple JSON string escaper helper
static std::string json_escape(const std::string& s) {
    std::string res;
    for (char c : s) {
        if (c == '\n') res += "\\n";
        else if (c == '\t') res += "\\t";
        else if (c == '\r') res += "\\r";
        else if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else res += c;
    }
    return res;
}

int main(int argc, char** argv) {
    std::string model_path = "";
    std::string prefix = "def quicksort(arr):\n    ";
    std::string suffix = "";
    std::string device = "cpu";
    int max_new_tokens = 32;
    int sampling_steps = 1;
    int n_threads = 4;
    int canvas_len = 256;
    float temp = 0.0f;
    float score_temp = 0.99f;
    bool run_bench = false;
    bool daemon_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = json_unescape(argv[++i]);
        } else if (arg == "--suffix" && i + 1 < argc) {
            suffix = json_unescape(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            sampling_steps = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--score-temp" && i + 1 < argc) {
            score_temp = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--canvas-len" && i + 1 < argc) {
            canvas_len = std::atoi(argv[++i]);
        } else if (arg == "--bench") {
            run_bench = true;
        } else if (arg == "--daemon") {
            daemon_mode = true;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model <path.gguf> is required.\n";
        print_usage(argv[0]);
        return 1;
    }

    tab_completion::TabCompletionEngine engine;
    if (!engine.load_model(model_path, device, canvas_len)) {
        std::cerr << "Failed to initialize tab completion engine.\n";
        return 1;
    }

    bool score_temp_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--score-temp") score_temp_set = true;
    }
    if (!score_temp_set) {
        score_temp = (sampling_steps > 1) ? 0.5f : 0.99945f;
    }

    tab_completion::CompletionOptions options;
    options.max_new_tokens = max_new_tokens;
    options.sampling_steps = sampling_steps;
    options.sample_temp = temp;
    options.score_temp = score_temp;
    options.n_threads = n_threads;

    if (run_bench) {
        engine.benchmark(5, 2, options);
        return 0;
    }

    if (daemon_mode) {
        std::cout << "{\"status\":\"ready\",\"device\":\"" << device << "\",\"canvas_len\":" << canvas_len << "}" << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            // Parse simple fields: prefix, suffix, max_tokens, id
            std::string req_id = "0";
            std::string req_prefix = "";
            std::string req_suffix = "";
            int req_tokens = max_new_tokens;

            size_t id_pos = line.find("\"id\":");
            if (id_pos != std::string::npos) {
                size_t val_start = line.find_first_not_of(" :\"", id_pos + 5);
                size_t val_end = line.find_first_of(",} \"", val_start);
                if (val_start != std::string::npos) {
                    req_id = line.substr(val_start, val_end - val_start);
                }
            }

            size_t pref_pos = line.find("\"prefix\":");
            if (pref_pos != std::string::npos) {
                size_t quote1 = line.find('"', pref_pos + 9);
                if (quote1 != std::string::npos) {
                    size_t quote2 = quote1 + 1;
                    while (quote2 < line.size()) {
                        if (line[quote2] == '"' && line[quote2 - 1] != '\\') break;
                        quote2++;
                    }
                    if (quote2 < line.size()) {
                        req_prefix = json_unescape(line.substr(quote1 + 1, quote2 - quote1 - 1));
                    }
                }
            }

            size_t suff_pos = line.find("\"suffix\":");
            if (suff_pos != std::string::npos) {
                size_t quote1 = line.find('"', suff_pos + 9);
                if (quote1 != std::string::npos) {
                    size_t quote2 = quote1 + 1;
                    while (quote2 < line.size()) {
                        if (line[quote2] == '"' && line[quote2 - 1] != '\\') break;
                        quote2++;
                    }
                    if (quote2 < line.size()) {
                        req_suffix = json_unescape(line.substr(quote1 + 1, quote2 - quote1 - 1));
                    }
                }
            }

            size_t tok_pos = line.find("\"max_tokens\":");
            if (tok_pos != std::string::npos) {
                size_t val_start = line.find_first_not_of(" :\"", tok_pos + 13);
                size_t val_end = line.find_first_of(",} ", val_start);
                if (val_start != std::string::npos) {
                    req_tokens = std::atoi(line.substr(val_start, val_end - val_start).c_str());
                }
            }

            auto req_opts = options;
            req_opts.max_new_tokens = req_tokens;

            auto res = engine.complete(req_prefix, req_suffix, req_opts);

            std::cout << "{\"id\":" << req_id
                      << ",\"completion\":\"" << json_escape(res.completion_text) << "\""
                      << ",\"total_ms\":" << res.total_time_ms
                      << ",\"tokens_per_second\":" << res.tokens_per_second
                      << "}" << std::endl;
        }
        return 0;
    }

    // Default single-shot execution
    std::cout << "\n--- Input Prefix ---" << std::endl;
    std::cout << prefix << std::endl;
    if (!suffix.empty()) {
        std::cout << "--- Input Suffix ---" << std::endl;
        std::cout << suffix << std::endl;
    }
    std::cout << "--------------------" << std::endl;

    auto result = engine.complete(prefix, suffix, options);

    std::cout << "\n=== Tab Completion Result ===" << std::endl;
    std::cout << result.completion_text << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Latency: " << result.total_time_ms << " ms"
              << " (DDIM: " << result.ddim_time_ms << " ms, Decode: " << result.decode_time_ms << " ms)"
              << " | Speed: " << result.tokens_per_second << " tokens/sec\n" << std::endl;

    return 0;
}
