#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <map>

#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "ggml.h"

struct BenchResult {
    std::string engine = "ggmlc";
    std::string model_filename;
    std::string model_type;
    uint64_t model_size = 0;
    uint64_t model_n_params = 0;
    std::string device;
    int n_threads = 1;
    int n_prompt = 0;
    int n_gen = 0;
    int n_depth = 0;
    std::string test_type; // "pp" or "tg"
    double avg_ns = 0.0;
    double stddev_ns = 0.0;
    double avg_ts = 0.0;
    double stddev_ts = 0.0;
    std::vector<double> samples_ts;
};

static std::vector<int> parse_int_list(const std::string& str) {
    std::vector<int> res;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            try {
                res.push_back(std::stoi(item));
            } catch (...) {}
        }
    }
    return res;
}

static std::string escape_json(const std::string& str) {
    std::string out;
    for (char c : str) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static void print_help(const char* prog_name) {
    std::cout << "================================================================================\n"
              << " ggmlc-bench : High-Performance Standalone Neural Computation Benchmark Tool\n"
              << "================================================================================\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  -h, --help                            Show this help menu and exit\n"
              << "  -m, --model <filename>                Path to compiled GGUF model\n"
              << "  -p, --n-prompt <n1,n2,...>            Prompt prefill lengths (default: 16,64,128,256,512,1024)\n"
              << "  -n, --n-gen <n1,n2,...>               Generation/decode token counts (default: 0,32,64,128)\n"
              << "  -pg <pp,tg>                           End-to-end: prefill pp then generate tg (repeatable;\n"
              << "                                        use -p 0 -n 0 to skip separate pp/tg tests)\n"
              << "  -r, --repetitions <n>                 Number of repetitions per test (default: 5)\n"
              << "  -o, --output <json|md|csv>            Output format printed to stdout (default: json)\n"
              << "  -d, --device <cpu|cuda>               Target execution backend (default: cpu)\n"
              << "  -t, --threads <n>                     CPU execution worker threads (default: 4)\n"
              << "  -ub, --ubatch <n>                     Prompt prefill physical chunk size (default: 512)\n"
              << "  --no-warmup                           Skip initial warmup passes\n"
              << "  --cuda-graph                          Enable CUDA graph capture on decode steps\n"
              << "  --unplanned                           Disable planned arena reuse (debug mode)\n"
              << "  --profile                             Print per-phase host/device timings\n"
              << "  --dump-graph                          Print serialized + runtime op histograms\n"
              << "  --full-logits                         Disable last-token lm_head gather (llama n_outputs=1)\n"
              << "  --diag-mode <normal|replay|host-only|no-pos-update>\n"
              << "                                        Isolate decode bottlenecks (tg tests only)\n"
              << "                                          normal: full prepare+KV mutate+run (default)\n"
              << "                                          replay: capture once, replay identical graph\n"
              << "                                          host-only: prepare/KV mutate, skip GPU run\n"
              << "                                          no-pos-update: run every step at frozen pos=1\n\n"
              << "Examples:\n"
              << "  " << prog_name << " -m model.gguf -p 16,64,128,256,512 -n 0 -r 5 -o md\n"
              << "  " << prog_name << " -m model.gguf -p 0 -n 32,64,128 -d cuda -o json\n"
              << "  " << prog_name << " -m model.gguf -p 0 -n 0 -pg 128,128 -pg 1024,32 -d cuda -o json\n"
              << "================================================================================\n";
}

static std::string make_test_label(const BenchResult& r) {
    if (r.test_type == "pg" || (r.n_prompt > 0 && r.n_gen > 0)) {
        return "pp" + std::to_string(r.n_prompt) + "+tg" + std::to_string(r.n_gen);
    }
    if (r.n_prompt > 0) {
        return "pp" + std::to_string(r.n_prompt);
    }
    return "tg" + std::to_string(r.n_gen);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    std::string model_path;
    std::vector<int> prompt_lens = {16, 64, 128, 256, 512, 1024};
    std::vector<int> gen_lens = {0, 32, 64, 128};
    std::vector<std::pair<int, int>> pg_cases;
    bool prompt_lens_set = false;
    bool gen_lens_set = false;
    int repetitions = 5;
    std::string output_format = "json";
    std::string device_name = "cpu";
    int n_threads = 4;
    int ubatch = 512;
    bool no_warmup = false;
    bool use_cuda_graph = false;
    bool unplanned = false;
    bool enable_profile = false;
    bool dump_graph = false;
    bool full_logits = false; // default: last-token gather (match llama-bench)
    std::string diag_mode = "normal";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--n-prompt") && i + 1 < argc) {
            prompt_lens = parse_int_list(argv[++i]);
            prompt_lens_set = true;
        } else if ((arg == "-n" || arg == "--n-gen") && i + 1 < argc) {
            gen_lens = parse_int_list(argv[++i]);
            gen_lens_set = true;
        } else if (arg == "-pg" && i + 1 < argc) {
            auto vals = parse_int_list(argv[++i]);
            if (vals.size() != 2) {
                std::cerr << "[ggmlc-bench ERROR] -pg expects exactly two ints: <pp,tg>\n";
                return 1;
            }
            pg_cases.emplace_back(vals[0], vals[1]);
        } else if ((arg == "-r" || arg == "--repetitions") && i + 1 < argc) {
            repetitions = std::max(1, std::stoi(argv[++i]));
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_format = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_name = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            n_threads = std::max(1, std::stoi(argv[++i]));
        } else if ((arg == "-ub" || arg == "--ubatch" || arg == "--chunk-size") && i + 1 < argc) {
            ubatch = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--no-warmup") {
            no_warmup = true;
        } else if (arg == "--cuda-graph") {
            use_cuda_graph = true;
        } else if (arg == "--unplanned") {
            unplanned = true;
        } else if (arg == "--profile") {
            enable_profile = true;
        } else if (arg == "--dump-graph") {
            dump_graph = true;
        } else if (arg == "--full-logits") {
            full_logits = true;
        } else if (arg == "--diag-mode" && i + 1 < argc) {
            diag_mode = argv[++i];
        } else if (model_path.empty() && arg.rfind("-", 0) != 0) {
            model_path = arg;
        }
    }

    // Match llama-bench: when only -pg is supplied, skip default separate pp/tg matrix.
    if (!pg_cases.empty() && !prompt_lens_set && !gen_lens_set) {
        prompt_lens = {0};
        gen_lens = {0};
    }

    if (model_path.empty()) {
        std::cerr << "[ggmlc-bench ERROR] No model file specified. Use '-m <path.gguf>'.\n";
        return 1;
    }

    std::ifstream file_check(model_path, std::ios::binary | std::ios::ate);
    if (!file_check.is_open()) {
        std::cerr << "[ggmlc-bench ERROR] Cannot open model file: " << model_path << "\n";
        return 1;
    }
    uint64_t file_size_bytes = static_cast<uint64_t>(file_check.tellg());
    file_check.close();

    try {
        auto model_graph = ggmlc::ModelLoader::load_from_file(model_path);

        // Deduce input and position IDs
        uint32_t in_tid = 0;
        int32_t pos_tid = -1;
        if (!model_graph.inputs.empty()) {
            in_tid = model_graph.inputs[0];
            for (size_t idx = 1; idx < model_graph.inputs.size(); ++idx) {
                uint32_t tid = model_graph.inputs[idx];
                auto it = model_graph.tensors.find(tid);
                if (it != model_graph.tensors.end()) {
                    if (it->second.name == "position_ids" || it->second.name.find("pos") != std::string::npos) {
                        pos_tid = static_cast<int32_t>(tid);
                    }
                }
            }
        }

        bool use_kv_cache = false;
        for (const auto& op : model_graph.ops) {
            if (op.opcode == 74) { // GGML_OP_FLASH_ATTN_EXT
                use_kv_cache = true;
                break;
            }
        }

        int max_prompt = 0;
        for (int p : prompt_lens) if (p > max_prompt) max_prompt = p;
        int max_gen = 0;
        for (int g : gen_lens) if (g > max_gen) max_gen = g;
        for (const auto& pg : pg_cases) {
            if (pg.first > max_prompt) max_prompt = pg.first;
            if (pg.second > max_gen) max_gen = pg.second;
        }
        int max_ctx = max_prompt + max_gen + 512;

        ggmlc::ModelExecutor executor(model_graph, device_name);
        if (use_cuda_graph) {
            executor.set_enable_cuda_graph(true);
        }
        if (enable_profile) {
            executor.set_enable_profile(true);
        }
        // Match llama-bench: only last-token logits unless --full-logits.
        executor.set_logits_last_only(!full_logits);
        if (use_kv_cache) {
            executor.init_kv_cache(max_ctx);
        }

        if (dump_graph) {
            std::map<std::string, int> ser_hist;
            for (const auto& op : model_graph.ops) {
                const char* name = (op.opcode >= 0 && op.opcode < GGML_OP_COUNT)
                    ? ggml_op_name(static_cast<enum ggml_op>(op.opcode))
                    : "CUSTOM";
                ser_hist[name]++;
            }
            std::cerr << "[ggmlc-bench] serialized_ops n=" << model_graph.ops.size()
                      << " ggml_cuda_graphs_compiled="
                      << (ggmlc::ModelExecutor::ggml_cuda_graphs_compiled() ? "1" : "0")
                      << " diag_mode=" << diag_mode << "\n";
            for (const auto& pair : ser_hist) {
                std::cerr << "  ser " << pair.first << " " << pair.second << "\n";
            }
        }

        uint64_t param_elements = 0;
        std::unordered_map<std::string, int64_t> dummy_sym_env;
        for (const auto& sym : model_graph.symbol_table) {
            dummy_sym_env[sym] = 1;
        }
        for (const auto& pair : model_graph.tensors) {
            if (pair.second.storage == ggmlc::StorageClass::PARAMETER || pair.second.data_ptr != nullptr) {
                uint64_t nelems = 1;
                for (const auto& d : pair.second.ne) {
                    if (d) {
                        try {
                            nelems *= std::max<int64_t>(1, d->evaluate(dummy_sym_env, model_graph.symbol_table));
                        } catch (...) {
                            nelems *= 1;
                        }
                    }
                }
                param_elements += nelems;
            }
        }

        std::vector<BenchResult> results;

        // ====================================================================
        // 1. Prompt Processing (pp) Benchmarks
        // ====================================================================
        for (int P : prompt_lens) {
            if (P <= 0) continue;

            int effective_chunk_size = (ubatch > 0 && use_kv_cache) ? ubatch : P;
            int n_chunks = (P + effective_chunk_size - 1) / effective_chunk_size;

            std::vector<int32_t> tokens(P, 1);
            std::vector<int32_t> pos_vec;
            if (pos_tid >= 0) {
                pos_vec.resize(P);
                for (int i = 0; i < P; ++i) pos_vec[i] = i;
            }

            // Warmup
            if (!no_warmup) {
                if (use_kv_cache) executor.reset_kv_cache();
                for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
                    int c_start = chunk_idx * effective_chunk_size;
                    int c_len = std::min(effective_chunk_size, P - c_start);
                    int pos = c_start;

                    std::unordered_map<std::string, int64_t> symbol_env;
                    symbol_env["s"] = c_len;
                    for (const auto& sym : model_graph.symbol_table) {
                        symbol_env[sym] = c_len;
                    }
                    if (use_kv_cache) symbol_env["pos"] = pos;

                    executor.prepare(symbol_env, !unplanned);
                    executor.set_input(in_tid, tokens.data() + c_start, c_len * sizeof(int32_t));
                    if (pos_tid >= 0) {
                        executor.set_input(pos_tid, pos_vec.data() + c_start, c_len * sizeof(int32_t));
                    }
                    executor.run(n_threads);
                }
                executor.synchronize();
            }

            // Graph dump once per prompt length (independent of warmup).
            if (dump_graph) {
                if (use_kv_cache) executor.reset_kv_cache();
                int c_len = std::min(effective_chunk_size, P);
                std::unordered_map<std::string, int64_t> symbol_env;
                symbol_env["s"] = c_len;
                for (const auto& sym : model_graph.symbol_table) {
                    symbol_env[sym] = c_len;
                }
                if (use_kv_cache) symbol_env["pos"] = 0;
                executor.prepare(symbol_env, !unplanned);
                std::cerr << "[ggmlc-bench] runtime prefill graph P=" << P
                          << " chunk=" << c_len << ": "
                          << executor.runtime_graph_summary() << "\n";
                std::cerr << "[ggmlc-bench] mul_mat/fa shapes: "
                          << executor.runtime_mul_mat_shape_summary() << "\n";
            }

            std::vector<double> samples_ns;
            std::vector<double> samples_ts;
            for (int r = 0; r < repetitions; ++r) {
                if (use_kv_cache) executor.reset_kv_cache();
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
                    int c_start = chunk_idx * effective_chunk_size;
                    int c_len = std::min(effective_chunk_size, P - c_start);
                    int pos = c_start;

                    std::unordered_map<std::string, int64_t> symbol_env;
                    symbol_env["s"] = c_len;
                    for (const auto& sym : model_graph.symbol_table) {
                        symbol_env[sym] = c_len;
                    }
                    if (use_kv_cache) symbol_env["pos"] = pos;

                    executor.prepare(symbol_env, !unplanned);
                    executor.set_input(in_tid, tokens.data() + c_start, c_len * sizeof(int32_t));
                    if (pos_tid >= 0) {
                        executor.set_input(pos_tid, pos_vec.data() + c_start, c_len * sizeof(int32_t));
                    }
                    executor.run(n_threads);
                }
                executor.synchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                samples_ns.push_back(ns);
                double ts = (double)P / (ns * 1e-9);
                samples_ts.push_back(ts);
            }

            double mean_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / samples_ns.size();
            double sq_sum_ns = 0.0;
            for (double s : samples_ns) sq_sum_ns += (s - mean_ns) * (s - mean_ns);
            double std_ns = samples_ns.size() > 1 ? std::sqrt(sq_sum_ns / (samples_ns.size() - 1)) : 0.0;

            double mean_ts = std::accumulate(samples_ts.begin(), samples_ts.end(), 0.0) / samples_ts.size();
            double sq_sum_ts = 0.0;
            for (double s : samples_ts) sq_sum_ts += (s - mean_ts) * (s - mean_ts);
            double std_ts = samples_ts.size() > 1 ? std::sqrt(sq_sum_ts / (samples_ts.size() - 1)) : 0.0;

            BenchResult br;
            br.model_filename = model_path;
            br.model_type = model_graph.name;
            br.model_size = file_size_bytes;
            br.model_n_params = param_elements;
            br.device = device_name;
            br.n_threads = n_threads;
            br.n_prompt = P;
            br.n_gen = 0;
            br.test_type = "pp";
            br.avg_ns = mean_ns;
            br.stddev_ns = std_ns;
            br.avg_ts = mean_ts;
            br.stddev_ts = std_ts;
            br.samples_ts = samples_ts;
            results.push_back(br);
            if (enable_profile) {
                auto p = executor.get_profile();
                std::cerr << "[ggmlc-bench] profile pp" << P
                          << " prepare_ms=" << p.prepare_ms
                          << " set_chunk_pos_ms=" << p.set_chunk_pos_ms
                          << " mask_fill_ms=" << p.mask_fill_ms
                          << " run_ms=" << p.run_ms
                          << " n_prepare=" << p.n_prepare
                          << " n_run=" << p.n_run << "\n";
                executor.reset_profile();
            }
        }

        // ====================================================================
        // 2. Text Generation / Token Decode (tg) Benchmarks (S=1 with KV cache)
        // ====================================================================
        for (int N : gen_lens) {
            if (N <= 0) continue;

            int32_t dummy_token = 1;
            int32_t cur_pos = 0;

            // Warmup 1 token
            if (!no_warmup) {
                if (use_kv_cache) executor.reset_kv_cache();
                std::unordered_map<std::string, int64_t> sym_env;
                sym_env["s"] = 1;
                for (const auto& sym : model_graph.symbol_table) {
                    sym_env[sym] = 1;
                }
                if (use_kv_cache) sym_env["pos"] = 0;

                executor.prepare(sym_env, !unplanned);
                executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                executor.run(n_threads);
                executor.synchronize();
            }

            std::vector<double> samples_ns;
            std::vector<double> samples_ts;

            for (int r = 0; r < repetitions; ++r) {
                if (use_kv_cache) executor.reset_kv_cache();

                // Initial token (pos 0) outside the timed decode loop
                std::unordered_map<std::string, int64_t> sym_env;
                sym_env["s"] = 1;
                for (const auto& sym : model_graph.symbol_table) {
                    sym_env[sym] = 1;
                }
                if (use_kv_cache) sym_env["pos"] = 0;

                cur_pos = 0;
                executor.prepare(sym_env, !unplanned);
                if (dump_graph && r == 0) {
                    std::cerr << "[ggmlc-bench] runtime decode graph: "
                              << executor.runtime_graph_summary() << "\n";
                    dump_graph = false;
                }
                executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                executor.run(n_threads);
                executor.synchronize();

                if (enable_profile) executor.reset_profile();

                // Time pure N generation steps
                auto t0 = std::chrono::high_resolution_clock::now();
                if (diag_mode == "replay") {
                    // Freeze the graph at pos=1 and replay identical compute.
                    cur_pos = 1;
                    if (use_kv_cache) sym_env["pos"] = cur_pos;
                    executor.prepare(sym_env, !unplanned);
                    executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                    if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                    for (int step = 1; step <= N; ++step) {
                        executor.run(n_threads);
                    }
                } else if (diag_mode == "no-pos-update") {
                    cur_pos = 1;
                    if (use_kv_cache) sym_env["pos"] = cur_pos;
                    executor.prepare(sym_env, !unplanned);
                    for (int step = 1; step <= N; ++step) {
                        executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                        if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                        executor.run(n_threads);
                    }
                } else if (diag_mode == "host-only") {
                    for (int step = 1; step <= N; ++step) {
                        cur_pos = step;
                        if (use_kv_cache) sym_env["pos"] = cur_pos;
                        executor.prepare(sym_env, !unplanned);
                        executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                        if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                    }
                } else {
                    for (int step = 1; step <= N; ++step) {
                        cur_pos = step;
                        if (use_kv_cache) sym_env["pos"] = cur_pos;
                        executor.prepare(sym_env, !unplanned);
                        executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                        if (pos_tid >= 0) executor.set_input(pos_tid, &cur_pos, sizeof(int32_t));
                        executor.run(n_threads);
                    }
                }
                executor.synchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                samples_ns.push_back(ns);
                double ts = (double)N / (ns * 1e-9);
                samples_ts.push_back(ts);
            }

            double mean_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / samples_ns.size();
            double sq_sum_ns = 0.0;
            for (double s : samples_ns) sq_sum_ns += (s - mean_ns) * (s - mean_ns);
            double std_ns = samples_ns.size() > 1 ? std::sqrt(sq_sum_ns / (samples_ns.size() - 1)) : 0.0;

            double mean_ts = std::accumulate(samples_ts.begin(), samples_ts.end(), 0.0) / samples_ts.size();
            double sq_sum_ts = 0.0;
            for (double s : samples_ts) sq_sum_ts += (s - mean_ts) * (s - mean_ts);
            double std_ts = samples_ts.size() > 1 ? std::sqrt(sq_sum_ts / (samples_ts.size() - 1)) : 0.0;

            BenchResult br;
            br.model_filename = model_path;
            br.model_type = model_graph.name;
            br.model_size = file_size_bytes;
            br.model_n_params = param_elements;
            br.device = device_name;
            br.n_threads = n_threads;
            br.n_prompt = 0;
            br.n_gen = N;
            br.test_type = "tg";
            br.avg_ns = mean_ns;
            br.stddev_ns = std_ns;
            br.avg_ts = mean_ts;
            br.stddev_ts = std_ts;
            br.samples_ts = samples_ts;
            results.push_back(br);
            if (enable_profile) {
                auto p = executor.get_profile();
                std::cerr << "[ggmlc-bench] profile tg" << N
                          << " diag=" << diag_mode
                          << " prepare_ms=" << p.prepare_ms
                          << " set_decode_pos_ms=" << p.set_decode_pos_ms
                          << " run_ms=" << p.run_ms
                          << " n_prepare=" << p.n_prepare
                          << " n_set_decode_pos=" << p.n_set_decode_pos
                          << " n_run=" << p.n_run << "\n";
                executor.reset_profile();
            }
        }

        // ====================================================================
        // 3. End-to-end prompt + generation (pg) — wall clock a user feels
        // ====================================================================
        for (const auto& pg : pg_cases) {
            const int P = pg.first;
            const int N = pg.second;
            if (P <= 0 && N <= 0) continue;

            int effective_chunk_size = (ubatch > 0 && use_kv_cache) ? ubatch : std::max(P, 1);
            int n_chunks = P > 0 ? (P + effective_chunk_size - 1) / effective_chunk_size : 0;

            std::vector<int32_t> tokens(std::max(P, 1), 1);
            std::vector<int32_t> pos_vec;
            if (pos_tid >= 0 && P > 0) {
                pos_vec.resize(P);
                for (int i = 0; i < P; ++i) pos_vec[i] = i;
            }
            int32_t dummy_token = 1;

            auto run_prefill = [&]() {
                for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
                    int c_start = chunk_idx * effective_chunk_size;
                    int c_len = std::min(effective_chunk_size, P - c_start);
                    int pos = c_start;

                    std::unordered_map<std::string, int64_t> symbol_env;
                    symbol_env["s"] = c_len;
                    for (const auto& sym : model_graph.symbol_table) {
                        symbol_env[sym] = c_len;
                    }
                    if (use_kv_cache) symbol_env["pos"] = pos;

                    executor.prepare(symbol_env, !unplanned);
                    executor.set_input(in_tid, tokens.data() + c_start, c_len * sizeof(int32_t));
                    if (pos_tid >= 0) {
                        executor.set_input(pos_tid, pos_vec.data() + c_start, c_len * sizeof(int32_t));
                    }
                    executor.run(n_threads);
                }
            };

            auto run_decode = [&](int start_pos, int n_steps) {
                std::unordered_map<std::string, int64_t> sym_env;
                sym_env["s"] = 1;
                for (const auto& sym : model_graph.symbol_table) {
                    sym_env[sym] = 1;
                }
                for (int step = 0; step < n_steps; ++step) {
                    int cur_pos = start_pos + step;
                    if (use_kv_cache) sym_env["pos"] = cur_pos;
                    executor.prepare(sym_env, !unplanned);
                    executor.set_input(in_tid, &dummy_token, sizeof(int32_t));
                    if (pos_tid >= 0) {
                        int32_t pos_i32 = cur_pos;
                        executor.set_input(pos_tid, &pos_i32, sizeof(int32_t));
                    }
                    executor.run(n_threads);
                }
            };

            if (!no_warmup) {
                if (use_kv_cache) executor.reset_kv_cache();
                if (P > 0) run_prefill();
                if (N > 0) run_decode(P, 1);
                executor.synchronize();
            }

            std::vector<double> samples_ns;
            std::vector<double> samples_ts;
            std::vector<double> samples_ttft_ns;
            std::vector<double> samples_decode_ns;

            for (int r = 0; r < repetitions; ++r) {
                if (use_kv_cache) executor.reset_kv_cache();

                auto t0 = std::chrono::high_resolution_clock::now();
                if (P > 0) run_prefill();
                executor.synchronize();
                auto t_mid = std::chrono::high_resolution_clock::now();
                if (N > 0) run_decode(P, N);
                executor.synchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                double ns_total = std::chrono::duration<double, std::nano>(t1 - t0).count();
                double ns_ttft = std::chrono::duration<double, std::nano>(t_mid - t0).count();
                double ns_decode = std::chrono::duration<double, std::nano>(t1 - t_mid).count();
                samples_ns.push_back(ns_total);
                samples_ttft_ns.push_back(ns_ttft);
                samples_decode_ns.push_back(ns_decode);
                int n_tokens = P + N;
                samples_ts.push_back(n_tokens > 0 ? (double)n_tokens / (ns_total * 1e-9) : 0.0);
            }

            auto mean_std = [](const std::vector<double>& v) -> std::pair<double, double> {
                double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
                double sq = 0.0;
                for (double s : v) sq += (s - mean) * (s - mean);
                double stdev = v.size() > 1 ? std::sqrt(sq / (v.size() - 1)) : 0.0;
                return {mean, stdev};
            };

            auto [mean_ns, std_ns] = mean_std(samples_ns);
            auto [mean_ts, std_ts] = mean_std(samples_ts);
            auto [mean_ttft, std_ttft] = mean_std(samples_ttft_ns);
            auto [mean_dec, std_dec] = mean_std(samples_decode_ns);
            (void)std_ttft;
            (void)std_dec;

            BenchResult br;
            br.model_filename = model_path;
            br.model_type = model_graph.name;
            br.model_size = file_size_bytes;
            br.model_n_params = param_elements;
            br.device = device_name;
            br.n_threads = n_threads;
            br.n_prompt = P;
            br.n_gen = N;
            br.test_type = "pg";
            br.avg_ns = mean_ns;
            br.stddev_ns = std_ns;
            br.avg_ts = mean_ts;
            br.stddev_ts = std_ts;
            br.samples_ts = samples_ts;
            results.push_back(br);

            // Store precise TTFT/decode ns for JSON emission (not tok/s samples).
            results.back().samples_ts.clear();
            results.back().samples_ts.push_back(mean_ttft);
            results.back().samples_ts.push_back(mean_dec);

            if (enable_profile) {
                std::cerr << "[ggmlc-bench] profile pg pp" << P << "+tg" << N
                          << " total_ms=" << (mean_ns * 1e-6)
                          << " ttft_ms=" << (mean_ttft * 1e-6)
                          << " decode_ms=" << (mean_dec * 1e-6) << "\n";
            }
        }

        // ====================================================================
        // 4. Render Output
        // ====================================================================
        if (output_format == "json") {
            std::cout << "[\n";
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                std::string test_name = make_test_label(r);
                double ttft_ms = 0.0;
                double decode_ms = 0.0;
                if (r.test_type == "pg" && r.samples_ts.size() >= 2) {
                    ttft_ms = r.samples_ts[0] * 1e-6;
                    decode_ms = r.samples_ts[1] * 1e-6;
                } else if (r.test_type == "pp") {
                    ttft_ms = r.avg_ns * 1e-6;
                } else if (r.test_type == "tg") {
                    decode_ms = r.avg_ns * 1e-6;
                }
                std::cout << "  {\n"
                          << "    \"engine\": \"" << escape_json(r.engine) << "\",\n"
                          << "    \"model_filename\": \"" << escape_json(r.model_filename) << "\",\n"
                          << "    \"model_type\": \"" << escape_json(r.model_type) << "\",\n"
                          << "    \"model_size\": " << r.model_size << ",\n"
                          << "    \"model_n_params\": " << r.model_n_params << ",\n"
                          << "    \"device\": \"" << r.device << "\",\n"
                          << "    \"n_threads\": " << r.n_threads << ",\n"
                          << "    \"n_prompt\": " << r.n_prompt << ",\n"
                          << "    \"n_gen\": " << r.n_gen << ",\n"
                          << "    \"n_depth\": " << r.n_depth << ",\n"
                          << "    \"test\": \"" << test_name << "\",\n"
                          << "    \"avg_ns\": " << std::fixed << std::setprecision(0) << r.avg_ns << ",\n"
                          << "    \"stddev_ns\": " << std::fixed << std::setprecision(0) << r.stddev_ns << ",\n"
                          << "    \"avg_ts\": " << std::fixed << std::setprecision(2) << r.avg_ts << ",\n"
                          << "    \"stddev_ts\": " << std::fixed << std::setprecision(2) << r.stddev_ts << ",\n"
                          << "    \"avg_ms\": " << std::fixed << std::setprecision(3) << (r.avg_ns * 1e-6) << ",\n"
                          << "    \"ttft_ms\": " << std::fixed << std::setprecision(3) << ttft_ms << ",\n"
                          << "    \"decode_ms\": " << std::fixed << std::setprecision(3) << decode_ms << "\n"
                          << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
            }
            std::cout << "]\n";
        } else if (output_format == "csv") {
            std::cout << "engine,model_filename,model_type,model_size,model_n_params,device,n_threads,n_prompt,n_gen,test,avg_ns,stddev_ns,avg_ts,stddev_ts,avg_ms,ttft_ms,decode_ms\n";
            for (const auto& r : results) {
                std::string test_name = make_test_label(r);
                double ttft_ms = 0.0;
                double decode_ms = 0.0;
                if (r.test_type == "pg" && r.samples_ts.size() >= 2) {
                    ttft_ms = r.samples_ts[0] * 1e-6;
                    decode_ms = r.samples_ts[1] * 1e-6;
                } else if (r.test_type == "pp") {
                    ttft_ms = r.avg_ns * 1e-6;
                } else if (r.test_type == "tg") {
                    decode_ms = r.avg_ns * 1e-6;
                }
                std::cout << r.engine << ",\"" << r.model_filename << "\",\"" << r.model_type << "\","
                          << r.model_size << "," << r.model_n_params << ",\"" << r.device << "\","
                          << r.n_threads << "," << r.n_prompt << "," << r.n_gen << ",\"" << test_name << "\","
                          << std::fixed << std::setprecision(0) << r.avg_ns << ","
                          << std::fixed << std::setprecision(0) << r.stddev_ns << ","
                          << std::fixed << std::setprecision(2) << r.avg_ts << ","
                          << std::fixed << std::setprecision(2) << r.stddev_ts << ","
                          << std::fixed << std::setprecision(3) << (r.avg_ns * 1e-6) << ","
                          << std::fixed << std::setprecision(3) << ttft_ms << ","
                          << std::fixed << std::setprecision(3) << decode_ms << "\n";
            }
        } else {
            // Markdown output format
            double size_gib = (double)file_size_bytes / (1024.0 * 1024.0 * 1024.0);
            double params_m = (double)param_elements / 1e6;
            std::cout << "| model | size | params | backend | threads | test | t/s | wall ms |\n"
                      << "| :--- | ---: | ---: | :--- | ---: | :--- | ---: | ---: |\n";
            for (const auto& r : results) {
                std::string test_label = make_test_label(r);
                std::cout << "| " << r.model_type << " | "
                          << std::fixed << std::setprecision(2) << size_gib << " GiB | "
                          << std::fixed << std::setprecision(1) << params_m << " M | "
                          << r.device << " | "
                          << r.n_threads << " | "
                          << test_label << " | "
                          << std::fixed << std::setprecision(2) << r.avg_ts << " ± "
                          << std::fixed << std::setprecision(2) << r.stddev_ts << " | "
                          << std::fixed << std::setprecision(1) << (r.avg_ns * 1e-6) << " |\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ggmlc-bench ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
