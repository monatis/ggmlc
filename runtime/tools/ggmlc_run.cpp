#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "ggmlc/pipeline/image.h"
#include "ggmlc/pipeline/tokenizer.h"

static void print_help(const char* prog_name) {
    std::cout << "================================================================================\n"
              << " ggmlc-run : High-Performance Standalone Neural Program Runner & Text Generator\n"
              << "================================================================================\n"
              << "Usage: " << prog_name << " <model.gguf> [options]\n\n"
              << "Inspection Options:\n"
              << "  -h, --help                  Show this comprehensive help message and exit\n"
              << "  --info                      Inspect GGUF metadata, tensor graph, and capabilities\n\n"
              << "Text Generation & Chat Options (SLMs):\n"
              << "  --chat <message>            Instruction chat generation with automatic template\n"
              << "  --prompt <string>           Autoregressive text completion from raw prompt\n"
              << "  --system <message>          System instruction prompt for chat template\n"
              << "  --generate                  Enable autoregressive token generation\n"
              << "  --max-tokens <N>            Maximum new tokens to generate (default: 32)\n"
              << "  --temperature <T>           Sampling temperature (0.0 = greedy argmax, default: 0.0)\n"
              << "  --top-p <P>                 Nucleus sampling probability (default: 0.9)\n"
              << "  --echo-prompt               Echo prompt before streaming response (for debugging)\n"
              << "  --show-special              Print special control tokens (e.g. <|im_end|>)\n\n"
              << "Preprocessing Options:\n"
              << "  --image <name:file.jpg>     Preprocess and set image tensor (bicubic + normalize)\n"
              << "  --text <name:string>        Tokenize and set text input tensor (BPE/WordPiece)\n\n"
              << "Execution & Hardware Options:\n"
              << "  --device <cpu|cuda>         Execution device (default: cpu)\n"
              << "  --threads <N>               Number of CPU execution threads (default: 1)\n"
              << "  --unplanned                 Disable memory arena reuse planning (for debugging)\n"
              << "  --symbol <key=value>        Bind dynamic symbol (e.g. s=128)\n\n"
              << "Raw Tensor I/O Options:\n"
              << "  --input <name:file.bin>     Load raw input tensor from binary file\n"
              << "  --output <id:file.bin>      Save computed output tensor ID to binary file\n"
              << "  --state-in <name:file.bin>  Load recurrent initial state from binary file\n"
              << "  --state-out <name:file.bin> Save recurrent final state to binary file\n\n"
              << "Examples:\n"
              << "  1. Instruction Chat with SmolLM2 / Llama:\n"
              << "     " << prog_name << " smollm2.gguf --chat \"What is gravity?\" --threads 4\n\n"
              << "  2. Chat with System Prompt on NVIDIA GPU:\n"
              << "     " << prog_name << " model.gguf --system \"You are concise.\" --chat \"Hello!\" --device cuda\n\n"
              << "  3. Text Completion:\n"
              << "     " << prog_name << " gpt2.gguf --prompt \"Once upon a time\" --max-tokens 24\n\n"
              << "  4. Inspect Model Capabilities & Tasks:\n"
              << "     " << prog_name << " model.gguf --info\n\n"
              << "  5. Image Classification / Vision Inference:\n"
              << "     " << prog_name << " resnet50.gguf --image x:cat.jpg --threads 4\n"
              << "================================================================================\n";
}

static void print_model_info(const std::string& model_path, const ggmlc::SerializedModelGraph& g) {
    std::cout << "================================================================================\n"
              << " Model Information: " << model_path << "\n"
              << "================================================================================\n"
              << " Name:            " << g.name << "\n";

    auto it_arch = g.metadata_str.find("general.architecture");
    if (it_arch != g.metadata_str.end()) {
        std::cout << " Architecture:    " << it_arch->second << "\n";
    }

    auto it_ver = g.metadata_str.find("ggmlc.version");
    if (it_ver != g.metadata_str.end()) {
        std::cout << " Compiler Ver:    " << it_ver->second << "\n";
    }

    auto tasks = g.get_tasks();
    if (!tasks.empty()) {
        std::cout << " Declared Tasks:  [";
        for (size_t i = 0; i < tasks.size(); ++i) {
            std::cout << tasks[i] << (i + 1 < tasks.size() ? ", " : "");
        }
        std::cout << "]\n";
    } else {
        std::cout << " Declared Tasks:  [none]\n";
    }

    std::cout << " Total Tensors:   " << g.tensors.size() << "\n"
              << " Graph Ops:       " << g.ops.size() << "\n";

    if (!g.symbol_table.empty()) {
        std::cout << " Dynamic Symbols: ";
        for (size_t i = 0; i < g.symbol_table.size(); ++i) {
            std::cout << g.symbol_table[i] << (i + 1 < g.symbol_table.size() ? ", " : "");
        }
        std::cout << "\n";
    }

    std::cout << "\n Capabilities:\n"
              << "   [" << (g.has_tokenizer() ? "x" : " ") << "] Tokenizer:      "
              << (g.has_tokenizer() ? "Present in GGUF metadata" : "Not present") << "\n"
              << "   [" << (g.has_chat_template() ? "x" : " ") << "] Chat Template:  "
              << (g.has_chat_template() ? "Present in GGUF metadata" : "Not present") << "\n"
              << "   [" << (g.has_vision() ? "x" : " ") << "] Vision Preproc: "
              << (g.has_vision() ? "Present in GGUF metadata" : "Not present") << "\n";

    std::cout << "\n Graph Inputs (" << g.inputs.size() << "):\n";
    for (uint32_t tid : g.inputs) {
        auto t_it = g.tensors.find(tid);
        if (t_it != g.tensors.end()) {
            std::cout << "   #" << tid << ": " << t_it->second.name << " (type: "
                      << static_cast<int>(t_it->second.type) << ")\n";
        }
    }

    std::cout << "\n Graph Outputs (" << g.outputs.size() << "):\n";
    for (uint32_t tid : g.outputs) {
        auto t_it = g.tensors.find(tid);
        if (t_it != g.tensors.end()) {
            std::cout << "   #" << tid << ": " << t_it->second.name << " (type: "
                      << static_cast<int>(t_it->second.type) << ")\n";
        }
    }
    std::cout << "================================================================================\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    std::string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h") {
        print_help(argv[0]);
        return 0;
    }

    std::string model_path = first_arg;
    std::unordered_map<std::string, std::string> input_files;
    std::unordered_map<std::string, std::string> image_files;
    std::unordered_map<std::string, std::string> text_inputs;
    std::unordered_map<uint32_t, std::string> output_files;
    std::unordered_map<std::string, std::string> state_in_files;
    std::unordered_map<std::string, std::string> state_out_files;
    std::unordered_map<std::string, int64_t> symbol_env;

    std::string prompt_text;
    std::string chat_text;
    std::string system_text;
    bool is_generate = false;
    bool show_info = false;
    bool echo_prompt = false;
    bool show_special = false;
    int max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 0.9f;

    std::string device_name = "cpu";
    int n_threads = 1;
    bool unplanned = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--info") {
            show_info = true;
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt_text = argv[++i];
            is_generate = true;
        } else if (arg == "--chat" && i + 1 < argc) {
            chat_text = argv[++i];
            is_generate = true;
        } else if (arg == "--system" && i + 1 < argc) {
            system_text = argv[++i];
        } else if (arg == "--generate") {
            is_generate = true;
        } else if (arg == "--echo-prompt") {
            echo_prompt = true;
        } else if (arg == "--show-special") {
            show_special = true;
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--input" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                input_files[val.substr(0, colon)] = val.substr(colon + 1);
            }
        } else if (arg == "--image" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                image_files[val.substr(0, colon)] = val.substr(colon + 1);
            }
        } else if (arg == "--text" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                text_inputs[val.substr(0, colon)] = val.substr(colon + 1);
            }
        } else if (arg == "--device" && i + 1 < argc) {
            device_name = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                uint32_t tid = std::stoul(val.substr(0, colon));
                output_files[tid] = val.substr(colon + 1);
            }
        } else if (arg == "--state-in" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                state_in_files[val.substr(0, colon)] = val.substr(colon + 1);
            }
        } else if (arg == "--state-out" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t colon = val.find(':');
            if (colon != std::string::npos) {
                state_out_files[val.substr(0, colon)] = val.substr(colon + 1);
            }
        } else if (arg == "--symbol" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t eq = val.find('=');
            if (eq != std::string::npos) {
                symbol_env[val.substr(0, eq)] = std::stoll(val.substr(eq + 1));
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (arg == "--unplanned") {
            unplanned = true;
        }
    }

    try {
        auto model_graph = ggmlc::ModelLoader::load_from_file(model_path);

        if (show_info) {
            print_model_info(model_path, model_graph);
            return 0;
        }

        // Validate argument compatibility against model capabilities
        if (!image_files.empty() && !model_graph.has_vision()) {
            std::cerr << "[ggmlc-run ERROR] Model '" << model_graph.name
                      << "' does not specify an image preprocessing pipeline in its GGUF metadata.\n"
                      << "  Cannot use '--image'. If this model requires raw image tensors, provide them via '--input <name:file.bin>'.\n";
            return 1;
        }

        if (!text_inputs.empty() && !model_graph.has_tokenizer()) {
            std::cerr << "[ggmlc-run ERROR] Model '" << model_graph.name
                      << "' does not specify tokenizer metadata in its GGUF container.\n"
                      << "  Cannot use '--text'. Provide token IDs via '--input <name:file.bin>'.\n";
            return 1;
        }

        if ((!prompt_text.empty() || is_generate) && chat_text.empty() && !model_graph.has_tokenizer()) {
            std::cerr << "[ggmlc-run ERROR] Model '" << model_graph.name
                      << "' does not contain tokenizer metadata for text generation.\n"
                      << "  Cannot use '--prompt' or '--generate'.\n";
            return 1;
        }

        if (!chat_text.empty()) {
            if (!model_graph.has_tokenizer()) {
                std::cerr << "[ggmlc-run ERROR] Model '" << model_graph.name
                          << "' does not contain tokenizer metadata. Cannot use '--chat'.\n";
                return 1;
            }
            if (!model_graph.has_chat_template()) {
                std::cerr << "[ggmlc-run NOTICE] Model '" << model_graph.name
                          << "' does not define a chat template in its GGUF metadata. Using standard prompt format.\n";
            }
        }

        std::cout << "[ggmlc-run] Loaded model '" << model_graph.name << "' with "
                  << model_graph.tensors.size() << " tensors and "
                  << model_graph.ops.size() << " operations"
                  << (unplanned ? " (UNPLANNED / NO REUSE)" : " (PLANNED ARENA REUSE)") << ".\n";

        // Initialize tokenizer from model GGUF file if available
        ggmlc::pipeline::BPETokenizer tokenizer;
        bool has_tokenizer = tokenizer.init_from_gguf_file(model_path);

        // ====================================================================
        // Mode A: Autoregressive Text Generation
        // ====================================================================
        if (is_generate || !chat_text.empty() || !prompt_text.empty()) {
            if (!has_tokenizer) {
                std::cerr << "[ggmlc-run ERROR] Failed to initialize tokenizer from GGUF metadata.\n";
                return 1;
            }

            if (model_graph.inputs.empty() || model_graph.outputs.empty()) {
                std::cerr << "[ggmlc-run ERROR] Model graph has no input or output tensors.\n";
                return 1;
            }

            uint32_t in_tid = model_graph.inputs[0];
            uint32_t out_tid = model_graph.outputs[0];

            std::string formatted_prompt;
            bool is_chat_mode = !chat_text.empty();
            if (is_chat_mode) {
                formatted_prompt = tokenizer.apply_chat_template(chat_text, system_text, true);
            } else if (!prompt_text.empty()) {
                formatted_prompt = prompt_text;
            } else {
                formatted_prompt = "Hello";
            }

            std::vector<int32_t> current_tokens = tokenizer.encode(formatted_prompt, 0, false, false);
            if (current_tokens.empty()) {
                current_tokens.push_back(tokenizer.bos_token_id() >= 0 ? tokenizer.bos_token_id() : 0);
            }

            std::cout << "[ggmlc-run] Prompt tokens: " << current_tokens.size() << "\n";
            std::cout << "[ggmlc-run] Generating (" << max_tokens << " max tokens, temp="
                      << temperature << ", device=" << device_name << "):\n\n";

            // If prompt echo was explicitly requested, or if in raw prompt mode, print prompt
            if (echo_prompt) {
                std::cout << formatted_prompt << std::flush;
            } else if (!is_chat_mode) {
                std::cout << formatted_prompt << std::flush;
            }

            ggmlc::ModelExecutor executor(model_graph, device_name);
            auto t_start = std::chrono::high_resolution_clock::now();
            int generated_count = 0;

            for (int step = 0; step < max_tokens; ++step) {
                int64_t S = static_cast<int64_t>(current_tokens.size());
                symbol_env["s"] = S;

                // Auto-deduce any symbolic dimensions in input tensor
                for (const auto& dim_expr : model_graph.tensors[in_tid].ne) {
                    if (dim_expr && dim_expr->type == ggmlc::DimType::SYMBOL) {
                        int64_t sym_idx = dim_expr->val;
                        if (sym_idx >= 0 && sym_idx < static_cast<int64_t>(model_graph.symbol_table.size())) {
                            symbol_env[model_graph.symbol_table[sym_idx]] = S;
                        }
                    }
                }

                executor.prepare(symbol_env, !unplanned);
                executor.set_input(in_tid, current_tokens.data(), current_tokens.size() * sizeof(int32_t));
                executor.run(n_threads);

                const float* logits_data = static_cast<const float*>(executor.get_output_data(out_tid));
                size_t total_elements = executor.get_tensor_size_bytes(out_tid) / sizeof(float);
                int64_t vocab_size = total_elements / S;
                if (vocab_size <= 0) {
                    vocab_size = static_cast<int64_t>(tokenizer.vocab_size());
                }

                const float* last_logits = logits_data + (S - 1) * vocab_size;

                int32_t next_token = 0;
                if (temperature <= 0.0f) {
                    // Greedy argmax
                    float max_val = -1e30f;
                    for (int64_t v = 0; v < vocab_size; ++v) {
                        if (last_logits[v] > max_val) {
                            max_val = last_logits[v];
                            next_token = static_cast<int32_t>(v);
                        }
                    }
                } else {
                    // Temperature + Top-P sampling
                    std::vector<std::pair<float, int32_t>> probs(vocab_size);
                    float max_l = -1e30f;
                    for (int64_t v = 0; v < vocab_size; ++v) {
                        if (last_logits[v] > max_l) max_l = last_logits[v];
                    }
                    float sum_exp = 0.0f;
                    for (int64_t v = 0; v < vocab_size; ++v) {
                        float p = std::exp((last_logits[v] - max_l) / std::max(temperature, 1e-5f));
                        probs[v] = {p, static_cast<int32_t>(v)};
                        sum_exp += p;
                    }
                    for (auto& pair : probs) {
                        pair.first /= sum_exp;
                    }

                    if (top_p < 1.0f) {
                        std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                        float cumsum = 0.0f;
                        size_t cutoff = 1;
                        for (size_t k = 0; k < probs.size(); ++k) {
                            cumsum += probs[k].first;
                            if (cumsum > top_p && k > 0) {
                                cutoff = k + 1;
                                break;
                            }
                        }
                        probs.resize(cutoff);
                        float new_sum = 0.0f;
                        for (const auto& p : probs) new_sum += p.first;
                        for (auto& p : probs) p.first /= new_sum;
                    }

                    float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                    float acc = 0.0f;
                    next_token = probs[0].second;
                    for (const auto& p : probs) {
                        acc += p.first;
                        if (r <= acc) {
                            next_token = p.second;
                            break;
                        }
                    }
                }

                current_tokens.push_back(next_token);
                generated_count++;

                // Stop immediately on EOS
                if (tokenizer.eos_token_id() >= 0 && next_token == tokenizer.eos_token_id()) {
                    break;
                }

                // If special token generated and show_special is not enabled, stop or skip
                if (tokenizer.is_special_token(next_token)) {
                    if (show_special) {
                        std::cout << tokenizer.decode({next_token}, false) << std::flush;
                    }
                    break;
                }

                std::string piece = tokenizer.decode_token(next_token, true);
                std::cout << piece << std::flush;
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "\n\n[ggmlc-run] Generated " << generated_count << " tokens in "
                      << std::fixed << std::setprecision(2) << elapsed_sec << "s ("
                      << (generated_count / std::max(elapsed_sec, 1e-6)) << " tok/s)\n";

            return 0;
        }

        // ====================================================================
        // Mode B: Standard One-Shot Graph Execution
        // ====================================================================
        ggmlc::ModelExecutor executor(model_graph, device_name);
        executor.prepare(symbol_env, !unplanned);

        // Load initial state data if provided
        for (const auto& pair : state_in_files) {
            std::ifstream fin(pair.second, std::ios::binary | std::ios::ate);
            if (!fin.is_open()) {
                std::cerr << "Failed to open state-in file: " << pair.second << "\n";
                return 1;
            }
            size_t sz = fin.tellg();
            fin.seekg(0, std::ios::beg);
            std::vector<uint8_t> buf(sz);
            fin.read(reinterpret_cast<char*>(buf.data()), sz);

            executor.set_state_by_name(pair.first, buf.data(), sz);
            std::cout << "[ggmlc-run] Loaded state '" << pair.first << "' (" << sz << " bytes)\n";
        }

        // Load input data
        for (const auto& pair : input_files) {
            std::ifstream fin(pair.second, std::ios::binary | std::ios::ate);
            if (!fin.is_open()) {
                std::cerr << "Failed to open input file: " << pair.second << "\n";
                return 1;
            }
            size_t sz = fin.tellg();
            fin.seekg(0, std::ios::beg);
            std::vector<uint8_t> buf(sz);
            fin.read(reinterpret_cast<char*>(buf.data()), sz);

            executor.set_input_by_name(pair.first, buf.data(), sz);
            std::cout << "[ggmlc-run] Loaded input '" << pair.first << "' (" << sz << " bytes)\n";
        }

        // Load and preprocess image inputs
        for (const auto& pair : image_files) {
            int target_w = 224;
            int target_h = 224;
            auto it_size = model_graph.metadata_int.find("clip.vision.image_size");
            if (it_size != model_graph.metadata_int.end() && it_size->second > 0) {
                target_w = static_cast<int>(it_size->second);
                target_h = target_w;
            }
            auto img_tensor = ggmlc::pipeline::ImagePreprocessor::preprocess_file(pair.second, target_w, target_h);
            executor.set_input_by_name(pair.first, img_tensor.data.data(), img_tensor.data.size() * sizeof(float));
            std::cout << "[ggmlc-run] Preprocessed image '" << pair.first << "' from " << pair.second
                      << " (" << img_tensor.channels << "x" << img_tensor.height << "x" << img_tensor.width << ")\n";
        }

        // Tokenize and load text inputs
        for (const auto& pair : text_inputs) {
            auto tokens = tokenizer.encode(pair.second, 0, true, false);
            executor.set_input_by_name(pair.first, tokens.data(), tokens.size() * sizeof(int32_t));
            std::cout << "[ggmlc-run] Tokenized text '" << pair.first << "' (" << tokens.size() << " tokens)\n";
        }

        // Execute
        executor.run(n_threads);
        std::cout << "[ggmlc-run] Execution completed successfully.\n";

        // Save output data to files if specified
        for (const auto& pair : output_files) {
            uint32_t tid = pair.first;
            const void* data = executor.get_output_data(tid);
            size_t sz = executor.get_tensor_size_bytes(tid);

            std::ofstream fout(pair.second, std::ios::binary);
            if (!fout.is_open()) {
                std::cerr << "Failed to write output file: " << pair.second << "\n";
                return 1;
            }
            fout.write(reinterpret_cast<const char*>(data), sz);
            std::cout << "[ggmlc-run] Saved output tensor " << tid << " (" << sz << " bytes) to " << pair.second << "\n";
        }

        // Save state-out data
        for (const auto& pair : state_out_files) {
            const void* data = executor.get_state_data_by_name(pair.first);
            size_t sz = 0;
            for (const auto& t_pair : model_graph.tensors) {
                if (t_pair.second.name == pair.first) {
                    sz = executor.get_tensor_size_bytes(t_pair.first);
                    break;
                }
            }
            if (sz > 0) {
                std::ofstream fout(pair.second, std::ios::binary);
                if (!fout.is_open()) {
                    std::cerr << "Failed to write state-out file: " << pair.second << "\n";
                    return 1;
                }
                fout.write(reinterpret_cast<const char*>(data), sz);
                std::cout << "[ggmlc-run] Saved state '" << pair.first << "' (" << sz << " bytes) to " << pair.second << "\n";
            }
        }

        // Explicit Task-Aware Output Formatting (when no explicit output file requested)
        if (output_files.empty() && !model_graph.outputs.empty()) {
            uint32_t primary_out_tid = model_graph.outputs[0];
            const float* out_f32 = static_cast<const float*>(executor.get_output_data(primary_out_tid));
            size_t n_elem = executor.get_tensor_size_bytes(primary_out_tid) / sizeof(float);

            if (model_graph.has_task("classification") || model_graph.has_task("image-classification")) {
                // Compute Softmax and display Top-5
                std::vector<std::pair<float, int>> ranked(n_elem);
                float max_logit = -1e30f;
                for (size_t i = 0; i < n_elem; ++i) {
                    if (out_f32[i] > max_logit) max_logit = out_f32[i];
                }
                float sum_exp = 0.0f;
                for (size_t i = 0; i < n_elem; ++i) {
                    float exp_val = std::exp(out_f32[i] - max_logit);
                    ranked[i] = {exp_val, static_cast<int>(i)};
                    sum_exp += exp_val;
                }
                for (auto& p : ranked) p.first /= sum_exp;

                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

                std::cout << "\n[ggmlc-run] Classification Results (Top 5):\n";
                for (size_t k = 0; k < std::min<size_t>(5, ranked.size()); ++k) {
                    std::cout << "  #" << (k + 1) << ": Class " << std::setw(5) << ranked[k].second
                              << "  - " << std::fixed << std::setprecision(2) << (ranked[k].first * 100.0f) << "%\n";
                }
            } else if (model_graph.has_task("embedding") || model_graph.has_task("text-embedding")) {
                // Compute L2 norm
                double sum_sq = 0.0;
                for (size_t i = 0; i < n_elem; ++i) {
                    sum_sq += static_cast<double>(out_f32[i]) * out_f32[i];
                }
                double l2_norm = std::sqrt(sum_sq);
                std::cout << "\n[ggmlc-run] Embedding Vector (dim=" << n_elem
                          << ", L2 norm=" << std::fixed << std::setprecision(4) << l2_norm << "):\n  [";
                for (size_t i = 0; i < std::min<size_t>(4, n_elem); ++i) {
                    std::cout << std::setprecision(4) << out_f32[i] << ", ";
                }
                std::cout << "... (" << n_elem << " float32 elements)]\n";
            } else if (model_graph.has_task("similarity")) {
                float logit_val = out_f32[0];
                std::cout << "\n[ggmlc-run] Similarity Logit: " << std::fixed << std::setprecision(4)
                          << logit_val << "\n";
            } else {
                for (uint32_t out_id : model_graph.outputs) {
                    size_t sz = executor.get_tensor_size_bytes(out_id);
                    std::cout << "[ggmlc-run] Graph output tensor " << out_id << ": " << sz << " bytes computed.\n";
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ggmlc-run ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
