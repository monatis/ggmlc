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
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "ggmlc/pipeline/image.h"
#include "ggmlc/pipeline/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ggmlc-run <model.gguf> [options]\n"
                  << "Options:\n"
                  << "  --prompt <string>           Autoregressive text generation with raw prompt\n"
                  << "  --chat <string>             Autoregressive chat generation with chat template\n"
                  << "  --system <string>           System prompt for chat template\n"
                  << "  --generate                  Enable autoregressive text generation\n"
                  << "  --max-tokens <N>            Maximum new tokens to generate (default: 32)\n"
                  << "  --temperature <T>           Sampling temperature (0.0 = greedy argmax, default: 0.0)\n"
                  << "  --top-p <P>                 Nucleus sampling probability (default: 0.9)\n"
                  << "  --input <name:file.bin>     Set input tensor from binary file\n"
                  << "  --image <name:file.jpg>     Preprocess and set image input tensor (bicubic + normalize)\n"
                  << "  --text <name:string>        Tokenize and set text input tensor (BPE)\n"
                  << "  --output <id:file.bin>      Save output tensor ID to binary file\n"
                  << "  --state-in <name:file.bin>  Load initial state tensor from binary file\n"
                  << "  --state-out <name:file.bin> Save final state tensor to binary file\n"
                  << "  --symbol <key=value>        Bind dynamic symbol\n"
                  << "  --device <cpu|cuda>         Device to execute on (default: cpu)\n"
                  << "  --threads <N>               Number of threads (default: 1)\n"
                  << "  --unplanned                 Disable memory arena reuse planning\n";
        return 1;
    }

    std::string model_path = argv[1];
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
    int max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 0.9f;

    std::string device_name = "cpu";
    int n_threads = 1;
    bool unplanned = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt_text = argv[++i];
            is_generate = true;
        } else if (arg == "--chat" && i + 1 < argc) {
            chat_text = argv[++i];
            is_generate = true;
        } else if (arg == "--system" && i + 1 < argc) {
            system_text = argv[++i];
        } else if (arg == "--generate") {
            is_generate = true;
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
        std::cout << "[ggmlc-run] Loaded model '" << model_graph.name << "' with "
                  << model_graph.tensors.size() << " tensors and "
                  << model_graph.ops.size() << " operations"
                  << (unplanned ? " (UNPLANNED / NO REUSE)" : " (PLANNED ARENA REUSE)") << ".\n";

        // Initialize tokenizer from model GGUF file
        ggmlc::pipeline::BPETokenizer tokenizer;
        bool has_tokenizer = tokenizer.init_from_gguf_file(model_path);

        // ====================================================================
        // Mode A: Autoregressive Text Generation
        // ====================================================================
        if (is_generate || !chat_text.empty() || !prompt_text.empty()) {
            if (!has_tokenizer) {
                std::cerr << "[ggmlc-run ERROR] Model does not contain GGUF tokenizer metadata for text generation.\n";
                return 1;
            }

            if (model_graph.inputs.empty() || model_graph.outputs.empty()) {
                std::cerr << "[ggmlc-run ERROR] Model graph has no input or output tensors.\n";
                return 1;
            }

            uint32_t in_tid = model_graph.inputs[0];
            uint32_t out_tid = model_graph.outputs[0];

            std::string formatted_prompt;
            if (!chat_text.empty()) {
                formatted_prompt = tokenizer.apply_chat_template(chat_text, system_text, true);
            } else if (!prompt_text.empty()) {
                formatted_prompt = prompt_text;
            } else {
                formatted_prompt = "Hello";
            }

            std::vector<int32_t> current_tokens = tokenizer.encode(formatted_prompt, 0, true, false);
            if (current_tokens.empty()) {
                current_tokens.push_back(tokenizer.bos_token_id() >= 0 ? tokenizer.bos_token_id() : 0);
            }

            std::cout << "[ggmlc-run] Prompt tokens: " << current_tokens.size() << "\n";
            std::cout << "[ggmlc-run] Generating (" << max_tokens << " max tokens, temp=" << temperature << ", device=" << device_name << "):\n\n";
            std::cout << formatted_prompt << std::flush;

            ggmlc::ModelExecutor executor(model_graph, device_name);
            auto t_start = std::chrono::high_resolution_clock::now();
            int generated_count = 0;

            for (int step = 0; step < max_tokens; ++step) {
                int64_t S = static_cast<int64_t>(current_tokens.size());
                symbol_env["s"] = S;

                // Auto-deduce any symbolic dimensions
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

                if (tokenizer.eos_token_id() >= 0 && next_token == tokenizer.eos_token_id()) {
                    break;
                }

                std::string piece = tokenizer.decode_token(next_token, false);
                std::cout << piece << std::flush;
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
            std::cout << "\n\n[ggmlc-run] Generated " << generated_count << " tokens in "
                      << elapsed_sec << "s (" << (generated_count / std::max(elapsed_sec, 1e-6)) << " tok/s)\n";

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
            auto img_tensor = ggmlc::pipeline::ImagePreprocessor::preprocess_file(pair.second, 224, 224);
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

        // Save output data
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

        // If no explicit output files, dump primary graph outputs
        if (output_files.empty() && !model_graph.outputs.empty()) {
            for (uint32_t out_id : model_graph.outputs) {
                size_t sz = executor.get_tensor_size_bytes(out_id);
                std::cout << "[ggmlc-run] Graph output tensor " << out_id << ": " << sz << " bytes computed.\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ggmlc-run ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
