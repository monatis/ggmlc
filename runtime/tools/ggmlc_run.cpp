#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "ggmlc/loader.h"
#include "ggmlc/executor.h"
#include "ggmlc/pipeline/image.h"
#include "ggmlc/pipeline/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ggmlc-run <model.gguf> [options]\n"
                  << "Options:\n"
                  << "  --input <name:file.bin>     Set input tensor from binary file\n"
                  << "  --image <name:file.jpg>     Preprocess and set image input tensor (bicubic + normalize)\n"
                  << "  --text <name:string>        Tokenize and set text input tensor (BPE)\n"
                  << "  --output <id:file.bin>      Save output tensor ID to binary file\n"
                  << "  --state-in <name:file.bin>  Load initial state tensor from binary file\n"
                  << "  --state-out <name:file.bin> Save final state tensor to binary file\n"
                  << "  --symbol <key=value>        Bind dynamic symbol\n"
                  << "  --device <cpu|cuda>         Device to execute on (default: cpu)\n"
                  << "  --threads <N>               Number of threads (default: 1)\n";
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
    std::string device_name = "cpu";
    int n_threads = 1;
    bool unplanned = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
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
            ggmlc::pipeline::BPETokenizer tokenizer;
            auto tokens = tokenizer.encode(pair.second, 77, true, true);
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
            // Find tensor size
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
