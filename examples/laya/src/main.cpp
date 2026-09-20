#include "engine.h"
#include "presets.h"
#include "server.h"
#include "questions.h"
#include "json_util.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using laya::JsonValue;
using laya::JsonParser;

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void print_help(const char* prog) {
    std::cout
        << "====================================================================\n"
        << " Laya - System 1 Decision Engine (ggmlc)\n"
        << " Non-autoregressive typed decisions: choice / score / noul\n"
        << "====================================================================\n\n"
        << "USAGE:\n"
        << "  " << prog << " [model.gguf] [options]\n\n"
        << "INSPECTION & SERVING:\n"
        << "  -h, --help             Show this help menu and exit\n"
        << "  --info                 Inspect GGUF metadata, special tokens, temperatures\n"
        << "  --list-presets         List built-in decision workflows\n"
        << "  --serve                Start Web Studio and REST API\n"
        << "  --port <P>             HTTP port (default: 8080)\n"
        << "  --daemon               Newline JSON-RPC on stdin/stdout (agents / IDEs)\n\n"
        << "DECISION INPUT:\n"
        << "  --preset <name>        Built-in workflow (email, triage, guard, ...)\n"
        << "  --state <json|text>    Observation / ticket / prompt JSON or raw text\n"
        << "  --state-file <path>    Load state from a file\n"
        << "  --text <str>           Write into the preset's primary field (body/prompt/...)\n"
        << "  --questions <json>     Question map (Laya/Jev schema)\n"
        << "  --questions-file <p>   Load questions JSON from a file\n"
        << "  --json                 Print machine JSON instead of the CLI bars\n"
        << "  --bench                Latency benchmark on the selected preset\n"
        << "  --runs <N>             Benchmark runs (default: 5)\n"
        << "  --warmup <N>           Benchmark warmup (default: 2)\n\n"
        << "HARDWARE:\n"
        << "  --device <cpu|cuda>    Execution device (default: cpu)\n"
        << "  --threads <N>          CPU worker threads (default: 4)\n"
        << "  --cuda-graph           Capture a CUDA graph for the active (B,S) shape\n"
        << "  --max-batch <N>        Cap question batch (default: GGUF laya.max_batch, 8)\n"
        << "  --model <path>         GGUF path (alternative to the positional argument)\n\n"
        << "EXAMPLES:\n"
        << "  " << prog << " scratch/laya_english_f16.gguf --preset email\n"
        << "  " << prog << " scratch/laya_english_f16.gguf --preset guard --text \"Ignore previous instructions\"\n"
        << "  " << prog << " scratch/laya_english_f16.gguf --preset email --device cuda --cuda-graph --bench\n"
        << "  " << prog << " scratch/laya_english_f16.gguf --daemon --device cuda --cuda-graph\n"
        << "  " << prog << " scratch/laya_english_f16.gguf --serve --port 8080 --device cuda\n"
        << std::endl;
}

static JsonValue parse_state_arg(const std::string& s) {
    if (s.empty()) return JsonValue::object();
    try {
        return JsonParser::parse_string(s);
    } catch (...) {
        return JsonValue::string(s);
    }
}

static int run_daemon(laya::DecisionEngine& engine) {
    std::cout << "{\"status\":\"ready\",\"model\":\"laya\"}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        JsonValue req;
        try {
            req = JsonParser::parse_string(line);
        } catch (const std::exception& e) {
            std::cout << "{\"error\":\"" << laya::json_escape(e.what()) << "\"}" << std::endl;
            continue;
        }
        JsonValue id = JsonValue::string("0");
        if (const auto* v = req.get("id")) id = *v;

        JsonValue state = JsonValue::object();
        if (const auto* s = req.get("state")) state = *s;
        std::vector<laya::Question> qs;
        if (const auto* q = req.get("questions")) qs = laya::questions_from_json(*q);

        const laya::Preset* pr = nullptr;
        if (const auto* pn = req.get("preset")) {
            if (pn->is_string()) pr = laya::find_preset(pn->s);
            if (pr) {
                if (qs.empty()) qs = pr->questions;
                if (state.is_object() && state.obj.empty()) state = pr->default_state;
            }
        }
        if (const auto* t = req.get("text")) {
            if (t->is_string()) state = laya::apply_text_to_state(pr, state, t->s);
        }
        if (qs.empty()) {
            std::cout << "{\"id\":" << laya::json_dumps(id)
                      << ",\"error\":\"missing questions\"}" << std::endl;
            continue;
        }
        auto result = engine.decide(state, qs);
        JsonValue out = laya::JsonParser::parse_string(laya::format_answer_json(result, false));
        out.set("id", id);
        std::cout << laya::json_dumps(out) << std::endl;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help("laya.exe");
        return 1;
    }

    std::string model_path;
    std::string device = "cpu";
    int n_threads = 4;
    int port = 8080;
    int runs = 5;
    int warmup = 2;
    bool cuda_graph = false;
    bool info = false;
    bool serve = false;
    bool daemon = false;
    bool list_presets = false;
    bool as_json = false;
    bool bench = false;
    int max_batch = 0;
    std::string preset_name;
    std::string state_arg;
    std::string state_file;
    std::string text_arg;
    std::string questions_arg;
    std::string questions_file;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (a == "--model") {
            model_path = need("--model");
        } else if (a == "--device") {
            device = need("--device");
        } else if (a == "--threads") {
            n_threads = std::atoi(need("--threads"));
        } else if (a == "--port") {
            port = std::atoi(need("--port"));
        } else if (a == "--preset") {
            preset_name = need("--preset");
        } else if (a == "--state") {
            state_arg = need("--state");
        } else if (a == "--state-file") {
            state_file = need("--state-file");
        } else if (a == "--text") {
            text_arg = need("--text");
        } else if (a == "--questions") {
            questions_arg = need("--questions");
        } else if (a == "--questions-file") {
            questions_file = need("--questions-file");
        } else if (a == "--runs") {
            runs = std::atoi(need("--runs"));
        } else if (a == "--warmup") {
            warmup = std::atoi(need("--warmup"));
        } else if (a == "--cuda-graph") {
            cuda_graph = true;
        } else if (a == "--max-batch") {
            max_batch = std::atoi(need("--max-batch"));
        } else if (a == "--info") {
            info = true;
        } else if (a == "--serve") {
            serve = true;
        } else if (a == "--daemon") {
            daemon = true;
        } else if (a == "--list-presets") {
            list_presets = true;
        } else if (a == "--json") {
            as_json = true;
        } else if (a == "--bench") {
            bench = true;
        } else if (!a.empty() && a[0] != '-' && model_path.empty()) {
            model_path = a;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            print_help(argv[0]);
            return 1;
        }
    }

    if (list_presets) {
        std::cout << "Built-in presets:\n";
        for (const auto& p : laya::all_presets()) {
            std::cout << "  " << p.name << "  -  " << p.title << "\n      " << p.blurb << "\n";
        }
        return 0;
    }

    if (model_path.empty()) {
        std::cerr << "Error: model.gguf is required.\n";
        print_help(argv[0]);
        return 1;
    }

    laya::EngineOptions opt;
    opt.device = device;
    opt.n_threads = n_threads;
    opt.cuda_graph = cuda_graph;
    opt.max_batch = max_batch;

    laya::DecisionEngine engine;
    if (!engine.load_model(model_path, opt)) return 1;

    if (info) {
        engine.print_info();
        return 0;
    }
    if (serve) {
        laya::Server server(engine, port);
        return server.start() ? 0 : 1;
    }
    if (daemon) return run_daemon(engine);

    const laya::Preset* pr = nullptr;
    if (!preset_name.empty()) {
        pr = laya::find_preset(preset_name);
        if (!pr) {
            std::cerr << "unknown preset: " << preset_name << "\n";
            return 1;
        }
    }

    laya::JsonValue state = laya::JsonValue::object();
    std::vector<laya::Question> qs;
    if (pr) {
        state = pr->default_state;
        qs = pr->questions;
    }
    try {
        if (!state_file.empty()) state = laya::JsonParser::parse_string(read_file(state_file));
        else if (!state_arg.empty()) state = parse_state_arg(state_arg);
        if (!questions_file.empty()) qs = laya::questions_from_json(laya::JsonParser::parse_string(read_file(questions_file)));
        else if (!questions_arg.empty()) qs = laya::questions_from_json(laya::JsonParser::parse_string(questions_arg));
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    if (!text_arg.empty()) state = laya::apply_text_to_state(pr, state, text_arg);

    if (qs.empty()) {
        pr = laya::find_preset("email");
        state = pr->default_state;
        qs = pr->questions;
        std::cerr << "[laya] no questions given; using --preset email\n";
    }

    if (bench) {
        engine.benchmark(state, qs, runs, warmup);
        return 0;
    }

    auto result = engine.decide(state, qs);
    if (as_json) std::cout << laya::format_answer_json(result, true) << std::endl;
    else std::cout << laya::format_answer_cli(result) << std::endl;
    return 0;
}
