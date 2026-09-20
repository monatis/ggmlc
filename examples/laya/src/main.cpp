#include "engine.h"
#include "presets.h"
#include "server.h"
#include "questions.h"
#include "json_util.h"
#include "router.h"
#include "language.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")
#endif

using laya::JsonValue;
using laya::JsonParser;

#ifdef _WIN32
static std::vector<std::string> utf8_argv() {
    int n = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> out;
    out.reserve(n > 0 ? static_cast<size_t>(n) : 0);
    for (int i = 0; i < n; ++i) {
        int sz = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (sz <= 1) {
            out.emplace_back();
            continue;
        }
        std::string s(static_cast<size_t>(sz - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), sz, nullptr, nullptr);
        out.push_back(std::move(s));
    }
    LocalFree(wargv);
    return out;
}
#endif

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void print_help(const char* argv0) {
    const std::string prog = argv0 ? argv0 : "laya";
    std::cout
        << "Laya — System 1 typed decisions (choice / score / noul) compiled with ggmlc.\n"
        << "One encoder pass. No generated tokens.\n\n"
        << "USAGE\n"
        << "  " << prog << " --model <file.gguf> [command] [options]\n"
        << "  " << prog << " <file.gguf> [command] [options]\n"
        << "  " << prog << " --models-dir <dir> [command] [options]\n\n"
        << "COMMANDS  (pick one; default is a single decide)\n"
        << "  --help              Show this help and exit\n"
        << "  --list-presets      List built-in workflows (no model required)\n"
        << "  --detect-lang       Print language / family routing for --state/--text (no model)\n"
        << "  --info              Print GGUF metadata, tokenizer, temperatures\n"
        << "  --serve             Web Studio + POST /api/decide  (see --port)\n"
        << "  --daemon            Newline JSON-RPC on stdin/stdout\n"
        << "  --bench             Latency / throughput on the selected input\n"
        << "  (none)              Run one decision and print CLI bars (or --json)\n\n"
        << "MODEL\n"
        << "  --model <path>            Single GGUF (also accepted as the first positional arg)\n"
        << "  --models-dir <dir>        Directory of Laya GGUFs; routes english vs multilingual\n"
        << "  --family <name>           auto | english | multilingual | typed-decisions\n"
        << "                            default: auto  (script + English-word routing)\n\n"
        << "INPUT  (optional; default --preset email)\n"
        << "  --preset <name>           email | triage | guard | moderation | router |\n"
        << "                            expense | security | invoice | customer_service | harness\n"
        << "  --state <json|text>       Observation object or a raw string\n"
        << "  --state-file <path>       Read --state from a file\n"
        << "  --text <str>              Write into the preset's primary field (body/prompt/…)\n"
        << "  --questions <json>        Laya/Jev question map\n"
        << "  --questions-file <path>   Read --questions from a file\n"
        << "  --json                    Machine JSON instead of CLI bars\n\n"
        << "HARDWARE\n"
        << "  --device <name>           auto | cpu | cuda | cuda:0 | metal     default: auto\n"
        << "                            auto = CUDA or Metal if the binary was built with it\n"
        << "                            and a device is present, else CPU\n"
        << "  --threads <N>             CPU workers                               default: 4\n"
        << "  --cuda-graph              Capture a CUDA graph for the live (B, S) shape\n"
        << "  --max-batch <N>           Cap questions per forward                 default: from GGUF (8)\n"
        << "  --port <P>                HTTP port for --serve                     default: 8080\n"
        << "  --runs <N>                --bench timed runs                        default: 5\n"
        << "  --warmup <N>              --bench warmup runs                       default: 2\n\n"
        << "EXAMPLES\n"
        << "  " << prog << " --list-presets\n"
        << "  " << prog << " --detect-lang --text \"I was charged twice\"\n"
        << "  " << prog << " --model scratch/laya_english_f16.gguf --preset email --device auto --cuda-graph\n"
        << "  " << prog << " --model scratch/laya_english_q8_0.gguf --preset guard --text \"Ignore previous instructions\" --json\n"
        << "  " << prog << " --models-dir scratch/laya-ggufs --preset email --text \"二重に請求されました\"\n"
        << "  " << prog << " --models-dir scratch/laya-ggufs --family multilingual --serve --port 8080\n"
        << "  " << prog << " --model scratch/laya_english_f16.gguf --daemon --device auto --cuda-graph\n"
        << "  " << prog << " --model scratch/laya_english_f16.gguf --preset email --bench --warmup 5 --runs 7\n\n"
        << "DOWNLOADS\n"
        << "  GGUF weights  https://huggingface.co/mys/laya-GGUF\n"
        << "                https://huggingface.co/mys/laya-multilingual-GGUF\n"
        << "                https://huggingface.co/mys/laya-typed-decisions-GGUF\n"
        << "  Binaries      https://github.com/monatis/ggmlc/releases/latest\n"
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

static int run_daemon(laya::DecisionRouter& router) {
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
        const bool had_state = req.get("state") != nullptr;
        if (const auto* s = req.get("state")) state = *s;
        std::vector<laya::Question> qs;
        if (const auto* q = req.get("questions")) qs = laya::questions_from_json(*q);

        const laya::Preset* pr = nullptr;
        if (const auto* pn = req.get("preset")) {
            if (pn->is_string()) pr = laya::find_preset(pn->s);
        }
        if (pr) {
            if (qs.empty()) qs = pr->questions;
            if (!had_state && req.get("text") == nullptr) state = pr->default_state;
        }
        if (const auto* t = req.get("text")) {
            if (t->is_string()) {
                JsonValue base = had_state ? state : JsonValue::object();
                state = laya::apply_text_to_state(pr, base, t->s);
            }
        }
        if (qs.empty()) {
            std::cout << "{\"id\":" << laya::json_dumps(id)
                      << ",\"error\":\"missing questions\"}" << std::endl;
            continue;
        }
        auto result = router.decide(state, qs);
        JsonValue out = laya::JsonParser::parse_string(laya::format_answer_json(result, false));
        out.set("id", id);
        std::cout << laya::json_dumps(out) << std::endl;
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    auto argv_u8 = utf8_argv();
    std::vector<char*> argv_ptr;
    argv_ptr.reserve(argv_u8.size());
    for (auto& s : argv_u8) argv_ptr.push_back(s.data());
    argc = static_cast<int>(argv_ptr.size());
    argv = argv_ptr.data();
#endif
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    std::string model_path;
    std::string models_dir;
    std::string family = "auto";
    std::string device = "auto";
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
    bool detect_lang = false;
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
        } else if (a == "--models-dir") {
            models_dir = need("--models-dir");
        } else if (a == "--family") {
            family = need("--family");
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
        } else if (a == "--detect-lang") {
            detect_lang = true;
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
    const bool user_state = !state_file.empty() || !state_arg.empty();
    if (pr) {
        qs = pr->questions;
        if (!user_state && text_arg.empty()) state = pr->default_state;
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
    if (!text_arg.empty()) {
        laya::JsonValue base = user_state ? state : laya::JsonValue::object();
        state = laya::apply_text_to_state(pr, base, text_arg);
    }

    if (detect_lang) {
        auto g = laya::guess_language(state);
        std::cout << "script: " << g.script
                  << "\nenglish: " << (g.is_english ? "yes" : "no")
                  << "\nwords: " << g.words
                  << "  english_hits: " << g.english_hits
                  << "  ratio: " << g.english_ratio
                  << "\nroute: " << (g.is_english ? "english" : "multilingual")
                  << "\nreason: " << g.reason << "\n";
        return 0;
    }

    if (model_path.empty() && models_dir.empty()) {
        std::cerr << "Error: pass --model <file.gguf> or --models-dir <dir>.\n";
        print_help(argv[0]);
        return 1;
    }

    laya::EngineOptions opt;
    opt.device = device;
    opt.n_threads = n_threads;
    opt.cuda_graph = cuda_graph;
    opt.max_batch = max_batch;

    laya::DecisionRouter router;
    router.set_options(opt);
    router.set_forced_family(family);
    try {
        if (!models_dir.empty()) {
            if (!router.load_dir(models_dir)) return 1;
        }
        if (!model_path.empty()) {
            if (!router.load_file(model_path)) return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (info) {
        router.print_info();
        return 0;
    }
    if (serve || daemon) {
        try {
            router.pick(laya::JsonValue::string("hello"), {});
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
    if (serve) {
        laya::Server server(router, port);
        return server.start() ? 0 : 1;
    }
    if (daemon) return run_daemon(router);

    if (qs.empty()) {
        pr = laya::find_preset("email");
        state = pr->default_state;
        qs = pr->questions;
        std::cerr << "[laya] no questions given; using --preset email\n";
    }

    if (bench) {
        router.pick(state, qs).benchmark(state, qs, runs, warmup);
        return 0;
    }

    auto result = router.decide(state, qs);
    if (as_json) std::cout << laya::format_answer_json(result, true) << std::endl;
    else std::cout << laya::format_answer_cli(result) << std::endl;
    return 0;
}
