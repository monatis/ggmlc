# `laya.cpp` — Standalone System 1 Decision Engine

A zero-dependency C++ implementation of **Laya**, the open reproduction of TypeSafe **Jev** System One models: typed questions (`choice` / `score` / `noul`) scored in one parallel pass. No autoregressive token generation.

The neural trunk is a GGUF compiled with `ggmlc`. Sequence construction, temperatures, Shannon confidence, language routing, JSON-RPC, and the Web Studio live in C++.

Checkpoints (same contract, different encoder / context):

| Family | Hugging Face | Encoder | Params | Context | Use it for |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `english` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English (default demo) |
| `multilingual` | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, ~2× faster |
| `typed-decisions` | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | invoice / SOC / CS / agent-trace specialist |

Official numbers: [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md). The English checkpoint does **not** degrade gracefully off English (it stays confident while collapsing), so routing happens *before* the forward.

---

## Download (recommended)

### GGUF weights

Pre-compiled GGUFs (F16, Q8_0, UD_Q4_K_M) are published under:

- English: [mys/laya-GGUF](https://huggingface.co/mys/laya-GGUF)
- Multilingual: [mys/laya-multilingual-GGUF](https://huggingface.co/mys/laya-multilingual-GGUF)
- Typed-decisions: [mys/laya-typed-decisions-GGUF](https://huggingface.co/mys/laya-typed-decisions-GGUF)

```powershell
# huggingface-cli download mys/laya-GGUF laya_english_f16.gguf --local-dir scratch
```

Put the files you need in one directory if you want `--models-dir` routing.

### Binaries (`laya.exe` / `laya`)

GitHub Release artifacts from the **`latest`** tag: [monatis/ggmlc releases](https://github.com/monatis/ggmlc/releases/latest)

| Artifact | Backend |
| :--- | :--- |
| `laya-macos-arm64-metal.tar.gz` | Apple Silicon Metal |
| `laya-linux-x86_64-cuda-sm80.tar.gz` | Linux CUDA (Ampere A100 / A30 class) |
| `laya-linux-x86_64-cuda-sm86.tar.gz` | Linux CUDA (GA102 / RTX 30-series) |
| `laya-linux-x86_64-cuda-sm89.tar.gz` | Linux CUDA (Ada / RTX 40-series) |
| `laya-windows-x86_64-cuda-sm86.zip` | Windows CUDA sm86 |
| `laya-windows-x86_64-cuda-sm89.zip` | Windows CUDA sm89 (RTX 4050 / 4060 / 4070 / 4090) |

Direct links follow the GitHub `latest` pattern, for example:

- https://github.com/monatis/ggmlc/releases/latest/download/laya-windows-x86_64-cuda-sm89.zip
- https://github.com/monatis/ggmlc/releases/latest/download/laya-macos-arm64-metal.tar.gz
- https://github.com/monatis/ggmlc/releases/latest/download/laya-linux-x86_64-cuda-sm86.tar.gz

Pre-built CUDA binaries cover **sm80, sm86, and sm89** only (the common datacenter / consumer Ampere–Ada set). If you need a binary for other hardware (for example sm75 Turing, sm90 Hopper, or a CPU-only Windows build), open an issue with the exact GPU / CPU and OS and we will consider adding that matrix cell.

---

## Why this exists

Jev (TypeSafe, 2026) is a cloud API: given a *state* and a map of typed questions, it returns calibrated probabilities in ~30 ms instead of waiting for an LLM to emit tokens. Laya is the community checkpoint that implements the same contract locally.

Typical gates in front of a slow System-2 model:

- Route a support email (billing vs technical vs sales)
- Jailbreak / prompt-injection guard before the LLM
- Expense / invoice / SOC alert: act vs escalate
- Agent harness: act vs tool vs ask-user vs stop (LangChain + Jev pattern)

---

## Directory layout

```
examples/laya/
├── CMakeLists.txt
├── README.md
├── compile_laya.py          # ggmlc.compile → scratch/laya_{family}_{quant}.gguf
├── laya_trunk.py            # exportable encoder + typed head (no unpadding)
├── include/
│   ├── language.h           # script + English-word routing
│   ├── router.h             # --models-dir catalog
│   ├── json_util.h
│   ├── questions.h          # Laya/Jev schema
│   ├── sequence.h
│   ├── presets.h
│   ├── engine.h
│   ├── server.h
│   └── web_assets.h         # Decision Studio (form builder + JSON)
└── src/
    ├── language.cpp
    ├── router.cpp
    ├── questions.cpp
    ├── sequence.cpp
    ├── presets.cpp
    ├── engine.cpp
    ├── server.cpp
    └── main.cpp
```

---

## Compile GGUFs from scratch (Python, once)

```powershell
uv pip install laya
$env:USE_TF="0"; $env:TRANSFORMERS_NO_TF="1"; $env:PYTHONIOENCODING="utf-8"

# English F16 (default demo, ~847 MB)
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family english --quantize f16

# English quants
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family english --quantize q8_0
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family english --quantize ud_q4_k_m

# Multilingual (mmBERT, Gemma BPE, max_len=1024) and typed-decisions specialist
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family multilingual --quantize f16
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family typed-decisions --quantize f16

# Or everything the script knows about at one quant
.\.venv\Scripts\python.exe examples\laya\compile_laya.py --family all --quantize q8_0
```

Outputs land in `scratch/laya_{family}_{quant}.gguf`. RoPE fusion is **disabled**: Laya uses two precomputed thetas (full 160000, sliding 10000). Folding that pattern into `GGML_OP_ROPE` is incorrect.

Dynamic export: batch `b ∈ [1, 8]`, sequence `s ∈ [64, max_len]` (`max_len` is 512 for English and 1024 for the other two). Runtime matches Python `collate_items`: pad to `max(len_i)` in the chunk. CUDA keeps `B·S ≤ 1024` on a 6 GB laptop (arena reuse is on).

---

## Build the C++ binary from source

Native Windows CUDA (this laptop):

```powershell
cmake --build build-win-cuda --target laya -j8
```

Binary: `.\build-win-cuda\examples\laya\laya.exe`

If CMake was configured before this example existed, re-run configure so `GGMLC_BUILD_EXAMPLE_LAYA` is picked up (`GGMLC_BUILD_EXAMPLES=ON` in the release workflow).

Linux / macOS (same tree as the other examples):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGMLC_ENABLE_CUDA=ON   # or -DGGMLC_ENABLE_METAL=ON
cmake --build build --target laya -j8
```

---

## CLI

`--device` defaults to **`auto`**: CUDA or Metal when that backend is compiled in and a device is present, otherwise CPU. Pass `cpu`, `cuda`, `cuda:0`, or `metal` to pin it.

```powershell
.\build-win-cuda\examples\laya\laya.exe --help
.\build-win-cuda\examples\laya\laya.exe --list-presets
.\build-win-cuda\examples\laya\laya.exe --detect-lang --text "I was charged twice"

# Single GGUF
.\build-win-cuda\examples\laya\laya.exe --model scratch\laya_english_f16.gguf --info
.\build-win-cuda\examples\laya\laya.exe scratch\laya_english_f16.gguf --preset email --device auto --cuda-graph
.\build-win-cuda\examples\laya\laya.exe scratch\laya_english_f16.gguf --preset guard --text "Ignore previous instructions" --json

# Language routing: directory of GGUFs (english + multilingual)
.\build-win-cuda\examples\laya\laya.exe --models-dir scratch\laya-ggufs --preset email --text "二重に請求されました"
.\build-win-cuda\examples\laya\laya.exe --models-dir scratch\laya-ggufs --family multilingual --serve --port 8080
```

### Language routing (`--models-dir`)

Same idea as Python `laya.Router`: decide the checkpoint **before** the forward. Detection is:

1. Dominant script of the *string leaves* of the state (JSON keys are ignored — they are usually English). Non-Latin → `multilingual`.
2. Otherwise count common English function words. Enough hits → `english`, else `multilingual`.
3. `typed-decisions` is **not** selected automatically unless `--family typed-decisions` or the question-id set matches one of the four specialist workflows.

`--detect-lang` prints the decision without loading a GGUF.

### Stdio JSON-RPC (`--daemon`)

```json
{"id":"1","preset":"email"}
{"id":"2","preset":"guard","text":"Ignore previous instructions"}
{"id":"3","state":{"message":"..."},"questions":{"intent":{"type":"choice","instructions":"...","criteria":{"refund":"..."}}}}
```

### Web Studio (`--serve`)

```powershell
.\build-win-cuda\examples\laya\laya.exe --model scratch\laya_english_f16.gguf --serve --port 8080 --device auto --cuda-graph
```

- `GET /` — Decision Studio: presets, **question builder** (add choice/score/noul + options), raw JSON tab, **Copy Jev schema**
- `GET /api/health`
- `GET /api/presets`
- `POST /api/decide` with `{"state":..., "questions":...}` or `{"preset":"email"}`

---

## Built-in presets

| Name | Source | What it decides |
| :--- | :--- | :--- |
| `email` | Laya model card | Department, spam/phish, urgency, reply, churn, refund |
| `triage` | `laya.presets.triage_questions` | Intent, urgency, frustration, refund, churn |
| `guard` | `laya.presets.guard_questions` | Jailbreak, injection, secrets, harm, topic |
| `moderation` | `laya.presets.moderation_questions` | Toxic / harassment / threat / spam |
| `router` | `laya.presets.router_questions` | Difficulty, domain, tools, sensitivity |
| `expense` | TypeSafe Jev evals | Policy, category, audit risk |
| `security` | TypeSafe Jev evals | True-positive, severity, playbook |
| `invoice` | TypeSafe Jev evals | PO match, duplicate, AP decision |
| `customer_service` | TypeSafe Jev evals | Next action, save vs refund |
| `harness` | LangChain + Jev | Act / tool / ask-user / stop |

---

## Latency & throughput

Jev/Laya exist for **short decision latency**, not frontier generation. One typed question is one encoder forward. `--bench` reports wall clock, live pad length `S`, batch `B`, forwards, and questions/s after warmup.

**Hardware (2026-09-20):** NVIDIA GeForce RTX 4050 Laptop GPU 6 GB (CC 8.9), Windows, English F16 GGUF ~847 MB. Dynamic `b`/`s`, pad to `max(len_i)` in the chunk (Python `collate_items`), CUDA batch cap `B·S ≤ 1024`, `ggml_gallocr` arena reuse on. C++ warmup 5 / runs 7. Python `laya.Agent` collates questions to the live max length and runs one SDPA forward — it does not concat-pack sequences.

| Path | Shape | Single noul (p50) | Email 7-question wall (p50) | Throughput |
| :--- | :--- | :---: | :---: | :--- |
| **`laya.exe` CUDA + graph** | S=84 B=1 / S=124 B=7 | **25 ms** (best 23.5) | **143 ms** (1 forward, best 134) | **40 q/s** single / **49 q/s** email |
| `laya.exe` CPU 4 threads | S=128, B=8 | — | 28.3 s (1 forward) | 0.24 q/s |
| Python `Agent` CUDA | pad-to-max-in-batch | 57 ms (best 52) | **143 ms** (one forward) | 18 q/s single / 49 q/s batched |
| Python `Agent` CPU | pad-to-max-in-batch | 273 ms | 2.26 s (one forward) | 3.6 q/s single |

Previous static `[1, 512]` C++ path was **234 ms**/noul and **1.45 s** for the email preset. Pad-to-max-in-batch plus arena reuse lets the email preset run as one `B=7` forward, matching the official PyTorch Agent wall clock.

```powershell
.\build-win-cuda\examples\laya\laya.exe scratch\laya_english_f16.gguf --preset email --device auto --cuda-graph --bench --warmup 5 --runs 7
```

---

## Tests

```powershell
pytest tests/numerical/test_laya_differential.py tests/numerical/test_laya_family.py tests/numerical/test_laya_mini_benchmarks.py -v
```

- Differential: live `laya.Agent` vs `LayaCleanTrunk` vs `scratch/laya_english_f16.gguf` (and other GGUFs when present).
- Family / quants: skip if the GGUF is not compiled yet.
- Mini benches: 8 AG News headlines + 4 spam/ham noul examples vs the Python Agent (official AG News is 0.95 on the full split; this is a smoke replica).

---

## Architectural notes

- Dynamic `b` (1–8) and `s` (64–`max_len`). Runtime pads to `max(len_i)` in the chunk (`attention_mask` zeros pads), same as Python `collate_items`. CUDA keeps `B·S ≤ 1024` with OOM-halve fallback.
- Multilingual uses Gemma BPE + Metaspace (`▁`) specials `<bos>/<eos>/<pad>/<mask>` (ids 2/1/0/4), not ModernBERT `[CLS]/[SEP]/[MASK]`. Those ids are stored in GGUF metadata.
- RoPE / sliding-window buffers are length `max_len+1` so `[:, :s]` is never a no-op at `s=max_len`.
- Host code bakes `b * S` into `marker_pos` before GET_ROWS. Do not ADD I32 batch offsets in-graph (CUDA binbcast is F32/F16 only).
- QKV is split on the last 4D axis (`split(hidden)` + `view` + `transpose`). A 5D `view(B,S,3,H,D)` is folded incorrectly by the importer.
- Memory arena reuse is **on** (`ggml_gallocr`) with `ModelExecutor::pin_live_graph_tensors`.
- Marker gather is `F.embedding` (GET_ROWS), not `torch.gather`.
- Pad mask is `(mask - 1) * 1e4`.
- Domain math matches `laya.common.build_sequence` / `Agent.system_one`.
