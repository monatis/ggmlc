# `laya.cpp` — Standalone System 1 Decision Engine

A zero-dependency C++ implementation of **Laya**, the open reproduction of TypeSafe **Jev** System One models: typed questions (`choice` / `score` / `noul`) scored in one parallel pass. No autoregressive token generation.

The neural trunk is a GGUF compiled with `ggmlc`. Sequence construction, temperatures, Shannon confidence, language routing, JSON-RPC, and the Web Studio live in C++.

**Preprocessing is GGUF-defined** (`ggmlc.decision`, the System One analogue of `tokenizer.chat_template`). New architectures (Kev, …) must bake tokenizer + sequence program + postprocess knobs at compile time. The runner does not switch on model family.

Already-distributed Laya GGUFs (english / multilingual / typed-decisions) predate that key. If `ggmlc.decision` is missing, `laya.exe` **assumes the built-in Laya preprocessor** so existing downloads keep working. Do not ship any non-Laya GGUF without `ggmlc.decision`.

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
- Kev 0.5B (Qwen2.5): [mys/kev-0.5b-GGUF](https://huggingface.co/mys/kev-0.5b-GGUF)
- Kev 0.8B (Qwen3.5 Gated DeltaNet): [mys/kev-0.8b-GGUF](https://huggingface.co/mys/kev-0.8b-GGUF)
- Kev 4B (Qwen3.5 Gated DeltaNet): [mys/kev-4b-GGUF](https://huggingface.co/mys/kev-4b-GGUF)

```powershell
# One-command cache via ggmlc (recommended)
ggmlc download mys/laya-GGUF:laya_english_f16.gguf

# Or huggingface-cli into a local folder for --models-dir routing
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

## CLI

`--device` defaults to **`auto`**: CUDA or Metal when that backend is compiled in and a device is present, otherwise CPU. Pass `cpu`, `cuda`, `cuda:0`, or `metal` to pin it.

```bash
laya help
laya list-presets
laya detect-lang --text "I was charged twice"

# Single GGUF
laya info scratch\laya_english_f16.gguf
laya decide scratch\laya_english_f16.gguf --preset email --device auto --cuda-graph
laya decide scratch\laya_english_f16.gguf --preset guard --text "Ignore previous instructions" --json

# Language routing: directory of GGUFs (english + multilingual)
laya decide --models-dir scratch\laya-ggufs --preset email --text "二重に請求されました"
laya serve --models-dir scratch\laya-ggufs --family multilingual --port 8080
```

### Language routing (`--models-dir`)

Same idea as Python `laya.Router`: decide the checkpoint **before** the forward. Detection is:

1. Dominant script of the *string leaves* of the state (JSON keys are ignored — they are usually English). Non-Latin → `multilingual`.
2. Otherwise count common English function words. Enough hits → `english`, else `multilingual`.
3. `typed-decisions` is **not** selected automatically unless `--family typed-decisions` or the question-id set matches one of the four specialist workflows.

`detect-lang` prints the decision without loading a GGUF.

### Stdio JSON-RPC (`daemon`)

```json
{"id":"1","preset":"email"}
{"id":"2","preset":"guard","text":"Ignore previous instructions"}
{"id":"3","state":{"message":"..."},"questions":{"intent":{"type":"choice","instructions":"...","criteria":{"refund":"..."}}}}
```

### Web Studio + TypeSafe API (`serve`)

`laya serve` is a TypeSafe-compatible System One server. The official [`typesafe-sdk`](https://github.com/typesafe-ai/typesafe-sdk-python) talks to it unchanged — set `base_url` at the local port. The Decision Studio uses the same routes.

```bash
laya serve scratch\laya_english_f16.gguf --port 8080 --device auto --cuda-graph
```

| Method | Path | Role |
| :--- | :--- | :--- |
| `GET` | `/` | Decision Studio: presets, question builder, **Copy Jev schema** |
| `GET` | `/health` | Liveness (`status`, `device`, `families`) |
| `GET` | `/v1/models` | TypeSafe model list (`name`, `description`, `release_date`) |
| `GET` | `/v1/presets` | Studio helper: built-in workflows |
| `POST` | `/v1/systemone` | TypeSafe System One ([OpenAPI](https://api.typesafe.ai/openapi.json)) |
| `POST` | `/v1/decide` | Same body/response as `/v1/systemone` |
| `POST` | `/v1/decide/batch` | `{states, questions}` — up to 256 states, shared questions |

Request body matches TypeSafe: `{ "state": ..., "model": "jev-latest", "questions": { "<id>": { "type": "choice"|"score"|"noul", "instructions": "...", "criteria": ... } } }`.

Response matches TypeSafe: `{ "model", "answers", "usage": { "input_tokens", "output_tokens" } }`. Extra Laya fields (`family`, `route`, `usage.latency_ms`, `action`) are ignored by the SDK.

Auth is off by default. Set `LAYA_API_KEY` or `TYPESAFE_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*` POSTs and `GET /v1/models`.

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080") as client:
    result = client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(
                instructions="What is the tone?",
                criteria={"calm": None, "angry": None},
            ),
        },
        model="jev-latest",
    )
    print(result.nouls["billing"].noul, result.choices["tone"].choice)
```

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

Jev/Laya exist for **short decision latency**, not frontier generation. One typed question is one encoder forward. `bench` reports wall clock, live pad length `S`, batch `B`, forwards, and questions/s after warmup.

**Hardware (2026-09-20):** NVIDIA GeForce RTX 4050 Laptop GPU 6 GB (CC 8.9), Windows, English F16 GGUF ~847 MB. Dynamic `b`/`s`, pad to `max(len_i)` across the request (Python `collate_items`), CUDA token budget `B·S ≤ 8192` then OOM-halve, `ggml_gallocr` arena reuse on. C++ warmup 5 / runs 7. Python `laya.Agent` collates questions to the live max length and runs one SDPA forward — it does not concat-pack sequences. A 2026-09-22 remeasure of the English email preset was **147 ms** p50 (still one `B=7, S=124` forward).

| Path | Shape | Single noul (p50) | Email 7-question wall (p50) | Throughput |
| :--- | :--- | :---: | :---: | :--- |
| **`laya.exe` CUDA + graph** | S=84 B=1 / S=124 B=7 | **25 ms** (best 23.5) | **143 ms** (1 forward, best 134) | **40 q/s** single / **49 q/s** email |
| `laya.exe` CPU 4 threads | S=128, B=8 | — | 28.3 s (1 forward) | 0.24 q/s |
| Python `Agent` CUDA | pad-to-max-in-batch | 57 ms (best 52) | **143 ms** (one forward) | 18 q/s single / 49 q/s batched |
| Python `Agent` CPU | pad-to-max-in-batch | 273 ms | 2.26 s (one forward) | 3.6 q/s single |

Previous static `[1, 512]` C++ path was **234 ms**/noul and **1.45 s** for the email preset. Pad-to-max-in-batch plus arena reuse lets the email preset run as one `B=7` forward, matching the official PyTorch Agent wall clock.

```powershell
laya bench scratch\laya_english_f16.gguf --preset email --device auto --cuda-graph --warmup 5 --runs 7
```

### Kev 0.5B and 0.8B

Same email preset and the same wall-clock measurement (`laya.exe bench` vs official `kev` `DecisionModel.forward`, merged LoRA, fp32, one CUDA forward). GGUFs are F16. Qwen3.5 PyTorch on this machine uses the reference Gated DeltaNet (`flash-linear-attention` is not installed); ggmlc lowers that op to `GGML_OP_GATED_DELTA_NET`.

**Hardware (2026-09-22):** same RTX 4050 Laptop. Every row is one forward.

| Model | Path | Single noul (p50) | Email 7-question wall (p50) | Shape |
| :--- | :--- | :---: | :---: | :--- |
| **Kev 0.5B** | **`laya.exe` CUDA + graph** | **17.9 ms** (best 17.5) | **101.5 ms** (best 97.9) | S=68 B=1 / S=122 B=7 |
| Kev 0.5B | PyTorch fp32 | 75.1 ms (best 43.5) | 109.4 ms (best 105.9) | one padded batch |
| **Kev 0.8B** | **`laya.exe` CUDA + graph** | **69.4 ms** (best 67.6) | **464.6 ms** (best 450.9) | S=176 B=1 / S=316 B=7 |
| Kev 0.8B | PyTorch fp32 | 114.3 ms (best 97.5) | 494.2 ms (best 460.2) | one padded batch |

Splitting the 0.8B email by length bucket was three weight reads (**~927 ms**). One pad-to-max batch (`B=7, S=316`) is what matches PyTorch.

```powershell
.\build-win-cuda\examples\laya\laya.exe bench scratch\kev_0.5b_f16.gguf --preset email --device cuda --cuda-graph --warmup 3 --runs 5
.\build-win-cuda\examples\laya\laya.exe bench scratch\kev_0.8b_f16.gguf --preset email --device cuda --cuda-graph --warmup 5 --runs 7
.\.venv\Scripts\python.exe scratch\bench_kev.py --repo jaredpalmer/kev-0.5b --device cuda --dtype fp32
.\.venv\Scripts\python.exe scratch\bench_kev.py --repo jaredpalmer/kev-0.8b --device cuda --dtype fp32
```

---

## Tests

```bash
pytest tests/numerical/test_laya_differential.py tests/numerical/test_laya_family.py tests/numerical/test_laya_mini_benchmarks.py -v
```

TypeSafe SDK smoke (server must already be listening):

```powershell
.\build-win-cuda\examples\laya\laya.exe serve scratch\laya_english_f16.gguf --port 18080 --device auto --cuda-graph
uv pip install typesafe-sdk
.venv\Scripts\python.exe scratch\test_laya_typesafe_sdk.py --base-url http://127.0.0.1:18080

# Same suite against a Kev GGUF (TypeSafe `model` must match GET /v1/models):
.\build-win-cuda\examples\laya\laya.exe serve scratch\kev_0.8b_f16.gguf --port 18081 --device auto --cuda-graph
.venv\Scripts\python.exe scratch\test_laya_typesafe_sdk.py --base-url http://127.0.0.1:18081 --model kev-0.8b
```

- Differential: live `laya.Agent` vs `LayaCleanTrunk` vs `scratch/laya_english_f16.gguf` (and other GGUFs when present).
- Family / quants: skip if the GGUF is not compiled yet.
- Mini benches: 8 AG News headlines + 4 spam/ham noul examples vs the Python Agent (official AG News is 0.95 on the full split; this is a smoke replica).

---

## Compile GGUFs from scratch (Python, once)

```bash
uv pip install laya

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

# Kev-0.5B (Qwen2.5) and Kev-0.8B (Qwen3.5 hybrid Gated DeltaNet).
# LoRA is merged into the base weights in fp32 before export. Adapters are refused.
uv pip install "kev @ git+https://github.com/jaredpalmer/kev.git"
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.5b --quantize f16
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.5b --quantize q8_0
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.5b --quantize ud_q4_k_m

.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.8b --quantize f16
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.8b --quantize q8_0
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 0.8b --quantize ud_q4_k_m

# Larger Qwen3.5 hybrids. Same flags.
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 4b --quantize f16 --device cpu
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 4b --quantize q8_0 --device cpu
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 4b --quantize ud_q4_k_m --device cpu
.\.venv\Scripts\python.exe examples\laya\compile_kev.py --family 9b --quantize f16 --device cpu
```

Outputs land in `scratch/laya_{family}_{quant}.gguf` or `scratch/kev_{size}_{quant}.gguf`. Laya RoPE fusion is **disabled**: Laya uses two precomputed thetas (full 160000, sliding 10000). Folding that pattern into `GGML_OP_ROPE` is incorrect. Kev compile also leaves `enable_rope` off (Qwen3.5 uses partial rotary). Qwen3.5 zero-centered RMS (`rms(x) * (1 + weight)`) is fused to `RMS_NORM`, and the gamma is baked into a following linear when that linear is the only consumer. Already-published Laya GGUFs without `ggmlc.decision` still run: the C++ binary falls back to the original Laya encoder.

Dynamic export: batch `b ∈ [1, 8]`, sequence `s ∈ [64, max_len]` (`max_len` is 512 for English, 1024 for the other Laya families, 2048 for Kev). Runtime matches Python `collate_items`: one batch padded to `max(len_i)`. CUDA keeps `B·S ≤ 8192`, then halves `B` on OOM (arena reuse is on).

---

## Build the C++ binary from source

Native Windows CUDA:

```powershell
cmake --build build --target laya -j8
```

Binary: `.\build\examples\laya\laya.exe`

If CMake was configured before this example existed, re-run configure so `GGMLC_BUILD_EXAMPLE_LAYA` is picked up (`GGMLC_BUILD_EXAMPLES=ON` in the release workflow).

Linux / macOS (same tree as the other examples):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGMLC_ENABLE_CUDA=ON   # or -DGGMLC_ENABLE_METAL=ON
cmake --build build --target laya -j8
```
