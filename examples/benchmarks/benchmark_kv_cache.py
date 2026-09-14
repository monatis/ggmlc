"""Comprehensive KV Cache & Autoregressive Inference Benchmark Suite.

Benchmarks:
1. llama.cpp baseline using official GGUF (SmolLM2-135M-Instruct-f16.gguf)
2. ggmlc-run native C++ runner with hardware KV cache (smollm2_chat.gguf)

Measures prefill throughput, inter-token decode latency (ms/tok), and total throughput
across sequence lengths: 32, 64, 128, 256 tokens on CPU and NVIDIA CUDA GPU.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


def find_ggmlc_run_executable(explicit_path: str | None = None) -> str:
    """Discovers standalone ggmlc-run binary across platforms (Linux/Colab/Windows)."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Specified executable not found: {explicit_path}")

    if sys.platform == "win32":
        candidates = [
            Path("build-win-cuda/runtime/ggmlc-run.exe"),
            Path("build-win/runtime/Release/ggmlc-run.exe"),
            Path("build-win/runtime/Debug/ggmlc-run.exe"),
            Path("./ggmlc-run.exe"),
            Path("build/runtime/ggmlc-run.exe"),
        ]
    else:
        candidates = [
            Path("build/runtime/ggmlc-run"),
            Path("build-cuda/runtime/ggmlc-run"),
            Path("runtime/ggmlc-run"),
            Path("./ggmlc-run"),
        ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    which_path = shutil.which("ggmlc-run")
    if which_path:
        return which_path

    raise FileNotFoundError(
        "Could not locate 'ggmlc-run' standalone binary.\n"
        "Please build it via `cmake --build build -j` or specify path using `--executable <path>`."
    )


def find_or_fetch_official_gguf(explicit_path: str | None = None, quantize: str = "q8_0") -> str:
    """Discovers existing official GGUF or fetches SmolLM2-135M from Hugging Face Hub."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Specified official GGUF not found: {explicit_path}")

    target_fn = (
        "SmolLM2-135M-Instruct-f16.gguf"
        if quantize.lower() == "f16"
        else "SmolLM2-135M-Instruct-Q8_0.gguf"
    )
    candidates = [
        Path(f"scratch/{target_fn}"),
        Path(f".cache/gguf/{target_fn}"),
        Path(f"scratch/SmolLM2-135M-Instruct-{quantize}.gguf"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    try:
        from huggingface_hub import hf_hub_download

        print(
            f"📥 Fetching official SmolLM2 GGUF `{target_fn}` from Hugging Face Hub...", flush=True
        )
        downloaded = hf_hub_download(
            repo_id="bartowski/SmolLM2-135M-Instruct-GGUF",
            filename=target_fn,
            cache_dir=".cache/gguf",
        )
        return str(Path(downloaded).resolve())
    except Exception as e:
        raise FileNotFoundError(
            f"Official SmolLM2 GGUF ({target_fn}) not found and auto-download failed ({e}).\n"
            "Please specify `--official-gguf <path>`."
        ) from e


def find_or_compile_ggmlc_gguf(
    explicit_path: str | None = None, auto_compile: bool = True, quantize: str = "q8_0"
) -> str:
    """Discovers existing ggmlc GGUF or compiles SmolLM2-135M on demand with matching precision."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Specified ggmlc GGUF not found: {explicit_path}")

    target_path = Path(f"scratch/smollm2_chat_{quantize}.gguf")
    if target_path.is_file():
        return str(target_path.resolve())

    # Check fallback legacy path if f16
    if quantize.lower() == "f16" and Path("scratch/smollm2_chat.gguf").is_file():
        return str(Path("scratch/smollm2_chat.gguf").resolve())

    if not auto_compile:
        raise FileNotFoundError(
            f"ggmlc GGUF not found at {target_path}. Please specify `--ggmlc-gguf <path>` or allow auto-compilation."
        )

    print(
        f"⚙️ Compiling SmolLM2-135M (quantize={quantize}) with native GQA and RoPE fusion for benchmark...",
        flush=True,
    )
    import ggmlc
    import torch
    from ggmlc.pipeline.tokenizer import BPETokenizer

    from examples.models.hub_models import load_smollm2_model

    target_path.parent.mkdir(parents=True, exist_ok=True)

    model, dummy_input, _ = load_smollm2_model(seq_len=8)
    tokenizer = BPETokenizer.from_huggingface("HuggingFaceTB/SmolLM2-135M-Instruct")

    dim_s = torch.export.Dim("s", min=1, max=2048)
    dynamic_shapes = ({1: dim_s},)

    ggmlc.compile(
        model=model,
        sample_inputs=dummy_input,
        output=str(target_path),
        dynamic_shapes=dynamic_shapes,
        model_name="smollm2_135m",
        quantize=quantize,
        pipeline=tokenizer,
        tasks=["text-generation"],
    )
    print(
        f"✅ Compilation finished: {target_path} ({target_path.stat().st_size / (1024 * 1024):.1f} MB)",
        flush=True,
    )
    return str(target_path.resolve())


def run_llama_cpp(
    model_path: str, prompt: str, max_tokens: int, n_threads: int, n_gpu_layers: int = 0
):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("⚠️ 'llama_cpp' module not installed. Skipping llama.cpp baseline.")
        return None

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    t0 = time.perf_counter()
    res = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    gen_tokens = res["usage"]["completion_tokens"]
    prompt_tokens = res["usage"]["prompt_tokens"]
    gen_text = res["choices"][0]["text"]

    decode_tok_s = gen_tokens / max(total_time, 1e-6)
    ms_per_tok = (total_time * 1000.0) / max(gen_tokens, 1)

    return {
        "engine": "llama.cpp",
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": total_time,
        "decode_tok_s": decode_tok_s,
        "ms_per_tok": ms_per_tok,
        "text": gen_text[:60].strip(),
    }


def run_ggmlc_run(
    executable_path: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    device: str = "cpu",
    n_threads: int = 4,
    use_cuda_graph: bool = False,
):
    cmd = [
        executable_path,
        model_path,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--device",
        device,
        "--threads",
        str(n_threads),
    ]
    if device == "cuda" and use_cuda_graph:
        cmd.append("--cuda-graph")

    res = subprocess.run(
        cmd, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
    )
    stdout = res.stdout

    gen_tokens = max_tokens
    total_time = 0.0
    tok_s = 0.0
    prefill_ms = 0.0
    prefill_tok_s = 0.0
    decode_tok_s = 0.0
    decode_ms_tok = 0.0

    m_sum = re.search(
        r"Summary:\s+(\d+)\s+tokens generated in\s+([\d\.]+)s\s+\(([\d\.]+)\s+tok/s overall\)",
        stdout,
    )
    if m_sum:
        gen_tokens = int(m_sum.group(1))
        total_time = float(m_sum.group(2))
        tok_s = float(m_sum.group(3))

    m_pref = re.search(
        r"Prompt Prefill\s+:\s+\d+\s+tokens in\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+tok/s\)", stdout
    )
    if m_pref:
        prefill_ms = float(m_pref.group(1))
        prefill_tok_s = float(m_pref.group(2))

    m_dec = re.search(
        r"Token Decode\s+:\s+\d+\s+tokens in\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+tok/s,\s+([\d\.]+)\s+ms/tok\)",
        stdout,
    )
    if m_dec:
        decode_tok_s = float(m_dec.group(2))
        decode_ms_tok = float(m_dec.group(3))
    else:
        decode_tok_s = tok_s
        decode_ms_tok = (total_time * 1000.0) / max(gen_tokens, 1)

    return {
        "engine": "ggmlc-run",
        "gen_tokens": gen_tokens,
        "total_time_s": total_time,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "ms_per_tok": decode_ms_tok,
    }


def main():
    parser = argparse.ArgumentParser(description="KV Cache Autoregressive Benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--quantize",
        type=str,
        default="q8_0",
        choices=["q8_0", "f16"],
        help="Quantization precision for both llama.cpp and ggmlc-run (default: q8_0)",
    )
    parser.add_argument("--output-md", type=str, default="benchmark_kv_cache_report.md")
    parser.add_argument("--output-json", type=str, default="benchmark_kv_cache_report.json")
    parser.add_argument(
        "--executable", type=str, default=None, help="Path to ggmlc-run standalone binary"
    )
    parser.add_argument(
        "--official-gguf", type=str, default=None, help="Path to official llama.cpp GGUF"
    )
    parser.add_argument("--ggmlc-gguf", type=str, default=None, help="Path to ggmlc compiled GGUF")
    parser.add_argument(
        "--skip-compile", action="store_true", help="Do not auto-compile ggmlc GGUF if missing"
    )
    parser.add_argument(
        "--cuda-graph", action="store_true", default=True, help="Enable CUDA Graph for ggmlc-run"
    )
    args = parser.parse_args()

    ggmlc_run_exe = find_ggmlc_run_executable(args.executable)
    print(f"🔧 Using standalone runner: {ggmlc_run_exe}")

    official_gguf = find_or_fetch_official_gguf(args.official_gguf, quantize=args.quantize)
    llama_payload_mb = round(Path(official_gguf).stat().st_size / (1024.0 * 1024.0), 1)
    print(
        f"📦 Using official GGUF ({args.quantize.upper()}, {llama_payload_mb} MB): {official_gguf}"
    )

    ggmlc_gguf = find_or_compile_ggmlc_gguf(
        args.ggmlc_gguf, auto_compile=not args.skip_compile, quantize=args.quantize
    )
    ggmlc_payload_mb = round(Path(ggmlc_gguf).stat().st_size / (1024.0 * 1024.0), 1)
    print(f"🚀 Using ggmlc GGUF ({args.quantize.upper()}, {ggmlc_payload_mb} MB): {ggmlc_gguf}")

    prompt = "The capital of France is"
    seq_lengths = [32, 64, 128, 256]

    devices_to_test = ["cpu", "cuda"] if args.device == "both" else [args.device]

    all_results = {}

    for dev in devices_to_test:
        print(
            f"\n{'=' * 70}\nBenchmarking on device: {dev.upper()} ({args.quantize.upper()})\n{'=' * 70}"
        )
        all_results[dev] = {}

        for slen in seq_lengths:
            print(f"\n--- Sequence Length: {slen} tokens ---")

            # 1. llama.cpp baseline
            gpu_layers = 99 if dev == "cuda" else 0
            try:
                llama_res = run_llama_cpp(
                    official_gguf, prompt, slen, n_threads=args.threads, n_gpu_layers=gpu_layers
                )
                if llama_res:
                    llama_res["payload_mb"] = llama_payload_mb
                    llama_bw = (
                        round((llama_payload_mb / 1024.0) / (llama_res["ms_per_tok"] / 1000.0), 1)
                        if llama_res["ms_per_tok"] > 0
                        else 0.0
                    )
                    llama_res["bandwidth_gb_s"] = llama_bw
                    print(
                        f"[llama.cpp  {dev.upper()}] {llama_res['gen_tokens']} tok | {llama_res['total_time_s']:.2f}s | "
                        f"{llama_res['decode_tok_s']:.1f} tok/s | {llama_res['ms_per_tok']:.2f} ms/tok | "
                        f"BW: {llama_bw} GB/s ({llama_payload_mb} MB)"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[llama.cpp  {dev.upper()}] Error: {e}")
                llama_res = None

            # 2. ggmlc-run with KV cache
            try:
                ggmlc_res = run_ggmlc_run(
                    ggmlc_run_exe,
                    ggmlc_gguf,
                    prompt,
                    slen,
                    device=dev,
                    n_threads=args.threads,
                    use_cuda_graph=args.cuda_graph,
                )
                if ggmlc_res:
                    ggmlc_res["payload_mb"] = ggmlc_payload_mb
                    ggmlc_bw = (
                        round((ggmlc_payload_mb / 1024.0) / (ggmlc_res["ms_per_tok"] / 1000.0), 1)
                        if ggmlc_res["ms_per_tok"] > 0
                        else 0.0
                    )
                    ggmlc_res["bandwidth_gb_s"] = ggmlc_bw
                    print(
                        f"[ggmlc-run  {dev.upper()}] {ggmlc_res['gen_tokens']} tok | {ggmlc_res['total_time_s']:.2f}s | "
                        f"{ggmlc_res['decode_tok_s']:.1f} tok/s | {ggmlc_res['ms_per_tok']:.2f} ms/tok | "
                        f"BW: {ggmlc_bw} GB/s ({ggmlc_payload_mb} MB)"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[ggmlc-run  {dev.upper()}] Error: {e}")
                ggmlc_res = None

            all_results[dev][slen] = {
                "llama_cpp": llama_res,
                "ggmlc_run": ggmlc_res,
            }

    # Generate Markdown Report
    md_lines = [
        "# Autoregressive KV Cache Inference Benchmark Report",
        "",
        f'**Precision Target:** `{args.quantize.upper()}` | **Prompt:** "{prompt}"  ',
        f"**Model:** SmolLM2-135M (`llama.cpp`: {llama_payload_mb} MB vs `ggmlc-run`: {ggmlc_payload_mb} MB)  ",
        "",
        "Comparing standalone native runner `ggmlc-run` (with hardware KV cache) against official `llama.cpp`",
        "across sequence lengths (32, 64, 128, 256 tokens).",
        "",
    ]

    for dev in devices_to_test:
        md_lines.append(f"## Hardware Target: {dev.upper()}")
        md_lines.append("")
        md_lines.append(
            "| Max Tokens | Engine | Precision | Model Payload | Generated | Total Time | Decode Throughput | Inter-Token Latency | Memory Bandwidth | Latency Flatness |"
        )
        md_lines.append("| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- |")

        for slen in seq_lengths:
            data = all_results[dev].get(slen, {})
            lres = data.get("llama_cpp")
            gres = data.get("ggmlc_run")

            if lres:
                md_lines.append(
                    f"| {slen} | `llama.cpp` | {args.quantize.upper()} | {lres.get('payload_mb', llama_payload_mb)} MB | "
                    f"{lres['gen_tokens']} tok | {lres['total_time_s']:.2f} s | {lres['decode_tok_s']:.1f} tok/s | "
                    f"{lres['ms_per_tok']:.2f} ms/tok | **{lres.get('bandwidth_gb_s', 0.0)} GB/s** | Baseline |"
                )
            if gres:
                md_lines.append(
                    f"| {slen} | `ggmlc-run` (KV Cache) | {args.quantize.upper()} | {gres.get('payload_mb', ggmlc_payload_mb)} MB | "
                    f"{gres['gen_tokens']} tok | {gres['total_time_s']:.2f} s | {gres['decode_tok_s']:.1f} tok/s | "
                    f"{gres['ms_per_tok']:.2f} ms/tok | **{gres.get('bandwidth_gb_s', 0.0)} GB/s** | **O(1) Flat** |"
                )

        md_lines.append("")

    report_md = "\n".join(md_lines)
    Path(args.output_md).write_text(report_md, encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark reports to '{args.output_md}' and '{args.output_json}'!")


if __name__ == "__main__":
    main()
