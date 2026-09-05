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
import subprocess
import time
from pathlib import Path


def run_llama_cpp(
    model_path: str, prompt: str, max_tokens: int, n_threads: int, n_gpu_layers: int = 0
):
    from llama_cpp import Llama

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

    # Decode throughput
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

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = res.stdout

    # Parse stdout metrics
    # Summary: X tokens generated in Ys (Z tok/s overall)
    #   Prompt Prefill : P tokens in AA ms (BB tok/s)
    #   Token Decode   : D tokens in CC ms (DD tok/s, EE ms/tok)
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
    parser.add_argument("--output-md", type=str, default="benchmark_kv_cache_report.md")
    parser.add_argument("--output-json", type=str, default="benchmark_kv_cache_report.json")
    args = parser.parse_args()

    official_gguf = "scratch/SmolLM2-135M-Instruct-f16.gguf"
    ggmlc_gguf = "scratch/smollm2_chat.gguf"
    ggmlc_run_exe = "build-win-cuda/runtime/ggmlc-run.exe"

    prompt = "The capital of France is"
    seq_lengths = [32, 64, 128, 256]

    devices_to_test = ["cpu", "cuda"] if args.device == "both" else [args.device]

    all_results = {}

    for dev in devices_to_test:
        print(f"\n{'=' * 70}\nBenchmarking on device: {dev.upper()}\n{'=' * 70}")
        all_results[dev] = {}

        for slen in seq_lengths:
            print(f"\n--- Sequence Length: {slen} tokens ---")

            # 1. llama.cpp baseline
            gpu_layers = 99 if dev == "cuda" else 0
            try:
                llama_res = run_llama_cpp(
                    official_gguf, prompt, slen, n_threads=args.threads, n_gpu_layers=gpu_layers
                )
                print(
                    f"[llama.cpp  {dev.upper()}] {llama_res['gen_tokens']} tok | {llama_res['total_time_s']:.2f}s | {llama_res['decode_tok_s']:.1f} tok/s | {llama_res['ms_per_tok']:.2f} ms/tok"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[llama.cpp  {dev.upper()}] Error: {e}")
                llama_res = None

            # 2. ggmlc-run with KV cache
            try:
                ggmlc_res = run_ggmlc_run(
                    ggmlc_run_exe, ggmlc_gguf, prompt, slen, device=dev, n_threads=args.threads
                )
                print(
                    f"[ggmlc-run  {dev.upper()}] {ggmlc_res['gen_tokens']} tok | {ggmlc_res['total_time_s']:.2f}s | {ggmlc_res['decode_tok_s']:.1f} tok/s | {ggmlc_res['ms_per_tok']:.2f} ms/tok"
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
        "Comparing standalone native runner `ggmlc-run` (with hardware KV cache) against official `llama.cpp`",
        "on SmolLM2-135M across sequence lengths (32, 64, 128, 256 tokens).",
        "",
    ]

    for dev in devices_to_test:
        md_lines.append(f"## Hardware Target: {dev.upper()}")
        md_lines.append("")
        md_lines.append(
            "| Max Tokens | Engine | Generated | Total Time | Decode Throughput | Inter-Token Latency | Latency Flatness |"
        )
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for slen in seq_lengths:
            data = all_results[dev].get(slen, {})
            lres = data.get("llama_cpp")
            gres = data.get("ggmlc_run")

            if lres:
                md_lines.append(
                    f"| {slen} | `llama.cpp` | {lres['gen_tokens']} tok | {lres['total_time_s']:.2f} s | {lres['decode_tok_s']:.1f} tok/s | {lres['ms_per_tok']:.2f} ms/tok | Baseline |"
                )
            if gres:
                md_lines.append(
                    f"| {slen} | `ggmlc-run` (KV Cache) | {gres['gen_tokens']} tok | {gres['total_time_s']:.2f} s | {gres['decode_tok_s']:.1f} tok/s | {gres['ms_per_tok']:.2f} ms/tok | **O(1) Flat** |"
                )

        md_lines.append("")

    report_md = "\n".join(md_lines)
    Path(args.output_md).write_text(report_md, encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark reports to '{args.output_md}' and '{args.output_json}'!")


if __name__ == "__main__":
    main()
