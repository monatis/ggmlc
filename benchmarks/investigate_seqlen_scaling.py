"""Fine-grained sequence length scaling and cache locality investigation for ggmlc.

Evaluates latency, throughput, latency-per-token, memory footprint, and speedup delta
between Planned Arena Reuse vs Unplanned Eager Execution across a fine-grained sequence length ladder.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.dtype import DType
from ggmlc.memory.planner import plan_memory_arena
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.transforms import create_standard_optimization_pipeline
from ggmlc.validation.numerical import run_compiled_model_wsl

from examples.models.hub_models import load_gpt2_model, load_minilm_model


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


@dataclass
class SeqLenDataPoint:
    model_name: str
    seq_len: int
    unplanned_act_mb: float
    planned_act_mb: float
    reuse_ratio: float
    memory_saved_pct: float
    # Unplanned timings
    unplanned_mean_lat_ms: float
    unplanned_p50_lat_ms: float
    unplanned_std_ms: float
    unplanned_throughput: float
    unplanned_ms_per_token: float
    # Planned timings
    planned_mean_lat_ms: float
    planned_p50_lat_ms: float
    planned_std_ms: float
    planned_throughput: float
    planned_ms_per_token: float
    # Delta & Accuracy
    speedup_ratio: float
    cosine_sim: float
    max_abs_diff: float


def benchmark_single_seqlen(
    model_name: str,
    seq_len: int,
    load_fn,
    iterations: int = 5,
    warmup: int = 2,
) -> SeqLenDataPoint:
    print("\n" + "=" * 75)
    print(f"🔬 Investigating: {model_name.upper()} @ SeqLen L={seq_len}")
    print("=" * 75)

    # 1. Load model with specific seq_len
    model, dummy_inputs, input_names = load_fn(seq_len=seq_len)
    model = model.eval()

    # 2. PyTorch reference forward pass
    with torch.no_grad():
        ref_out = model(*dummy_inputs)
    if isinstance(ref_out, (tuple, list)):
        ref_arr = ref_out[0].detach().cpu().numpy()
    elif isinstance(ref_out, dict):
        ref_arr = next(iter(ref_out.values())).detach().cpu().numpy()
    else:
        ref_arr = ref_out.detach().cpu().numpy()

    # 3. Export to Canonical IR & apply optimizations
    print(f"Exporting {model_name} (L={seq_len}) to Canonical IR...")
    exported = export_torch_model(model, dummy_inputs, model_name=f"{model_name}_L{seq_len}")
    canonical_graph = exported.main_graph
    pipeline = create_standard_optimization_pipeline()
    opt_res = pipeline.run(canonical_graph)
    opt_graph = opt_res.graph

    # 4. Memory Arena Planning
    symbol_env = {"seq_len": seq_len, "s": seq_len}
    mem_plan = plan_memory_arena(opt_graph, symbol_env=symbol_env)
    unplanned_mb = mem_plan.unplanned_activation_bytes / (1024 * 1024)
    planned_mb = mem_plan.peak_activation_bytes / (1024 * 1024)
    pct_saved = (1.0 - (planned_mb / max(1e-6, unplanned_mb))) * 100.0 if unplanned_mb > 0 else 0.0

    print(
        f"  Memory Footprint: Unplanned = {unplanned_mb:.2f} MB -> "
        f"Planned = {planned_mb:.2f} MB ({mem_plan.reuse_ratio:.2f}x reuse, -{pct_saved:.1f}% savings)"
    )

    # 5. Lower to GGML & Serialize
    ggml_graph = lower_to_ggml(opt_graph)
    serialized_bytes = serialize_ggml_graph(ggml_graph)
    print(f"  Serialized model: {len(serialized_bytes) / (1024 * 1024):.2f} MB")

    input_names = [exported.main_graph.tensors[tid].name for tid in exported.main_graph.inputs]
    input_feed = {input_names[i]: dummy_inputs[i].detach().cpu().numpy() for i in range(len(dummy_inputs))}
    output_ids = [exported.main_graph.outputs[0]]

    # 6. Verify Numerical Parity (Planned Mode)
    print("Verifying numerical parity vs PyTorch...")
    val_out = run_compiled_model_wsl(
        serialized_bytes,
        input_feed,
        output_ids,
        extra_flags=[],
    )
    out_tensor = val_out[output_ids[0]].reshape(ref_arr.shape)
    cos_sim = compute_cosine_similarity(ref_arr, out_tensor)
    max_diff = float(np.max(np.abs(ref_arr - out_tensor)))
    print(f"  Accuracy (Planned): Max Diff = {max_diff:.6f}, Cosine Sim = {cos_sim:.6f}")

    # 7. Benchmark UNPLANNED Execution
    print(f"Benchmarking UNPLANNED execution ({iterations} runs, warmup={warmup})...")
    for _ in range(warmup):
        run_compiled_model_wsl(
            serialized_bytes,
            input_feed,
            output_ids,
            extra_flags=["--unplanned"],
        )
    unplanned_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        run_compiled_model_wsl(
            serialized_bytes,
            input_feed,
            output_ids,
            extra_flags=["--unplanned"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        unplanned_times.append(elapsed_ms)
        print(f"    [Unplanned Run {i + 1}/{iterations}]: {elapsed_ms:.2f} ms")

    unpl_mean = float(np.mean(unplanned_times))
    unpl_p50 = float(np.median(unplanned_times))
    unpl_std = float(np.std(unplanned_times))
    unpl_tp = (seq_len / (unpl_mean / 1000.0)) if model_name == "gpt2" else (1000.0 / unpl_mean)
    unpl_ms_per_tok = unpl_mean / max(1, seq_len)

    # 8. Benchmark PLANNED Execution
    print(f"Benchmarking PLANNED execution ({iterations} runs, warmup={warmup})...")
    for _ in range(warmup):
        run_compiled_model_wsl(
            serialized_bytes,
            input_feed,
            output_ids,
            extra_flags=[],
        )
    planned_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        run_compiled_model_wsl(
            serialized_bytes,
            input_feed,
            output_ids,
            extra_flags=[],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        planned_times.append(elapsed_ms)
        print(f"    [Planned Run {i + 1}/{iterations}]:   {elapsed_ms:.2f} ms")

    plan_mean = float(np.mean(planned_times))
    plan_p50 = float(np.median(planned_times))
    plan_std = float(np.std(planned_times))
    plan_tp = (seq_len / (plan_mean / 1000.0)) if model_name == "gpt2" else (1000.0 / plan_mean)
    plan_ms_per_tok = plan_mean / max(1, seq_len)

    speedup = unpl_mean / max(1e-6, plan_mean)

    print(
        f"  [Summary @ L={seq_len}] Unplanned: {unpl_mean:.2f}ms | Planned: {plan_mean:.2f}ms | "
        f"Speedup: {speedup:.2f}x | Memory Saved: -{pct_saved:.1f}%"
    )

    return SeqLenDataPoint(
        model_name=model_name,
        seq_len=seq_len,
        unplanned_act_mb=unplanned_mb,
        planned_act_mb=planned_mb,
        reuse_ratio=mem_plan.reuse_ratio,
        memory_saved_pct=pct_saved,
        unplanned_mean_lat_ms=unpl_mean,
        unplanned_p50_lat_ms=unpl_p50,
        unplanned_std_ms=unpl_std,
        unplanned_throughput=unpl_tp,
        unplanned_ms_per_token=unpl_ms_per_tok,
        planned_mean_lat_ms=plan_mean,
        planned_p50_lat_ms=plan_p50,
        planned_std_ms=plan_std,
        planned_throughput=plan_tp,
        planned_ms_per_token=plan_ms_per_tok,
        speedup_ratio=speedup,
        cosine_sim=cos_sim,
        max_abs_diff=max_diff,
    )


def generate_markdown_report(results: list[SeqLenDataPoint]) -> str:
    lines = []
    lines.append("# Sequence Length Scaling & Cache Locality Investigation Report\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(
        "This empirical investigation measures the impact of sequence length scaling on memory reuse, "
        "cache locality, per-token latency, and execution speedup comparing Planned Arena Reuse vs Unplanned Execution.\n"
    )

    # Group by model
    models = sorted(list({r.model_name for r in results}))
    for model in models:
        m_results = [r in results and r for r in results if r.model_name == model]
        lines.append(f"### Model: {model.upper()}\n")

        # Table 1: Memory Scaling
        lines.append("#### Memory Footprint Scaling")
        lines.append(
            "| Seq Length (L) | Unplanned Activations | Planned Arena | Reuse Ratio | Memory Saved (%) | Peak vs L1/L2/L3 Working Set |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in m_results:
            ws_class = (
                "Fits L1 (<32KB)"
                if r.planned_act_mb < 0.032
                else (
                    "Fits L2 (<1MB)"
                    if r.planned_act_mb < 1.0
                    else ("Fits L3 (<32MB)" if r.planned_act_mb < 32.0 else "Exceeds L3 (>32MB)")
                )
            )
            lines.append(
                f"| **L={r.seq_len}** | {r.unplanned_act_mb:.2f} MB | **{r.planned_act_mb:.2f} MB** | "
                f"**{r.reuse_ratio:.2f}x** | **-{r.memory_saved_pct:.1f}%** | {ws_class} |"
            )
        lines.append("")

        # Table 2: Latency & Throughput Scaling
        lines.append("#### Latency, Throughput & Locality Factor")
        tp_unit = "tokens/s" if model == "gpt2" else "inf/s"
        lines.append(
            f"| Seq Length (L) | Unplanned Latency | Planned Latency | Unplanned TP | Planned TP | Latency / Token | Speedup Factor | Cosine Sim |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in m_results:
            lines.append(
                f"| **L={r.seq_len}** | {r.unplanned_mean_lat_ms:.2f} ms | **{r.planned_mean_lat_ms:.2f} ms** | "
                f"{r.unplanned_throughput:.2f} {tp_unit} | **{r.planned_throughput:.2f} {tp_unit}** | "
                f"**{r.planned_ms_per_token:.2f} ms/tok** | **{r.speedup_ratio:.2f}x** | **{r.cosine_sim:.6f}** |"
            )
        lines.append("\n---\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Investigate sequence length scaling & cache locality in ggmlc."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2"],
        choices=["gpt2", "minilm"],
        help="Models to investigate",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128],
        help="Sequence lengths ladder to test",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/benchmarks/sequence_length_scaling_analysis.md",
        help="Path to save Markdown report",
    )
    args = parser.parse_args()

    results: list[SeqLenDataPoint] = []

    load_map = {
        "gpt2": load_gpt2_model,
        "minilm": load_minilm_model,
    }

    for model_name in args.models:
        load_fn = load_map[model_name]
        for seq_len in args.seq_lens:
            try:
                res = benchmark_single_seqlen(
                    model_name=model_name,
                    seq_len=seq_len,
                    load_fn=load_fn,
                    iterations=args.iterations,
                    warmup=args.warmup,
                )
                results.append(res)
            except Exception as e:
                print(f"Error benchmarking {model_name} @ L={seq_len}: {e}")
                import traceback

                traceback.print_exc()

    md_report = generate_markdown_report(results)
    print("\n" + "=" * 80)
    print(md_report)
    print("=" * 80)

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_report, encoding="utf-8")
    print(f"\nReport successfully written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
