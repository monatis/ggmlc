"""Comprehensive comparative benchmarking suite for ggmlc.

Measures Accuracy, Latency, Throughput, and Peak Memory across vision, encoder,
and autoregressive LLM architectures comparing Planned Arena Reuse vs Unplanned Execution
across multiple sequence lengths.
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
from ggmlc.quantization import quantize_graph_parameters
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.transforms import create_standard_optimization_pipeline
from ggmlc.validation.numerical import run_compiled_model_wsl

from examples.models.hub_models import (
    load_bge_m3_distill_model,
    load_gpt2_model,
    load_minilm_model,
    load_qwen_model,
    load_resnet_model,
)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


@dataclass
class ComparativeBenchmarkResult:
    model_name: str
    seq_config: str
    num_params_m: float
    param_bytes_mb: float
    unplanned_act_mb: float
    planned_act_mb: float
    memory_reuse_ratio: float
    memory_saved_pct: float
    unplanned_mean_lat_ms: float
    unplanned_p50_lat_ms: float
    unplanned_p90_lat_ms: float
    unplanned_p99_lat_ms: float
    unplanned_throughput: float
    planned_mean_lat_ms: float
    planned_p50_lat_ms: float
    planned_p90_lat_ms: float
    planned_p99_lat_ms: float
    planned_throughput: float
    speedup_ratio: float
    throughput_unit: str
    cos_similarity: float
    max_abs_diff: float


def run_comparative_model_benchmark(
    name: str,
    seq_config: str,
    load_fn,
    iterations: int = 5,
    warmup: int = 2,
    dtype: str = "f32",
) -> ComparativeBenchmarkResult:
    print("\n" + "=" * 70)
    print(f"Benchmarking: {name.upper()} | Config: {seq_config} (dtype={dtype.upper()})")
    print("=" * 70)

    # 1. Load PyTorch model
    print(f"Loading pretrained model '{name}' ({seq_config})...")
    model, dummy_inputs, _ = load_fn()
    model = model.eval()

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    num_params_m = total_params / 1e6

    # 2. PyTorch reference forward pass
    with torch.no_grad():
        ref_out_raw = model(*dummy_inputs)

    if isinstance(ref_out_raw, torch.Tensor):
        ref_out = ref_out_raw.detach().cpu().numpy()
    elif isinstance(ref_out_raw, (tuple, list)):
        ref_out = ref_out_raw[0].detach().cpu().numpy()
    elif hasattr(ref_out_raw, "last_hidden_state"):
        ref_out = ref_out_raw.last_hidden_state.detach().cpu().numpy()
    elif hasattr(ref_out_raw, "logits"):
        ref_out = ref_out_raw.logits.detach().cpu().numpy()
    else:
        ref_out = np.array(ref_out_raw)

    # 3. Export to Canonical IR
    print("Exporting PyTorch model to Canonical IR...")
    exported = export_torch_model(model, dummy_inputs, model_name=f"{name}_{seq_config}")
    canonical_graph = exported.main_graph

    # 4. Optimization passes
    pipeline = create_standard_optimization_pipeline()
    opt_res = pipeline.run(canonical_graph)
    canonical_graph = opt_res.graph

    # 5. Static Memory Planning
    mem_plan = plan_memory_arena(canonical_graph, alignment=32)
    unplanned_act_mb = mem_plan.unplanned_activation_bytes / (1024 * 1024)
    planned_act_mb = mem_plan.peak_activation_bytes / (1024 * 1024)
    reuse_ratio = mem_plan.reuse_ratio
    saved_pct = (1.0 - (planned_act_mb / unplanned_act_mb)) * 100.0 if unplanned_act_mb > 0 else 0.0
    print(
        f"  Memory Footprint: Unplanned = {unplanned_act_mb:.2f} MB -> Planned = {planned_act_mb:.2f} MB "
        f"({reuse_ratio:.2f}x reuse, -{saved_pct:.1f}% savings)"
    )

    # 6. Lower to GGML dialect
    ggml_graph = lower_to_ggml(canonical_graph)

    # 7. Optional Quantization
    if dtype == "q4_0":
        ggml_graph, q_stats = quantize_graph_parameters(ggml_graph, target_dtype=DType.Q4_0)
        print(f"  Quantized to Q4_0: {q_stats['compression_ratio']:.2f}x compression")
    elif dtype == "q8_0":
        ggml_graph, q_stats = quantize_graph_parameters(ggml_graph, target_dtype=DType.Q8_0)
        print(f"  Quantized to Q8_0: {q_stats['compression_ratio']:.2f}x compression")

    # 8. Serialize
    binary = serialize_ggml_graph(ggml_graph)
    serialized_size_mb = len(binary) / (1024 * 1024)
    print(f"  Serialized .ggmlc size: {serialized_size_mb:.2f} MB")

    # 9. Prepare Input Buffers
    input_names = [exported.main_graph.tensors[tid].name for tid in exported.main_graph.inputs]
    input_dict = {}
    for i, name_k in enumerate(input_names):
        inp_t = dummy_inputs[i]
        if isinstance(inp_t, torch.Tensor):
            input_dict[name_k] = inp_t.detach().cpu().numpy()

    output_ids = [exported.main_graph.outputs[0]]

    # 10. Accuracy Verification (Planned mode)
    print("Verifying numerical parity vs PyTorch...")
    res_planned = run_compiled_model_wsl(binary, input_dict, output_ids)
    out_tensor = res_planned[output_ids[0]].reshape(ref_out.shape)

    max_diff = float(np.max(np.abs(ref_out - out_tensor)))
    mean_diff = float(np.mean(np.abs(ref_out - out_tensor)))
    cos_sim = compute_cosine_similarity(ref_out, out_tensor)
    print(
        f"  Accuracy (Planned): Max Diff = {max_diff:.6f}, Mean Diff = {mean_diff:.6f}, Cosine Sim = {cos_sim:.6f}"
    )

    # Determine throughput unit
    if name in ("gpt2", "qwen"):
        seq_len_val = dummy_inputs[0].shape[-1] if hasattr(dummy_inputs[0], "shape") else 1
        throughput_unit = "tokens/s"
    else:
        seq_len_val = 1
        throughput_unit = "inf/s"

    # 11. Benchmark Mode 1: Unplanned Execution (no arena reuse)
    print(f"Benchmarking UNPLANNED execution across {iterations} runs (warmup={warmup})...")
    for _ in range(warmup):
        run_compiled_model_wsl(binary, input_dict, output_ids, extra_flags=["--unplanned"])

    unplanned_lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        run_compiled_model_wsl(binary, input_dict, output_ids, extra_flags=["--unplanned"])
        unplanned_lats.append((time.perf_counter() - t0) * 1000.0)

    unplanned_arr = np.array(unplanned_lats)
    unplanned_mean = float(np.mean(unplanned_arr))
    unplanned_p50 = float(np.percentile(unplanned_arr, 50))
    unplanned_p90 = float(np.percentile(unplanned_arr, 90))
    unplanned_p99 = float(np.percentile(unplanned_arr, 99))
    unplanned_tp = (
        (seq_len_val / (unplanned_mean / 1000.0))
        if throughput_unit == "tokens/s"
        else (1000.0 / unplanned_mean)
    )

    print(
        f"  [Unplanned] Latency: Mean = {unplanned_mean:.2f}ms | p50 = {unplanned_p50:.2f}ms | Throughput = {unplanned_tp:.2f} {throughput_unit}"
    )

    # 12. Benchmark Mode 2: Planned Arena Reuse Execution
    print(f"Benchmarking PLANNED arena execution across {iterations} runs (warmup={warmup})...")
    for _ in range(warmup):
        run_compiled_model_wsl(binary, input_dict, output_ids)

    planned_lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        run_compiled_model_wsl(binary, input_dict, output_ids)
        planned_lats.append((time.perf_counter() - t0) * 1000.0)

    planned_arr = np.array(planned_lats)
    planned_mean = float(np.mean(planned_arr))
    planned_p50 = float(np.percentile(planned_arr, 50))
    planned_p90 = float(np.percentile(planned_arr, 90))
    planned_p99 = float(np.percentile(planned_arr, 99))
    planned_tp = (
        (seq_len_val / (planned_mean / 1000.0))
        if throughput_unit == "tokens/s"
        else (1000.0 / planned_mean)
    )

    speedup = unplanned_mean / planned_mean if planned_mean > 0 else 1.0
    print(
        f"  [Planned]   Latency: Mean = {planned_mean:.2f}ms | p50 = {planned_p50:.2f}ms | Throughput = {planned_tp:.2f} {throughput_unit}"
    )
    print(
        f"  [Delta]     Speedup / Locality Factor: {speedup:.2f}x | Memory Saved: -{saved_pct:.1f}%"
    )

    return ComparativeBenchmarkResult(
        model_name=name,
        seq_config=seq_config,
        num_params_m=num_params_m,
        param_bytes_mb=mem_plan.persistent_bytes / (1024 * 1024),
        unplanned_act_mb=unplanned_act_mb,
        planned_act_mb=planned_act_mb,
        memory_reuse_ratio=reuse_ratio,
        memory_saved_pct=saved_pct,
        unplanned_mean_lat_ms=unplanned_mean,
        unplanned_p50_lat_ms=unplanned_p50,
        unplanned_p90_lat_ms=unplanned_p90,
        unplanned_p99_lat_ms=unplanned_p99,
        unplanned_throughput=unplanned_tp,
        planned_mean_lat_ms=planned_mean,
        planned_p50_lat_ms=planned_p50,
        planned_p90_lat_ms=planned_p90,
        planned_p99_lat_ms=planned_p99,
        planned_throughput=planned_tp,
        speedup_ratio=speedup,
        throughput_unit=throughput_unit,
        cos_similarity=cos_sim,
        max_abs_diff=max_diff,
    )


def print_comparative_markdown_report(results: list[ComparativeBenchmarkResult]):
    print("\n\n" + "=" * 90)
    print("## 📊 ggmlc Multi-Sequence Length Comparative Benchmark Results (Planned vs Unplanned)")
    print("=" * 90 + "\n")

    # Table 1: Memory Footprint Across Sequence Lengths
    print("### 1. Memory Footprint & Arena Reuse Across Sequence Lengths")
    print(
        "| Model Architecture | Sequence / Spatial Config | Params | Weight Size | Unplanned Act | Planned Arena | Reuse Ratio | Memory Saved (%) |"
    )
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(
            f"| **{r.model_name}** | `{r.seq_config}` | {r.num_params_m:.1f}M | {r.param_bytes_mb:.1f} MB | "
            f"{r.unplanned_act_mb:.2f} MB | **{r.planned_act_mb:.2f} MB** | **{r.memory_reuse_ratio:.2f}x** | "
            f"**-{r.memory_saved_pct:.1f}%** |"
        )

    # Table 2: Latency, Throughput & Performance Comparison
    print("\n### 2. Latency, Throughput & Cache Performance Comparison (Planned vs Unplanned)")
    print(
        "| Model Architecture | Config | Unplanned Latency | Planned Latency | Unplanned TP | Planned TP | Speedup / Locality | Cosine Sim |"
    )
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(
            f"| **{r.model_name}** | `{r.seq_config}` | {r.unplanned_mean_lat_ms:.2f} ms | "
            f"**{r.planned_mean_lat_ms:.2f} ms** | {r.unplanned_throughput:.2f} {r.throughput_unit} | "
            f"**{r.planned_throughput:.2f} {r.throughput_unit}** | **{r.speedup_ratio:.2f}x** | "
            f"**{r.cos_similarity:.6f}** |"
        )
    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="ggmlc Multi-Sequence Length Comparative Benchmark Suite"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "minilm", "gpt2", "bge_m3"],
        choices=["resnet18", "minilm", "gpt2", "qwen", "bge_m3"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of benchmark iterations per mode"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per mode")
    parser.add_argument("--dtype", type=str, default="f32", choices=["f32", "q4_0", "q8_0"])
    args = parser.parse_args()

    # Configurations per model architecture (at least 2 sequence lengths or resolutions)
    model_configs = {
        "resnet18": [
            ("224x224 (standard)", lambda: load_resnet_model("resnet18", resolution=224)),
            ("112x112 (scaled)", lambda: load_resnet_model("resnet18", resolution=112)),
        ],
        "minilm": [
            ("L=16 (short)", lambda: load_minilm_model(seq_len=16)),
            ("L=64 (long)", lambda: load_minilm_model(seq_len=64)),
        ],
        "gpt2": [
            ("L=8 (prompt)", lambda: load_gpt2_model(seq_len=8)),
            ("L=32 (generation)", lambda: load_gpt2_model(seq_len=32)),
        ],
        "bge_m3": [
            ("L=8 (query)", lambda: load_bge_m3_distill_model(seq_len=8)),
            ("L=32 (passage)", lambda: load_bge_m3_distill_model(seq_len=32)),
        ],
        "qwen": [
            ("L=8 (short)", lambda: load_qwen_model("Qwen/Qwen2.5-0.5B", seq_len=8)),
            ("L=32 (long)", lambda: load_qwen_model("Qwen/Qwen2.5-0.5B", seq_len=32)),
        ],
    }

    results = []
    for model_name in args.models:
        configs = model_configs.get(model_name, [])
        for seq_config, loader in configs:
            try:
                res = run_comparative_model_benchmark(
                    model_name,
                    seq_config,
                    loader,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    dtype=args.dtype,
                )
                results.append(res)
            except (RuntimeError, ValueError, AttributeError, KeyError) as e:
                print(f"Error benchmarking {model_name} ({seq_config}): {e}")
                import traceback

                traceback.print_exc()

    if results:
        print_comparative_markdown_report(results)


if __name__ == "__main__":
    main()
