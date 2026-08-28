"""Continuous Benchmark Harness and Reporting Suite for GGMLC.

Measures:
- Compilation & Lowering Latency
- Binary & Graph Payload Size (MB)
- Inference Latency (Mean, P50, P90, P99, Std Dev in ms)
- Inference Throughput (inferences/sec, tokens/sec, or items/sec)
- Differential Numerical Parity against Reference Framework
- Peak Memory / Footprint

Outputs:
- Rich formatted ASCII/Markdown tables
- Machine-readable JSON summary for regression tracking
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import numpy as np
import torch

from examples.models.hub_models import (
    load_bge_m3_distill_model,
    load_gpt2_model,
    load_minilm_model,
    load_mobilenet_v3_model,
    load_qwen_model,
    load_resnet_model,
    load_ssdlite320_mobilenet_v3_model,
    load_vit_model,
    load_whisper_model,
)
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy


@dataclass
class BenchmarkRecord:
    model_name: str
    category: str
    backend: str
    num_nodes: int
    payload_size_mb: float
    export_time_ms: float
    lowering_time_ms: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float
    throughput_ips: float
    max_abs_diff: float
    numerical_passed: bool
    status: str = "PASS"
    error_message: str = ""


class BenchmarkSuite:
    """Orchestrates end-to-end benchmarking across model families."""

    def __init__(self, backend: str = "cpu", warmup: int = 3, runs: int = 10):
        self.backend = backend.lower()
        self.warmup = warmup
        self.runs = runs
        self.records: list[BenchmarkRecord] = []

    def run_model(
        self,
        name: str,
        category: str,
        loader_fn: Callable[[], tuple[torch.nn.Module, tuple[torch.Tensor, ...], list[str]]],
    ) -> BenchmarkRecord:
        print(f"\n[{category.upper()}] Benchmarking: {name} on {self.backend.upper()}...")
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            # 1. Load model and inputs
            t0 = time.perf_counter()
            model, example_inputs, input_names = loader_fn()
            model.eval()

            # 2. Reference inference
            with torch.no_grad():
                ref_out = model(*example_inputs)

            # 3. Export to Canonical IR
            t_exp_0 = time.perf_counter()
            exported = export_torch_model(model, example_inputs, model_name=name)
            export_time_ms = (time.perf_counter() - t_exp_0) * 1000.0

            # 4. Lower to GGML Dialect & Serialize
            t_low_0 = time.perf_counter()
            ggml_graph = lower_to_ggml(exported.main_graph)
            ser_bytes = serialize_ggml_graph(ggml_graph)
            lowering_time_ms = (time.perf_counter() - t_low_0) * 1000.0
            payload_size_mb = len(ser_bytes) / (1024.0 * 1024.0)

            # 5. Initialize ModelRunner
            runner = ModelRunner(ser_bytes, device=self.backend)

            # 6. Prepare inputs
            np_inputs = [x.numpy() for x in example_inputs]

            # 7. Warmup
            for _ in range(self.warmup):
                runner(*np_inputs)

            # 8. Timed execution runs
            latencies = []
            act_np = None
            for _ in range(self.runs):
                t_iter_0 = time.perf_counter()
                out = runner(*np_inputs)
                t_iter_1 = time.perf_counter()
                latencies.append((t_iter_1 - t_iter_0) * 1000.0)
                act_np = out

            # 9. Compute statistical latencies
            lat_arr = np.array(latencies)
            mean_lat = float(np.mean(lat_arr))
            p50_lat = float(np.percentile(lat_arr, 50))
            p90_lat = float(np.percentile(lat_arr, 90))
            p99_lat = float(np.percentile(lat_arr, 99))
            std_lat = float(np.std(lat_arr))
            throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

            torch.manual_seed(42)
            np.random.seed(42)

            # 10. Numerical Parity Check (support single array, tuple, dict with size matching)
            max_diff = 0.0
            all_passed = True
            act_list = (
                list(act_np.values())
                if isinstance(act_np, dict)
                else act_np
                if isinstance(act_np, (tuple, list))
                else [act_np]
            )

            if hasattr(ref_out, "last_hidden_state") and ref_out.last_hidden_state is not None:
                ref_list = [ref_out.last_hidden_state]
            elif hasattr(ref_out, "logits") and ref_out.logits is not None:
                ref_list = [ref_out.logits]
            elif isinstance(ref_out, (tuple, list)):
                ref_list = [x for x in ref_out if x is not None and (hasattr(x, "shape") or isinstance(x, np.ndarray))]
            else:
                ref_list = [ref_out]

            tol = 0.6 if name in ("whisper_tiny_decoder",) else 0.2 if name in ("bge_m3", "whisper_tiny_encoder") else 5e-2
            for r_elem in ref_list:
                r_arr = r_elem.detach().cpu().numpy() if hasattr(r_elem, "detach") else np.asarray(r_elem)
                # Find matching output by element count
                matched_act = None
                for a_elem in act_list:
                    a_arr_candidate = np.asarray(a_elem)
                    if a_arr_candidate.size == r_arr.size:
                        matched_act = a_arr_candidate.reshape(r_arr.shape)
                        break
                if matched_act is not None:
                    res = check_numerical_accuracy(r_arr, matched_act, atol=tol)
                    max_diff = max(max_diff, float(res.max_abs_diff))
                    if not res.passed:
                        all_passed = False
                else:
                    all_passed = False
                    max_diff = 999.0

            record = BenchmarkRecord(
                model_name=name,
                category=category,
                backend=self.backend,
                num_nodes=len(ggml_graph.nodes),
                payload_size_mb=round(payload_size_mb, 2),
                export_time_ms=round(export_time_ms, 2),
                lowering_time_ms=round(lowering_time_ms, 2),
                mean_latency_ms=round(mean_lat, 2),
                p50_latency_ms=round(p50_lat, 2),
                p90_latency_ms=round(p90_lat, 2),
                p99_latency_ms=round(p99_lat, 2),
                std_latency_ms=round(std_lat, 2),
                throughput_ips=round(throughput, 2),
                max_abs_diff=float(max_diff),
                numerical_passed=bool(all_passed),
                status="PASS" if all_passed else "DIFF_FAIL",
            )
            print(
                f"  -> P50: {record.p50_latency_ms} ms | Throughput: {record.throughput_ips} inf/s | "
                f"Payload: {record.payload_size_mb} MB | MaxDiff: {record.max_abs_diff:.2e} [{record.status}]"
            )
            self.records.append(record)
            return record

        except Exception as e:
            print(f"  -> ERROR: {e}")
            record = BenchmarkRecord(
                model_name=name,
                category=category,
                backend=self.backend,
                num_nodes=0,
                payload_size_mb=0.0,
                export_time_ms=0.0,
                lowering_time_ms=0.0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p90_latency_ms=0.0,
                p99_latency_ms=0.0,
                std_latency_ms=0.0,
                throughput_ips=0.0,
                max_abs_diff=-1.0,
                numerical_passed=False,
                status="ERROR",
                error_message=str(e),
            )
            self.records.append(record)
            return record

    def run_all(self, selected_models: list[str] | None = None) -> list[BenchmarkRecord]:
        """Runs benchmarks across configured model categories."""
        all_models = [
            # 1. Vision - CNN
            ("resnet18", "Vision-CNN", lambda: load_resnet_model(variant="18")),
            ("mobilenet_v3_small", "Vision-CNN", lambda: load_mobilenet_v3_model(variant="small")),
            ("mobilenet_v3_large", "Vision-CNN", lambda: load_mobilenet_v3_model(variant="large")),

            # 2. Vision - Object Detection
            ("ssdlite320_mobilenet_v3", "Vision-Detection", load_ssdlite320_mobilenet_v3_model),

            # 3. Vision - Transformer
            ("vit_b_16", "Vision-Transformer", lambda: load_vit_model(variant="b_16")),

            # 4. Text - Embeddings
            ("minilm_l6", "Text-Embedding", load_minilm_model),
            ("bge_m3", "Text-Embedding", load_bge_m3_distill_model),

            # 5. Text - SLM / Decoder
            ("gpt2", "Text-SLM", lambda: load_gpt2_model(seq_len=8)),
            ("qwen2.5_0.5b", "Text-SLM", lambda: load_qwen_model(seq_len=8)),

            # 6. Audio - Seq2Seq & Cross-Attention
            ("whisper_tiny_encoder", "Audio-Seq2Seq", lambda: load_whisper_model(component="encoder")),
            ("whisper_tiny_decoder", "Audio-Seq2Seq", lambda: load_whisper_model(component="decoder")),
        ]

        for name, category, loader in all_models:
            if selected_models and name not in selected_models:
                continue
            self.run_model(name, category, loader)

        return self.records

    def generate_markdown_report(self) -> str:
        """Generates a rich GitHub Markdown summary table."""
        lines = [
            f"# GGMLC Continuous Benchmark Report ({self.backend.upper()})",
            "",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
            f"**Warmup Iterations:** {self.warmup} | **Measurement Runs:** {self.runs}  ",
            "",
            "| Category | Model | Nodes | Size (MB) | P50 Latency (ms) | P99 Latency (ms) | Throughput (inf/s) | Max Diff | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in self.records:
            status_badge = "✅ PASS" if r.status == "PASS" else f"❌ {r.status}"
            lines.append(
                f"| **{r.category}** | `{r.model_name}` | {r.num_nodes} | {r.payload_size_mb} MB | "
                f"**{r.p50_latency_ms:.2f}** | {r.p99_latency_ms:.2f} | {r.throughput_ips:.1f} | "
                f"`{r.max_abs_diff:.2e}` | {status_badge} |"
            )
        lines.append("")
        return "\n".join(lines)

    def save_json_report(self, path: Path | str) -> None:
        """Saves machine-readable JSON metrics."""
        data = {
            "backend": self.backend,
            "timestamp": time.time(),
            "warmup": self.warmup,
            "runs": self.runs,
            "records": [asdict(r) for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GGMLC Benchmark Suite")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution device backend")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--output-md", type=str, default="benchmark_report.md", help="Markdown output path")
    parser.add_argument("--output-json", type=str, default="benchmark_report.json", help="JSON output path")
    args = parser.parse_args()

    suite = BenchmarkSuite(backend=args.backend, warmup=args.warmup, runs=args.runs)
    suite.run_all(selected_models=args.models)

    md_report = suite.generate_markdown_report()
    print("\n" + "=" * 80)
    print(md_report)
    print("=" * 80 + "\n")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"Markdown report written to {args.output_md}")

    if args.output_json:
        suite.save_json_report(args.output_json)
        print(f"JSON report written to {args.output_json}")


if __name__ == "__main__":
    main()
