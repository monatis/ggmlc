"""CLI command to quantize neural network models."""

from __future__ import annotations

import argparse
from pathlib import Path

from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.dtype import DType
from ggmlc.quantization.model_quantizer import quantize_graph_parameters
from ggmlc.serialization.gguf import serialize_to_gguf
from ggmlc.transforms import create_standard_optimization_pipeline


def main():
    parser = argparse.ArgumentParser(description="Quantize neural network models with ggmlc")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "qwen", "minilm", "resnet18", "bge_m3"],
        help="Pretrained model architecture to load and quantize",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="q4_0",
        choices=["q4_0", "q8_0"],
        help="Target quantization data type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .gguf binary path",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Run standard optimization pass pipeline before quantization",
    )
    args = parser.parse_args()

    target_dtype = DType.Q4_0 if args.dtype == "q4_0" else DType.Q8_0

    print(f"Loading pretrained model '{args.model}'...")
    if args.model == "gpt2":
        from examples.models.hub_models import load_gpt2_model

        model, dummy_inputs, _ = load_gpt2_model()
    elif args.model == "qwen":
        from examples.models.hub_models import load_qwen_model

        model, dummy_inputs, _ = load_qwen_model()
    elif args.model == "minilm":
        from examples.models.hub_models import load_minilm_model

        model, dummy_inputs, _ = load_minilm_model()
    elif args.model == "resnet18":
        from examples.models.hub_models import load_resnet_model

        model, dummy_inputs, _ = load_resnet_model("resnet18")
    elif args.model == "bge_m3":
        from examples.models.hub_models import load_bge_m3_distill_model

        model, dummy_inputs, _ = load_bge_m3_distill_model()
    else:
        msg = f"Unknown model: {args.model}"
        raise ValueError(msg)

    print("Exporting PyTorch model to Canonical IR...")
    exported = export_torch_model(model, dummy_inputs, model_name=args.model)
    canonical_graph = exported.main_graph

    if args.optimize:
        print("Running Canonical IR Optimization Pipeline...")
        pipeline = create_standard_optimization_pipeline()
        res = pipeline.run(canonical_graph)
        print(
            f"  Optimization: {res.stats.nodes_before} -> {res.stats.nodes_after} nodes "
            f"(-{res.stats.node_reduction_pct:.1f}%), {res.stats.fusions_applied} fusions, "
            f"{res.stats.constants_folded} folded constants in {res.stats.duration_ms:.1f}ms."
        )
        canonical_graph = res.graph

    print("Lowering to GGML dialect...")
    ggml_graph = lower_to_ggml(canonical_graph)

    print(f"Quantizing parameters to {target_dtype.name}...")
    quant_ggml_graph, stats = quantize_graph_parameters(ggml_graph, target_dtype=target_dtype)
    print(
        f"  Quantized {stats['tensors_quantized']} weight tensors: "
        f"{stats['orig_bytes'] / (1024 * 1024):.2f} MB -> {stats['quant_bytes'] / (1024 * 1024):.2f} MB "
        f"({stats['compression_ratio']:.2f}x compression ratio)"
    )
    binary = serialize_to_gguf(quant_ggml_graph)

    out_path = Path(args.output or f"{args.model}_{args.dtype}.gguf")
    out_path.write_bytes(binary)
    print(f"Saved quantized model to '{out_path}' ({len(binary) / (1024 * 1024):.2f} MB).")


if __name__ == "__main__":
    main()
