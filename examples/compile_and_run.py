"""Unified CLI demonstration script for compiling and executing full models with ggmlc."""

import argparse
from pathlib import Path

import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl

from examples.models.hub_models import (
    load_bge_m3_distill_model,
    load_gpt2_model,
    load_minilm_model,
    load_qwen_model,
    load_resnet_model,
)


def get_model(name: str):
    name = name.lower()
    if name in ("resnet", "resnet18"):
        return load_resnet_model("resnet18")
    elif name == "resnet50":
        return load_resnet_model("resnet50")
    elif name in ("minilm", "all-minilm-l6-v2", "bert"):
        return load_minilm_model()
    elif name in ("gpt2", "gpt"):
        return load_gpt2_model()
    elif name in ("qwen", "qwen2.5", "qwen3"):
        return load_qwen_model("Qwen/Qwen2.5-0.5B")
    elif name in ("bge", "bge-m3", "bge_m3_distill"):
        return load_bge_m3_distill_model()
    else:
        raise ValueError(f"Unknown model name: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and execute neural network models with ggmlc"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=[
            "resnet",
            "resnet18",
            "resnet50",
            "minilm",
            "bert",
            "gpt2",
            "qwen",
            "bge_m3_distill",
        ],
        help="Model architecture to compile and execute",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save compiled .gguf artifact",
    )
    args = parser.parse_args()

    print(f"=== [ggmlc] Compiling Hub model '{args.model}' ===")
    torch.manual_seed(42)
    model, example_inputs, input_names = get_model(args.model)
    model.eval()

    # 1. Frontend export
    print("[1/4] Ingesting PyTorch graph via torch.export...")
    exported = export_torch_model(model, example_inputs, model_name=args.model)
    print(
        f"      Canonical IR graph constructed: {len(exported.main_graph.nodes)} ops, {len(exported.main_graph.tensors)} tensors."
    )

    # 2. Lower to GGML dialect
    print("[2/4] Lowering Canonical IR to GGML dialect...")
    ggml_graph = lower_to_ggml(exported.main_graph)
    print(f"      GGML execution graph: {len(ggml_graph.ops)} ops scheduled.")

    # 3. Serialize to GGUF v3 binary container
    print("[3/4] Serializing to GGUF v3 binary container...")
    ser_bytes = serialize_ggml_graph(ggml_graph)
    print(f"      Serialized GGUF size: {len(ser_bytes):,} bytes.")

    if args.output:
        out_p = Path(args.output)
        out_p.write_bytes(ser_bytes)
        print(f"      Saved artifact to: {out_p.resolve()}")

    # 4. Execute and differential verify
    print("[4/4] Executing via generic C++ runtime in WSL...")
    with torch.no_grad():
        ref_out = model(*example_inputs)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        ref_np = ref_out.numpy()

    inputs_dict = {}
    for name, tensor_val in zip(input_names, example_inputs, strict=False):
        inputs_dict[name] = tensor_val.numpy()

    out_id = exported.main_graph.outputs[0]
    exec_results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs=inputs_dict,
        output_tensor_ids=[out_id],
    )

    actual_np = exec_results[out_id]
    cmp_res = check_numerical_accuracy(ref_np, actual_np, atol=1e-4)

    print("\n=== Execution Verification Results ===")
    print(f"Status:        {'PASSED [OK]' if cmp_res.passed else 'FAILED [FAIL]'}")
    print(f"Max Abs Diff:  {cmp_res.max_abs_diff:.6e}")
    print(f"Mean Abs Diff: {cmp_res.mean_abs_diff:.6e}")
    print(f"Relative Diff: {cmp_res.rel_diff:.6e}")


if __name__ == "__main__":
    main()
