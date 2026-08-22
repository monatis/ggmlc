import argparse
from pathlib import Path

import ggmlc
import torch
from ggmlc.validation.numerical import check_numerical_accuracy

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

    # 1. Compile model directly with ggmlc
    print("[1/2] Compiling model with ggmlc...")
    out_file = args.output or f"{args.model}.gguf"
    model_path = ggmlc.compile(
        model=model,
        sample_inputs=example_inputs,
        output=out_file,
        model_name=args.model,
    )
    print(f"      Saved GGUF binary container to: {Path(model_path).resolve()}")

    # 2. Native high-speed evaluation and differential verification
    print("[2/2] Loading native ModelRunner and executing...")
    runner = ggmlc.load(model_path)

    with torch.no_grad():
        ref_out = model(*example_inputs)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        ref_np = ref_out.numpy()

    inputs_dict = {
        name: tensor_val.numpy()
        for name, tensor_val in zip(input_names, example_inputs, strict=False)
    }
    actual_np = runner(**inputs_dict)

    cmp_res = check_numerical_accuracy(ref_np, actual_np, atol=1e-4)

    print("\n=== Execution Verification Results ===")
    print(f"Status:        {'PASSED [OK]' if cmp_res.passed else 'FAILED [FAIL]'}")
    print(f"Max Abs Diff:  {cmp_res.max_abs_diff:.6e}")
    print(f"Mean Abs Diff: {cmp_res.mean_abs_diff:.6e}")
    print(f"Relative Diff: {cmp_res.rel_diff:.6e}")


if __name__ == "__main__":
    main()
