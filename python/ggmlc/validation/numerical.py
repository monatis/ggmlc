from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch.exporter import export_torch_model
from ggmlc.ir.graph import Graph
from ggmlc.ir.model import Model
from ggmlc.serialization.graph import serialize_ggml_graph


@dataclass
class NumericalComparisonResult:
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    rel_diff: float
    message: str


def run_compiled_model_wsl(
    serialized_bytes: bytes,
    inputs: Dict[str, np.ndarray],
    output_tensor_ids: List[int],
    symbols: Optional[Dict[str, int]] = None,
    executable_path: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Executes a serialized model via the generic C++ ggmlc-run binary in WSL."""
    if executable_path is None:
        # Default to build path
        executable_path = "/mnt/c/Users/ailabs/ggmlc/build/runtime/ggmlc-run"

    symbols = symbols or {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_file = tmp_path / "model.ggmlc"
        model_file.write_bytes(serialized_bytes)

        # Write inputs
        input_args = []
        for name, arr in inputs.items():
            in_file = tmp_path / f"in_{name}.bin"
            # Ensure C-contiguous
            arr_c = np.ascontiguousarray(arr)
            in_file.write_bytes(arr_c.tobytes())
            # Convert Windows path to WSL path
            wsl_in = in_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            input_args.extend(["--input", f"{name}:{wsl_in}"])

        # Prepare outputs
        output_args = []
        out_files: Dict[int, Path] = {}
        for tid in output_tensor_ids:
            out_file = tmp_path / f"out_{tid}.bin"
            out_files[tid] = out_file
            wsl_out = out_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            output_args.extend(["--output", f"{tid}:{wsl_out}"])

        # Symbol args
        symbol_args = []
        for k, v in symbols.items():
            symbol_args.extend(["--symbol", f"{k}={v}"])

        wsl_model = model_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")

        cmd = [
            "wsl",
            "bash",
            "-c",
            f"{executable_path} {wsl_model} {' '.join(input_args)} {' '.join(output_args)} {' '.join(symbol_args)}",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ggmlc-run failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}")

        results: Dict[int, np.ndarray] = {}
        for tid, out_file in out_files.items():
            if not out_file.exists():
                raise RuntimeError(f"Expected output file not generated for tensor {tid}")
            raw_bytes = out_file.read_bytes()
            results[tid] = np.frombuffer(raw_bytes, dtype=np.float32)

        return results


def check_numerical_accuracy(
    ref: np.ndarray,
    actual: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> NumericalComparisonResult:
    """Compares reference and actual numpy arrays within tolerances."""
    ref_flat = ref.flatten()
    act_flat = actual.flatten()

    if ref_flat.shape != act_flat.shape:
        return NumericalComparisonResult(
            passed=False,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            rel_diff=float("inf"),
            message=f"Shape mismatch: reference has {ref_flat.shape[0]} elements, actual has {act_flat.shape[0]} elements",
        )

    abs_diff = np.abs(ref_flat - act_flat)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rel_diff = float(np.max(abs_diff / (np.abs(ref_flat) + 1e-8)))

    passed = np.allclose(ref_flat, act_flat, atol=atol, rtol=rtol)
    msg = f"max_diff={max_abs:.6e}, mean_diff={mean_abs:.6e}, rel_diff={rel_diff:.6e} (passed={passed})"

    return NumericalComparisonResult(
        passed=bool(passed),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rel_diff=rel_diff,
        message=msg,
    )
