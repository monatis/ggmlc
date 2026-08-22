from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NumericalComparisonResult:
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    rel_diff: float
    message: str


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two flattened arrays."""
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def run_compiled_model_wsl(
    serialized_bytes: bytes,
    inputs: dict[str, np.ndarray],
    output_tensor_ids: list[int],
    symbols: dict[str, int] | None = None,
    states_in: dict[str, np.ndarray] | None = None,
    states_out: list[str] | None = None,
    executable_path: str | None = None,
    n_threads: int = 1,
    extra_flags: list[str] | None = None,
) -> dict[int, np.ndarray] | tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Executes a serialized model via the generic C++ ggmlc-run binary in WSL."""
    if executable_path is None:
        # Default to build path
        executable_path = "/mnt/c/Users/ailabs/ggmlc/build/runtime/ggmlc-run"

    symbols = symbols or {}
    states_in = states_in or {}
    states_out = states_out or []
    extra_flags = extra_flags or []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(serialized_bytes)

        # Write inputs
        input_args = []
        for name, arr in inputs.items():
            in_file = tmp_path / f"in_{name}.bin"
            arr_c = np.ascontiguousarray(arr)
            in_file.write_bytes(arr_c.tobytes())
            wsl_in = in_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            input_args.extend(["--input", f"{name}:{wsl_in}"])

        # Write initial states
        state_in_args = []
        for name, arr in states_in.items():
            s_file = tmp_path / f"state_in_{name}.bin"
            arr_c = np.ascontiguousarray(arr)
            s_file.write_bytes(arr_c.tobytes())
            wsl_s = s_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            state_in_args.extend(["--state-in", f"{name}:{wsl_s}"])

        # Prepare outputs
        output_args = []
        out_files: dict[int, Path] = {}
        for tid in output_tensor_ids:
            out_file = tmp_path / f"out_{tid}.bin"
            out_files[tid] = out_file
            wsl_out = out_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            output_args.extend(["--output", f"{tid}:{wsl_out}"])

        # Prepare state outputs
        state_out_args = []
        sout_files: dict[str, Path] = {}
        for sname in states_out:
            s_file = tmp_path / f"state_out_{sname}.bin"
            sout_files[sname] = s_file
            wsl_s = s_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
            state_out_args.extend(["--state-out", f"{sname}:{wsl_s}"])

        # Symbol args
        symbol_args = []
        for k, v in symbols.items():
            symbol_args.extend(["--symbol", f"{k}={v}"])

        thread_args = ["--threads", str(n_threads)]

        wsl_model = model_file.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")

        all_args = (
            input_args
            + state_in_args
            + output_args
            + state_out_args
            + symbol_args
            + thread_args
            + extra_flags
        )
        cmd = [
            "wsl",
            "bash",
            "-c",
            f"{executable_path} {wsl_model} {' '.join(all_args)}",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"ggmlc-run failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}"
            )

        results: dict[int, np.ndarray] = {}
        for tid, out_file in out_files.items():
            if not out_file.exists():
                raise RuntimeError(f"Expected output file not generated for tensor {tid}")
            raw_bytes = out_file.read_bytes()
            results[tid] = np.frombuffer(raw_bytes, dtype=np.float32)

        if states_out:
            res_states: dict[str, np.ndarray] = {}
            for sname, sfile in sout_files.items():
                if not sfile.exists():
                    raise RuntimeError(f"Expected state file not generated for state {sname}")
                raw_bytes = sfile.read_bytes()
                res_states[sname] = np.frombuffer(raw_bytes, dtype=np.float32)
            return results, res_states

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
