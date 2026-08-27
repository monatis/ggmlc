"""High-performance Python runtime interface for ggmlc models."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ggmlc import _runtime

    _RUNTIME_AVAILABLE = True
except ImportError:
    _runtime = None
    _RUNTIME_AVAILABLE = False

from ggmlc.dialect.ggml.ops import GGMLType
from ggmlc.serialization.gguf import deserialize_ggml_graph

# Map GGMLType integer values to numpy dtypes
GGML_TYPE_TO_NUMPY: dict[int, np.dtype] = {
    int(GGMLType.GGML_TYPE_F32): np.dtype(np.float32),
    int(GGMLType.GGML_TYPE_F16): np.dtype(np.float16),
    int(GGMLType.GGML_TYPE_I8): np.dtype(np.int8),
    int(GGMLType.GGML_TYPE_I16): np.dtype(np.int16),
    int(GGMLType.GGML_TYPE_I32): np.dtype(np.int32),
    int(GGMLType.GGML_TYPE_I64): np.dtype(np.int64),
    int(GGMLType.GGML_TYPE_F64): np.dtype(np.float64),
}


def get_available_devices() -> list[str]:
    """Returns a list of hardware execution devices supported by the runtime (e.g. ['cpu', 'cuda:0'])."""
    if _RUNTIME_AVAILABLE and hasattr(_runtime, "get_available_devices"):
        return list(_runtime.get_available_devices())
    return ["cpu"]


class ModelRunner:
    """Zero-dependency high-performance runner for compiled GGUF models.

    Wraps the native nanobind C++ execution engine for in-memory model evaluation
    with zero-copy numpy buffers, device placement (CPU, CUDA), and multi-threading.
    """

    def __init__(
        self,
        model_source: str | Path | bytes,
        n_threads: int = 1,
        device: str = "cpu",
    ) -> None:
        if not _RUNTIME_AVAILABLE:
            raise RuntimeError(
                "Native ggmlc C++ runtime extension '_runtime' is not available. "
                "Build it using 'cmake --build build-win --target _runtime' (Windows) "
                "or 'cmake --build build --target _runtime' (Linux)."
            )

        self.n_threads = n_threads

        if isinstance(model_source, (str, Path)):
            p = Path(model_source).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            raw_bytes = p.read_bytes()
            self.graph = _runtime.ModelLoader.load_from_file(str(p))
        elif isinstance(model_source, (bytes, bytearray)):
            raw_bytes = bytes(model_source)
            self.graph = _runtime.ModelLoader.load_from_bytes(raw_bytes)
        else:
            raise TypeError(f"Expected file path or bytes, got {type(model_source)}")

        try:
            self.py_graph = deserialize_ggml_graph(raw_bytes)
        except (ValueError, KeyError, struct.error, OSError):
            self.py_graph = None

        self.executor = _runtime.ModelExecutor(self.graph, device)
        self.device = getattr(self.executor, "device", device)
        self.name = self.graph.name
        self.symbol_table = list(self.graph.symbol_table)
        self.inputs = list(self.graph.inputs)
        self.outputs = list(self.graph.outputs)
        self.parameters = list(self.graph.parameters)

        # Build name to ID maps
        self.input_name_to_id: dict[str, int] = {}
        self.tensor_info: dict[int, Any] = {}
        for tid, t in self.graph.tensors.items():
            self.tensor_info[tid] = t
            if tid in self.inputs:
                self.input_name_to_id[t.name] = tid

    def __call__(
        self,
        *args: np.ndarray,
        symbols: dict[str, int] | None = None,
        n_threads: int | None = None,
        **kwargs: np.ndarray,
    ) -> np.ndarray | dict[str | int, np.ndarray]:
        """Runs model inference synchronously on input tensors.

        Args:
            *args: Positional input arrays (matched to graph inputs in order).
            symbols: Optional dictionary of dynamic symbol values (e.g. {'seq_len': 32}).
            n_threads: Number of CPU execution threads (defaults to self.n_threads).
            **kwargs: Named input arrays (matched to input tensor names).

        Returns:
            Computed output numpy array (if single output) or dict of outputs.
        """
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            kwargs = args[0]
            args = ()
        threads = n_threads if n_threads is not None else self.n_threads
        symbol_env: dict[str, int] = {}
        if symbols:
            symbol_env.update(symbols)

        # Auto-deduce symbols from positional inputs using Python graph metadata
        if self.py_graph:
            for idx, arr in enumerate(args):
                if idx < len(self.inputs):
                    tid = self.inputs[idx]
                    t = self.py_graph.tensors.get(tid)
                    if t and hasattr(arr, "shape"):
                        arr_shape = arr.shape
                        for i, dim_obj in enumerate(t.ne):
                            rm_idx = len(arr_shape) - 1 - i
                            if 0 <= rm_idx < len(arr_shape):
                                dim_val = arr_shape[rm_idx]
                                for sym in dim_obj.free_symbols():
                                    if sym not in symbol_env:
                                        symbol_env[sym] = int(dim_val)

        # Fallback if symbols passed and count matches symbol_table
        if symbols and len(self.symbol_table) == len(symbols):
            for reg_sym, val in zip(self.symbol_table, symbols.values()):
                if reg_sym not in symbol_env:
                    symbol_env[reg_sym] = int(val)

        # 1. Prepare context for dynamic symbols
        self.executor.prepare(symbol_env)

        # 2. Bind positional inputs
        for idx, arr in enumerate(args):
            if idx >= len(self.inputs):
                raise ValueError(
                    f"Too many positional arguments: model expects {len(self.inputs)} inputs, got {len(args)}"
                )
            tid = self.inputs[idx]
            arr_c = np.ascontiguousarray(arr)
            self.executor.set_input_by_id(tid, arr_c)

        # 3. Bind keyword inputs
        for name, arr in kwargs.items():
            arr_c = np.ascontiguousarray(arr)
            if name in self.input_name_to_id:
                tid = self.input_name_to_id[name]
                self.executor.set_input_by_id(tid, arr_c)
            elif len(self.inputs) == 1 and len(kwargs) == 1:
                tid = self.inputs[0]
                self.executor.set_input_by_id(tid, arr_c)
            else:
                self.executor.set_input_by_name(name, arr_c)

        # 4. Execute forward graph
        self.executor.run(threads)

        # 5. Extract output tensors
        results: dict[str | int, np.ndarray] = {}
        for out_tid in self.outputs:
            t = self.tensor_info.get(out_tid)
            raw_bytes = self.executor.get_output_bytes(out_tid)
            ne_shape = self.executor.get_tensor_shape(out_tid)

            # Determine numpy dtype
            dtype_val = t.type if t else 0
            np_dtype = GGML_TYPE_TO_NUMPY.get(dtype_val, np.dtype(np.float32))

            # ne is in GGML column-major order [ne0, ne1, ne2, ne3]
            # Convert to PyTorch / C-contiguous row-major shape [ne3, ne2, ne1, ne0]
            full_c_shape = [ne_shape[3], ne_shape[2], ne_shape[1], ne_shape[0]]
            py_t = self.py_graph.tensors.get(out_tid) if self.py_graph else None
            rank = getattr(py_t, "original_rank", None)
            if rank is None or rank <= 0:
                input_ndim = (
                    args[0].ndim
                    if len(args) > 0
                    else (next(iter(kwargs.values())).ndim if kwargs else 2)
                )
                rank = max(2, input_ndim)
            c_shape = full_c_shape[-rank:]

            # Reconstruct numpy array
            arr = np.frombuffer(raw_bytes, dtype=np_dtype)
            try:
                arr = arr.reshape(c_shape)
            except ValueError:
                pass

            out_name = t.name if t else str(out_tid)
            results[out_name] = arr

        if len(self.outputs) == 1:
            return next(iter(results.values()))
        return results

    def reset_state(self) -> None:
        """Resets all persistent state buffers in the executor."""
        self.executor.reset_state()


def load(
    model_source: str | Path | bytes,
    n_threads: int = 1,
    device: str = "cpu",
) -> ModelRunner:
    """Loads a compiled GGUF model into a high-performance native runner.

    Args:
        model_source: Path to .gguf file or raw GGUF bytes.
        n_threads: Number of CPU threads to use during execution.
        device: Hardware device to execute on ("cpu", "cuda", "cuda:0", "auto").

    Returns:
        Instantiated ModelRunner instance.
    """
    return ModelRunner(model_source, n_threads=n_threads, device=device)
