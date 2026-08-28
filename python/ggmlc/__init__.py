import os
import sys
from pathlib import Path

# On Windows, register CUDA toolkit and build binary directories for native DLL loading
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for cuda_cand in (
        os.environ.get("CUDA_PATH", ""),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    ):
        if cuda_cand:
            bin_p = Path(cuda_cand) / "bin"
            if bin_p.exists():
                try:
                    os.add_dll_directory(str(bin_p))
                except (OSError, ValueError):
                    pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    for build_cand in (
        repo_root / "build-win-cuda" / "bin",
        repo_root / "build-win-cuda" / "runtime",
        repo_root / "build-win" / "bin",
        repo_root / "build-win" / "Release",
        repo_root / "build-win" / "runtime" / "Release",
    ):
        if build_cand.exists():
            try:
                os.add_dll_directory(str(build_cand))
            except (OSError, ValueError):
                pass

from typing import Any

from ggmlc.runtime.runner import ModelRunner, get_available_devices, load

__version__ = "0.1.0"

__all__ = [
    "DType",
    "Graph",
    "ModelRunner",
    "OpCode",
    "StorageClass",
    "Tensor",
    "codegen",
    "compile",
    "compile_to_bytes",
    "generate_cpp_project",
    "get_available_devices",
    "graph_to_mermaid",
    "load",
    "visualize",
]


def __getattr__(name: str) -> Any:
    """Lazy-loads compiler, IR, codegen, and visualization modules on demand."""
    if name in ("compile", "compile_to_bytes", "codegen"):
        import ggmlc.compiler as compiler_mod

        return getattr(compiler_mod, name)
    if name == "generate_cpp_project":
        import ggmlc.codegen as codegen_mod

        return getattr(codegen_mod, name)
    if name in ("visualize", "graph_to_mermaid"):
        import ggmlc.visualization as viz_mod

        return getattr(viz_mod, name)
    if name in ("DType", "Graph", "OpCode", "StorageClass", "Tensor"):
        if name == "DType":
            from ggmlc.ir.dtype import DType

            return DType
        if name == "Graph":
            from ggmlc.ir.graph import Graph

            return Graph
        if name == "OpCode":
            from ggmlc.ir.op import OpCode

            return OpCode
        if name == "StorageClass":
            from ggmlc.ir.tensor import StorageClass

            return StorageClass
        if name == "Tensor":
            from ggmlc.ir.tensor import Tensor

            return Tensor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
