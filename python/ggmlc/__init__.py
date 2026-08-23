"""ggmlc: High-Performance Neural Network Tensor Program Compiler to GGML."""

from __future__ import annotations

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
