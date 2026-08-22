"""ggmlc: Neural-Network Compiler Targeting GGML."""

from __future__ import annotations

from ggmlc.codegen import generate_cpp_project
from ggmlc.compiler import codegen, compile
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.tensor import StorageClass, Tensor
from ggmlc.runtime.runner import ModelRunner, load

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
    "generate_cpp_project",
    "load",
]
