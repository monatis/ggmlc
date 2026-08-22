"""Tests for FusionOptions and OperatorFusionPass graph rewriting."""

from __future__ import annotations

from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.fusion import FusionOptions


def test_fusion_options_toggle():
    """Verify that fusion options can be independently toggled on and off."""
    # Build a graph with an Add -> GELU pattern and a SILU -> Mul pattern
    g = Graph(name="toggle_test")
    t_x = g.add_tensor("x", Shape.from_tuple((2, 4, 32)), DType.F32, StorageClass.INPUT)
    t_bias = g.add_tensor("b", Shape.from_tuple((32,)), DType.F32, StorageClass.PARAMETER)
    t_add = g.add_tensor(
        "add_out", Shape.from_tuple((2, 4, 32)), DType.F32, StorageClass.ACTIVATION
    )
    t_gelu = g.add_tensor("gelu_out", Shape.from_tuple((2, 4, 32)), DType.F32, StorageClass.OUTPUT)

    g.inputs = [t_x.id]
    g.parameters = [t_bias.id]
    g.outputs = [t_gelu.id]

    g.add_op(OpCode.ADD, [t_x.id, t_bias.id], [t_add.id])
    g.add_op(OpCode.GELU, [t_add.id], [t_gelu.id])

    # 1. When enable_bias_gelu is False
    opt_disabled = FusionOptions(enable_bias_gelu=False)
    ggml_disabled = lower_to_ggml(g, enable_fusion=True, fusion_options=opt_disabled)
    opcodes_disabled = [op.opcode for op in ggml_disabled.nodes]
    assert GGMLOpCode.GGML_OP_CUSTOM_BIAS_GELU not in opcodes_disabled

    # 2. When enable_bias_gelu is True
    opt_enabled = FusionOptions(enable_bias_gelu=True)
    ggml_enabled = lower_to_ggml(g, enable_fusion=True, fusion_options=opt_enabled)
    opcodes_enabled = [op.opcode for op in ggml_enabled.nodes]
    assert GGMLOpCode.GGML_OP_CUSTOM_BIAS_GELU in opcodes_enabled
