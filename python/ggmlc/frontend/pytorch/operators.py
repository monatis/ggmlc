from __future__ import annotations

from typing import Any

from ggmlc.ir.op import OpCode

# Map ATen Target -> (OpCode, attribute_extractor)
ATEN_OP_MAP: dict[str, OpCode] = {
    "aten.add.Tensor": OpCode.ADD,
    "aten.add.Scalar": OpCode.ADD,
    "aten.sub.Tensor": OpCode.SUB,
    "aten.sub.Scalar": OpCode.SUB,
    "aten.mul.Tensor": OpCode.MUL,
    "aten.mul.Scalar": OpCode.MUL,
    "aten.div.Tensor": OpCode.DIV,
    "aten.div.Scalar": OpCode.DIV,
    "aten.neg.default": OpCode.NEG,
    "aten.sqrt.default": OpCode.SQRT,
    "aten.rsqrt.default": OpCode.RSQRT,
    "aten.exp.default": OpCode.EXP,
    "aten.log.default": OpCode.LOG,
    "aten.abs.default": OpCode.ABS,
    "aten.maximum.default": OpCode.MAXIMUM,
    "aten.minimum.default": OpCode.MINIMUM,
    "aten.relu.default": OpCode.RELU,
    "aten.gelu.default": OpCode.GELU,
    "aten.silu.default": OpCode.SILU,
    "aten.sigmoid.default": OpCode.SIGMOID,
    "aten._softmax.default": OpCode.SOFTMAX,
    "aten.softmax.int": OpCode.SOFTMAX,
    "aten.mm.default": OpCode.MATMUL,
    "aten.bmm.default": OpCode.MATMUL,
    "aten.matmul.default": OpCode.MATMUL,
    "aten.linear.default": OpCode.LINEAR,
    "aten.embedding.default": OpCode.EMBEDDING,
    "aten.view.default": OpCode.VIEW,
    "aten._unsafe_view.default": OpCode.VIEW,
    "aten.reshape.default": OpCode.RESHAPE,
    "aten.permute.default": OpCode.PERMUTE,
    "aten.transpose.int": OpCode.TRANSPOSE,
    "aten.t.default": OpCode.TRANSPOSE,
    "aten.slice.Tensor": OpCode.SLICE,
    "aten.cat.default": OpCode.CONCAT,
    "aten.split.Tensor": OpCode.SPLIT,
    "aten.expand.default": OpCode.EXPAND,
    "aten.squeeze.dim": OpCode.SQUEEZE,
    "aten.unsqueeze.default": OpCode.UNSQUEEZE,
    "aten.sum.dim_IntList": OpCode.SUM,
    "aten.mean.dim": OpCode.MEAN,
    "aten.scaled_dot_product_attention.default": OpCode.SDPA,
    "aten.native_layer_norm.default": OpCode.LAYER_NORM,
    "aten.layer_norm.default": OpCode.LAYER_NORM,
    "aten.rms_norm.default": OpCode.RMS_NORM,
    "aten.contiguous.default": OpCode.CONTIGUOUS,
    "aten.pow.Tensor_Scalar": OpCode.POW,
    "aten.pow.Tensor_Tensor": OpCode.POW,
}


def get_opcode_for_aten(target: Any) -> OpCode | None:
    name = str(target)
    if name in ATEN_OP_MAP:
        return ATEN_OP_MAP[name]
    # Fallback to string name matching
    for k, v in ATEN_OP_MAP.items():
        if k in name:
            return v
    return None
