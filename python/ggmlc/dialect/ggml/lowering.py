from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ggmlc.dialect.ggml.ops import GGMLOpCode, GGMLType, GGMLUnaryOpCode
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.shape import Dim, Shape, StaticDim
from ggmlc.ir.tensor import StorageClass


def dtype_to_ggml_type(dtype: DType) -> GGMLType:
    mapping = {
        DType.F32: GGMLType.GGML_TYPE_F32,
        DType.F16: GGMLType.GGML_TYPE_F16,
        DType.BF16: GGMLType.GGML_TYPE_BF16,
        DType.I32: GGMLType.GGML_TYPE_I32,
        DType.I64: GGMLType.GGML_TYPE_I64,
        DType.I8: GGMLType.GGML_TYPE_I8,
        DType.BOOL: GGMLType.GGML_TYPE_I8,
    }
    return mapping.get(dtype, GGMLType.GGML_TYPE_F32)


def canonical_shape_to_ggml_ne(shape: Shape) -> tuple[Dim, Dim, Dim, Dim]:
    """Converts a Canonical N-D shape [d0, d1, ..., d_k] (row-major) to GGML 4D ne[0..3].

    GGML ne[0] is the innermost dimension (contiguous, stride 1), so we reverse the dimensions.
    Unused dimensions are set to StaticDim(1).
    """
    dims = list(shape.dims)
    rev_dims = dims[::-1]
    while len(rev_dims) < 4:
        rev_dims.append(StaticDim(1))
    return (rev_dims[0], rev_dims[1], rev_dims[2], rev_dims[3])


@dataclass
class GGMLTensorDef:
    id: int
    name: str
    ggml_type: GGMLType
    ne: tuple[Dim, Dim, Dim, Dim]
    storage: StorageClass
    producer_id: int | None = None
    data: np.ndarray | None = None
    role: str | None = None


@dataclass
class GGMLOpDef:
    id: int
    opcode: GGMLOpCode
    inputs: list[int]
    outputs: list[int]
    attributes: dict[str, Any] = field(default_factory=dict)
    name: str | None = None


@dataclass
class GGMLExecutionGraph:
    """Target execution graph ready for binary serialization and generic C++ runtime."""

    name: str
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    parameters: list[int] = field(default_factory=list)
    tensors: dict[int, GGMLTensorDef] = field(default_factory=dict)
    nodes: list[GGMLOpDef] = field(default_factory=list)
    symbol_table: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def get_tensor(self, tid: int) -> GGMLTensorDef:
        return self.tensors[tid]


def lower_to_ggml(canonical_graph: Graph) -> GGMLExecutionGraph:
    """Lowers a Canonical IR Graph into a GGML Execution Graph."""
    ggml_graph = GGMLExecutionGraph(
        name=canonical_graph.name,
        inputs=list(canonical_graph.inputs),
        outputs=list(canonical_graph.outputs),
        parameters=list(canonical_graph.parameters),
        metadata=dict(canonical_graph.metadata),
    )

    symbols: set[str] = set()

    # 1. Lower all tensors
    for tid, t in canonical_graph.tensors.items():
        ne = canonical_shape_to_ggml_ne(t.shape)
        for d in ne:
            symbols |= d.free_symbols()

        ggml_tensor = GGMLTensorDef(
            id=t.id,
            name=t.name,
            ggml_type=dtype_to_ggml_type(t.dtype),
            ne=ne,
            storage=t.storage,
            producer_id=t.producer_id,
            data=t.data,
            role=t.role,
        )
        ggml_graph.tensors[tid] = ggml_tensor

    # Optimize MATMUL weight tensor layouts (for JAX [K, N] weights -> GGML [K, N] with transposed data)
    for op in canonical_graph.nodes:
        if op.opcode == OpCode.MATMUL:
            b_id = op.inputs[1]
            b_t = ggml_graph.tensors[b_id]
            if (
                b_t.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                and b_t.data is not None
                and b_t.data.ndim == 2
            ):
                k_val, n_val = b_t.data.shape
                c_dims = canonical_graph.get_tensor(b_id).shape.dims
                if (
                    len(c_dims) == 2
                    and c_dims[0].evaluate({}) == k_val
                    and c_dims[1].evaluate({}) == n_val
                ):
                    b_t.data = np.ascontiguousarray(b_t.data.T)
                    b_t.ne = (StaticDim(k_val), StaticDim(n_val), StaticDim(1), StaticDim(1))

    ggml_graph.symbol_table = sorted(symbols)

    # 2. Lower operations
    for op in canonical_graph.nodes:
        ggml_op = _lower_op(op, canonical_graph, ggml_graph)
        ggml_graph.nodes.append(ggml_op)

    return ggml_graph


def _lower_op(op: Operation, c_graph: Graph, g_graph: GGMLExecutionGraph) -> GGMLOpDef:
    opcode = op.opcode
    in_ids = list(op.inputs)
    out_ids = list(op.outputs)
    attrs = dict(op.attributes)

    if opcode == OpCode.ADD:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_ADD, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SUB:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SUB, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.MUL:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_MUL, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.DIV:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_DIV, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SQRT:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SQRT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.RSQRT:
        return GGMLOpDef(
            op.id, GGMLOpCode.GGML_OP_SQRT, in_ids, out_ids, {"is_rsqrt": 1, **attrs}, op.name
        )
    elif opcode == OpCode.POW:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SQR, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.MEAN:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_MEAN, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CONTIGUOUS:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.LOG:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_LOG, in_ids, out_ids, attrs, op.name)
    elif opcode in (OpCode.RELU, OpCode.MAXIMUM):
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_RELU)},
            op.name,
        )
    elif opcode == OpCode.GELU:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_GELU)},
            op.name,
        )
    elif opcode == OpCode.SILU:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_SILU)},
            op.name,
        )
    elif opcode == OpCode.RMS_NORM:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RMS_NORM, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SOFTMAX:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SOFT_MAX, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.LINEAR:
        # PyTorch linear: in_ids = [x, weight, bias (optional)]
        # In GGML: out = ggml_mul_mat(weight, x) + bias
        x_id = in_ids[0]
        w_id = in_ids[1]
        b_id = in_ids[2] if len(in_ids) > 2 else None
        if b_id is not None:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_MUL_MAT, [w_id, x_id, b_id], out_ids, attrs, op.name
            )
        else:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_MUL_MAT, [w_id, x_id], out_ids, attrs, op.name
            )
    elif opcode == OpCode.MATMUL:
        t_in1 = c_graph.get_tensor(in_ids[1])
        is_param = t_in1.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
        attrs["transpose_in0"] = 0 if is_param else 1
        mapped_inputs = [in_ids[1], in_ids[0]]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_MUL_MAT, mapped_inputs, out_ids, attrs, op.name)
    elif opcode in (OpCode.RESHAPE, OpCode.VIEW, OpCode.SQUEEZE, OpCode.UNSQUEEZE):
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RESHAPE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.PERMUTE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        p = attrs.get("dims", list(range(R)))
        axes = [0, 1, 2, 3]
        for i in range(4):
            if i < R:
                in_k = R - 1 - i
                out_j = p.index(in_k)
                axes[i] = R - 1 - out_j
            else:
                axes[i] = i
        attrs["axis0"] = axes[0]
        attrs["axis1"] = axes[1]
        attrs["axis2"] = axes[2]
        attrs["axis3"] = axes[3]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_PERMUTE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.TRANSPOSE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        d0 = attrs.get("dim0", 0)
        d1 = attrs.get("dim1", 1)
        if d0 < 0:
            d0 += R
        if d1 < 0:
            d1 += R
        p = list(range(R))
        if 0 <= d0 < R and 0 <= d1 < R:
            p[d0], p[d1] = p[d1], p[d0]
        axes = [0, 1, 2, 3]
        for i in range(4):
            if i < R:
                in_k = R - 1 - i
                out_j = p.index(in_k)
                axes[i] = R - 1 - out_j
            else:
                axes[i] = i
        attrs["axis0"] = axes[0]
        attrs["axis1"] = axes[1]
        attrs["axis2"] = axes[2]
        attrs["axis3"] = axes[3]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_PERMUTE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SLICE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", 0)
        ggml_dim = R - 1 - dim if dim >= 0 else -1 - dim
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_VIEW, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CONCAT:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", 0)
        ggml_dim = R - 1 - dim if dim >= 0 else -1 - dim
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONCAT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.EXPAND:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_REPEAT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.LAYER_NORM:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_NORM, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.EMBEDDING:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_GET_ROWS, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SDPA:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_FLASH_ATTN_EXT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.ROPE:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_ROPE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SWIGLU:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_GLU, in_ids, out_ids, attrs, op.name)
    else:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_NONE, in_ids, out_ids, attrs, op.name)
