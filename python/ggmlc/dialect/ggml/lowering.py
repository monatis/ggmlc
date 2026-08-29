from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ggmlc.dialect.ggml.ops import GGMLOpCode, GGMLType, GGMLUnaryOpCode
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.shape import Dim, Shape, StaticDim
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def dtype_to_ggml_type(dtype: DType) -> GGMLType:
    mapping = {
        DType.F32: GGMLType.GGML_TYPE_F32,
        DType.F16: GGMLType.GGML_TYPE_F16,
        DType.BF16: GGMLType.GGML_TYPE_BF16,
        DType.I32: GGMLType.GGML_TYPE_I32,
        DType.I64: GGMLType.GGML_TYPE_I32,
        DType.I8: GGMLType.GGML_TYPE_F32,
        DType.BOOL: GGMLType.GGML_TYPE_F32,
        DType.Q4_0: GGMLType.GGML_TYPE_Q4_0,
        DType.Q4_K: GGMLType.GGML_TYPE_Q4_K,
        DType.Q8_0: GGMLType.GGML_TYPE_Q8_0,
    }
    return mapping.get(dtype, GGMLType.GGML_TYPE_F32)


def canonical_shape_to_ggml_ne(shape: Shape) -> tuple[Dim, Dim, Dim, Dim]:
    """Converts a Canonical N-D shape [d0, d1, ..., d_k] (row-major) to GGML 4D ne[0..3].

    GGML ne[0] is the innermost dimension (contiguous, stride 1), so we reverse the dimensions.
    For N > 4, outer dimensions are folded into ne[3]. Unused dimensions are set to StaticDim(1).
    """
    dims = list(shape.dims)
    if len(dims) == 0:
        return (StaticDim(1), StaticDim(1), StaticDim(1), StaticDim(1))
    elif len(dims) == 1:
        return (dims[0], StaticDim(1), StaticDim(1), StaticDim(1))
    elif len(dims) == 2:
        return (dims[1], dims[0], StaticDim(1), StaticDim(1))
    elif len(dims) == 3:
        return (dims[2], dims[1], dims[0], StaticDim(1))
    elif len(dims) == 4:
        return (dims[3], dims[2], dims[1], dims[0])
    else:
        outer_val = 1
        for d in dims[:-3]:
            if isinstance(d, StaticDim):
                outer_val *= d.value
            else:
                outer_val *= int(d)
        return (dims[-1], dims[-2], dims[-3], StaticDim(outer_val))


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
    original_rank: int = 4


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

    symbols: set[str] = field(default_factory=set)

    def to_mermaid(self) -> str:
        """Converts this lowered GGML execution graph to Mermaid flowchart format."""
        from ggmlc.visualization.mermaid import graph_to_mermaid

        return graph_to_mermaid(self)

    def visualize(self, output_path: str | Path, format: str = "html") -> Path:
        """Renders and exports this execution graph to HTML, SVG, PNG, or Mermaid MMD."""
        from ggmlc.visualization.mermaid import visualize

        return visualize(self, output_path, format=format)


def lower_to_ggml(
    canonical_graph: Graph,
    enable_fusion: bool = True,
    fusion_options: FusionOptions | None = None,
) -> GGMLExecutionGraph:
    """Lowers Canonical IR Graph to target-specific GGML execution semantics."""
    if enable_fusion:
        fuse_operations(canonical_graph, fusion_options)

    # 1.0 Ensure integer indices for EMBEDDING are I32
    for op in canonical_graph.nodes:
        if op.opcode == OpCode.EMBEDDING and len(op.inputs) > 1:
            idx_id = op.inputs[1]
            idx_t = canonical_graph.get_tensor(idx_id)
            if idx_t and idx_t.dtype in (DType.I32, DType.I64):
                idx_t.dtype = DType.I32
                if idx_t.data is not None:
                    idx_t.data = np.ascontiguousarray(idx_t.data.astype(np.int32))

        # Promote mixed int/float operands in binary arithmetic to F32
        if op.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV) and len(op.inputs) >= 2:
            t0 = canonical_graph.get_tensor(op.inputs[0])
            t1 = canonical_graph.get_tensor(op.inputs[1])
            if t0 and t1:
                if t0.dtype == DType.F32 and t1.dtype in (DType.I32, DType.I64, DType.BOOL):
                    t1.dtype = DType.F32
                    if t1.data is not None:
                        t1.data = np.ascontiguousarray(t1.data.astype(np.float32))
                elif t1.dtype == DType.F32 and t0.dtype in (DType.I32, DType.I64, DType.BOOL):
                    t0.dtype = DType.F32
                    if t0.data is not None:
                        t0.data = np.ascontiguousarray(t0.data.astype(np.float32))

    ggml_graph = GGMLExecutionGraph(
        name=canonical_graph.name,
        inputs=list(canonical_graph.inputs),
        outputs=list(canonical_graph.outputs),
        parameters=list(canonical_graph.parameters),
    )
    symbols: set[str] = set()

    # 1. Lower Tensors
    for tid, t in canonical_graph.tensors.items():
        ne = canonical_shape_to_ggml_ne(t.shape)
        for d in ne:
            symbols |= d.free_symbols()

        t_data = t.data
        if t_data is not None and not isinstance(t_data, np.ndarray):
            t_data = np.asarray(t_data)
        t_type = dtype_to_ggml_type(t.dtype)
        if t_data is not None:
            if t_type == GGMLType.GGML_TYPE_I32 and t_data.dtype == np.int64:
                t_data = np.ascontiguousarray(t_data.astype(np.int32))
            elif t_type == GGMLType.GGML_TYPE_F32 and t_data.dtype in (
                np.bool_,
                np.int8,
                np.uint8,
                np.int32,
                np.int64,
            ):
                t_data = np.ascontiguousarray(t_data.astype(np.float32))

        ggml_tensor = GGMLTensorDef(
            id=t.id,
            name=t.name,
            ggml_type=t_type,
            ne=ne,
            storage=t.storage,
            producer_id=t.producer_id,
            data=t_data,
            role=t.role,
            original_rank=len(t.shape.dims),
        )
        ggml_graph.tensors[tid] = ggml_tensor

    # 1.1 Convert parameter weights for Linear layers
    converted_weights: set[int] = set()
    for op in canonical_graph.nodes:
        if op.opcode == OpCode.LINEAR:
            x_id = op.inputs[0]
            w_id = op.inputs[1]
            if w_id in converted_weights:
                continue
            converted_weights.add(w_id)
            x_t = canonical_graph.get_tensor(x_id)
            w_t = ggml_graph.tensors[w_id]
            c_w = canonical_graph.get_tensor(w_id)
            if w_t.data is not None and hasattr(w_t.data, "ndim") and w_t.data.ndim == 2:
                k_val = x_t.shape.dims[-1].evaluate({})
                d0 = c_w.shape.dims[0].evaluate({})
                d1 = c_w.shape.dims[1].evaluate({})
                is_addmm = op.attributes.get("is_addmm", 0) != 0
                if is_addmm or (d0 == k_val and d1 != k_val):
                    # Hugging Face Conv1D / addmm: weight shape in PyTorch is (in_features, out_features)
                    # We transpose to (out_features, in_features) so GGML ne = [in_features, out_features]
                    # gets contiguous rows of in_features elements.
                    w_t.data = np.ascontiguousarray(w_t.data.T)
                    w_t.ne = (
                        StaticDim(k_val),
                        StaticDim(d1 if d0 == k_val else d0),
                        StaticDim(1),
                        StaticDim(1),
                    )
                else:
                    # Standard PyTorch Linear: weight shape in PyTorch is (out_features, in_features)
                    # PyTorch C-order memory already has out_features rows of in_features elements.
                    # GGML ne = [in_features, out_features] consumes these contiguous rows directly.
                    w_t.data = np.ascontiguousarray(w_t.data)
                    w_t.ne = (
                        StaticDim(k_val),
                        StaticDim(d0 if d1 == k_val else d1),
                        StaticDim(1),
                        StaticDim(1),
                    )

    ggml_graph.symbol_table = sorted(symbols)

    # 2. Lower operations
    op_by_id = {op.id: op for op in canonical_graph.nodes}
    for op in canonical_graph.nodes:
        ggml_op = _lower_op(op, canonical_graph, ggml_graph, op_by_id, fusion_options)
        ggml_graph.nodes.append(ggml_op)

    return ggml_graph


def _lower_op(
    op: Operation,
    c_graph: Graph,
    g_graph: GGMLExecutionGraph,
    op_by_id: dict[int, Operation],
    fusion_options: FusionOptions | None = None,
) -> GGMLOpDef:
    if fusion_options is None:
        fusion_options = FusionOptions()

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
        exp = attrs.get("exponent", attrs.get("y", 2))
        attrs["exponent"] = int(exp)
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SQR, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.MEAN:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", -1)
        if dim < 0:
            dim += R
        ggml_dim = R - 1 - dim
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_MEAN, in_ids, out_ids, attrs, op.name)
    elif opcode in (OpCode.SUM, OpCode.AMAX, OpCode.AMIN):
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        axes = attrs.get("axes", None) or attrs.get("dim", -1)
        if isinstance(axes, (tuple, list)):
            dim = axes[0] if len(axes) > 0 else -1
        else:
            dim = int(axes)
        if dim < 0:
            dim += R
        ggml_dim = R - 1 - dim
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SUM_ROWS, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CONTIGUOUS:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.LOG:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_LOG, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SOFTMAX:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SOFT_MAX, in_ids, out_ids, attrs, op.name)
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
    elif opcode == OpCode.SIGMOID:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_SIGMOID)},
            op.name,
        )
    elif opcode == OpCode.HARDSWISH:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_HARDSWISH)},
            op.name,
        )
    elif opcode == OpCode.HARDSIGMOID:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_HARDSIGMOID)},
            op.name,
        )
    elif opcode == OpCode.CLAMP:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_CLAMP,
            in_ids,
            out_ids,
            attrs,
            op.name,
        )
    elif opcode == OpCode.TANH:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_TANH)},
            op.name,
        )
    elif opcode == OpCode.NEG:
        return GGMLOpDef(
            op.id,
            GGMLOpCode.GGML_OP_UNARY,
            in_ids,
            out_ids,
            {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_NEG)},
            op.name,
        )
    elif opcode == OpCode.SIN:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_SIN, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.COS:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_COS, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.BIAS_GELU:
        if fusion_options.enable_bias_gelu:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_CUSTOM_BIAS_GELU, in_ids, out_ids, attrs, op.name
            )
        else:
            return GGMLOpDef(
                op.id,
                GGMLOpCode.GGML_OP_UNARY,
                in_ids,
                out_ids,
                {"unary_op": int(GGMLUnaryOpCode.GGML_UNARY_OP_GELU)},
                op.name,
            )
    elif opcode == OpCode.LAYER_NORM:
        if fusion_options.enable_layer_norm:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_CUSTOM_LAYER_NORM, in_ids, out_ids, attrs, op.name
            )
        else:
            return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_NORM, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.RMS_NORM:
        if fusion_options.enable_rms_norm:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_CUSTOM_RMS_NORM, in_ids, out_ids, attrs, op.name
            )
        else:
            return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RMS_NORM, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SWIGLU:
        if fusion_options.enable_swiglu:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_CUSTOM_SWIGLU, in_ids, out_ids, attrs, op.name
            )
        else:
            return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_GLU, in_ids, out_ids, attrs, op.name)
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
        if "transpose_in1" in op.attributes:
            # In Canonical IR: lhs @ rhs.T (transpose_in1=1) -> GGML mul_mat(rhs, lhs) = lhs @ rhs.T (transpose_in0=0)
            # In Canonical IR: lhs @ rhs (transpose_in1=0) -> GGML mul_mat(rhs.T, lhs) = lhs @ rhs (transpose_in0=1)
            attrs["transpose_in0"] = 0 if op.attributes["transpose_in1"] != 0 else 1
        elif "transpose_in0" in op.attributes:
            attrs["transpose_in0"] = int(op.attributes["transpose_in0"])

        mapped_inputs = [in_ids[1], in_ids[0]]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_MUL_MAT, mapped_inputs, out_ids, attrs, op.name)
    elif opcode == OpCode.CONV2D:
        # PyTorch conv2d: in_ids = [x, weight, bias (optional)]
        # In GGML conv_2d: first arg is weight [KW, KH, IC, OC], second arg is x [W, H, C, N]
        x_id = in_ids[0]
        w_id = in_ids[1]
        mapped_inputs = [w_id, x_id]
        if len(in_ids) > 2:
            mapped_inputs.append(in_ids[2])
        if "stride" in attrs:
            stride = attrs["stride"]
            if isinstance(stride, (list, tuple)):
                attrs["stride_h"] = int(stride[0])
                attrs["stride_w"] = int(stride[1]) if len(stride) > 1 else int(stride[0])
            else:
                attrs["stride_h"] = int(stride)
                attrs["stride_w"] = int(stride)
        if "padding" in attrs:
            padding = attrs["padding"]
            if isinstance(padding, (list, tuple)):
                attrs["pad_h"] = int(padding[0])
                attrs["pad_w"] = int(padding[1]) if len(padding) > 1 else int(padding[0])
            else:
                attrs["pad_h"] = int(padding)
                attrs["pad_w"] = int(padding)
        if "dilation" in attrs:
            dilation = attrs["dilation"]
            if isinstance(dilation, (list, tuple)):
                attrs["dilation_h"] = int(dilation[0])
                attrs["dilation_w"] = int(dilation[1]) if len(dilation) > 1 else int(dilation[0])
            else:
                attrs["dilation_h"] = int(dilation)
                attrs["dilation_w"] = int(dilation)
        groups = attrs.get("groups")
        groups = int(groups) if groups is not None else 1
        w_t = c_graph.get_tensor(w_id)
        x_t = c_graph.get_tensor(x_id)
        is_dw = False
        if (
            len(w_t.shape.dims) == 4
            and len(x_t.shape.dims) == 4
            and w_t.shape.dims[1].is_static()
            and x_t.shape.dims[1].is_static()
        ):
            w_ic = w_t.shape.dims[1].evaluate({})
            x_ic = x_t.shape.dims[1].evaluate({})
            if w_ic == 1 and (groups > 1 or x_ic > 1):
                is_dw = True
        elif groups > 1:
            is_dw = True

        if is_dw:
            return GGMLOpDef(
                op.id, GGMLOpCode.GGML_OP_CONV_2D_DW, mapped_inputs, out_ids, attrs, op.name
            )
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONV_2D, mapped_inputs, out_ids, attrs, op.name)
    elif opcode in (OpCode.MAX_POOL2D, OpCode.AVG_POOL2D):
        attrs["is_max"] = 1 if opcode == OpCode.MAX_POOL2D else 0
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_POOL_2D, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.ADAPTIVE_AVG_POOL2D:
        attrs["is_adaptive"] = 1
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_POOL_2D, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.PAD:
        attrs["pad_w"] = int(attrs.get("pad_w", 0))
        attrs["pad_h"] = int(attrs.get("pad_h", 0))
        attrs["pad_c"] = int(attrs.get("pad_c", 0))
        attrs["pad_n"] = int(attrs.get("pad_n", 0))
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_PAD, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CONCAT:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", attrs.get("axis", attrs.get("dimension", 0)))
        if dim < 0:
            dim += R
        ggml_dim = R - 1 - dim if R > 0 else 0
        attrs["ggml_dim"] = int(ggml_dim)
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONCAT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SLICE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", attrs.get("axis", 0))
        start = attrs.get("start", 0)
        if "start_indices" in attrs:
            s_indices = attrs["start_indices"]
            l_indices = attrs.get("limit_indices", [])
            in_shape = [d.value if isinstance(d, StaticDim) else 1 for d in in_t.shape.dims]
            for i in range(len(s_indices)):
                if s_indices[i] > 0 or (
                    i < len(l_indices) and i < len(in_shape) and l_indices[i] < in_shape[i]
                ):
                    dim = i
                    start = s_indices[i]
                    break
        if dim < 0:
            dim += R
        ggml_dim = R - 1 - dim if R > 0 else 0
        attrs["ggml_dim"] = int(ggml_dim)
        attrs["start"] = int(start)
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_VIEW, in_ids, out_ids, attrs, op.name)
    elif opcode in (OpCode.RESHAPE, OpCode.VIEW, OpCode.SQUEEZE, OpCode.UNSQUEEZE):
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RESHAPE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.PERMUTE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        p = attrs.get("axes") or attrs.get("dims") or attrs.get("permutation") or list(range(R))
        if isinstance(p, tuple):
            p = list(p)
        axes = [0, 1, 2, 3]
        if R == 5 and p[0] == 0:
            # 5D tensor with batch=1: fold leading batch dimension into 4D permute
            q = [x - 1 for x in p[1:]]
            for i in range(4):
                axes[i] = 3 - q.index(3 - i)
        else:
            for i in range(4):
                if i < R:
                    axes[i] = R - 1 - p.index(R - 1 - i)
                else:
                    axes[i] = i
        if any(ax >= 4 for ax in axes):
            return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RESHAPE, in_ids, out_ids, attrs, op.name)
        attrs["axis0"] = axes[0]
        attrs["axis1"] = axes[1]
        attrs["axis2"] = axes[2]
        attrs["axis3"] = axes[3]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_PERMUTE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.TRANSPOSE:
        in_t = c_graph.get_tensor(in_ids[0])
        d0 = attrs.get("dim0", 0)
        d1 = attrs.get("dim1", 1)
        R = len(in_t.shape.dims)
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
                axes[i] = R - 1 - p[R - 1 - i]
            else:
                axes[i] = i
        if R > 4 or any(ax >= 4 for ax in axes):
            return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_RESHAPE, in_ids, out_ids, attrs, op.name)
        attrs["axis0"] = axes[0]
        attrs["axis1"] = axes[1]
        attrs["axis2"] = axes[2]
        attrs["axis3"] = axes[3]
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_PERMUTE, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SLICE:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", 0)
        if dim < 0:
            dim += R
        ggml_dim = max(0, R - 1 - dim)
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_VIEW, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CONCAT:
        in_t = c_graph.get_tensor(in_ids[0])
        R = len(in_t.shape.dims)
        dim = attrs.get("dim", 0)
        if dim < 0:
            dim += R
        ggml_dim = max(0, R - 1 - dim)
        attrs["ggml_dim"] = ggml_dim
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CONCAT, in_ids, out_ids, attrs, op.name)
    elif opcode in (OpCode.EXPAND, OpCode.REPEAT):
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_REPEAT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.EMBEDDING:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_GET_ROWS, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.CAST:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_CPY, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.SDPA:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_FLASH_ATTN_EXT, in_ids, out_ids, attrs, op.name)
    elif opcode == OpCode.ROPE:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_ROPE, in_ids, out_ids, attrs, op.name)
    else:
        return GGMLOpDef(op.id, GGMLOpCode.GGML_OP_NONE, in_ids, out_ids, attrs, op.name)
