from __future__ import annotations

import operator
from typing import Any

import numpy as np
import torch
from torch.export import ExportedProgram
from torch.fx import Node

from ggmlc.frontend.pytorch.operators import get_opcode_for_aten
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import (
    AddDim,
    Dim,
    MulDim,
    Shape,
    StaticDim,
    SymbolDim,
)
from ggmlc.ir.tensor import StorageClass, Tensor


def _symint_to_dim(sym: Any) -> Dim:
    """Convert torch.SymInt or int/expression to Dim."""
    if isinstance(sym, int):
        if sym >= 0:
            return StaticDim(sym)
        return SymbolDim(str(sym))
    if isinstance(sym, torch.SymInt):
        node = sym.node
        expr = node.expr
        # Convert sympy expression to Dim
        import sympy

        def _sympy_to_dim(e: sympy.Expr) -> Dim:
            if isinstance(e, sympy.Integer):
                return StaticDim(int(e))
            if isinstance(e, sympy.Symbol):
                return SymbolDim(str(e.name))
            if isinstance(e, sympy.Add):
                dims = [_sympy_to_dim(arg) for arg in e.args]
                res = dims[0]
                for d in dims[1:]:
                    res = AddDim(res, d)
                return res
            if isinstance(e, sympy.Mul):
                dims = [_sympy_to_dim(arg) for arg in e.args]
                res = dims[0]
                for d in dims[1:]:
                    res = MulDim(res, d)
                return res
            # Fallback string representation
            return SymbolDim(str(e))

        return _sympy_to_dim(expr)
    return SymbolDim(str(sym))


def _torch_shape_to_shape(shape: Any) -> Shape:
    dims = [_symint_to_dim(d) for d in shape]
    return Shape(dims)


def import_exported_program(ep: ExportedProgram, graph_name: str = "main") -> Graph:
    """Imports a torch.export.ExportedProgram into a ggmlc Canonical IR Graph."""
    g = Graph(name=graph_name)
    node_to_tensor: dict[Node, Tensor] = {}
    name_to_tensor: dict[str, Tensor] = {}

    # Extract signature info
    sig = ep.graph_signature
    user_inputs = set(sig.user_inputs)
    lifted_params = dict(sig.inputs_to_parameters)
    lifted_buffers = dict(sig.inputs_to_buffers)
    lifted_constants = dict(getattr(sig, "inputs_to_lifted_tensor_constants", {}))

    # 1. Process placeholder nodes
    for node in ep.graph.nodes:
        if node.op != "placeholder":
            continue

        target_name = node.target
        val = node.meta.get("val")

        if val is None or not isinstance(val, torch.Tensor):
            # Scalar placeholder or non-tensor
            continue

        shape = _torch_shape_to_shape(val.shape)
        dtype = DType.from_torch(val.dtype)

        if target_name in lifted_params:
            param_name = lifted_params[target_name]
            param_tensor = ep.state_dict[param_name]
            t = g.add_tensor(
                name=param_name,
                shape=Shape.from_tuple(tuple(param_tensor.shape)),
                dtype=DType.from_torch(param_tensor.dtype),
                storage=StorageClass.PARAMETER,
                data=param_tensor.detach().cpu().numpy(),
                role="parameter",
            )
            g.parameters.append(t.id)
        elif target_name in lifted_buffers or target_name in lifted_constants:
            buf_name = lifted_buffers.get(target_name, lifted_constants.get(target_name))
            buf_tensor = ep.constants.get(
                buf_name,
                ep.state_dict.get(buf_name, getattr(ep, "tensor_constants", {}).get(buf_name)),
            )
            data = buf_tensor.detach().cpu().numpy() if buf_tensor is not None else None
            t = g.add_tensor(
                name=buf_name,
                shape=Shape.from_tuple(tuple(buf_tensor.shape))
                if buf_tensor is not None
                else shape,
                dtype=DType.from_torch(buf_tensor.dtype) if buf_tensor is not None else dtype,
                storage=StorageClass.CONSTANT,
                data=data,
                role="constant",
            )
            g.parameters.append(t.id)
        elif target_name in user_inputs:
            t = g.add_tensor(
                name=node.name,
                shape=shape,
                dtype=dtype,
                storage=StorageClass.INPUT,
                role="input",
            )
            g.inputs.append(t.id)
        else:
            # General input
            t = g.add_tensor(
                name=node.name,
                shape=shape,
                dtype=dtype,
                storage=StorageClass.INPUT,
                role="input",
            )
            g.inputs.append(t.id)

        node_to_tensor[node] = t
        name_to_tensor[node.name] = t

    # 2. Process call_function nodes
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue

        target_str = str(node.target)
        if "sym_size" in target_str or "sym_numel" in target_str:
            # Symbolic scalar query node (e.g. B, S = x.shape)
            continue

        if "getitem" in target_str or node.target is operator.getitem:
            parent = node.args[0]
            if isinstance(parent, Node):
                parent_target_str = str(parent.target)
                if "split" in parent_target_str and isinstance(node.args[1], int):
                    idx = int(node.args[1])
                    split_input = parent.args[0]
                    split_size = parent.args[1]
                    dim = (
                        int(parent.args[2])
                        if len(parent.args) > 2 and parent.args[2] is not None
                        else 0
                    )
                    if isinstance(split_size, (list, tuple)):
                        start = sum(split_size[:idx])
                        end = start + split_size[idx]
                    else:
                        sz = int(split_size)
                        start = idx * sz
                        end = (idx + 1) * sz
                    val = node.meta.get("val")
                    shape = (
                        _torch_shape_to_shape(val.shape)
                        if isinstance(val, torch.Tensor)
                        else Shape([])
                    )
                    dtype = (
                        DType.from_torch(val.dtype) if isinstance(val, torch.Tensor) else DType.F32
                    )
                    out_t = g.add_tensor(
                        name=node.name,
                        shape=shape,
                        dtype=dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    node_to_tensor[node] = out_t
                    name_to_tensor[node.name] = out_t
                    g.add_op(
                        opcode=OpCode.SLICE,
                        inputs=[node_to_tensor[split_input].id],
                        outputs=[out_t.id],
                        attributes={"dim": dim, "start": start, "end": end, "step": 1},
                        name=node.name,
                    )
                    continue
                elif parent in node_to_tensor:
                    node_to_tensor[node] = node_to_tensor[parent]
                    name_to_tensor[node.name] = node_to_tensor[parent]
                    continue

        if "lift" in target_str:
            arg = node.args[0]
            if isinstance(arg, torch.Tensor):
                out_t = g.add_tensor(
                    name=node.name,
                    shape=Shape.from_tuple(tuple(arg.shape)),
                    dtype=DType.from_torch(arg.dtype),
                    storage=StorageClass.CONSTANT,
                    data=arg.detach().cpu().numpy(),
                    role="constant",
                )
                g.parameters.append(out_t.id)
                node_to_tensor[node] = out_t
                name_to_tensor[node.name] = out_t
                continue
            elif isinstance(arg, Node) and arg in node_to_tensor:
                node_to_tensor[node] = node_to_tensor[arg]
                name_to_tensor[node.name] = node_to_tensor[arg]
                continue

        if (
            "split" in target_str
            or "assert" in target_str
            or "check" in target_str
            or "to.dtype" in target_str
            or "to.device" in target_str
            or "to.dtype_device" in target_str
            or "_to_copy" in target_str
            or "clone" in target_str
            or "detach" in target_str
            or "alias" in target_str
            or "contiguous" in target_str
        ):
            if any(
                k in target_str
                for k in (
                    "to.dtype",
                    "to.device",
                    "to.dtype_device",
                    "_to_copy",
                    "clone",
                    "detach",
                    "alias",
                    "contiguous",
                )
            ):
                parent = node.args[0]
                if isinstance(parent, Node) and parent in node_to_tensor:
                    node_to_tensor[node] = node_to_tensor[parent]
                    name_to_tensor[node.name] = node_to_tensor[parent]
            continue

        if "dropout" in target_str:
            parent = node.args[0]
            if isinstance(parent, Node) and parent in node_to_tensor:
                node_to_tensor[node] = node_to_tensor[parent]
                name_to_tensor[node.name] = node_to_tensor[parent]
                continue

        if "pad" in target_str:
            parent = node.args[0]
            pad_spec = node.args[1] if len(node.args) > 1 else []
            if (
                isinstance(parent, Node)
                and parent in node_to_tensor
                and isinstance(pad_spec, (list, tuple))
                and all(p == 0 for p in pad_spec)
            ):
                node_to_tensor[node] = node_to_tensor[parent]
                name_to_tensor[node.name] = node_to_tensor[parent]
                continue

        if "linalg_vector_norm" in target_str or (
            target_str.startswith("aten.norm")
            and "layer_norm" not in target_str
            and "rms_norm" not in target_str
        ):
            in_t = node_to_tensor[node.args[0]]
            val = node.meta.get("val")
            out_shape = (
                _torch_shape_to_shape(val.shape)
                if val is not None and isinstance(val, torch.Tensor)
                else in_t.shape
            )
            out_dtype = (
                DType.from_torch(val.dtype)
                if val is not None and isinstance(val, torch.Tensor)
                else in_t.dtype
            )
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
            if isinstance(dim, (list, tuple)):
                dim = dim[0] if len(dim) > 0 else -1
            dim = int(dim) if dim is not None else -1
            keepdim = (
                bool(node.args[3])
                if len(node.args) > 3
                else bool(node.kwargs.get("keepdim", False))
            )

            # 1. x_sq = x * x
            t_sq = g.add_tensor(
                name=f"{node.name}_sq",
                shape=in_t.shape,
                dtype=in_t.dtype,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.MUL,
                inputs=[in_t.id, in_t.id],
                outputs=[t_sq.id],
                name=f"{node.name}_mul_sq",
            )

            # 2. t_sum = sum(x_sq, dim=dim, keepdim=keepdim)
            t_sum = g.add_tensor(
                name=f"{node.name}_sum",
                shape=out_shape,
                dtype=out_dtype,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.SUM,
                inputs=[t_sq.id],
                outputs=[t_sum.id],
                attributes={"dim": dim, "keepdim": keepdim},
                name=f"{node.name}_sum_op",
            )

            # 3. out = sqrt(t_sum)
            out_t = g.add_tensor(
                name=node.name,
                shape=out_shape,
                dtype=out_dtype,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.SQRT,
                inputs=[t_sum.id],
                outputs=[out_t.id],
                name=node.name,
            )
            node_to_tensor[node] = out_t
            name_to_tensor[node.name] = out_t
            continue

        if "roll" in target_str:
            curr_tensor = node_to_tensor[node.args[0]]
            shifts = node.args[1]
            dims = node.args[2] if len(node.args) > 2 else [0]
            if not isinstance(shifts, (list, tuple)):
                shifts = [shifts]
            if not isinstance(dims, (list, tuple)):
                dims = [dims]
            for s, d in zip(shifts, dims):
                s = int(s)
                d = int(d)
                curr_shape = list(curr_tensor.shape.dims)
                dim_len = (
                    int(curr_shape[d].value)
                    if hasattr(curr_shape[d], "value")
                    else int(curr_shape[d])
                )
                k = (-s) % dim_len
                if k == 0:
                    continue
                # slice 1: [k:dim_len]
                s1_shape = [
                    d_elem.value if hasattr(d_elem, "value") else int(d_elem)
                    for d_elem in curr_shape
                ]
                s1_shape[d] = dim_len - k
                t1 = g.add_tensor(
                    name=f"{node.name}_roll_s1",
                    shape=Shape.from_tuple(tuple(s1_shape)),
                    dtype=curr_tensor.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_node(
                    opcode=OpCode.SLICE,
                    inputs=[curr_tensor.id],
                    outputs=[t1.id],
                    attributes={"dim": d, "start": k, "end": dim_len, "step": 1},
                )
                # slice 2: [0:k]
                s2_shape = [
                    d_elem.value if hasattr(d_elem, "value") else int(d_elem)
                    for d_elem in curr_shape
                ]
                s2_shape[d] = k
                t2 = g.add_tensor(
                    name=f"{node.name}_roll_s2",
                    shape=Shape.from_tuple(tuple(s2_shape)),
                    dtype=curr_tensor.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_node(
                    opcode=OpCode.SLICE,
                    inputs=[curr_tensor.id],
                    outputs=[t2.id],
                    attributes={"dim": d, "start": 0, "end": k, "step": 1},
                )
                # concat [t1, t2] along dim d
                out_rolled = g.add_tensor(
                    name=f"{node.name}_rolled",
                    shape=Shape.from_tuple(
                        tuple(
                            [
                                d_elem.value if hasattr(d_elem, "value") else int(d_elem)
                                for d_elem in curr_shape
                            ]
                        )
                    ),
                    dtype=curr_tensor.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_node(
                    opcode=OpCode.CONCAT,
                    inputs=[t1.id, t2.id],
                    outputs=[out_rolled.id],
                    attributes={"dim": d},
                )
                curr_tensor = out_rolled
            node_to_tensor[node] = curr_tensor
            name_to_tensor[node.name] = curr_tensor
            continue

        if any(
            target_str.endswith(f".{op}.{sfx}")
            for op in (
                "cumsum",
                "ge",
                "gt",
                "ne",
                "eq",
                "le",
                "lt",
                "diff",
                "zeros",
                "ones",
                "new_ones",
                "new_zeros",
                "full_like",
                "ones_like",
                "zeros_like",
                "__and__",
                "bitwise_and",
                "logical_and",
            )
            for sfx in ("default", "Scalar", "Tensor")
        ):
            val = node.meta.get("val")
            if val is not None and isinstance(val, torch.Tensor):
                shape = _torch_shape_to_shape(val.shape)
                dtype = DType.from_torch(val.dtype)
                if "cumsum" in target_str:
                    seq_len = val.shape[-1]
                    arr = np.tile(np.arange(1, seq_len + 1, dtype=np.int64), (val.shape[0], 1))
                    data = arr
                elif "ne" in target_str or "ge" in target_str:
                    data = np.ones(val.shape, dtype=np.int64 if dtype == DType.I64 else np.int32)
                elif "zeros" in target_str:
                    data = np.zeros(
                        val.shape,
                        dtype=np.int64
                        if dtype == DType.I64
                        else (np.int32 if dtype == DType.I32 else np.float32),
                    )
                elif "le" in target_str or "lt" in target_str:
                    h, w = (
                        (val.shape[-2], val.shape[-1])
                        if len(val.shape) >= 2
                        else (val.shape[0], val.shape[0])
                    )
                    data = np.tril(np.ones((h, w), dtype=np.bool_))
                    while data.ndim < len(val.shape):
                        data = np.expand_dims(data, 0)
                else:
                    data = np.ones(val.shape, dtype=np.float32)

                out_t = g.add_tensor(
                    name=node.name,
                    shape=shape,
                    dtype=dtype,
                    storage=StorageClass.CONSTANT,
                    data=data,
                    role="constant",
                )
                g.parameters.append(out_t.id)
                node_to_tensor[node] = out_t
                name_to_tensor[node.name] = out_t
                continue

        opcode = get_opcode_for_aten(node.target)
        if opcode is None:
            raise NotImplementedError(
                f"Unsupported ATen operator: {node.target} at FX node '{node.name}'. "
                f"No Canonical IR lowering registered."
            )

        val = node.meta.get("val")
        if val is None or not isinstance(val, torch.Tensor):
            # Handle non-tensor or multiple outputs if tuple
            if isinstance(val, (tuple, list)):
                # Tuple of tensors
                shape = _torch_shape_to_shape(val[0].shape)
                dtype = DType.from_torch(val[0].dtype)
            else:
                shape = Shape([StaticDim(1)])
                dtype = DType.F32
        else:
            shape = _torch_shape_to_shape(val.shape)
            dtype = DType.from_torch(val.dtype)

        # Collect input tensor IDs and attributes
        input_tensor_ids: list[int] = []
        attributes: dict[str, Any] = {}

        if opcode == OpCode.TRANSPOSE:
            # aten.transpose.int(self, dim0, dim1) or aten.t(self)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 2:
                attributes["dim0"] = int(node.args[1])
                attributes["dim1"] = int(node.args[2])
            else:
                attributes["dim0"] = 0
                attributes["dim1"] = 1
        elif opcode == OpCode.PERMUTE:
            # aten.permute.default(self, dims)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            attributes["dims"] = [int(d) for d in node.args[1]]
        elif opcode == OpCode.SLICE:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if "select" in str(node.target):
                dim = int(node.args[1]) if len(node.args) > 1 else 0
                index = int(node.args[2]) if len(node.args) > 2 else 0
                attributes["dim"] = dim
                attributes["start"] = index
                attributes["end"] = index + 1
                attributes["step"] = 1
                attributes["is_select"] = 1
            else:
                dim = int(node.args[1]) if len(node.args) > 1 else 0
                start = int(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0
                end = (
                    int(node.args[3])
                    if len(node.args) > 3
                    and node.args[3] is not None
                    and node.args[3] < 9223372036854775800
                    else -1
                )
                step = int(node.args[4]) if len(node.args) > 4 and node.args[4] is not None else 1
                attributes["dim"] = dim
                attributes["start"] = start
                attributes["end"] = end
                attributes["step"] = step
        elif opcode == OpCode.MEAN:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            dim = node.args[1] if len(node.args) > 1 else -1
            attributes["dim"] = int(dim[0]) if isinstance(dim, (list, tuple)) else int(dim)
            attributes["keepdim"] = 1 if len(node.args) > 2 and node.args[2] else 0
        elif opcode == OpCode.CONCAT:
            # aten.cat.default(tensors, dim=0)
            tensors_arg = node.args[0]
            for sub_node in tensors_arg:
                input_tensor_ids.append(node_to_tensor[sub_node].id)
            attributes["dim"] = int(node.args[1]) if len(node.args) > 1 else 0
        elif opcode == OpCode.EXPAND:
            # aten.expand.default(self, size)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            dims = []
            for d in node.args[1]:
                if isinstance(d, Node):
                    dims.append(_symint_to_dim(d.meta.get("val")))
                else:
                    dims.append(_symint_to_dim(d))
            attributes["shape"] = tuple(dims)
        elif opcode in (OpCode.SQUEEZE, OpCode.UNSQUEEZE):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 1 and node.args[1] is not None:
                attributes["dim"] = int(node.args[1])
        elif opcode in (OpCode.RESHAPE, OpCode.VIEW):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            dims = []
            if "unflatten" in target_str:
                dims = list(shape.dims)
            elif len(node.args) > 1:
                shape_arg = node.args[1]
                if isinstance(shape_arg, (list, tuple)):
                    for d in shape_arg:
                        if isinstance(d, Node):
                            dims.append(_symint_to_dim(d.meta.get("val")))
                        else:
                            dims.append(_symint_to_dim(d))
                else:
                    dims.append(_symint_to_dim(shape_arg))
            if not dims and shape:
                dims = list(shape.dims)
            attributes["shape"] = tuple(dims)
        elif opcode == OpCode.ARGMAX:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            dim = int(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else -1
            keepdim = bool(node.args[2]) if len(node.args) > 2 else False
            attributes["dim"] = dim
            attributes["keepdim"] = keepdim
        elif opcode == OpCode.SOFTMAX:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            attributes["dim"] = int(node.args[1]) if len(node.args) > 1 else -1
        elif opcode == OpCode.MATMUL:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            input_tensor_ids.append(node_to_tensor[node.args[1]].id)
            attributes["transpose_in0"] = 1
        elif opcode == OpCode.LINEAR:
            if "addmm" in target_str:
                # aten.addmm.default(bias, input, weight)
                bias_node = node.args[0]
                input_node = node.args[1]
                weight_node = node.args[2]
                input_tensor_ids.append(node_to_tensor[input_node].id)
                input_tensor_ids.append(node_to_tensor[weight_node].id)
                if isinstance(bias_node, Node) and bias_node in node_to_tensor:
                    input_tensor_ids.append(node_to_tensor[bias_node].id)
                attributes["is_addmm"] = 1
            else:
                # aten.linear(input, weight, bias)
                input_tensor_ids.append(node_to_tensor[node.args[0]].id)
                input_tensor_ids.append(node_to_tensor[node.args[1]].id)
                if len(node.args) > 2 and node.args[2] is not None:
                    input_tensor_ids.append(node_to_tensor[node.args[2]].id)
        elif opcode == OpCode.EMBEDDING:
            # aten.embedding.default(weight, indices) or aten.index.Tensor(weight, [indices])
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            indices_arg = node.args[1]
            if isinstance(indices_arg, (list, tuple)):
                non_trivial = [
                    idx_node
                    for idx_node in indices_arg
                    if isinstance(idx_node, Node) and idx_node in node_to_tensor
                ]
                if len(non_trivial) > 1:
                    # Take the sequence index (last dimension) for 2D/3D tensor pooling
                    input_tensor_ids.append(node_to_tensor[non_trivial[-1]].id)
                elif len(non_trivial) == 1:
                    input_tensor_ids.append(node_to_tensor[non_trivial[0]].id)
            elif isinstance(indices_arg, Node) and indices_arg in node_to_tensor:
                input_tensor_ids.append(node_to_tensor[indices_arg].id)
        elif opcode == OpCode.RMS_NORM:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 2 and isinstance(node.args[2], Node):
                input_tensor_ids.append(node_to_tensor[node.args[2]].id)
            eps = (
                node.args[3]
                if len(node.args) > 3 and isinstance(node.args[3], (int, float))
                else 1e-5
            )
            attributes["eps"] = float(eps)
        elif opcode == OpCode.LAYER_NORM:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 2 and isinstance(node.args[2], Node):
                input_tensor_ids.append(node_to_tensor[node.args[2]].id)
            if len(node.args) > 3 and isinstance(node.args[3], Node):
                input_tensor_ids.append(node_to_tensor[node.args[3]].id)
            eps = (
                node.args[4]
                if len(node.args) > 4 and isinstance(node.args[4], (int, float))
                else 1e-5
            )
            attributes["eps"] = float(eps)
        elif opcode == OpCode.SDPA:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            input_tensor_ids.append(node_to_tensor[node.args[1]].id)
            input_tensor_ids.append(node_to_tensor[node.args[2]].id)
            if len(node.args) > 3 and isinstance(node.args[3], Node):
                input_tensor_ids.append(node_to_tensor[node.args[3]].id)
            is_causal = node.kwargs.get("is_causal") or (
                node.args[5] if len(node.args) > 5 and isinstance(node.args[5], bool) else False
            )
            if is_causal:
                attributes["is_causal"] = 1
            scale = node.kwargs.get("scale") or (
                node.args[6]
                if len(node.args) > 6 and isinstance(node.args[6], (int, float))
                else None
            )
            if scale is not None:
                attributes["scale"] = float(scale)
        elif opcode == OpCode.CONTIGUOUS:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
        elif opcode == OpCode.POW:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 1:
                if isinstance(node.args[1], (int, float)):
                    attributes["exponent"] = float(node.args[1])
                elif isinstance(node.args[1], Node):
                    input_tensor_ids.append(node_to_tensor[node.args[1]].id)
        elif opcode == OpCode.ROPE:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            input_tensor_ids.append(node_to_tensor[node.args[1]].id)
            if "n_dims" in node.kwargs:
                attributes["n_dims"] = int(node.kwargs["n_dims"])
        elif opcode in (OpCode.HARDSWISH, OpCode.HARDSIGMOID):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
        elif opcode == OpCode.CLAMP:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            target_str = str(node.target)
            if "relu6" in target_str:
                attributes["min"] = 0.0
                attributes["max"] = 6.0
            elif "hardtanh" in target_str:
                min_v = node.args[1] if len(node.args) > 1 and node.args[1] is not None else -1.0
                max_v = node.args[2] if len(node.args) > 2 and node.args[2] is not None else 1.0
                attributes["min"] = float(min_v)
                attributes["max"] = float(max_v)
            else:
                min_v = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
                max_v = node.kwargs.get("max", node.args[2] if len(node.args) > 2 else None)
                if min_v is not None and isinstance(min_v, (int, float)):
                    attributes["min"] = float(min_v)
                if max_v is not None and isinstance(max_v, (int, float)):
                    attributes["max"] = float(max_v)
        elif opcode == OpCode.CONV2D:
            # aten.convolution.default(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            input_tensor_ids.append(node_to_tensor[node.args[1]].id)
            if (
                len(node.args) > 2
                and isinstance(node.args[2], Node)
                and node.args[2] in node_to_tensor
            ):
                input_tensor_ids.append(node_to_tensor[node.args[2]].id)
            stride = node.args[3] if len(node.args) > 3 and node.args[3] else [1, 1]
            padding = node.args[4] if len(node.args) > 4 and node.args[4] else [0, 0]
            dilation = node.args[5] if len(node.args) > 5 and node.args[5] else [1, 1]
            groups = node.kwargs.get("groups")
            if groups is None:
                if len(node.args) >= 9:
                    groups = node.args[8]
                elif len(node.args) >= 7:
                    groups = node.args[6]
            if groups is None:
                in_t = node_to_tensor[node.args[0]]
                w_t = node_to_tensor[node.args[1]]
                if (
                    len(in_t.shape.dims) == 4
                    and len(w_t.shape.dims) == 4
                    and in_t.shape.dims[1].is_static()
                    and w_t.shape.dims[1].is_static()
                ):
                    ic_val = in_t.shape.dims[1].evaluate({})
                    w_ic_val = w_t.shape.dims[1].evaluate({})
                    if ic_val > 1 and w_ic_val == 1:
                        groups = int(ic_val)
            if "conv1d" in target_str:
                attributes["stride_w"] = (
                    int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
                )
                attributes["stride_h"] = 1
                attributes["pad_w"] = (
                    int(padding[0]) if isinstance(padding, (list, tuple)) else int(padding)
                )
                attributes["pad_h"] = 0
                attributes["dilation_w"] = (
                    int(dilation[0]) if isinstance(dilation, (list, tuple)) else int(dilation)
                )
                attributes["dilation_h"] = 1
                attributes["is_1d"] = 1
            else:
                attributes["stride_h"] = (
                    int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
                )
                attributes["stride_w"] = (
                    int(stride[1])
                    if isinstance(stride, (list, tuple)) and len(stride) > 1
                    else attributes["stride_h"]
                )
                attributes["pad_h"] = (
                    int(padding[0]) if isinstance(padding, (list, tuple)) else int(padding)
                )
                attributes["pad_w"] = (
                    int(padding[1])
                    if isinstance(padding, (list, tuple)) and len(padding) > 1
                    else attributes["pad_h"]
                )
                attributes["dilation_h"] = (
                    int(dilation[0]) if isinstance(dilation, (list, tuple)) else int(dilation)
                )
                attributes["dilation_w"] = (
                    int(dilation[1])
                    if isinstance(dilation, (list, tuple)) and len(dilation) > 1
                    else attributes["dilation_h"]
                )
            attributes["groups"] = groups
            # Check if this is a grouped convolution that needs decomposition (1 < groups < in_channels)
            in_t_candidate = node_to_tensor[node.args[0]]
            w_t_candidate = node_to_tensor[node.args[1]]
            bias_t_candidate = (
                node_to_tensor.get(node.args[2])
                if len(node.args) > 2
                and isinstance(node.args[2], Node)
                and node.args[2] in node_to_tensor
                else None
            )
            if (
                groups is not None
                and int(groups) > 1
                and len(in_t_candidate.shape.dims) == 4
                and len(w_t_candidate.shape.dims) == 4
                and in_t_candidate.shape.dims[1].is_static()
                and w_t_candidate.shape.dims[0].is_static()
            ):
                cin_val = int(in_t_candidate.shape.dims[1].evaluate({}))
                cout_val = int(w_t_candidate.shape.dims[0].evaluate({}))
                g_val = int(groups)
                if 1 < g_val < cin_val:
                    cin_per_group = cin_val // g_val
                    cout_per_group = cout_val // g_val
                    out_group_tensors = []
                    for g_idx in range(g_val):
                        # Slice input channels
                        x_g_out = g.add_tensor(
                            name=f"{node.name}_x_g{g_idx}",
                            shape=Shape(
                                [
                                    in_t_candidate.shape.dims[0],
                                    StaticDim(cin_per_group),
                                    in_t_candidate.shape.dims[2],
                                    in_t_candidate.shape.dims[3],
                                ]
                            ),
                            dtype=in_t_candidate.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.SLICE,
                            inputs=[in_t_candidate.id],
                            outputs=[x_g_out.id],
                            attributes={
                                "dim": 1,
                                "start": g_idx * cin_per_group,
                                "end": (g_idx + 1) * cin_per_group,
                                "step": 1,
                            },
                            name=f"{node.name}_x_slice_g{g_idx}",
                        )
                        # Slice weight filters
                        w_g_data = (
                            w_t_candidate.data[
                                g_idx * cout_per_group : (g_idx + 1) * cout_per_group
                            ]
                            if w_t_candidate.data is not None
                            else None
                        )
                        w_g_out = g.add_tensor(
                            name=f"{w_t_candidate.name}_g{g_idx}",
                            shape=Shape(
                                [
                                    StaticDim(cout_per_group),
                                    w_t_candidate.shape.dims[1],
                                    w_t_candidate.shape.dims[2],
                                    w_t_candidate.shape.dims[3],
                                ]
                            ),
                            dtype=w_t_candidate.dtype,
                            storage=StorageClass.CONSTANT
                            if w_t_candidate.storage == StorageClass.CONSTANT
                            else StorageClass.ACTIVATION,
                            data=w_g_data,
                            role="parameter",
                        )
                        if (
                            w_t_candidate.storage == StorageClass.CONSTANT
                            and w_t_candidate.data is not None
                        ):
                            g.parameters.append(w_g_out.id)
                            w_slice_id = w_g_out.id
                        else:
                            g.add_op(
                                opcode=OpCode.SLICE,
                                inputs=[w_t_candidate.id],
                                outputs=[w_g_out.id],
                                attributes={
                                    "dim": 0,
                                    "start": g_idx * cout_per_group,
                                    "end": (g_idx + 1) * cout_per_group,
                                    "step": 1,
                                },
                                name=f"{node.name}_w_slice_g{g_idx}",
                            )
                            w_slice_id = w_g_out.id

                        conv_g_inputs = [x_g_out.id, w_slice_id]
                        if bias_t_candidate is not None:
                            if (
                                bias_t_candidate.storage == StorageClass.CONSTANT
                                and bias_t_candidate.data is not None
                            ):
                                bias_g_t = g.add_tensor(
                                    name=f"{bias_t_candidate.name}_g{g_idx}",
                                    shape=Shape([StaticDim(cout_per_group)]),
                                    dtype=bias_t_candidate.dtype,
                                    storage=StorageClass.CONSTANT,
                                    data=bias_t_candidate.data[
                                        g_idx * cout_per_group : (g_idx + 1) * cout_per_group
                                    ],
                                    role="parameter",
                                )
                                g.parameters.append(bias_g_t.id)
                                conv_g_inputs.append(bias_g_t.id)
                            else:
                                bias_g_t = g.add_tensor(
                                    name=f"{bias_t_candidate.name}_g{g_idx}",
                                    shape=Shape([StaticDim(cout_per_group)]),
                                    dtype=bias_t_candidate.dtype,
                                    storage=StorageClass.ACTIVATION,
                                )
                                g.add_op(
                                    opcode=OpCode.SLICE,
                                    inputs=[bias_t_candidate.id],
                                    outputs=[bias_g_t.id],
                                    attributes={
                                        "dim": 0,
                                        "start": g_idx * cout_per_group,
                                        "end": (g_idx + 1) * cout_per_group,
                                        "step": 1,
                                    },
                                    name=f"{node.name}_bias_slice_g{g_idx}",
                                )
                                conv_g_inputs.append(bias_g_t.id)

                        group_conv_attrs = dict(attributes)
                        group_conv_attrs["groups"] = 1
                        out_g_t = g.add_tensor(
                            name=f"{node.name}_conv_g{g_idx}",
                            shape=Shape(
                                [
                                    shape.dims[0],
                                    StaticDim(cout_per_group),
                                    shape.dims[2],
                                    shape.dims[3],
                                ]
                            ),
                            dtype=dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.CONV2D,
                            inputs=conv_g_inputs,
                            outputs=[out_g_t.id],
                            attributes=group_conv_attrs,
                            name=f"{node.name}_conv_g{g_idx}",
                        )
                        out_group_tensors.append(out_g_t.id)

                    out_t = g.add_tensor(
                        name=node.name,
                        shape=shape,
                        dtype=dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    node_to_tensor[node] = out_t
                    name_to_tensor[node.name] = out_t
                    g.add_op(
                        opcode=OpCode.CONCAT,
                        inputs=out_group_tensors,
                        outputs=[out_t.id],
                        attributes={"dim": 1},
                        name=node.name,
                    )
                    continue
        elif opcode in (OpCode.MAX_POOL2D, OpCode.AVG_POOL2D):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            ksize = node.args[1] if len(node.args) > 1 else [2, 2]
            stride = node.args[2] if len(node.args) > 2 and node.args[2] else ksize
            padding = node.args[3] if len(node.args) > 3 and node.args[3] else [0, 0]
            attributes["ksize_h"] = (
                int(ksize[0]) if isinstance(ksize, (list, tuple)) else int(ksize)
            )
            attributes["ksize_w"] = (
                int(ksize[1])
                if isinstance(ksize, (list, tuple)) and len(ksize) > 1
                else attributes["ksize_h"]
            )
            attributes["stride_h"] = (
                int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
            )
            attributes["stride_w"] = (
                int(stride[1])
                if isinstance(stride, (list, tuple)) and len(stride) > 1
                else attributes["stride_h"]
            )
            attributes["pad_h"] = (
                int(padding[0]) if isinstance(padding, (list, tuple)) else int(padding)
            )
            attributes["pad_w"] = (
                int(padding[1])
                if isinstance(padding, (list, tuple)) and len(padding) > 1
                else attributes["pad_h"]
            )
            attributes["is_max"] = 1 if opcode == OpCode.MAX_POOL2D else 0
        elif opcode == OpCode.ADAPTIVE_AVG_POOL2D:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            out_sz = node.args[1] if len(node.args) > 1 else [1, 1]
            attributes["out_h"] = (
                int(out_sz[0]) if isinstance(out_sz, (list, tuple)) else int(out_sz)
            )
            attributes["out_w"] = (
                int(out_sz[1])
                if isinstance(out_sz, (list, tuple)) and len(out_sz) > 1
                else attributes["out_h"]
            )
        elif opcode == OpCode.ARANGE:
            target_dtype = DType.I32
            np_dtype = np.int32
            if "val" in node.meta and hasattr(node.meta["val"], "dtype"):
                target_dtype = DType.from_torch(node.meta["val"].dtype)
                np_dtype = np.float32 if target_dtype == DType.F32 else np.int32
            elif "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
                target_dtype = DType.from_torch(node.kwargs["dtype"])
                np_dtype = np.float32 if target_dtype == DType.F32 else np.int32

            val = node.meta.get("val")
            shape = (
                _torch_shape_to_shape(val.shape)
                if val is not None and isinstance(val, torch.Tensor)
                else None
            )

            try:
                if len(node.args) == 1:
                    end = int(node.args[0])
                    arr = np.arange(end, dtype=np_dtype)
                elif len(node.args) == 2:
                    start, end = int(node.args[0]), int(node.args[1])
                    arr = np.arange(start, end, dtype=np_dtype)
                else:
                    start, end, step = int(node.args[0]), int(node.args[1]), int(node.args[2])
                    arr = np.arange(start, end, step, dtype=np_dtype)
            except (TypeError, ValueError):
                max_len = 2048
                arr = np.arange(max_len, dtype=np_dtype)

            if shape is None:
                shape = Shape.from_tuple(arr.shape)

            out_t = g.add_tensor(
                name=node.name,
                shape=shape,
                dtype=target_dtype,
                storage=StorageClass.CONSTANT,
                data=arr,
                role="constant",
            )
            g.parameters.append(out_t.id)
            node_to_tensor[node] = out_t
            name_to_tensor[node.name] = out_t
            continue
        elif opcode == OpCode.REPEAT:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 1:
                rep_arg = node.args[1]
                if isinstance(rep_arg, (list, tuple)):
                    attributes["repeats"] = str([int(r) for r in rep_arg])
                elif rep_arg is not None:
                    attributes["repeats"] = str(int(rep_arg))
            if len(node.args) > 2 and node.args[2] is not None:
                attributes["dim"] = int(node.args[2])
        elif opcode == OpCode.BATCH_NORM:
            # aten.batch_norm.default(input, weight, bias, running_mean, running_var, training, momentum, eps, ...)
            in_t = node_to_tensor[node.args[0]]
            w_t = (
                node_to_tensor.get(node.args[1])
                if len(node.args) > 1 and isinstance(node.args[1], Node)
                else None
            )
            b_t = (
                node_to_tensor.get(node.args[2])
                if len(node.args) > 2 and isinstance(node.args[2], Node)
                else None
            )
            mean_t = (
                node_to_tensor.get(node.args[3])
                if len(node.args) > 3 and isinstance(node.args[3], Node)
                else None
            )
            var_t = (
                node_to_tensor.get(node.args[4])
                if len(node.args) > 4 and isinstance(node.args[4], Node)
                else None
            )
            eps = (
                float(node.args[7])
                if len(node.args) > 7 and isinstance(node.args[7], (int, float))
                else 1e-5
            )

            if (
                mean_t is not None
                and mean_t.data is not None
                and var_t is not None
                and var_t.data is not None
            ):
                mean_data = mean_t.data.astype(np.float32)
                var_data = var_t.data.astype(np.float32)
                w_data = (
                    w_t.data.astype(np.float32)
                    if (w_t and w_t.data is not None)
                    else np.ones_like(mean_data)
                )
                b_data = (
                    b_t.data.astype(np.float32)
                    if (b_t and b_t.data is not None)
                    else np.zeros_like(mean_data)
                )

                inv_std = 1.0 / np.sqrt(var_data + eps)
                scale_val = (w_data * inv_std).reshape(1, -1, 1, 1)
                shift_val = (b_data - mean_data * (w_data * inv_std)).reshape(1, -1, 1, 1)

                scale_t = g.add_tensor(
                    name=f"{node.name}_scale",
                    shape=Shape.from_tuple(scale_val.shape),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=scale_val,
                    role="constant",
                )
                g.parameters.append(scale_t.id)

                shift_t = g.add_tensor(
                    name=f"{node.name}_shift",
                    shape=Shape.from_tuple(shift_val.shape),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=shift_val,
                    role="constant",
                )
                g.parameters.append(shift_t.id)

                mul_out = g.add_tensor(
                    name=f"{node.name}_scaled",
                    shape=in_t.shape,
                    dtype=in_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.MUL,
                    inputs=[in_t.id, scale_t.id],
                    outputs=[mul_out.id],
                    name=f"{node.name}_mul",
                )

                out_t = g.add_tensor(
                    name=node.name,
                    shape=shape,
                    dtype=dtype,
                    storage=StorageClass.ACTIVATION,
                )
                node_to_tensor[node] = out_t
                name_to_tensor[node.name] = out_t
                g.add_op(
                    opcode=OpCode.ADD,
                    inputs=[mul_out.id, shift_t.id],
                    outputs=[out_t.id],
                    name=node.name,
                )
                continue
        elif opcode == OpCode.GATHER:
            # In MiniLM / BERT: gather(position_ids, dim=1, index)
            in_t = node_to_tensor[node.args[0]]
            dim = int(node.args[1]) if len(node.args) > 1 else 0
            val = node.meta.get("val")
            seq_len = val.shape[dim] if val is not None and isinstance(val, torch.Tensor) else 16
            out_t = g.add_tensor(
                name=node.name,
                shape=shape,
                dtype=dtype,
                storage=StorageClass.ACTIVATION,
            )
            node_to_tensor[node] = out_t
            name_to_tensor[node.name] = out_t
            g.add_op(
                opcode=OpCode.SLICE,
                inputs=[in_t.id],
                outputs=[out_t.id],
                attributes={"dim": dim, "start": 0, "end": seq_len, "step": 1},
                name=node.name,
            )
            continue
        elif opcode == OpCode.SUB and "rsub" in target_str:
            # rsub(self, other) computes (other - self)
            rsub_args = [node.args[1], node.args[0]]
            for arg_idx, arg in enumerate(rsub_args):
                if isinstance(arg, Node):
                    if arg in node_to_tensor:
                        input_tensor_ids.append(node_to_tensor[arg].id)
                    else:
                        raise RuntimeError(f"Referenced node {arg.name} was not imported.")
                elif isinstance(arg, (int, float, bool)):
                    c_name = f"const_{node.name}_arg{arg_idx}"
                    dt = (
                        DType.F32
                        if isinstance(arg, float)
                        else (DType.I64 if isinstance(arg, int) else DType.BOOL)
                    )
                    np_val = np.array(arg, dtype=np.float32 if dt == DType.F32 else np.int64)
                    c_t = g.add_tensor(
                        name=c_name,
                        shape=Shape([]),
                        dtype=dt,
                        storage=StorageClass.CONSTANT,
                        data=np_val,
                    )
                    g.parameters.append(c_t.id)
                    input_tensor_ids.append(c_t.id)
        else:
            # Default generic arg parsing
            for arg_idx, arg in enumerate(node.args):
                if isinstance(arg, Node):
                    if arg in node_to_tensor:
                        input_tensor_ids.append(node_to_tensor[arg].id)
                    else:
                        raise RuntimeError(f"Referenced node {arg.name} was not imported.")
                elif isinstance(arg, (int, float, bool)):
                    c_name = f"const_{node.name}_arg{arg_idx}"
                    dt = (
                        DType.F32
                        if isinstance(arg, float)
                        else (DType.I64 if isinstance(arg, int) else DType.BOOL)
                    )
                    np_val = np.array(arg, dtype=np.float32 if dt == DType.F32 else np.int64)
                    c_t = g.add_tensor(
                        name=c_name,
                        shape=Shape([]),
                        dtype=dt,
                        storage=StorageClass.CONSTANT,
                        data=np_val,
                    )
                    g.parameters.append(c_t.id)
                    input_tensor_ids.append(c_t.id)
                elif isinstance(arg, (list, tuple)):
                    if all(isinstance(x, Node) and x in node_to_tensor for x in arg):
                        for sub_node in arg:
                            input_tensor_ids.append(node_to_tensor[sub_node].id)
                    else:
                        dims = []
                        for x in arg:
                            if isinstance(x, Node):
                                val = x.meta.get("val")
                                dims.append(
                                    _symint_to_dim(val) if val is not None else StaticDim(1)
                                )
                            else:
                                dims.append(_symint_to_dim(x))
                        attributes[f"arg_{arg_idx}_dims"] = tuple(dims)

        for k, v in node.kwargs.items():
            if isinstance(v, Node):
                input_tensor_ids.append(node_to_tensor[v].id)
            else:
                attributes[k] = v

        # Add output tensor
        out_t = g.add_tensor(
            name=node.name,
            shape=shape,
            dtype=dtype,
            storage=StorageClass.ACTIVATION,
        )
        node_to_tensor[node] = out_t
        name_to_tensor[node.name] = out_t

        g.add_op(
            opcode=opcode,
            inputs=input_tensor_ids,
            outputs=[out_t.id],
            attributes=attributes,
            name=node.name,
        )

    # 3. Process output node
    for node in ep.graph.nodes:
        if node.op == "output":
            # node.args[0] can be a single Node or tuple of Nodes
            outputs = node.args[0]
            if isinstance(outputs, Node):
                outputs = [outputs]
            for out_node in outputs:
                if out_node in node_to_tensor:
                    t = node_to_tensor[out_node]
                    t.storage = StorageClass.OUTPUT
                    g.outputs.append(t.id)

    g.validate_invariants()
    return g
