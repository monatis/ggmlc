from __future__ import annotations

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
        return StaticDim(sym)
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
        elif target_name in lifted_buffers:
            buf_name = lifted_buffers[target_name]
            buf_tensor = ep.constants.get(buf_name, ep.state_dict.get(buf_name))
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
            # aten.slice.Tensor(self, dim=0, start=None, end=None, step=1)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
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
        elif opcode == OpCode.CONCAT:
            # aten.cat.default(tensors, dim=0)
            tensors_arg = node.args[0]
            for sub_node in tensors_arg:
                input_tensor_ids.append(node_to_tensor[sub_node].id)
            attributes["dim"] = int(node.args[1]) if len(node.args) > 1 else 0
        elif opcode == OpCode.EXPAND:
            # aten.expand.default(self, size)
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            attributes["shape"] = tuple(_symint_to_dim(d) for d in node.args[1])
        elif opcode in (OpCode.SQUEEZE, OpCode.UNSQUEEZE):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            if len(node.args) > 1 and node.args[1] is not None:
                attributes["dim"] = int(node.args[1])
        elif opcode in (OpCode.RESHAPE, OpCode.VIEW):
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            attributes["shape"] = tuple(_symint_to_dim(d) for d in node.args[1])
        elif opcode == OpCode.SOFTMAX:
            input_tensor_ids.append(node_to_tensor[node.args[0]].id)
            attributes["dim"] = int(node.args[1]) if len(node.args) > 1 else -1
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
                    if all(isinstance(x, Node) for x in arg):
                        for sub_node in arg:
                            input_tensor_ids.append(node_to_tensor[sub_node].id)
                    elif all(isinstance(x, (int, torch.SymInt)) for x in arg):
                        attributes[f"arg_{arg_idx}_dims"] = tuple(_symint_to_dim(x) for x in arg)

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
