from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass, Tensor

# JAX Primitive -> Canonical OpCode mapping
JAX_PRIMITIVE_MAP: dict[str, OpCode] = {
    "add": OpCode.ADD,
    "add_any": OpCode.ADD,
    "sub": OpCode.SUB,
    "mul": OpCode.MUL,
    "div": OpCode.DIV,
    "neg": OpCode.NEG,
    "exp": OpCode.EXP,
    "log": OpCode.LOG,
    "sqrt": OpCode.SQRT,
    "rsqrt": OpCode.RSQRT,
    "abs": OpCode.ABS,
    "max": OpCode.MAXIMUM,
    "min": OpCode.MINIMUM,
    "sin": OpCode.SIN,
    "cos": OpCode.COS,
    "tanh": OpCode.TANH,
    "relu": OpCode.RELU,
    "silu": OpCode.SILU,
    "gelu": OpCode.GELU,
    "dot_general": OpCode.MATMUL,
    "reshape": OpCode.RESHAPE,
    "transpose": OpCode.PERMUTE,
    "rev": OpCode.TRANSPOSE,
    "slice": OpCode.SLICE,
    "concatenate": OpCode.CONCAT,
    "broadcast_in_dim": OpCode.EXPAND,
    "reduce_sum": OpCode.SUM,
    "reduce_max": OpCode.AMAX,
    "reduce_min": OpCode.AMIN,
}


def _jax_dtype_to_dtype(dtype: Any) -> DType:
    return DType.from_numpy(np.dtype(dtype))


def _import_equations(
    eqns: Sequence[Any],
    g: Graph,
    var_to_tensor: dict[Any, Tensor],
) -> None:
    for eqn in eqns:
        prim_name = eqn.primitive.name

        # Handle inlining of higher-order primitives like jit, pjit, custom_jvp_call
        if prim_name in ("jit", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call"):
            inner_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if inner_jaxpr is not None:
                if hasattr(inner_jaxpr, "jaxpr"):
                    inner_jaxpr = inner_jaxpr.jaxpr
                inner_var_map = dict(var_to_tensor)
                for inner_in, outer_in in zip(inner_jaxpr.invars, eqn.invars):
                    if hasattr(outer_in, "val"):
                        np_val = np.asarray(outer_in.val)
                        c_t = g.add_tensor(
                            name=f"lit_{len(g.tensors)}",
                            shape=Shape.from_tuple(tuple(np_val.shape)),
                            dtype=_jax_dtype_to_dtype(np_val.dtype),
                            storage=StorageClass.CONSTANT,
                            data=np_val,
                        )
                        g.parameters.append(c_t.id)
                        inner_var_map[inner_in] = c_t
                    elif outer_in in var_to_tensor:
                        inner_var_map[inner_in] = var_to_tensor[outer_in]

                _import_equations(inner_jaxpr.eqns, g, inner_var_map)

                # Map inner outvars to outer outvars
                for inner_out, outer_out in zip(inner_jaxpr.outvars, eqn.outvars):
                    if inner_out in inner_var_map:
                        var_to_tensor[outer_out] = inner_var_map[inner_out]
                continue

        opcode = JAX_PRIMITIVE_MAP.get(prim_name)
        if opcode is None:
            raise NotImplementedError(
                f"Unsupported JAX primitive: '{prim_name}'. No Canonical IR lowering registered."
            )

        # Collect inputs
        in_tids: list[int] = []
        for in_var in eqn.invars:
            if hasattr(in_var, "val"):
                np_val = np.asarray(in_var.val)
                c_t = g.add_tensor(
                    name=f"lit_{len(g.tensors)}",
                    shape=Shape.from_tuple(tuple(np_val.shape)),
                    dtype=_jax_dtype_to_dtype(np_val.dtype),
                    storage=StorageClass.CONSTANT,
                    data=np_val,
                )
                g.parameters.append(c_t.id)
                in_tids.append(c_t.id)
            elif in_var in var_to_tensor:
                in_tids.append(var_to_tensor[in_var].id)

        # Output
        out_var = eqn.outvars[0]
        out_aval = out_var.aval
        out_shape = Shape.from_tuple(tuple(out_aval.shape))
        out_dtype = _jax_dtype_to_dtype(out_aval.dtype)

        out_t = g.add_tensor(
            name=f"t_{out_var}",
            shape=out_shape,
            dtype=out_dtype,
            storage=StorageClass.ACTIVATION,
        )
        var_to_tensor[out_var] = out_t

        g.add_op(
            opcode=opcode,
            inputs=in_tids,
            outputs=[out_t.id],
            attributes=dict(eqn.params),
            name=f"{prim_name}_{out_var}",
        )


def import_jaxpr(
    closed_jaxpr: Any,
    graph_name: str = "main",
    input_names: Sequence[str] | None = None,
    params: dict[str, np.ndarray] | None = None,
) -> Graph:
    """Imports a JAX ClosedJaxpr into a ggmlc Canonical IR Graph."""
    g = Graph(name=graph_name)
    var_to_tensor: dict[Any, Tensor] = {}
    params = params or {}

    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.consts

    # 1. Process constant variables
    for c_var, c_val in zip(jaxpr.constvars, consts):
        np_arr = np.asarray(c_val)
        t = g.add_tensor(
            name=f"const_{c_var}",
            shape=Shape.from_tuple(tuple(np_arr.shape)),
            dtype=_jax_dtype_to_dtype(np_arr.dtype),
            storage=StorageClass.CONSTANT,
            data=np_arr,
        )
        g.parameters.append(t.id)
        var_to_tensor[c_var] = t

    # 2. Process input variables
    for i, var in enumerate(jaxpr.invars):
        aval = var.aval
        shape = Shape.from_tuple(tuple(aval.shape))
        dtype = _jax_dtype_to_dtype(aval.dtype)
        name = input_names[i] if input_names and i < len(input_names) else f"in_{var}"

        if name in params:
            t = g.add_tensor(
                name=name,
                shape=shape,
                dtype=dtype,
                storage=StorageClass.PARAMETER,
                data=params[name],
                role="parameter",
            )
            g.parameters.append(t.id)
        else:
            t = g.add_tensor(
                name=name,
                shape=shape,
                dtype=dtype,
                storage=StorageClass.INPUT,
                role="input",
            )
            g.inputs.append(t.id)

        var_to_tensor[var] = t

    # 3. Process equations
    _import_equations(jaxpr.eqns, g, var_to_tensor)

    # 4. Outputs
    for out_var in jaxpr.outvars:
        if out_var in var_to_tensor:
            t = var_to_tensor[out_var]
            t.storage = StorageClass.OUTPUT
            g.outputs.append(t.id)

    g.validate_invariants()
    return g
