from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape, StaticDim
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
    "convert_element_type": OpCode.CAST,
    "squeeze": OpCode.RESHAPE,
    "square": OpCode.POW,
    "integer_pow": OpCode.POW,
    "pow": OpCode.POW,
    "erf": OpCode.GELU,
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

        # Handle identity primitives like stop_gradient, copy
        if prim_name in ("stop_gradient", "copy"):
            in_var = eqn.invars[0]
            out_var = eqn.outvars[0]
            if in_var in var_to_tensor:
                var_to_tensor[out_var] = var_to_tensor[in_var]
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

        # Special handling for dot_general multi-head / multidimensional projections
        if prim_name == "dot_general":
            dim_numbers = eqn.params.get("dimension_numbers")
            lhs_t = g.get_tensor(in_tids[0])
            rhs_t = g.get_tensor(in_tids[1])
            (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dim_numbers

            # Check if this is a general dense / multi-head projection with 3D/4D weight and no batch dim
            if (len(rhs_t.shape.dims) > 2 or len(lhs_contracting) > 1) and len(lhs_batch) == 0:

                def _dim_val(d):
                    return d.value if isinstance(d, StaticDim) else int(d)

                lhs_in_id = in_tids[0]
                lhs_batch_dims = [
                    _dim_val(d) for i, d in enumerate(lhs_t.shape.dims) if i not in lhs_contracting
                ]
                lhs_k = int(
                    np.prod(
                        [
                            _dim_val(d)
                            for i, d in enumerate(lhs_t.shape.dims)
                            if i in lhs_contracting
                        ]
                    )
                )
                lhs_flat_shape = tuple(lhs_batch_dims + [lhs_k])
                if tuple(_dim_val(d) for d in lhs_t.shape.dims) != lhs_flat_shape:
                    lhs_flat_t = g.add_tensor(
                        name=f"flat_lhs_{out_var}",
                        shape=Shape.from_tuple(lhs_flat_shape),
                        dtype=lhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[lhs_in_id],
                        outputs=[lhs_flat_t.id],
                        name=f"flat_lhs_{out_var}",
                    )
                    lhs_in_id = lhs_flat_t.id

                rhs_in_id = in_tids[1]
                rhs_k = int(
                    np.prod(
                        [
                            _dim_val(d)
                            for i, d in enumerate(rhs_t.shape.dims)
                            if i in rhs_contracting
                        ]
                    )
                )
                rhs_out = int(
                    np.prod(
                        [
                            _dim_val(d)
                            for i, d in enumerate(rhs_t.shape.dims)
                            if i not in rhs_contracting
                        ]
                    )
                )
                rhs_flat_shape = (rhs_k, rhs_out)
                if tuple(_dim_val(d) for d in rhs_t.shape.dims) != rhs_flat_shape:
                    if rhs_t.data is not None:
                        rhs_t.data = np.ascontiguousarray(
                            np.asarray(rhs_t.data).reshape(rhs_flat_shape)
                        )
                        rhs_t.shape = Shape.from_tuple(rhs_flat_shape)
                    else:
                        rhs_flat_t = g.add_tensor(
                            name=f"flat_rhs_{out_var}",
                            shape=Shape.from_tuple(rhs_flat_shape),
                            dtype=rhs_t.dtype,
                            storage=rhs_t.storage,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[rhs_in_id],
                            outputs=[rhs_flat_t.id],
                            name=f"flat_rhs_{out_var}",
                        )
                        rhs_in_id = rhs_flat_t.id

                matmul_out_shape = tuple(lhs_batch_dims + [rhs_out])
                if matmul_out_shape == tuple(out_aval.shape):
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[lhs_in_id, rhs_in_id],
                        outputs=[out_t.id],
                        attributes=dict(eqn.params),
                        name=f"matmul_{out_var}",
                    )
                else:
                    matmul_t = g.add_tensor(
                        name=f"matmul_{out_var}",
                        shape=Shape.from_tuple(matmul_out_shape),
                        dtype=out_dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[lhs_in_id, rhs_in_id],
                        outputs=[matmul_t.id],
                        attributes=dict(eqn.params),
                        name=f"matmul_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[matmul_t.id],
                        outputs=[out_t.id],
                        name=f"reshape_{out_var}",
                    )
                continue

            elif len(lhs_batch) > 0:

                def _dim_val(d):
                    return d.value if isinstance(d, StaticDim) else int(d)

                # 1. Permute LHS to (batch_dims..., seq_dim, contracting_dim)
                lhs_in_id = in_tids[0]
                lhs_ndim = len(lhs_t.shape.dims)
                lhs_seq_dims = [
                    i for i in range(lhs_ndim) if i not in lhs_batch and i not in lhs_contracting
                ]
                target_lhs_perm = tuple(list(lhs_batch) + lhs_seq_dims + list(lhs_contracting))
                if target_lhs_perm != tuple(range(lhs_ndim)):
                    perm_lhs_shape = tuple([_dim_val(lhs_t.shape.dims[i]) for i in target_lhs_perm])
                    perm_lhs_t = g.add_tensor(
                        name=f"perm_lhs_{out_var}",
                        shape=Shape.from_tuple(perm_lhs_shape),
                        dtype=lhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[lhs_in_id],
                        outputs=[perm_lhs_t.id],
                        attributes={"dims": list(target_lhs_perm)},
                        name=f"perm_lhs_{out_var}",
                    )
                    lhs_in_id = perm_lhs_t.id

                # 2. Permute RHS to (batch_dims..., seq_dim, contracting_dim)
                rhs_in_id = in_tids[1]
                rhs_ndim = len(rhs_t.shape.dims)
                rhs_seq_dims = [
                    i for i in range(rhs_ndim) if i not in rhs_batch and i not in rhs_contracting
                ]
                target_rhs_perm = tuple(list(rhs_batch) + rhs_seq_dims + list(rhs_contracting))
                if target_rhs_perm != tuple(range(rhs_ndim)):
                    perm_rhs_shape = tuple([_dim_val(rhs_t.shape.dims[i]) for i in target_rhs_perm])
                    perm_rhs_t = g.add_tensor(
                        name=f"perm_rhs_{out_var}",
                        shape=Shape.from_tuple(perm_rhs_shape),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[rhs_in_id],
                        outputs=[perm_rhs_t.id],
                        attributes={"dims": list(target_rhs_perm)},
                        name=f"perm_rhs_{out_var}",
                    )
                    rhs_in_id = perm_rhs_t.id

                g.add_op(
                    opcode=OpCode.MATMUL,
                    inputs=[lhs_in_id, rhs_in_id],
                    outputs=[out_t.id],
                    attributes=dict(eqn.params),
                    name=f"bmm_{out_var}",
                )
                continue

        # Special handling for broadcast_in_dim
        if prim_name == "broadcast_in_dim":
            bcast_dims = tuple(eqn.params.get("broadcast_dimensions", ()))
            out_shape_tuple = tuple(out_aval.shape)
            in_shape_tuple = tuple(eqn.invars[0].aval.shape)

            intermediate_shape = [1] * len(out_shape_tuple)
            for in_idx, out_idx in enumerate(bcast_dims):
                intermediate_shape[out_idx] = in_shape_tuple[in_idx]
            inter_tuple = tuple(intermediate_shape)

            if inter_tuple != in_shape_tuple and inter_tuple != out_shape_tuple:
                # Insert intermediate reshape
                inter_t = g.add_tensor(
                    name=f"reshaped_{out_var}",
                    shape=Shape.from_tuple(inter_tuple),
                    dtype=out_dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.RESHAPE,
                    inputs=[in_tids[0]],
                    outputs=[inter_t.id],
                    name=f"reshape_{out_var}",
                )
                g.add_op(
                    opcode=OpCode.EXPAND,
                    inputs=[inter_t.id],
                    outputs=[out_t.id],
                    attributes=dict(eqn.params),
                    name=f"expand_{out_var}",
                )
                continue
            elif inter_tuple != in_shape_tuple and inter_tuple == out_shape_tuple:
                # Pure reshape
                g.add_op(
                    opcode=OpCode.RESHAPE,
                    inputs=[in_tids[0]],
                    outputs=[out_t.id],
                    name=f"reshape_{out_var}",
                )
                continue

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
