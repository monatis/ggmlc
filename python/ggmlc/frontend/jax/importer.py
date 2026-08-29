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
    "logistic": OpCode.SIGMOID,
    "sigmoid": OpCode.SIGMOID,
    "conv_general_dilated": OpCode.CONV2D,
}


def _jax_dtype_to_dtype(dtype: Any) -> DType:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.float64):
        return DType.F32
    return DType.from_numpy(dt)


def _import_equations(
    eqns: Sequence[Any],
    g: Graph,
    var_to_tensor: dict[Any, Tensor],
    padded_vars: dict[Any, tuple[Any, tuple[int, int, int, int]]] | None = None,
) -> None:
    if padded_vars is None:
        padded_vars = {}

    for eqn in eqns:
        prim_name = eqn.primitive.name

        # Handle inlining of higher-order primitives like jit, pjit, custom_jvp_call
        if prim_name in ("jit", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call"):
            inner_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if inner_jaxpr is not None:
                if hasattr(inner_jaxpr, "jaxpr"):
                    inner_jaxpr = inner_jaxpr.jaxpr

                # 1. Pattern: nn.Embed wrapped in jit (gather over table and indices)
                if (
                    any(eq.primitive.name == "gather" for eq in inner_jaxpr.eqns)
                    and len(eqn.invars) == 2
                    and len(eqn.outvars) == 1
                ):
                    in0_t = var_to_tensor.get(eqn.invars[0])
                    in1_t = var_to_tensor.get(eqn.invars[1])
                    if in0_t is not None and in1_t is not None:
                        wte_t = (
                            in0_t
                            if in0_t.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                            else in1_t
                        )
                        idx_t = in1_t if wte_t == in0_t else in0_t
                        out_var = eqn.outvars[0]
                        out_aval = out_var.aval
                        out_shape = Shape.from_tuple(tuple(out_aval.shape))
                        out_t = g.add_tensor(
                            name=f"embed_{out_var}",
                            shape=out_shape,
                            dtype=_jax_dtype_to_dtype(out_aval.dtype),
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.EMBEDDING,
                            inputs=[wte_t.id, idx_t.id],
                            outputs=[out_t.id],
                            attributes={"dim": 0},
                            name=f"embed_{out_var}",
                        )
                        var_to_tensor[out_var] = out_t
                        continue

                # 2. Pattern: attention masking select_n wrapped in jit (where(mask, logits, -1e10))
                if (
                    any(eq.primitive.name == "select_n" for eq in inner_jaxpr.eqns)
                    and len(eqn.outvars) == 1
                ):
                    # Find mask, false_val, and logits among eqn.invars
                    mask_t = None
                    false_val = -1e10
                    logits_t = None
                    for invar in eqn.invars:
                        if hasattr(invar, "val"):
                            val = np.asarray(invar.val)
                            if val.size == 1 and val.item() < -1e3:
                                false_val = float(val.item())
                        elif not hasattr(invar, "val") and invar in var_to_tensor:
                            t = var_to_tensor[invar]
                            if t.data is not None:
                                mask_t = t
                            else:
                                logits_t = t
                    if mask_t is not None and mask_t.data is not None and logits_t is not None:
                        mask_arr = mask_t.data.astype(bool)
                        bias_arr = np.where(mask_arr, 0.0, false_val).astype(np.float32)
                        out_var = eqn.outvars[0]
                        out_aval = out_var.aval
                        out_shape = Shape.from_tuple(tuple(out_aval.shape))
                        bias_t = g.add_tensor(
                            name=f"mask_bias_{len(g.tensors)}",
                            shape=Shape.from_tuple(tuple(bias_arr.shape)),
                            dtype=DType.F32,
                            storage=StorageClass.CONSTANT,
                            data=bias_arr,
                            role="constant",
                        )
                        g.parameters.append(bias_t.id)
                        out_t = g.add_tensor(
                            name=f"masked_attn_{out_var}",
                            shape=out_shape,
                            dtype=_jax_dtype_to_dtype(out_aval.dtype),
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.ADD,
                            inputs=[logits_t.id, bias_t.id],
                            outputs=[out_t.id],
                            name=f"masked_attn_{out_var}",
                        )
                        var_to_tensor[out_var] = out_t
                        continue

                # Standard inlining for composite subgraphs
                inner_var_map: dict[Any, Tensor] = {}
                inner_padded_vars: dict[Any, tuple[Any, tuple[int, int, int, int]]] = {}
                outer_in_map: dict[Any, Any] = {}
                for inner_in, outer_in in zip(inner_jaxpr.invars, eqn.invars):
                    outer_in_map[inner_in] = outer_in
                    if hasattr(outer_in, "val"):
                        np_val = np.asarray(outer_in.val)
                        if np_val.dtype == np.float64:
                            np_val = np_val.astype(np.float32)
                        c_t = g.add_tensor(
                            name=f"lit_{len(g.tensors)}",
                            shape=Shape.from_tuple(tuple(np_val.shape)),
                            dtype=_jax_dtype_to_dtype(np_val.dtype),
                            storage=StorageClass.CONSTANT,
                            data=np_val,
                        )
                        g.parameters.append(c_t.id)
                        inner_var_map[inner_in] = c_t
                    elif not hasattr(outer_in, "val") and outer_in in var_to_tensor:
                        inner_var_map[inner_in] = var_to_tensor[outer_in]
                    if not hasattr(outer_in, "val") and outer_in in padded_vars:
                        inner_padded_vars[inner_in] = padded_vars[outer_in]

                _import_equations(inner_jaxpr.eqns, g, inner_var_map, inner_padded_vars)

                # Map inner outvars to outer outvars
                for inner_out, outer_out in zip(inner_jaxpr.outvars, eqn.outvars):
                    if inner_out in inner_var_map:
                        var_to_tensor[outer_out] = inner_var_map[inner_out]
                    if not hasattr(inner_out, "val") and inner_out in inner_padded_vars:
                        in_orig, p_cfg = inner_padded_vars[inner_out]
                        padded_vars[outer_out] = (outer_in_map.get(in_orig, in_orig), p_cfg)
                continue

        # Handle identity primitives like stop_gradient, copy
        if prim_name in ("stop_gradient", "copy"):
            in_var = eqn.invars[0]
            out_var = eqn.outvars[0]
            if not hasattr(in_var, "val") and in_var in var_to_tensor:
                var_to_tensor[out_var] = var_to_tensor[in_var]
            if not hasattr(in_var, "val") and in_var in padded_vars:
                padded_vars[out_var] = padded_vars[in_var]
            continue

        # Handle compile-time constant folding for static sub-expressions (e.g., masks, iota, shapes)
        all_const = True
        const_inputs = []
        for in_var in eqn.invars:
            if hasattr(in_var, "val"):
                const_inputs.append(np.asarray(in_var.val))
            elif (
                not hasattr(in_var, "val")
                and in_var in var_to_tensor
                and var_to_tensor[in_var].data is not None
            ):
                const_inputs.append(var_to_tensor[in_var].data)
            else:
                all_const = False
                break

        if all_const:
            try:
                eval_res = eqn.primitive.bind(*const_inputs, **eqn.params)
                if not isinstance(eval_res, (list, tuple)):
                    eval_res = [eval_res]
                for out_var, res_val in zip(eqn.outvars, eval_res):
                    np_val = np.asarray(res_val)
                    if np_val.dtype == np.float64:
                        np_val = np_val.astype(np.float32)
                    c_t = g.add_tensor(
                        name=f"const_{len(g.tensors)}",
                        shape=Shape.from_tuple(tuple(np_val.shape)),
                        dtype=_jax_dtype_to_dtype(np_val.dtype),
                        storage=StorageClass.CONSTANT,
                        data=np_val,
                    )
                    g.parameters.append(c_t.id)
                    var_to_tensor[out_var] = c_t
                continue
            except Exception:  # noqa: BLE001, S110
                pass

        opcode = JAX_PRIMITIVE_MAP.get(prim_name)
        if opcode is None and prim_name not in (
            "gather",
            "select_n",
            "conv_general_dilated",
            "pad",
            "reduce_window_max",
            "reduce_window_sum",
            "erfc",
        ):
            raise NotImplementedError(
                f"Unsupported JAX primitive: '{prim_name}'. No Canonical IR lowering registered."
            )

        # Collect inputs
        in_tids: list[int] = []
        for in_var in eqn.invars:
            if hasattr(in_var, "val"):
                np_val = np.asarray(in_var.val)
                if np_val.dtype == np.float64:
                    np_val = np_val.astype(np.float32)
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
                matmul_attrs = dict(eqn.params)
                matmul_attrs["transpose_in0"] = 1
                if matmul_out_shape == tuple(out_aval.shape):
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[lhs_in_id, rhs_in_id],
                        outputs=[out_t.id],
                        attributes=matmul_attrs,
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
                        attributes=matmul_attrs,
                        name=f"matmul_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[matmul_t.id],
                        outputs=[out_t.id],
                        name=f"reshape_{out_var}",
                    )
                continue

            elif len(lhs_batch) == 0:
                attrs = dict(eqn.params)
                attrs["transpose_in0"] = 1
                g.add_op(
                    opcode=OpCode.MATMUL,
                    inputs=in_tids,
                    outputs=[out_t.id],
                    attributes=attrs,
                    name=f"matmul_{out_var}",
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

                bmm_attrs = dict(eqn.params)
                bmm_attrs["transpose_in0"] = 0
                g.add_op(
                    opcode=OpCode.MATMUL,
                    inputs=[lhs_in_id, rhs_in_id],
                    outputs=[out_t.id],
                    attributes=bmm_attrs,
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

        if prim_name == "conv_general_dilated":
            lhs_in_var = eqn.invars[0]
            params_dict = eqn.params
            window_strides = params_dict.get("window_strides", (1, 1))
            padding = list(params_dict.get("padding", ((0, 0), (0, 0))))
            rhs_dilation = params_dict.get("rhs_dilation", (1, 1))
            feature_group_count = params_dict.get("feature_group_count", 1)

            if not hasattr(lhs_in_var, "val") and lhs_in_var in padded_vars:
                orig_in_var, (p_top, p_bot, p_left, p_right) = padded_vars[lhs_in_var]
                lhs_t = var_to_tensor[orig_in_var]
                padding = [
                    (int(padding[0][0]) + p_top, int(padding[0][1]) + p_bot),
                    (int(padding[1][0]) + p_left, int(padding[1][1]) + p_right),
                ]
            else:
                lhs_t = g.get_tensor(in_tids[0])

            rhs_t = g.get_tensor(in_tids[1])

            def _dim_val(d):
                return d.value if isinstance(d, StaticDim) else int(d)

            # 1. Permute LHS (input) from NHWC to NCHW: (0, 3, 1, 2)
            lhs_dims = [_dim_val(d) for d in lhs_t.shape.dims]
            if len(lhs_dims) == 4:
                lhs_nchw_shape = (lhs_dims[0], lhs_dims[3], lhs_dims[1], lhs_dims[2])
                lhs_nchw_t = g.add_tensor(
                    name=f"nchw_lhs_{out_var}",
                    shape=Shape.from_tuple(lhs_nchw_shape),
                    dtype=lhs_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.PERMUTE,
                    inputs=[lhs_t.id],
                    outputs=[lhs_nchw_t.id],
                    attributes={"axes": (0, 3, 1, 2)},
                    name=f"perm_lhs_{out_var}",
                )
                lhs_conv_id = lhs_nchw_t.id
            else:
                lhs_conv_id = lhs_t.id

            # 2. Permute RHS (weights) from HWIO to OIHW: (3, 2, 0, 1)
            rhs_dims = [_dim_val(d) for d in rhs_t.shape.dims]
            if len(rhs_dims) == 4:
                rhs_oihw_shape = (rhs_dims[3], rhs_dims[2], rhs_dims[0], rhs_dims[1])
                if rhs_t.data is not None:
                    rhs_t.data = np.ascontiguousarray(np.transpose(rhs_t.data, (3, 2, 0, 1)))
                    rhs_t.shape = Shape.from_tuple(rhs_oihw_shape)
                    rhs_conv_id = rhs_t.id
                else:
                    rhs_oihw_t = g.add_tensor(
                        name=f"oihw_rhs_{out_var}",
                        shape=Shape.from_tuple(rhs_oihw_shape),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[rhs_t.id],
                        outputs=[rhs_oihw_t.id],
                        attributes={"axes": (3, 2, 0, 1)},
                        name=f"perm_rhs_{out_var}",
                    )
                    rhs_conv_id = rhs_oihw_t.id
            else:
                rhs_conv_id = rhs_t.id

            # 3. Intermediate NCHW Conv output
            out_dims = [_dim_val(d) for d in out_t.shape.dims]
            if len(out_dims) == 4:
                out_nchw_shape = (out_dims[0], out_dims[3], out_dims[1], out_dims[2])
                out_nchw_t = g.add_tensor(
                    name=f"nchw_conv_{out_var}",
                    shape=Shape.from_tuple(out_nchw_shape),
                    dtype=out_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                conv_attrs = {
                    "stride": tuple(window_strides),
                    "padding": (padding[0][0], padding[1][0]),
                    "dilation": tuple(rhs_dilation),
                    "groups": int(feature_group_count),
                }
                g.add_op(
                    opcode=OpCode.CONV2D,
                    inputs=[lhs_conv_id, rhs_conv_id],
                    outputs=[out_nchw_t.id],
                    attributes=conv_attrs,
                    name=f"conv_{out_var}",
                )
                # 4. Permute NCHW back to NHWC (0, 2, 3, 1) into out_t
                g.add_op(
                    opcode=OpCode.PERMUTE,
                    inputs=[out_nchw_t.id],
                    outputs=[out_t.id],
                    attributes={"axes": (0, 2, 3, 1)},
                    name=f"perm_out_{out_var}",
                )
            else:
                conv_attrs = {
                    "stride": tuple(window_strides),
                    "padding": (padding[0][0], padding[1][0]),
                    "dilation": tuple(rhs_dilation),
                    "groups": int(feature_group_count),
                }
                g.add_op(
                    opcode=OpCode.CONV2D,
                    inputs=[lhs_conv_id, rhs_conv_id],
                    outputs=[out_t.id],
                    attributes=conv_attrs,
                    name=f"conv_{out_var}",
                )
            continue

        if prim_name == "gather":
            # JAX gather(table, indices) -> Canonical IR OpCode.EMBEDDING
            table_t = var_to_tensor[eqn.invars[0]]
            indices_t = var_to_tensor[eqn.invars[1]]
            g.add_op(
                opcode=OpCode.EMBEDDING,
                inputs=[table_t.id, indices_t.id],
                outputs=[out_t.id],
                attributes={"dim": 0},
                name=f"embed_{out_var}",
            )
            continue

        if prim_name == "select_n":
            # select_n(cond, false_branch, true_branch)
            # In attention masking: cond is constant boolean mask, false_branch is constant -1e10, true_branch is dynamic logits
            cond_t = var_to_tensor.get(eqn.invars[0])
            false_t = var_to_tensor.get(eqn.invars[1])
            true_t = var_to_tensor.get(eqn.invars[2])

            if (
                cond_t is not None
                and cond_t.data is not None
                and false_t is not None
                and false_t.data is not None
                and true_t is not None
            ):
                # Lower to additive attention bias: true_t + where(cond, 0.0, false_val)
                cond_arr = cond_t.data.astype(bool)
                false_val = float(
                    false_t.data.item() if false_t.data.ndim == 0 else false_t.data.flatten()[0]
                )
                bias_arr = np.where(cond_arr, 0.0, false_val).astype(np.float32)

                bias_t = g.add_tensor(
                    name=f"mask_bias_{len(g.tensors)}",
                    shape=Shape.from_tuple(tuple(bias_arr.shape)),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=bias_arr,
                    role="constant",
                )
                g.parameters.append(bias_t.id)
                g.add_op(
                    opcode=OpCode.ADD,
                    inputs=[true_t.id, bias_t.id],
                    outputs=[out_t.id],
                    name=f"masked_attn_{out_var}",
                )
                continue

        if prim_name in ("reduce_window_max", "reduce_window_sum"):
            in_var_0 = eqn.invars[0]
            params_dict = eqn.params
            window_dimensions = params_dict.get("window_dimensions", (1, 1, 1, 1))
            window_strides = params_dict.get("window_strides", (1, 1, 1, 1))
            padding = list(params_dict.get("padding", ((0, 0), (0, 0), (0, 0), (0, 0))))

            def _dim_val(d):
                return d.value if isinstance(d, StaticDim) else int(d)

            if not hasattr(in_var_0, "val") and in_var_0 in padded_vars:
                orig_in_var, (p_top, p_bot, p_left, p_right) = padded_vars[in_var_0]
                in_t = var_to_tensor[orig_in_var]
                p_h = int(padding[1][0]) + p_top if len(padding) > 1 else p_top
                p_w = int(padding[2][0]) + p_left if len(padding) > 2 else p_left
            else:
                in_t = g.get_tensor(in_tids[0])
                p_h = int(padding[1][0]) if len(padding) > 1 else 0
                p_w = int(padding[2][0]) if len(padding) > 2 else 0

            lhs_dims = [_dim_val(d) for d in in_t.shape.dims]
            if len(lhs_dims) == 4:
                lhs_nchw_shape = (lhs_dims[0], lhs_dims[3], lhs_dims[1], lhs_dims[2])
                lhs_nchw_t = g.add_tensor(
                    name=f"nchw_in_{out_var}",
                    shape=Shape.from_tuple(lhs_nchw_shape),
                    dtype=in_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.PERMUTE,
                    inputs=[in_t.id],
                    outputs=[lhs_nchw_t.id],
                    attributes={"axes": (0, 3, 1, 2)},
                    name=f"perm_in_{out_var}",
                )
                pool_in_id = lhs_nchw_t.id

                out_dims = [_dim_val(d) for d in out_t.shape.dims]
                out_nchw_shape = (out_dims[0], out_dims[3], out_dims[1], out_dims[2])
                out_nchw_t = g.add_tensor(
                    name=f"nchw_pool_{out_var}",
                    shape=Shape.from_tuple(out_nchw_shape),
                    dtype=out_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                pool_opcode = (
                    OpCode.MAX_POOL2D if prim_name == "reduce_window_max" else OpCode.AVG_POOL2D
                )
                k_h = int(window_dimensions[1])
                k_w = int(window_dimensions[2])
                s_h = int(window_strides[1])
                s_w = int(window_strides[2])

                pool_attrs = {
                    "ksize_h": k_h,
                    "ksize_w": k_w,
                    "stride_h": s_h,
                    "stride_w": s_w,
                    "pad_h": p_h,
                    "pad_w": p_w,
                }
                g.add_op(
                    opcode=pool_opcode,
                    inputs=[pool_in_id],
                    outputs=[out_nchw_t.id],
                    attributes=pool_attrs,
                    name=f"pool_{out_var}",
                )
                g.add_op(
                    opcode=OpCode.PERMUTE,
                    inputs=[out_nchw_t.id],
                    outputs=[out_t.id],
                    attributes={"axes": (0, 2, 3, 1)},
                    name=f"perm_out_{out_var}",
                )
            else:
                pool_opcode = (
                    OpCode.MAX_POOL2D if prim_name == "reduce_window_max" else OpCode.AVG_POOL2D
                )
                g.add_op(
                    opcode=pool_opcode,
                    inputs=[in_t.id],
                    outputs=[out_t.id],
                    attributes=dict(eqn.params),
                    name=f"pool_{out_var}",
                )
            continue

        if prim_name == "pad":
            in_var = eqn.invars[0]
            out_var = eqn.outvars[0]
            padding_config = eqn.params.get("padding_config", ())
            is_zero_pad = all(
                int(c[0]) == 0 and int(c[1]) == 0 and int(c[2]) == 0 for c in padding_config
            )
            if is_zero_pad:
                if not hasattr(in_var, "val") and in_var in var_to_tensor:
                    var_to_tensor[out_var] = var_to_tensor[in_var]
                continue

            if len(padding_config) == 4:
                p_top = int(padding_config[1][0])
                p_bot = int(padding_config[1][1])
                p_left = int(padding_config[2][0])
                p_right = int(padding_config[2][1])
                padded_vars[out_var] = (in_var, (p_top, p_bot, p_left, p_right))
                if not hasattr(in_var, "val") and in_var in var_to_tensor:
                    var_to_tensor[out_var] = var_to_tensor[in_var]
                continue

            in_t = g.get_tensor(in_tids[0])
            g.add_op(
                opcode=OpCode.PAD,
                inputs=[in_t.id],
                outputs=[out_t.id],
                attributes=dict(eqn.params),
                name=f"pad_{out_var}",
            )
            continue

        if prim_name == "erfc":
            # erfc(x) = 1.0 - erf(x)
            in_t = g.get_tensor(in_tids[0])
            one_t = g.add_tensor(
                name=f"one_{len(g.tensors)}",
                shape=Shape.from_tuple(()),
                dtype=DType.F32,
                storage=StorageClass.CONSTANT,
                data=np.array(1.0, dtype=np.float32),
                role="constant",
            )
            g.parameters.append(one_t.id)
            erf_t = g.add_tensor(
                name=f"erf_{out_var}",
                shape=in_t.shape,
                dtype=in_t.dtype,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.GELU,
                inputs=[in_t.id],
                outputs=[erf_t.id],
                name=f"erf_{out_var}",
            )
            g.add_op(
                opcode=OpCode.SUB,
                inputs=[one_t.id, erf_t.id],
                outputs=[out_t.id],
                name=f"erfc_{out_var}",
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
            name=f"const_{len(g.tensors)}",
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
