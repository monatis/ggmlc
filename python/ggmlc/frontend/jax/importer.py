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
    "reduce_and": OpCode.AMIN,
    "reduce_or": OpCode.AMAX,
    "and": OpCode.MUL,
    "or": OpCode.MAXIMUM,
    "not": OpCode.NEG,
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
    if dt in (np.dtype(np.float64), np.dtype(bool), np.dtype(np.uint8), np.dtype(np.int8)):
        return DType.F32
    return DType.from_numpy(dt)


def _import_equations(
    eqns: Sequence[Any],
    g: Graph,
    var_to_tensor: dict[Any, Tensor],
    padded_vars: dict[Any, tuple[Any, tuple[int, int, int, int]]] | None = None,
    gelu_state: dict[str, Any] | None = None,
    softmax_state: dict[str, Any] | None = None,
) -> None:
    if padded_vars is None:
        padded_vars = {}
    if gelu_state is None:
        gelu_state = {
            "half_vars": {},
            "neg_vars": {},
            "scaled_neg_vars": {},
            "erfc_vars": {},
        }
    if softmax_state is None:
        softmax_state = {
            "max_vars": {},
            "sub_vars": {},
            "exp_vars": {},
            "sum_vars": {},
        }

    for eqn in eqns:
        prim_name = eqn.primitive.name

        if prim_name == "reduce_max":
            softmax_state["max_vars"][eqn.outvars[0]] = (
                eqn.invars[0],
                eqn.params.get("axes", (-1,)),
            )
        elif prim_name in ("broadcast_in_dim", "stop_gradient"):
            in_v = eqn.invars[0]
            out_v = eqn.outvars[0]
            if not hasattr(in_v, "val") and in_v in softmax_state["max_vars"]:
                softmax_state["max_vars"][out_v] = softmax_state["max_vars"][in_v]
            if not hasattr(in_v, "val") and in_v in softmax_state["sum_vars"]:
                softmax_state["sum_vars"][out_v] = softmax_state["sum_vars"][in_v]
        elif prim_name == "sub" and len(eqn.invars) == 2:
            v0, v1 = eqn.invars[0], eqn.invars[1]
            if (
                not hasattr(v1, "val")
                and v1 in softmax_state["max_vars"]
                and softmax_state["max_vars"][v1][0] == v0
            ):
                softmax_state["sub_vars"][eqn.outvars[0]] = softmax_state["max_vars"][v1]
        elif prim_name == "exp":
            in_v = eqn.invars[0]
            if not hasattr(in_v, "val") and in_v in softmax_state["sub_vars"]:
                softmax_state["exp_vars"][eqn.outvars[0]] = softmax_state["sub_vars"][in_v]
        elif prim_name == "reduce_sum":
            in_v = eqn.invars[0]
            if not hasattr(in_v, "val") and in_v in softmax_state["exp_vars"]:
                softmax_state["sum_vars"][eqn.outvars[0]] = softmax_state["exp_vars"][in_v]

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
            "dynamic_slice",
            "gt",
            "ge",
            "lt",
            "le",
            "eq",
            "ne",
            "split",
            "stack",
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
                rhs_contracting_dims = [
                    i for i in range(len(rhs_t.shape.dims)) if i in rhs_contracting
                ]
                rhs_other_dims = [
                    i for i in range(len(rhs_t.shape.dims)) if i not in rhs_contracting
                ]
                rhs_perm = tuple(rhs_contracting_dims + rhs_other_dims)

                rhs_flat_shape = (rhs_k, rhs_out)
                if tuple(_dim_val(d) for d in rhs_t.shape.dims) != rhs_flat_shape:
                    if rhs_t.data is not None:
                        arr = np.asarray(rhs_t.data)
                        if rhs_perm != tuple(range(len(rhs_t.shape.dims))):
                            arr = np.transpose(arr, rhs_perm)
                        rhs_t.data = np.ascontiguousarray(arr.reshape(rhs_flat_shape))
                        rhs_t.shape = Shape.from_tuple(rhs_flat_shape)
                    else:
                        cur_in_id = rhs_in_id
                        if rhs_perm != tuple(range(len(rhs_t.shape.dims))):
                            perm_t = g.add_tensor(
                                name=f"perm_rhs_{out_var}",
                                shape=Shape.from_tuple(
                                    tuple(_dim_val(rhs_t.shape.dims[i]) for i in rhs_perm)
                                ),
                                dtype=rhs_t.dtype,
                                storage=rhs_t.storage,
                            )
                            g.add_op(
                                opcode=OpCode.PERMUTE,
                                inputs=[cur_in_id],
                                outputs=[perm_t.id],
                                attributes={"dims": list(rhs_perm)},
                                name=f"perm_rhs_{out_var}",
                            )
                            cur_in_id = perm_t.id
                        rhs_flat_t = g.add_tensor(
                            name=f"flat_rhs_{out_var}",
                            shape=Shape.from_tuple(rhs_flat_shape),
                            dtype=rhs_t.dtype,
                            storage=rhs_t.storage,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[cur_in_id],
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

                # GQA Case 1: Q-K Scores calculation: LHS=Key(4D), RHS=Query(5D with GQA group)
                if (
                    len(lhs_t.shape.dims) == 4
                    and len(rhs_t.shape.dims) == 5
                    and tuple(lhs_batch) == (0, 2)
                    and tuple(rhs_batch) == (0, 2)
                    and tuple(lhs_contracting) == (3,)
                    and tuple(rhs_contracting) == (4,)
                ):
                    B = _dim_val(lhs_t.shape.dims[0])
                    S_k = _dim_val(lhs_t.shape.dims[1])
                    H_kv = _dim_val(lhs_t.shape.dims[2])
                    D = _dim_val(lhs_t.shape.dims[3])
                    S_q = _dim_val(rhs_t.shape.dims[1])
                    G = _dim_val(rhs_t.shape.dims[3])
                    H_q = H_kv * G

                    # 1. Permute query (0, 2, 3, 1, 4) -> (B, H_kv, G, S_q, D)
                    q_perm_t = g.add_tensor(
                        name=f"q_perm_{out_var}",
                        shape=Shape.from_tuple((B, H_kv, G, S_q, D)),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[in_tids[1]],
                        outputs=[q_perm_t.id],
                        attributes={"dims": [0, 2, 3, 1, 4]},
                        name=f"q_perm_{out_var}",
                    )
                    q_4d_t = g.add_tensor(
                        name=f"q4d_{out_var}",
                        shape=Shape.from_tuple((B, H_q, S_q, D)),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[q_perm_t.id],
                        outputs=[q_4d_t.id],
                        name=f"q4d_{out_var}",
                    )

                    # 2. Permute key (0, 2, 1, 3) -> (B, H_kv, S_k, D)
                    k_perm_t = g.add_tensor(
                        name=f"k_perm_{out_var}",
                        shape=Shape.from_tuple((B, H_kv, S_k, D)),
                        dtype=lhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[in_tids[0]],
                        outputs=[k_perm_t.id],
                        attributes={"dims": [0, 2, 1, 3]},
                        name=f"k_perm_{out_var}",
                    )
                    if G > 1:
                        k_5d_t = g.add_tensor(
                            name=f"k5d_{out_var}",
                            shape=Shape.from_tuple((B, H_kv, 1, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[k_perm_t.id],
                            outputs=[k_5d_t.id],
                            name=f"reshape_k5d_{out_var}",
                        )
                        k_exp_t = g.add_tensor(
                            name=f"kexp_{out_var}",
                            shape=Shape.from_tuple((B, H_kv, G, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.EXPAND,
                            inputs=[k_5d_t.id],
                            outputs=[k_exp_t.id],
                            attributes={"shape": (B, H_kv, G, S_k, D)},
                            name=f"exp_k_{out_var}",
                        )
                        k_4d_t = g.add_tensor(
                            name=f"k4d_{out_var}",
                            shape=Shape.from_tuple((B, H_q, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[k_exp_t.id],
                            outputs=[k_4d_t.id],
                            name=f"k4d_{out_var}",
                        )
                        k_final_id = k_4d_t.id
                    else:
                        k_final_id = k_perm_t.id

                    # 3. 4D BMM: k @ q.T -> (B, H_q, S_k, S_q)
                    scores_4d_t = g.add_tensor(
                        name=f"scores4d_{out_var}",
                        shape=Shape.from_tuple((B, H_q, S_k, S_q)),
                        dtype=out_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[k_final_id, q_4d_t.id],
                        outputs=[scores_4d_t.id],
                        attributes={"transpose_in0": 0, "transpose_in1": 1},
                        name=f"bmm_{out_var}",
                    )

                    # 4. Reshape to (B, H_kv, G, S_k, S_q) then permute to (B, H_kv, S_k, S_q, G)
                    scores_5d_t = g.add_tensor(
                        name=f"scores5d_{out_var}",
                        shape=Shape.from_tuple((B, H_kv, G, S_k, S_q)),
                        dtype=out_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[scores_4d_t.id],
                        outputs=[scores_5d_t.id],
                        name=f"reshape_scores5d_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[scores_5d_t.id],
                        outputs=[out_t.id],
                        attributes={"dims": [0, 1, 3, 4, 2]},
                        name=f"perm_out_{out_var}",
                    )
                    continue

                # GQA Case 2: Attention Context Projection: LHS=Value(4D), RHS=Weights(5D)
                if (
                    len(lhs_t.shape.dims) == 4
                    and len(rhs_t.shape.dims) == 5
                    and tuple(lhs_batch) == (0, 2)
                    and tuple(rhs_batch) == (0, 1)
                    and tuple(lhs_contracting) == (1,)
                    and tuple(rhs_contracting) == (4,)
                ):
                    B = _dim_val(lhs_t.shape.dims[0])
                    S_k = _dim_val(lhs_t.shape.dims[1])
                    H_kv = _dim_val(lhs_t.shape.dims[2])
                    D = _dim_val(lhs_t.shape.dims[3])
                    G = _dim_val(rhs_t.shape.dims[2])
                    S_q = _dim_val(rhs_t.shape.dims[3])
                    H_q = H_kv * G

                    # 1. Permute value (0, 2, 1, 3) -> (B, H_kv, S_k, D)
                    v_perm_t = g.add_tensor(
                        name=f"v_perm_{out_var}",
                        shape=Shape.from_tuple((B, H_kv, S_k, D)),
                        dtype=lhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[in_tids[0]],
                        outputs=[v_perm_t.id],
                        attributes={"dims": [0, 2, 1, 3]},
                        name=f"v_perm_{out_var}",
                    )
                    if G > 1:
                        v_5d_t = g.add_tensor(
                            name=f"v5d_{out_var}",
                            shape=Shape.from_tuple((B, H_kv, 1, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[v_perm_t.id],
                            outputs=[v_5d_t.id],
                            name=f"reshape_v5d_{out_var}",
                        )
                        v_exp_t = g.add_tensor(
                            name=f"vexp_{out_var}",
                            shape=Shape.from_tuple((B, H_kv, G, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.EXPAND,
                            inputs=[v_5d_t.id],
                            outputs=[v_exp_t.id],
                            attributes={"shape": (B, H_kv, G, S_k, D)},
                            name=f"exp_v_{out_var}",
                        )
                        v_4d_t = g.add_tensor(
                            name=f"v4d_{out_var}",
                            shape=Shape.from_tuple((B, H_q, S_k, D)),
                            dtype=lhs_t.dtype,
                            storage=StorageClass.ACTIVATION,
                        )
                        g.add_op(
                            opcode=OpCode.RESHAPE,
                            inputs=[v_exp_t.id],
                            outputs=[v_4d_t.id],
                            name=f"v4d_{out_var}",
                        )
                        v_final_id = v_4d_t.id
                    else:
                        v_final_id = v_perm_t.id

                    # 2. Reshape weights (B, H_kv, G, S_q, S_k) -> (B, H_q, S_q, S_k)
                    w_4d_t = g.add_tensor(
                        name=f"w4d_{out_var}",
                        shape=Shape.from_tuple((B, H_q, S_q, S_k)),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[in_tids[1]],
                        outputs=[w_4d_t.id],
                        name=f"w4d_{out_var}",
                    )

                    # 3. 4D BMM: w_4d @ v_4d -> (B, H_q, S_q, D)
                    ctx_4d_t = g.add_tensor(
                        name=f"ctx4d_{out_var}",
                        shape=Shape.from_tuple((B, H_q, S_q, D)),
                        dtype=out_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[w_4d_t.id, v_final_id],
                        outputs=[ctx_4d_t.id],
                        attributes={"transpose_in0": 0, "transpose_in1": 0},
                        name=f"bmm_{out_var}",
                    )

                    # 4. Reshape to (B, H_kv, G, S_q, D) then permute to (B, H_kv, D, G, S_q)
                    ctx_5d_t = g.add_tensor(
                        name=f"ctx5d_{out_var}",
                        shape=Shape.from_tuple((B, H_kv, G, S_q, D)),
                        dtype=out_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[ctx_4d_t.id],
                        outputs=[ctx_5d_t.id],
                        name=f"reshape_ctx5d_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.PERMUTE,
                        inputs=[ctx_5d_t.id],
                        outputs=[out_t.id],
                        attributes={"dims": [0, 1, 4, 2, 3]},
                        name=f"perm_out_{out_var}",
                    )
                    continue

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

                lhs_dims_val = [_dim_val(lhs_t.shape.dims[i]) for i in target_lhs_perm]
                rhs_dims_val = [_dim_val(rhs_t.shape.dims[i]) for i in target_rhs_perm]
                out_dims_val = [_dim_val(d) for d in out_t.shape.dims]

                if len(lhs_dims_val) > 4:
                    non_one_lhs = [d for d in lhs_dims_val if d != 1]
                    while len(non_one_lhs) < 4:
                        non_one_lhs.insert(0, 1)
                    lhs_4d_t = g.add_tensor(
                        name=f"lhs4d_{out_var}",
                        shape=Shape.from_tuple(tuple(non_one_lhs[:4])),
                        dtype=lhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[lhs_in_id],
                        outputs=[lhs_4d_t.id],
                        name=f"reshape_lhs4d_{out_var}",
                    )
                    lhs_in_id = lhs_4d_t.id

                if len(rhs_dims_val) > 4:
                    non_one_rhs = [d for d in rhs_dims_val if d != 1]
                    while len(non_one_rhs) < 4:
                        non_one_rhs.insert(0, 1)
                    rhs_4d_t = g.add_tensor(
                        name=f"rhs4d_{out_var}",
                        shape=Shape.from_tuple(tuple(non_one_rhs[:4])),
                        dtype=rhs_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[rhs_in_id],
                        outputs=[rhs_4d_t.id],
                        name=f"reshape_rhs4d_{out_var}",
                    )
                    rhs_in_id = rhs_4d_t.id

                if len(out_dims_val) > 4:
                    non_one_out = [d for d in out_dims_val if d != 1]
                    while len(non_one_out) < 4:
                        non_one_out.insert(0, 1)
                    bmm_4d_t = g.add_tensor(
                        name=f"bmm4d_{out_var}",
                        shape=Shape.from_tuple(tuple(non_one_out[:4])),
                        dtype=out_t.dtype,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.MATMUL,
                        inputs=[lhs_in_id, rhs_in_id],
                        outputs=[bmm_4d_t.id],
                        attributes=bmm_attrs,
                        name=f"bmm_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.RESHAPE,
                        inputs=[bmm_4d_t.id],
                        outputs=[out_t.id],
                        name=f"reshape_out_{out_var}",
                    )
                else:
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

        if prim_name == "integer_pow":
            in_t = g.get_tensor(in_tids[0])
            y = int(eqn.params.get("y", 1))
            if y == -1:
                one_t = g.add_tensor(
                    name=f"one_{len(g.tensors)}",
                    shape=Shape.from_tuple(()),
                    dtype=in_t.dtype,
                    storage=StorageClass.CONSTANT,
                    data=np.array(1.0, dtype=np.float32 if in_t.dtype == DType.F32 else np.int32),
                )
                g.parameters.append(one_t.id)
                g.add_op(
                    opcode=OpCode.DIV,
                    inputs=[one_t.id, in_t.id],
                    outputs=[out_t.id],
                    name=f"reciprocal_{out_var}",
                )
                continue
            elif y == 2:
                g.add_op(
                    opcode=OpCode.POW,
                    inputs=[in_t.id],
                    outputs=[out_t.id],
                    attributes={"exponent": 2},
                    name=f"sqr_{out_var}",
                )
                continue
            elif y == 3:
                sqr_t = g.add_tensor(
                    name=f"sqr_{out_var}",
                    shape=in_t.shape,
                    dtype=in_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.POW,
                    inputs=[in_t.id],
                    outputs=[sqr_t.id],
                    attributes={"exponent": 2},
                    name=f"sqr_{out_var}",
                )
                g.add_op(
                    opcode=OpCode.MUL,
                    inputs=[in_t.id, sqr_t.id],
                    outputs=[out_t.id],
                    name=f"cube_{out_var}",
                )
                continue

        if prim_name in ("gt", "ge", "lt", "le", "eq", "ne"):
            in0_t = g.get_tensor(in_tids[0])
            in1_t = g.get_tensor(in_tids[1])
            if in1_t.data is not None and np.all(in1_t.data == 0) and prim_name == "ne" or in1_t.data is not None and np.all(in1_t.data == 1) and prim_name == "eq":
                var_to_tensor[out_var] = in0_t
                continue
            else:
                t0_id = in0_t.id
                if in0_t.dtype != DType.F32:
                    c0_t = g.add_tensor(
                        name=f"cast_cmp0_{out_var}",
                        shape=in0_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[in0_t.id],
                        outputs=[c0_t.id],
                        attributes={"dtype": DType.F32},
                        name=f"cast_cmp0_{out_var}",
                    )
                    t0_id = c0_t.id

                t1_id = in1_t.id
                if in1_t.dtype != DType.F32:
                    c1_t = g.add_tensor(
                        name=f"cast_cmp1_{out_var}",
                        shape=in1_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[in1_t.id],
                        outputs=[c1_t.id],
                        attributes={"dtype": DType.F32},
                        name=f"cast_cmp1_{out_var}",
                    )
                    t1_id = c1_t.id

                diff_t = g.add_tensor(
                    name=f"diff_{out_var}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.SUB,
                    inputs=[t0_id, t1_id] if prim_name not in ("lt", "le") else [t1_id, t0_id],
                    outputs=[diff_t.id],
                    name=f"diff_{out_var}",
                )
                if prim_name in ("eq", "ne"):
                    abs_t = g.add_tensor(
                        name=f"abs_{out_var}",
                        shape=out_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.ABS,
                        inputs=[diff_t.id],
                        outputs=[abs_t.id],
                        name=f"abs_{out_var}",
                    )
                    clamp_t = g.add_tensor(
                        name=f"clamp_{out_var}",
                        shape=out_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CLAMP,
                        inputs=[abs_t.id],
                        outputs=[clamp_t.id],
                        attributes={"min": 0.0, "max": 1.0},
                        name=f"clamp_{out_var}",
                    )
                    if prim_name == "ne":
                        var_to_tensor[out_var] = clamp_t
                    else:
                        one_t = g.add_tensor(
                            name=f"one_{len(g.tensors)}",
                            shape=Shape.from_tuple(()),
                            dtype=DType.F32,
                            storage=StorageClass.CONSTANT,
                            data=np.array(1.0, dtype=np.float32),
                        )
                        g.parameters.append(one_t.id)
                        g.add_op(
                            opcode=OpCode.SUB,
                            inputs=[one_t.id, clamp_t.id],
                            outputs=[out_t.id],
                            name=f"eq_{out_var}",
                        )
                else:
                    g.add_op(
                        opcode=OpCode.CLAMP,
                        inputs=[diff_t.id],
                        outputs=[out_t.id],
                        attributes={"min": 0.0, "max": 1.0},
                        name=f"cmp_{out_var}",
                    )
                continue

        if prim_name in ("reduce_and", "reduce_or"):
            in_t = g.get_tensor(in_tids[0])
            axes = eqn.params.get("axes", (0,))
            dim = int(axes[0]) if isinstance(axes, (tuple, list)) and len(axes) > 0 else int(axes)
            in_dims = [_dim_val(d) for d in in_t.shape.dims]
            if dim < 0:
                dim += len(in_dims)
            N = in_dims[dim]

            sum_t = g.add_tensor(
                name=f"sum_{out_var}",
                shape=out_t.shape,
                dtype=DType.F32,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.SUM,
                inputs=[in_t.id],
                outputs=[sum_t.id],
                attributes={"axes": [dim]},
                name=f"sum_{out_var}",
            )
            if prim_name == "reduce_and":
                n_const = g.add_tensor(
                    name=f"n_const_{len(g.tensors)}",
                    shape=Shape.from_tuple(()),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=np.array(float(N), dtype=np.float32),
                )
                g.parameters.append(n_const.id)
                diff_n = g.add_tensor(
                    name=f"diff_n_{out_var}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.SUB,
                    inputs=[sum_t.id, n_const.id],
                    outputs=[diff_n.id],
                    name=f"diff_n_{out_var}",
                )
                abs_n = g.add_tensor(
                    name=f"abs_n_{out_var}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.ABS,
                    inputs=[diff_n.id],
                    outputs=[abs_n.id],
                    name=f"abs_n_{out_var}",
                )
                clamp_n = g.add_tensor(
                    name=f"clamp_n_{out_var}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.CLAMP,
                    inputs=[abs_n.id],
                    outputs=[clamp_n.id],
                    attributes={"min": 0.0, "max": 1.0},
                    name=f"clamp_n_{out_var}",
                )
                one_t = g.add_tensor(
                    name=f"one_{len(g.tensors)}",
                    shape=Shape.from_tuple(()),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=np.array(1.0, dtype=np.float32),
                )
                g.parameters.append(one_t.id)
                g.add_op(
                    opcode=OpCode.SUB,
                    inputs=[one_t.id, clamp_n.id],
                    outputs=[out_t.id],
                    name=f"reduce_and_{out_var}",
                )
            else:
                g.add_op(
                    opcode=OpCode.CLAMP,
                    inputs=[sum_t.id],
                    outputs=[out_t.id],
                    attributes={"min": 0.0, "max": 1.0},
                    name=f"reduce_or_{out_var}",
                )
            continue

        if prim_name == "select_n":
            # select_n(cond, false_branch, true_branch)
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
            elif cond_t is not None and cond_t.data is not None:
                # Static boolean scalar condition: choose branch directly
                cond_val = bool(
                    cond_t.data.item() if cond_t.data.ndim == 0 else cond_t.data.flatten()[0]
                )
                selected_t = true_t if cond_val else false_t
                if selected_t is not None:
                    var_to_tensor[out_var] = selected_t
                    continue
            elif cond_t is not None and true_t is not None and false_t is not None:
                # Dynamic condition: true_t * cond + false_t * (1 - cond)
                # Cast cond, true_t, false_t to F32 for arithmetic
                cond_float_t = cond_t
                if cond_t.dtype != DType.F32:
                    cond_float_t = g.add_tensor(
                        name=f"cond_f32_{len(g.tensors)}",
                        shape=cond_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[cond_t.id],
                        outputs=[cond_float_t.id],
                        attributes={"dtype": DType.F32},
                        name=f"cast_cond_{out_var}",
                    )

                true_float_t = true_t
                if true_t.dtype != DType.F32:
                    true_float_t = g.add_tensor(
                        name=f"true_f32_{len(g.tensors)}",
                        shape=true_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[true_t.id],
                        outputs=[true_float_t.id],
                        attributes={"dtype": DType.F32},
                        name=f"cast_true_{out_var}",
                    )

                false_float_t = false_t
                if false_t.dtype != DType.F32:
                    false_float_t = g.add_tensor(
                        name=f"false_f32_{len(g.tensors)}",
                        shape=false_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[false_t.id],
                        outputs=[false_float_t.id],
                        attributes={"dtype": DType.F32},
                        name=f"cast_false_{out_var}",
                    )

                if false_float_t.data is not None:
                    arr = np.asarray(false_float_t.data)
                    if arr.size == 1 and float(arr.item()) < -10000.0:
                        false_float_t.data = np.array(-10000.0, dtype=np.float32)

                if true_float_t.data is not None:
                    arr = np.asarray(true_float_t.data)
                    if arr.size == 1 and float(arr.item()) < -10000.0:
                        true_float_t.data = np.array(-10000.0, dtype=np.float32)

                one_t = g.add_tensor(
                    name=f"one_{len(g.tensors)}",
                    shape=Shape.from_tuple(()),
                    dtype=DType.F32,
                    storage=StorageClass.CONSTANT,
                    data=np.array(1.0, dtype=np.float32),
                    role="constant",
                )
                g.parameters.append(one_t.id)

                inv_cond_t = g.add_tensor(
                    name=f"inv_cond_{len(g.tensors)}",
                    shape=cond_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.SUB,
                    inputs=[one_t.id, cond_float_t.id],
                    outputs=[inv_cond_t.id],
                    name=f"sub_cond_{out_var}",
                )

                false_term_t = g.add_tensor(
                    name=f"false_term_{len(g.tensors)}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.MUL,
                    inputs=[inv_cond_t.id, false_float_t.id],
                    outputs=[false_term_t.id],
                    name=f"mul_false_{out_var}",
                )

                true_term_t = g.add_tensor(
                    name=f"true_term_{len(g.tensors)}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.MUL,
                    inputs=[cond_float_t.id, true_float_t.id],
                    outputs=[true_term_t.id],
                    name=f"mul_true_{out_var}",
                )

                if out_t.dtype == DType.F32:
                    g.add_op(
                        opcode=OpCode.ADD,
                        inputs=[true_term_t.id, false_term_t.id],
                        outputs=[out_t.id],
                        name=f"where_{out_var}",
                    )
                else:
                    sum_f32_t = g.add_tensor(
                        name=f"sum_f32_{len(g.tensors)}",
                        shape=out_t.shape,
                        dtype=DType.F32,
                        storage=StorageClass.ACTIVATION,
                    )
                    g.add_op(
                        opcode=OpCode.ADD,
                        inputs=[true_term_t.id, false_term_t.id],
                        outputs=[sum_f32_t.id],
                        name=f"where_f32_{out_var}",
                    )
                    g.add_op(
                        opcode=OpCode.CAST,
                        inputs=[sum_f32_t.id],
                        outputs=[out_t.id],
                        attributes={"dtype": out_t.dtype},
                        name=f"where_cast_{out_var}",
                    )
                continue

        if prim_name == "div" and len(eqn.invars) == 2:
            num_v, den_v = eqn.invars[0], eqn.invars[1]
            if (
                not hasattr(num_v, "val")
                and not hasattr(den_v, "val")
                and num_v in softmax_state["exp_vars"]
                and den_v in softmax_state["sum_vars"]
                and softmax_state["exp_vars"][num_v] == softmax_state["sum_vars"][den_v]
            ):
                x_var, axes = softmax_state["exp_vars"][num_v]
                x_t = var_to_tensor[x_var]
                dim = (
                    int(axes[0]) if isinstance(axes, (tuple, list)) and len(axes) > 0 else int(axes)
                )
                g.add_op(
                    opcode=OpCode.SOFTMAX,
                    inputs=[x_t.id],
                    outputs=[out_t.id],
                    attributes={"dim": dim},
                    name=f"softmax_{out_var}",
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

        if prim_name == "neg":
            gelu_state["neg_vars"][eqn.outvars[0]] = eqn.invars[0]

        if prim_name == "mul" and len(eqn.invars) == 2:
            v0, v1 = eqn.invars[0], eqn.invars[1]
            val0 = (
                v0.val
                if hasattr(v0, "val")
                else (
                    var_to_tensor[v0].data
                    if v0 in var_to_tensor and var_to_tensor[v0].data is not None
                    else None
                )
            )
            val1 = (
                v1.val
                if hasattr(v1, "val")
                else (
                    var_to_tensor[v1].data
                    if v1 in var_to_tensor and var_to_tensor[v1].data is not None
                    else None
                )
            )

            if val0 is not None and np.allclose(val0, 0.5):
                gelu_state["half_vars"][eqn.outvars[0]] = v1
            elif val1 is not None and np.allclose(val1, 0.5):
                gelu_state["half_vars"][eqn.outvars[0]] = v0

            if (
                val0 is not None
                and np.allclose(val0, 0.70710678, atol=1e-4)
                and not hasattr(v1, "val")
                and v1 in gelu_state["neg_vars"]
            ):
                gelu_state["scaled_neg_vars"][eqn.outvars[0]] = gelu_state["neg_vars"][v1]
            elif (
                val1 is not None
                and np.allclose(val1, 0.70710678, atol=1e-4)
                and not hasattr(v0, "val")
                and v0 in gelu_state["neg_vars"]
            ):
                gelu_state["scaled_neg_vars"][eqn.outvars[0]] = gelu_state["neg_vars"][v0]

            x_orig = None
            if (
                not hasattr(v0, "val")
                and v0 in gelu_state["erfc_vars"]
                and not hasattr(v1, "val")
                and gelu_state["half_vars"].get(v1) == gelu_state["erfc_vars"][v0]
            ):
                x_orig = gelu_state["erfc_vars"][v0]
            elif (
                not hasattr(v1, "val")
                and v1 in gelu_state["erfc_vars"]
                and not hasattr(v0, "val")
                and gelu_state["half_vars"].get(v0) == gelu_state["erfc_vars"][v1]
            ):
                x_orig = gelu_state["erfc_vars"][v1]

            if x_orig is not None and x_orig in var_to_tensor:
                x_t = var_to_tensor[x_orig]
                g.add_op(
                    opcode=OpCode.GELU,
                    inputs=[x_t.id],
                    outputs=[out_t.id],
                    name=f"gelu_{out_var}",
                )
                continue

        if prim_name == "erfc":
            in_var = eqn.invars[0]
            if not hasattr(in_var, "val") and in_var in gelu_state["scaled_neg_vars"]:
                gelu_state["erfc_vars"][eqn.outvars[0]] = gelu_state["scaled_neg_vars"][in_var]
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

        if prim_name == "min" and len(in_tids) == 2:
            in0_t = g.get_tensor(in_tids[0])
            in1_t = g.get_tensor(in_tids[1])
            t0_id = in0_t.id
            t1_id = in1_t.id
            if in0_t.dtype != DType.F32:
                c0_t = g.add_tensor(
                    name=f"cast_min0_{out_var}",
                    shape=in0_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.CAST,
                    inputs=[in0_t.id],
                    outputs=[c0_t.id],
                    attributes={"dtype": DType.F32},
                    name=f"cast_min0_{out_var}",
                )
                t0_id = c0_t.id

            if in1_t.dtype != DType.F32:
                c1_t = g.add_tensor(
                    name=f"cast_min1_{out_var}",
                    shape=in1_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.CAST,
                    inputs=[in1_t.id],
                    outputs=[c1_t.id],
                    attributes={"dtype": DType.F32},
                    name=f"cast_min1_{out_var}",
                )
                t1_id = c1_t.id

            diff_t = g.add_tensor(
                name=f"diff_min_{out_var}",
                shape=out_t.shape,
                dtype=DType.F32,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.SUB,
                inputs=[t0_id, t1_id],
                outputs=[diff_t.id],
                name=f"diff_min_{out_var}",
            )
            relu_t = g.add_tensor(
                name=f"relu_min_{out_var}",
                shape=out_t.shape,
                dtype=DType.F32,
                storage=StorageClass.ACTIVATION,
            )
            g.add_op(
                opcode=OpCode.RELU,
                inputs=[diff_t.id],
                outputs=[relu_t.id],
                name=f"relu_min_{out_var}",
            )
            sub_out_id = out_t.id
            if out_t.dtype != DType.F32:
                sub_f32_t = g.add_tensor(
                    name=f"sub_min_f32_{out_var}",
                    shape=out_t.shape,
                    dtype=DType.F32,
                    storage=StorageClass.ACTIVATION,
                )
                sub_out_id = sub_f32_t.id

            g.add_op(
                opcode=OpCode.SUB,
                inputs=[t0_id, relu_t.id],
                outputs=[sub_out_id],
                name=f"sub_min_{out_var}",
            )

            if out_t.dtype != DType.F32:
                g.add_op(
                    opcode=OpCode.CAST,
                    inputs=[sub_out_id],
                    outputs=[out_t.id],
                    attributes={"dtype": out_t.dtype},
                    name=f"cast_min_out_{out_var}",
                )
            continue

        if prim_name == "dynamic_slice":
            in_t = g.get_tensor(in_tids[0])
            slice_sizes = eqn.params.get("slice_sizes", ())
            start_indices = []
            for v in eqn.invars[1:]:
                if hasattr(v, "val"):
                    start_indices.append(int(v.val))
                elif (
                    not hasattr(v, "val")
                    and v in var_to_tensor
                    and var_to_tensor[v].data is not None
                ):
                    start_indices.append(int(var_to_tensor[v].data.item()))
                else:
                    start_indices.append(0)
            limit_indices = [s + sz for s, sz in zip(start_indices, slice_sizes)]
            attrs = {
                "start_indices": list(start_indices),
                "limit_indices": list(limit_indices),
                "strides": [1] * len(start_indices),
            }
            g.add_op(
                opcode=OpCode.SLICE,
                inputs=[in_t.id],
                outputs=[out_t.id],
                attributes=attrs,
                name=f"dynamic_slice_{out_var}",
            )
            continue

        if prim_name == "split":
            in_t = g.get_tensor(in_tids[0])
            axis = int(eqn.params.get("axis", eqn.params.get("dimension", -1)))
            sizes = tuple(eqn.params.get("sizes", ()))
            in_shape_dims = [_dim_val(d) for d in in_t.shape.dims]
            if axis < 0:
                axis += len(in_shape_dims)

            offset = 0
            for i, out_v in enumerate(eqn.outvars):
                split_sz = (
                    int(sizes[i]) if i < len(sizes) else (in_shape_dims[axis] // len(eqn.outvars))
                )
                out_av = out_v.aval
                out_sh = Shape.from_tuple(tuple(out_av.shape))
                out_sp_t = g.add_tensor(
                    name=f"split_{out_v}",
                    shape=out_sh,
                    dtype=_jax_dtype_to_dtype(out_av.dtype),
                    storage=StorageClass.ACTIVATION,
                )
                var_to_tensor[out_v] = out_sp_t

                start_indices = [0] * len(in_shape_dims)
                limit_indices = list(in_shape_dims)
                start_indices[axis] = offset
                limit_indices[axis] = offset + split_sz

                g.add_op(
                    opcode=OpCode.SLICE,
                    inputs=[in_t.id],
                    outputs=[out_sp_t.id],
                    attributes={
                        "start_indices": start_indices,
                        "limit_indices": limit_indices,
                        "strides": [1] * len(start_indices),
                    },
                    name=f"split_slice_{out_v}",
                )
                offset += split_sz
            continue

        if prim_name == "stack":
            dim = int(eqn.params.get("dimension", eqn.params.get("axis", 0)))
            out_shape_dims = [_dim_val(d) for d in out_t.shape.dims]
            if dim < 0:
                dim += len(out_shape_dims)

            expanded_ids = []
            for i, in_tid in enumerate(in_tids):
                in_t = g.get_tensor(in_tid)
                in_dims = [_dim_val(d) for d in in_t.shape.dims]
                exp_dims = list(in_dims)
                exp_dims.insert(dim, 1)
                exp_t = g.add_tensor(
                    name=f"exp_stack_{i}_{out_var}",
                    shape=Shape.from_tuple(tuple(exp_dims)),
                    dtype=in_t.dtype,
                    storage=StorageClass.ACTIVATION,
                )
                g.add_op(
                    opcode=OpCode.RESHAPE,
                    inputs=[in_t.id],
                    outputs=[exp_t.id],
                    name=f"reshape_stack_{i}_{out_var}",
                )
                expanded_ids.append(exp_t.id)

            g.add_op(
                opcode=OpCode.CONCAT,
                inputs=expanded_ids,
                outputs=[out_t.id],
                attributes={"dim": dim},
                name=f"stack_concat_{out_var}",
            )
            continue

        if opcode is None:
            raise NotImplementedError(
                f"Unhandled primitive '{prim_name}' reached op emission with None opcode."
            )

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
