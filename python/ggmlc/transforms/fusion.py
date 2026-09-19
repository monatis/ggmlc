"""Graph transformation passes for targeted operator fusion (ggmlc-fused)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.shape import Shape, StaticDim
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


@dataclass
class FusionOptions:
    """Configuration options for enabling or disabling individual fusion passes."""

    enable_bias_gelu: bool = True
    enable_approx_gelu: bool = True
    enable_layer_norm: bool = True
    enable_rms_norm: bool = True
    enable_swiglu: bool = True
    enable_conv2d_relu: bool = True
    enable_softmax: bool = True
    enable_horizontal_mlp: bool = True
    enable_horizontal_qkv: bool = True
    enable_rope: bool = True
    enable_sdpa_transpose: bool = True
    # Bake RMSNorm gamma into following Linear weight columns at compile time,
    # then emit weightless RMS_NORM. Removes the post-norm MUL (and gamma tensor)
    # without new CUDA kernels. Must run after horizontal fusion; quantize sees
    # already-scaled W. Default ON after SmolLM/Qwen/LLaMA A/B (stable, often faster).
    enable_bake_rms_into_linear: bool = True


class OperatorFusionPass(Pass):
    """Compiler transformation pass that fuses eligible operator patterns."""

    def __init__(self, options: FusionOptions | None = None, name: str = "OperatorFusionPass"):
        super().__init__(name)
        self.options = options or FusionOptions()

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        fuse_operations(graph, self.options)

        nodes_after = len(graph.nodes)
        tensors_after = len(graph.tensors)
        fusions_applied = max(0, nodes_before - nodes_after)

        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            tensors_before=tensors_before,
            tensors_after=tensors_after,
            fusions_applied=fusions_applied,
        )
        return GraphTransformResult(
            graph=graph,
            modified=(nodes_before != nodes_after),
            stats=stats,
        )


def fuse_operations(graph: Graph, options: FusionOptions | None = None) -> Graph:
    """Applies pattern-matching fusion rewrites to a Canonical IR Graph.

    Returns a new or modified Graph with fused operations where applicable.
    """
    if options is None:
        options = FusionOptions()

    if options.enable_rope:
        _fuse_rope_patterns(graph)

    if options.enable_layer_norm:
        _fuse_layer_norm_patterns(graph)

    if options.enable_rms_norm:
        _fuse_rms_norm_patterns(graph)

    if options.enable_softmax:
        _fuse_softmax_patterns(graph)

    if options.enable_conv2d_relu:
        _fuse_conv2d_relu_patterns(graph)

    if options.enable_swiglu:
        _fuse_swiglu_patterns(graph)

    if options.enable_approx_gelu or options.enable_bias_gelu:
        _fuse_approx_gelu_patterns(graph)

    if options.enable_bias_gelu:
        _fuse_bias_gelu_patterns(graph)

    if options.enable_horizontal_mlp or options.enable_horizontal_qkv:
        import os

        if os.environ.get("GGMLC_DEBUG_FUSION"):
            print(
                f"[fusion] horizontal mlp={options.enable_horizontal_mlp} "
                f"qkv={options.enable_horizontal_qkv}",
                flush=True,
            )
        _fuse_horizontal_linear_patterns(graph, options)

    if options.enable_swiglu and options.enable_horizontal_mlp:
        _fuse_swiglu_concatenated_patterns(graph)

    if options.enable_sdpa_transpose:
        _fuse_sdpa_transpose_patterns(graph)

    # After horizontal fusion so QKV / gate+up are single Linears (one bake each).
    if options.enable_bake_rms_into_linear:
        _bake_rms_weights_into_linears(graph)

    return graph


def _bake_rms_weights_into_linears(graph: Graph) -> int:
    """Absorb RMSNorm gamma into consumer Linear weights; leave weightless RMS_NORM.

    Math (PyTorch Linear ``W[out, in]``, ``y = x_scaled @ W.T``):
      ``x_scaled[..., k] = rms(x)[..., k] * gamma[k]``
      ``W'[j, k] = W[j, k] * gamma[k]``  (scale columns / in_features)

    For HF Conv1D / ``aten.addmm`` weights stored as ``W[in, out]``, scale rows instead.

    Only rewrites when every consumer of the RMS output is a LINEAR whose activation
    input is that tensor (no residual / view fanout). Returns number of RMS ops baked.
    """
    consumers: dict[int, list[Operation]] = {}
    for op in graph.nodes:
        for in_id in op.inputs:
            consumers.setdefault(in_id, []).append(op)

    baked = 0
    for op in graph.nodes:
        if op.opcode != OpCode.RMS_NORM or len(op.inputs) < 2:
            continue

        x_id, gamma_id = op.inputs[0], op.inputs[1]
        gamma_t = graph.get_tensor(gamma_id)
        if (
            gamma_t is None
            or gamma_t.data is None
            or gamma_t.storage not in (StorageClass.PARAMETER, StorageClass.CONSTANT)
        ):
            continue

        gamma = np.asarray(gamma_t.data, dtype=np.float32).reshape(-1)
        if gamma.size == 0:
            continue

        rms_out = op.outputs[0]
        outs = consumers.get(rms_out, [])
        if not outs:
            continue
        if any(c.opcode != OpCode.LINEAR or not c.inputs or c.inputs[0] != rms_out for c in outs):
            continue

        # Validate every Linear can absorb gamma before mutating any weight.
        # Skip when a Linear weight is shared with non-consumer ops (e.g. tied
        # embed_tokens.weight used by both EMBEDDING and lm_head) — in-place
        # bake would corrupt the other use.
        allowed_op_ids = {c.id for c in outs}
        planned: list[tuple[int, np.ndarray]] = []  # weight_id → scaled data
        seen_w: set[int] = set()
        ok = True
        for lin in outs:
            w_id = lin.inputs[1]
            w_t = graph.get_tensor(w_id)
            if (
                w_t is None
                or w_t.data is None
                or not hasattr(w_t.data, "ndim")
                or w_t.data.ndim != 2
                or w_t.storage != StorageClass.PARAMETER
            ):
                ok = False
                break
            for other in graph.nodes:
                if w_id in other.inputs and other.id not in allowed_op_ids:
                    ok = False
                    break
            if not ok:
                break
            if w_id in seen_w:
                continue
            seen_w.add(w_id)
            W = np.asarray(w_t.data, dtype=np.float32)
            is_addmm = lin.attributes.get("is_addmm", 0) != 0
            # addmm / Conv1D: (in, out); standard Linear: (out, in)
            if is_addmm or (W.shape[0] == gamma.size and W.shape[1] != gamma.size):
                if W.shape[0] != gamma.size:
                    ok = False
                    break
                W_new = W * gamma[:, np.newaxis]
            else:
                if W.shape[1] != gamma.size:
                    ok = False
                    break
                W_new = W * gamma[np.newaxis, :]
            planned.append((w_id, W_new))

        if not ok or not planned:
            continue

        for w_id, W_new in planned:
            graph.get_tensor(w_id).data = np.ascontiguousarray(W_new)

        op.inputs = [x_id]
        baked += 1

    return baked


def _fuse_rope_patterns(graph: Graph) -> None:
    """Matches decomposed Rotary Position Embedding (RoPE) subgraphs from PyTorch:
    x * cos + rotate_half(x) * sin -> ROPE(x, pos)
    where rotate_half(x) = cat((-x[..., d//2:], x[..., :d//2]), dim=-1).
    """
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.ADD and len(op.inputs) == 2:
            in0_id, in1_id = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0_id)
            prod1 = producer_map.get(in1_id)

            if prod0 and prod1 and prod0.opcode == OpCode.MUL and prod1.opcode == OpCode.MUL:
                matched = False
                for prod_cos, prod_sin in [(prod0, prod1), (prod1, prod0)]:
                    sin_input_cands = [producer_map.get(inp) for inp in prod_sin.inputs]
                    rot_cand = next(
                        (p for p in sin_input_cands if p and p.opcode == OpCode.CONCAT), None
                    )
                    if not rot_cand or len(rot_cand.inputs) != 2:
                        continue

                    neg_cand = producer_map.get(rot_cand.inputs[0])
                    slice1_cand = producer_map.get(rot_cand.inputs[1])
                    if not (neg_cand and neg_cand.opcode == OpCode.NEG):
                        neg_cand = producer_map.get(rot_cand.inputs[1])
                        slice1_cand = producer_map.get(rot_cand.inputs[0])

                    if not (
                        neg_cand
                        and neg_cand.opcode == OpCode.NEG
                        and slice1_cand
                        and slice1_cand.opcode == OpCode.SLICE
                    ):
                        continue

                    slice2_cand = producer_map.get(neg_cand.inputs[0])
                    if not (slice2_cand and slice2_cand.opcode == OpCode.SLICE):
                        continue

                    if slice1_cand.inputs[0] != slice2_cand.inputs[0]:
                        continue
                    x_id = slice1_cand.inputs[0]

                    if x_id not in prod_cos.inputs:
                        continue

                    cos_id = (
                        prod_cos.inputs[0] if prod_cos.inputs[1] == x_id else prod_cos.inputs[1]
                    )
                    s1_end = slice1_cand.attributes.get("end", 0)
                    n_dims = int(s1_end * 2) if s1_end > 0 else 64

                    # Trace up from cos_id to find pos and freq_base
                    curr = producer_map.get(cos_id)
                    while curr and curr.opcode in (OpCode.UNSQUEEZE, OpCode.VIEW, OpCode.RESHAPE):
                        curr = producer_map.get(curr.inputs[0]) if curr.inputs else None

                    pos_id = None
                    freq_base = 10000.0
                    if curr and curr.opcode == OpCode.COS:
                        emb_prod = producer_map.get(curr.inputs[0])
                        while emb_prod and emb_prod.opcode in (
                            OpCode.CONCAT,
                            OpCode.VIEW,
                            OpCode.RESHAPE,
                        ):
                            emb_prod = (
                                producer_map.get(emb_prod.inputs[0]) if emb_prod.inputs else None
                            )
                        if emb_prod and emb_prod.opcode == OpCode.MUL:
                            for inp in emb_prod.inputs:
                                inp_op = producer_map.get(inp)
                                actual_t_id = (
                                    inp_op.inputs[0]
                                    if inp_op and inp_op.opcode == OpCode.UNSQUEEZE
                                    else inp
                                )
                                t = graph.get_tensor(actual_t_id)
                                if (
                                    t
                                    and t.data is not None
                                    and t.data.ndim == 1
                                    and len(t.data) > 1
                                ):
                                    arr = t.data
                                    if len(arr) > 1 and arr[1] > 0:
                                        freq_base = float((1.0 / arr[1]) ** (n_dims / 2.0))
                                else:
                                    pos_id = actual_t_id

                    # If pos_id was not discovered via graph tracing, search for arange or pos symbol
                    if pos_id is None:
                        for t in graph.tensors.values():
                            if t.name in ("pos", "positions", "position_ids") or "arange" in t.name:
                                pos_id = t.id
                                break

                    # Ensure position tensor is I32 for GGML rope kernel
                    if pos_id is not None:
                        t_pos = graph.get_tensor(pos_id)
                        if t_pos is not None:
                            t_pos.dtype = DType.I32
                            if t_pos.data is not None and hasattr(t_pos.data, "astype"):
                                t_pos.data = np.ascontiguousarray(t_pos.data.astype(np.int32))

                    ops_to_remove.add(prod_cos.id)
                    ops_to_remove.add(prod_sin.id)
                    ops_to_remove.add(rot_cand.id)
                    ops_to_remove.add(neg_cand.id)
                    ops_to_remove.add(slice1_cand.id)
                    ops_to_remove.add(slice2_cand.id)
                    ops_to_remove.add(op.id)
                    # Check if x_id comes from a TRANSPOSE(pre_trans, dim0=1, dim1=2)
                    trans_prod = producer_map.get(x_id)
                    is_pre_trans = (
                        trans_prod is not None
                        and trans_prod.opcode == OpCode.TRANSPOSE
                        and len(trans_prod.inputs) == 1
                        and (
                            (
                                trans_prod.attributes.get("dim0") == 1
                                and trans_prod.attributes.get("dim1") == 2
                            )
                            or (
                                trans_prod.attributes.get("dim0") == 2
                                and trans_prod.attributes.get("dim1") == 1
                            )
                        )
                    )

                    if is_pre_trans:
                        pre_trans_id = trans_prod.inputs[0]
                        pre_trans_t = graph.get_tensor(pre_trans_id)

                        rope_out_t = graph.add_tensor(
                            name=f"{pre_trans_t.name if pre_trans_t else 'act'}_rope_fused",
                            shape=pre_trans_t.shape
                            if pre_trans_t
                            else graph.get_tensor(x_id).shape,
                            dtype=graph.get_tensor(x_id).dtype,
                            storage=StorageClass.ACTIVATION,
                        )

                        fused_rope_op = Operation(
                            id=graph.new_op_id(),
                            opcode=OpCode.ROPE,
                            inputs=[pre_trans_id, pos_id if pos_id is not None else 0],
                            outputs=[rope_out_t.id],
                            attributes={
                                "n_dims": n_dims,
                                "mode": 2,  # GGML_ROPE_TYPE_NEOX
                                "freq_base": freq_base,
                                "freq_scale": 1.0,
                            },
                            name=f"{op.name or 'rope'}_fused",
                        )
                        rope_out_t.producer_id = fused_rope_op.id

                        trans_op = Operation(
                            id=graph.new_op_id(),
                            opcode=OpCode.TRANSPOSE,
                            inputs=[rope_out_t.id],
                            outputs=list(op.outputs),
                            attributes=dict(trans_prod.attributes),
                            name=f"{op.name or 'rope'}_transposed",
                        )
                        out_t = graph.get_tensor(op.outputs[0])
                        if out_t is not None:
                            out_t.producer_id = trans_op.id

                        if consumer_counts.get(x_id, 0) <= 3:
                            ops_to_remove.add(trans_prod.id)

                        new_nodes.append(fused_rope_op)
                        new_nodes.append(trans_op)
                    else:
                        fused_rope_op = Operation(
                            id=graph.new_op_id(),
                            opcode=OpCode.ROPE,
                            inputs=[x_id, pos_id if pos_id is not None else 0],
                            outputs=list(op.outputs),
                            attributes={
                                "n_dims": n_dims,
                                "mode": 2,  # GGML_ROPE_TYPE_NEOX
                                "freq_base": freq_base,
                                "freq_scale": 1.0,
                            },
                            name=f"{op.name or 'rope'}_fused",
                        )
                        out_t = graph.get_tensor(op.outputs[0])
                        if out_t is not None:
                            out_t.producer_id = fused_rope_op.id
                        new_nodes.append(fused_rope_op)

                    matched = True
                    break

                if matched:
                    continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


def _fuse_layer_norm_patterns(graph: Graph) -> None:
    """Matches decomposed LayerNorm subgraphs (e.g. from JAX/XLA) and fuses into LAYER_NORM."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.ADD and len(op.inputs) == 2:
            in0, in1 = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0)
            prod1 = producer_map.get(in1)

            mul_x_op = None
            bias_term_op = None
            if prod0 and prod0.opcode == OpCode.MUL:
                mul_x_op = prod0
                bias_term_op = prod1
            elif prod1 and prod1.opcode == OpCode.MUL:
                mul_x_op = prod1
                bias_term_op = prod0

            if mul_x_op is not None and len(mul_x_op.inputs) == 2:
                cand_x_0, cand_rstd_gamma_0 = mul_x_op.inputs[0], mul_x_op.inputs[1]
                prod_rg = producer_map.get(cand_rstd_gamma_0)
                x_id = cand_x_0
                if prod_rg is None or prod_rg.opcode != OpCode.MUL:
                    prod_rg = producer_map.get(cand_x_0)
                    x_id = cand_rstd_gamma_0

                # If x_id is SUB(x, mean), extract true input x
                prod_x = producer_map.get(x_id)
                if prod_x and prod_x.opcode == OpCode.SUB and len(prod_x.inputs) == 2:
                    x_id = prod_x.inputs[0]
                    prod_x = producer_map.get(x_id)

                # Ensure x_id is valid and rank <= 3
                x_t = graph.get_tensor(x_id) if x_id in graph.tensors else None
                is_valid_x = prod_x is None or prod_x.opcode not in (
                    OpCode.NEG,
                    OpCode.DIV,
                    OpCode.SUM,
                    OpCode.RSQRT,
                )
                is_valid_rank = x_t is not None and len(x_t.shape.dims) <= 3

                if (
                    is_valid_x
                    and is_valid_rank
                    and prod_rg is not None
                    and prod_rg.opcode == OpCode.MUL
                ):
                    rg_in0, rg_in1 = prod_rg.inputs[0], prod_rg.inputs[1]
                    rsqrt_op = producer_map.get(rg_in0)
                    gamma_id = rg_in1
                    if rsqrt_op is None or rsqrt_op.opcode != OpCode.RSQRT:
                        rsqrt_op = producer_map.get(rg_in1)
                        gamma_id = rg_in0

                    if rsqrt_op is not None and rsqrt_op.opcode == OpCode.RSQRT:
                        var_add_op = producer_map.get(rsqrt_op.inputs[0])
                        eps = 1e-5
                        if var_add_op and var_add_op.opcode == OpCode.ADD:
                            for inp_t_id in var_add_op.inputs:
                                t = graph.get_tensor(inp_t_id)
                                if (
                                    t
                                    and t.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)
                                    and t.data is not None
                                    and float(t.data) > 0.0
                                ):
                                    eps = float(t.data)

                        # Verify that var_add_op is a direct reduction of x_id (not across conv/matmul layers)
                        is_direct_norm = False
                        curr = rsqrt_op.inputs[0] if rsqrt_op.inputs else None
                        for _ in range(6):
                            if curr == x_id:
                                is_direct_norm = True
                                break
                            p = producer_map.get(curr)
                            if not p or p.opcode in (OpCode.CONV2D, OpCode.MATMUL, OpCode.LINEAR):
                                break
                            if x_id in p.inputs:
                                is_direct_norm = True
                                break
                            curr = p.inputs[0] if p.inputs else None

                        if is_direct_norm:
                            beta_id = None
                            if bias_term_op and bias_term_op.opcode == OpCode.ADD:
                                b_in0, b_in1 = bias_term_op.inputs[0], bias_term_op.inputs[1]
                                t0 = graph.get_tensor(b_in0)
                                t1 = graph.get_tensor(b_in1)
                                if t1 and t1.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = b_in1
                                elif t0 and t0.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = b_in0
                            elif bias_term_op is None:
                                # Direct beta parameter (e.g. in Flax)
                                other_in = in1 if mul_x_op == prod0 else in0
                                t_other = graph.get_tensor(other_in)
                                if t_other and t_other.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = other_in

                            # Squeeze/match gamma and beta shapes to 1D if needed
                            for param_cand_id in (gamma_id, beta_id):
                                if param_cand_id is not None:
                                    p_t = graph.get_tensor(param_cand_id)
                                    if p_t and len(p_t.shape.dims) > 1:
                                        # Reshape to 1D
                                        from ggmlc.ir.shape import Shape

                                        last_d = p_t.shape.dims[-1]
                                        p_t.shape = Shape([last_d])
                                        if p_t.data is not None:
                                            p_t.data = p_t.data.reshape(-1)

                            ops_to_remove.add(op.id)

                            ln_inputs = [x_id]
                            if gamma_id is not None:
                                ln_inputs.append(gamma_id)
                            if beta_id is not None:
                                ln_inputs.append(beta_id)

                            fused_id = graph.new_op_id()
                            fused_op = Operation(
                                id=fused_id,
                                opcode=OpCode.LAYER_NORM,
                                inputs=ln_inputs,
                                outputs=list(op.outputs),
                                attributes={"eps": eps},
                                name=f"{op.name or 'layer_norm'}_fused",
                            )
                            new_nodes.append(fused_op)
                            continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


def _fuse_rms_norm_patterns(graph: Graph) -> None:
    """Matches decomposed RMSNorm subgraphs (e.g. from JAX/XLA) and fuses into RMS_NORM."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.MUL and len(op.inputs) == 2:
            in0, in1 = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0)
            prod1 = producer_map.get(in1)

            # RMSNorm output is MUL(MUL(x, rstd), gamma) or MUL(x, MUL(rstd, gamma))
            rstd_op = None
            x_id = None
            gamma_id = None

            if prod0 and prod0.opcode == OpCode.MUL:
                # in0 is MUL(x, rstd) or MUL(rstd, gamma)
                gamma_cand = graph.get_tensor(in1)
                if gamma_cand and gamma_cand.storage in (
                    StorageClass.PARAMETER,
                    StorageClass.CONSTANT,
                ):
                    gamma_id = in1
                    sub_prod0 = producer_map.get(prod0.inputs[0])
                    sub_prod1 = producer_map.get(prod0.inputs[1])
                    if sub_prod0 and sub_prod0.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod0
                        x_id = prod0.inputs[1]
                    elif sub_prod1 and sub_prod1.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod1
                        x_id = prod0.inputs[0]
            elif prod1 and prod1.opcode == OpCode.MUL:
                gamma_cand = graph.get_tensor(in0)
                if gamma_cand and gamma_cand.storage in (
                    StorageClass.PARAMETER,
                    StorageClass.CONSTANT,
                ):
                    gamma_id = in0
                    sub_prod0 = producer_map.get(prod1.inputs[0])
                    sub_prod1 = producer_map.get(prod1.inputs[1])
                    if sub_prod0 and sub_prod0.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod0
                        x_id = prod1.inputs[1]
                    elif sub_prod1 and sub_prod1.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod1
                        x_id = prod1.inputs[0]

            if rstd_op is not None and x_id is not None and gamma_id is not None:
                # Ensure tensor is 1D, 2D, 3D, or 4D where innermost dimension is ne0
                x_t = graph.get_tensor(x_id)
                if x_t and len(x_t.shape.dims) <= 4:
                    var_add_op = producer_map.get(rstd_op.inputs[0])
                    eps = 1e-5
                    if var_add_op and var_add_op.opcode == OpCode.ADD:
                        for inp_t_id in var_add_op.inputs:
                            t = graph.get_tensor(inp_t_id)
                            if (
                                t
                                and t.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)
                                and t.data is not None
                                and float(t.data) > 0.0
                            ):
                                eps = float(t.data)

                    # Verify that var_add_op is a direct reduction of x_id
                    is_direct_norm = False
                    curr = rstd_op.inputs[0] if rstd_op.inputs else None
                    for _ in range(6):
                        if curr == x_id:
                            is_direct_norm = True
                            break
                        p = producer_map.get(curr)
                        if not p or p.opcode in (OpCode.CONV2D, OpCode.MATMUL, OpCode.LINEAR):
                            break
                        if x_id in p.inputs:
                            is_direct_norm = True
                            break
                        curr = p.inputs[0] if p.inputs else None

                    if is_direct_norm:
                        # Reshape gamma to 1D if needed
                        p_t = graph.get_tensor(gamma_id)
                        if p_t and len(p_t.shape.dims) > 1:
                            from ggmlc.ir.shape import Shape

                            last_d = p_t.shape.dims[-1]
                            p_t.shape = Shape([last_d])
                            if p_t.data is not None:
                                p_t.data = p_t.data.reshape(-1)

                        ops_to_remove.add(op.id)

                        fused_id = graph.new_op_id()
                        fused_op = Operation(
                            id=fused_id,
                            opcode=OpCode.RMS_NORM,
                            inputs=[x_id, gamma_id],
                            outputs=list(op.outputs),
                            attributes={"eps": eps},
                            name=f"{op.name or 'rms_norm'}_fused",
                        )
                        new_nodes.append(fused_op)
                        continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


def _fuse_conv2d_relu_patterns(graph: Graph) -> None:
    """Fuses CONV2D followed by RELU into a single CONV2D with fused_activation attribute."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()

    for op in graph.nodes:
        if op.opcode == OpCode.RELU and len(op.inputs) == 1:
            inp_id = op.inputs[0]
            prod = producer_map.get(inp_id)
            if (
                prod
                and prod.opcode == OpCode.CONV2D
                and consumer_counts.get(prod.outputs[0], 0) <= 1
            ):
                prod.attributes["fused_relu"] = True
                prod.attributes["fused_activation"] = "relu"
                prod.outputs = list(op.outputs)
                if op.outputs[0] in graph.tensors:
                    graph.tensors[op.outputs[0]].producer_id = prod.id
                ops_to_remove.add(op.id)

    graph.nodes = [n for n in graph.nodes if n.id not in ops_to_remove]


def _fuse_swiglu_patterns(graph: Graph) -> None:
    """Matches Silu(gate) * up and replaces with SWIGLU(gate, up)."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.MUL and len(op.inputs) == 2:
            in0_id, in1_id = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0_id)
            prod1 = producer_map.get(in1_id)

            gate_id = None
            up_id = None
            silu_op = None

            if prod0 and prod0.opcode == OpCode.SILU:
                silu_op = prod0
                gate_id = prod0.inputs[0]
                up_id = in1_id
            elif prod1 and prod1.opcode == OpCode.SILU:
                silu_op = prod1
                gate_id = prod1.inputs[0]
                up_id = in0_id

            if silu_op is not None and gate_id is not None and up_id is not None:
                gate_t = graph.get_tensor(gate_id)
                up_t = graph.get_tensor(up_id)
                # SwiGLU is only valid for identical shape MLP projections (not Squeeze-and-Excitation or broadcasted gates)
                if (
                    gate_t is not None
                    and up_t is not None
                    and gate_t.shape == up_t.shape
                    and (prod0 is None or prod0.opcode != OpCode.SIGMOID)
                    and (prod1 is None or prod1.opcode != OpCode.SIGMOID)
                ):
                    # If the intermediate silu output is only consumed by this mul, we can prune it
                    if consumer_counts.get(silu_op.outputs[0], 0) <= 1:
                        ops_to_remove.add(silu_op.id)

                    fused_op = Operation(
                        id=op.id,
                        opcode=OpCode.SWIGLU,
                        inputs=[gate_id, up_id],
                        outputs=list(op.outputs),
                        attributes=dict(op.attributes),
                        name=f"{op.name or 'swiglu'}_fused",
                    )
                    new_nodes.append(fused_op)
                    continue

        new_nodes.append(op)

    # Filter out nodes marked for removal
    final_nodes = [n for n in new_nodes if n.id not in ops_to_remove]
    graph.nodes = final_nodes


def _fuse_approx_gelu_patterns(graph: Graph) -> None:
    """Matches polynomial NewGELU approximation subgraphs:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    and folds the entire elementary operation sub-DAG into a single OpCode.GELU(approximate="tanh").
    """
    producer_map: dict[int, Operation] = {}
    consumer_map: dict[int, list[Operation]] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_map.setdefault(in_id, []).append(op)

    def get_const_val(t_id: int) -> float | None:
        t = graph.get_tensor(t_id)
        if (
            t
            and t.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)
            and t.data is not None
        ):
            if hasattr(t.data, "item"):
                try:
                    return float(t.data.item())
                except (TypeError, ValueError):
                    return None
            try:
                return float(t.data)
            except (TypeError, ValueError):
                return None

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.TANH and len(op.inputs) == 1:
            tanh_in = op.inputs[0]
            prod_mul_c = producer_map.get(tanh_in)
            if not prod_mul_c or prod_mul_c.opcode != OpCode.MUL or len(prod_mul_c.inputs) != 2:
                new_nodes.append(op)
                continue

            # Check for sqrt(2/pi) approx 0.79788
            c_val = None
            poly_sum_id = None
            for inp in prod_mul_c.inputs:
                v = get_const_val(inp)
                if v is not None and abs(v - 0.79788) < 0.02:
                    c_val = v
                else:
                    poly_sum_id = inp

            if c_val is None or poly_sum_id is None:
                new_nodes.append(op)
                continue

            prod_poly_add = producer_map.get(poly_sum_id)
            if (
                not prod_poly_add
                or prod_poly_add.opcode != OpCode.ADD
                or len(prod_poly_add.inputs) != 2
            ):
                new_nodes.append(op)
                continue

            # In ADD(x, 0.044715 * x^3), identify x and cube_term
            x_cand = None
            cube_op = None
            for inp in prod_poly_add.inputs:
                p = producer_map.get(inp)
                if p and p.opcode == OpCode.MUL:
                    cube_op = p
                else:
                    x_cand = inp

            if not cube_op or x_cand is None:
                new_nodes.append(op)
                continue

            # Check 0.044715 in cube_op
            c_coeff = None
            pow_id = None
            for inp in cube_op.inputs:
                v = get_const_val(inp)
                if v is not None and abs(v - 0.044715) < 0.01:
                    c_coeff = v
                else:
                    pow_id = inp

            if c_coeff is None or pow_id is None:
                new_nodes.append(op)
                continue

            # Check downstream: TANH -> ADD(..., 1.0) -> MUL(..., 0.5 * x)
            tanh_consumers = consumer_map.get(op.outputs[0], [])
            if len(tanh_consumers) != 1:
                new_nodes.append(op)
                continue

            add1_op = tanh_consumers[0]
            if add1_op.opcode != OpCode.ADD or len(add1_op.inputs) != 2:
                new_nodes.append(op)
                continue

            has_one = any(
                get_const_val(inp) is not None and abs(get_const_val(inp) - 1.0) < 1e-3
                for inp in add1_op.inputs
            )
            if not has_one:
                new_nodes.append(op)
                continue

            add1_consumers = consumer_map.get(add1_op.outputs[0], [])
            if len(add1_consumers) != 1:
                new_nodes.append(op)
                continue

            mul_final_op = add1_consumers[0]
            if mul_final_op.opcode != OpCode.MUL or len(mul_final_op.inputs) != 2:
                new_nodes.append(op)
                continue

            # Find the other input to mul_final_op
            other_inp = (
                mul_final_op.inputs[0]
                if mul_final_op.inputs[1] == add1_op.outputs[0]
                else mul_final_op.inputs[1]
            )
            # other_inp should be 0.5 * x or x
            prod_half = producer_map.get(other_inp)
            final_out_op = mul_final_op
            matched_gelu = False

            if prod_half and prod_half.opcode == OpCode.MUL:
                # Check if it multiplies x_cand by 0.5
                if x_cand in prod_half.inputs:
                    for inp in prod_half.inputs:
                        v = get_const_val(inp)
                        if v is not None and abs(v - 0.5) < 1e-3:
                            matched_gelu = True
                            ops_to_remove.add(prod_half.id)
                            break
            elif other_inp == x_cand:
                # Might have an outer MUL by 0.5
                outer_consumers = consumer_map.get(mul_final_op.outputs[0], [])
                if len(outer_consumers) == 1 and outer_consumers[0].opcode == OpCode.MUL:
                    outer_mul = outer_consumers[0]
                    for inp in outer_mul.inputs:
                        v = get_const_val(inp)
                        if v is not None and abs(v - 0.5) < 1e-3:
                            matched_gelu = True
                            final_out_op = outer_mul
                            ops_to_remove.add(mul_final_op.id)
                            break

            if not matched_gelu:
                new_nodes.append(op)
                continue

            # Mark all intermediate polynomial nodes for removal
            ops_to_remove.add(op.id)
            ops_to_remove.add(prod_mul_c.id)
            ops_to_remove.add(prod_poly_add.id)
            ops_to_remove.add(cube_op.id)
            prod_pow = producer_map.get(pow_id)
            if prod_pow and len(consumer_map.get(pow_id, [])) <= 1:
                ops_to_remove.add(prod_pow.id)
            ops_to_remove.add(add1_op.id)
            ops_to_remove.add(final_out_op.id)

            fused_op = Operation(
                id=graph.new_op_id(),
                opcode=OpCode.GELU,
                inputs=[x_cand],
                outputs=list(final_out_op.outputs),
                attributes={"approximate": "tanh"},
                name=f"{final_out_op.name or 'gelu'}_approx_fused",
            )
            new_nodes.append(fused_op)
            continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


def _fuse_bias_gelu_patterns(graph: Graph) -> None:
    """Matches Linear(x, w, bias) -> GELU or Add(x, bias) -> GELU and fuses into BIAS_GELU."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.GELU and len(op.inputs) == 1:
            inp_id = op.inputs[0]
            prod = producer_map.get(inp_id)

            if prod and prod.opcode == OpCode.LINEAR and len(prod.inputs) >= 3:
                # Linear(x, w, bias) -> GELU(out)
                # Split into Linear(x, w) and BiasGELU(linear_out, bias)
                x_id = prod.inputs[0]
                w_id = prod.inputs[1]
                b_id = prod.inputs[2]
                prod.inputs = [x_id, w_id]

                fused_op = Operation(
                    id=op.id,
                    opcode=OpCode.BIAS_GELU,
                    inputs=[prod.outputs[0], b_id],
                    outputs=list(op.outputs),
                    attributes=dict(op.attributes),
                    name=f"{op.name or 'bias_gelu'}_fused",
                )
                new_nodes.append(fused_op)
                continue

            elif prod and prod.opcode == OpCode.ADD and len(prod.inputs) == 2:
                # Check if one input is an activation and the other is a bias/parameter
                in0 = graph.get_tensor(prod.inputs[0])
                in1 = graph.get_tensor(prod.inputs[1])

                act_id = None
                bias_id = None
                if (
                    in1.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                    or len(in1.shape.dims) == 1
                ):
                    act_id = in0.id
                    bias_id = in1.id
                elif (
                    in0.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                    or len(in0.shape.dims) == 1
                ):
                    act_id = in1.id
                    bias_id = in0.id

                if act_id is not None and bias_id is not None:
                    if consumer_counts.get(prod.outputs[0], 0) <= 1:
                        ops_to_remove.add(prod.id)

                    fused_op = Operation(
                        id=op.id,
                        opcode=OpCode.BIAS_GELU,
                        inputs=[act_id, bias_id],
                        outputs=list(op.outputs),
                        attributes=dict(op.attributes),
                        name=f"{op.name or 'bias_gelu'}_fused",
                    )
                    new_nodes.append(fused_op)
                    continue

        new_nodes.append(op)

    final_nodes = [n for n in new_nodes if n.id not in ops_to_remove]
    graph.nodes = final_nodes


def _fuse_softmax_patterns(graph: Graph) -> None:
    """Matches decomposed softmax: div(exp(x - max(x)), sum(exp(x - max(x)))) -> SOFTMAX(x)."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()

    for op in graph.nodes:
        if op.opcode == OpCode.DIV and len(op.inputs) == 2:
            num_id, denom_id = op.inputs[0], op.inputs[1]
            exp_op = producer_map.get(num_id)
            if not exp_op or exp_op.opcode != OpCode.EXP:
                continue

            sub_op = producer_map.get(exp_op.inputs[0])
            if sub_op and sub_op.opcode == OpCode.SUB:
                x_id = sub_op.inputs[0]
            else:
                x_id = exp_op.inputs[0]

            denom_chain: list[int] = []
            curr_denom = producer_map.get(denom_id)
            sum_op = None
            while curr_denom and curr_denom.opcode in (
                OpCode.RESHAPE,
                OpCode.VIEW,
                OpCode.SQUEEZE,
                OpCode.UNSQUEEZE,
                OpCode.EXPAND,
                OpCode.REPEAT,
            ):
                denom_chain.append(curr_denom.id)
                next_op = producer_map.get(curr_denom.inputs[0])
                if next_op and next_op.opcode == OpCode.SUM:
                    sum_op = next_op
                    break
                curr_denom = next_op
            if not sum_op and curr_denom and curr_denom.opcode == OpCode.SUM:
                sum_op = curr_denom

            if sum_op and sum_op.inputs[0] == exp_op.outputs[0]:
                op.opcode = OpCode.SOFTMAX
                op.inputs = [x_id]
                op.attributes["dim"] = -1
                ops_to_remove.add(exp_op.id)
                if sub_op:
                    ops_to_remove.add(sub_op.id)
                    max_in = sub_op.inputs[1]
                    curr_max = producer_map.get(max_in)
                    while curr_max:
                        ops_to_remove.add(curr_max.id)
                        if curr_max.opcode in (
                            OpCode.RESHAPE,
                            OpCode.VIEW,
                            OpCode.SQUEEZE,
                            OpCode.UNSQUEEZE,
                            OpCode.EXPAND,
                            OpCode.REPEAT,
                        ):
                            curr_max = producer_map.get(curr_max.inputs[0])
                        elif curr_max.opcode in (OpCode.AMAX, OpCode.MAXIMUM):
                            break
                        else:
                            break
                for d_id in denom_chain:
                    ops_to_remove.add(d_id)
                ops_to_remove.add(sum_op.id)

    if ops_to_remove:
        graph.nodes = [n for n in graph.nodes if n.id not in ops_to_remove]


def _fuse_horizontal_linear_patterns(graph: Graph, options: FusionOptions) -> None:
    """Matches parallel LINEAR operations consuming the same input activation and fuses them.

    Specifically targets:
    1. Attention Q + K + V projections:
       Linear(x, W_q) & Linear(x, W_k) & Linear(x, W_v) -> Linear(x, [W_q; W_k; W_v]) followed by slices.
    2. MLP Gate + Up projections:
       Linear(x, W_gate) & Linear(x, W_up) -> Linear(x, [W_gate; W_up]) followed by slices.
    """
    linear_ops = [n for n in graph.nodes if n.opcode == OpCode.LINEAR and len(n.inputs) >= 2]
    if len(linear_ops) < 2:
        return

    # Group linear ops by input activation ID
    by_input: dict[int, list[Operation]] = {}
    for op in linear_ops:
        w = graph.get_tensor(op.inputs[1])
        if (
            w is not None
            and w.storage == StorageClass.PARAMETER
            and w.data is not None
            and hasattr(w.data, "ndim")
            and w.data.ndim == 2
        ):
            by_input.setdefault(op.inputs[0], []).append(op)

    node_index_map = {op.id: i for i, op in enumerate(graph.nodes)}
    nodes_to_replace: dict[int, list[Operation]] = {}
    ops_to_remove: set[int] = set()

    def _is_q(op: Operation) -> bool:
        w = graph.get_tensor(op.inputs[1])
        names = f"{op.name or ''} {w.name if w else ''}".lower()
        return bool(re.search(r"(?:^|[._])(?:q|query)(?:_proj|[._]|$)", names))

    def _is_k(op: Operation) -> bool:
        w = graph.get_tensor(op.inputs[1])
        names = f"{op.name or ''} {w.name if w else ''}".lower()
        return bool(re.search(r"(?:^|[._])(?:k|key)(?:_proj|[._]|$)", names))

    def _is_v(op: Operation) -> bool:
        w = graph.get_tensor(op.inputs[1])
        names = f"{op.name or ''} {w.name if w else ''}".lower()
        return bool(re.search(r"(?:^|[._])(?:v|value)(?:_proj|[._]|$)", names))

    def _is_gate(op: Operation) -> bool:
        w = graph.get_tensor(op.inputs[1])
        names = f"{op.name or ''} {w.name if w else ''}".lower()
        return bool(re.search(r"(?:^|[._])(?:gate|w1)(?:_proj|[._]|$)", names))

    def _is_up(op: Operation) -> bool:
        w = graph.get_tensor(op.inputs[1])
        names = f"{op.name or ''} {w.name if w else ''}".lower()
        return bool(re.search(r"(?:^|[._])(?:up|w3)(?:_proj|[._]|$)", names))

    def _fuse_subgroup(cands: list[Operation], in_id: int, tag: str) -> bool:
        if len(cands) < 2:
            return False

        in_act = graph.get_tensor(in_id)
        if in_act is None:
            return False

        weights = [graph.get_tensor(op.inputs[1]) for op in cands]
        if any(w is None or w.data is None or w.data.ndim != 2 for w in weights):
            return False

        d_in = weights[0].data.shape[1]
        if not all(w.data.shape[1] == d_in for w in weights):
            return False

        # Bias verification
        has_bias = [len(op.inputs) > 2 for op in cands]
        if any(has_bias) and not all(has_bias):
            return False

        biases = None
        if all(has_bias):
            biases = [graph.get_tensor(op.inputs[2]) for op in cands]
            if any(b is None or b.data is None or b.data.ndim != 1 for b in biases):
                return False
            if not all(b.data.shape[0] == w.data.shape[0] for b, w in zip(biases, weights)):
                return False

        out_dims = [int(w.data.shape[0]) for w in weights]

        # 1. Concatenate weights along dimension 0 (out_features)
        fused_w_data = np.ascontiguousarray(np.concatenate([w.data for w in weights], axis=0))
        total_out_dim = int(fused_w_data.shape[0])
        fused_w_shape = Shape([StaticDim(total_out_dim), StaticDim(d_in)])
        fused_w_name = weights[0].name
        for old_part in (
            "gate_proj",
            "up_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "w1",
            "w3",
        ):
            if old_part in fused_w_name:
                fused_w_name = fused_w_name.replace(old_part, f"fused_{tag}")
                break
        else:
            fused_w_name = f"{weights[0].name}_{tag}_fused"

        fused_w = graph.add_tensor(
            name=fused_w_name,
            shape=fused_w_shape,
            dtype=weights[0].dtype,
            storage=StorageClass.PARAMETER,
            data=fused_w_data,
        )
        graph.parameters.append(fused_w.id)
        for w in weights:
            if w.id in graph.parameters:
                graph.parameters.remove(w.id)
            w.data = None
            graph.tensors.pop(w.id, None)

        # 2. Concatenate biases along dimension 0 if present
        fused_b_id = None
        if biases is not None:
            fused_b_data = np.ascontiguousarray(np.concatenate([b.data for b in biases], axis=0))
            fused_b_shape = Shape([StaticDim(total_out_dim)])
            fused_b_name = biases[0].name
            for old_part in (
                "gate_proj",
                "up_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "query",
                "key",
                "value",
                "w1",
                "w3",
            ):
                if old_part in fused_b_name:
                    fused_b_name = fused_b_name.replace(old_part, f"fused_{tag}")
                    break
            else:
                fused_b_name = f"{biases[0].name}_{tag}_fused"

            fused_b = graph.add_tensor(
                name=fused_b_name,
                shape=fused_b_shape,
                dtype=biases[0].dtype,
                storage=StorageClass.PARAMETER,
                data=fused_b_data,
            )
            graph.parameters.append(fused_b.id)
            for b in biases:
                if b.id in graph.parameters:
                    graph.parameters.remove(b.id)
                b.data = None
                graph.tensors.pop(b.id, None)
            fused_b_id = fused_b.id

        # 3. Create fused activation tensor
        fused_act_shape = Shape(list(in_act.shape.dims[:-1]) + [StaticDim(total_out_dim)])
        fused_act = graph.add_tensor(
            name=f"{in_act.name}_{tag}_fused",
            shape=fused_act_shape,
            dtype=in_act.dtype,
            storage=StorageClass.ACTIVATION,
        )

        # 4. Create fused LINEAR operation
        linear_inputs = [in_id, fused_w.id]
        if fused_b_id is not None:
            linear_inputs.append(fused_b_id)

        fused_linear_op = Operation(
            id=graph.new_op_id(),
            opcode=OpCode.LINEAR,
            inputs=linear_inputs,
            outputs=[fused_act.id],
            name=f"linear_fused_{tag}_{d_in}_to_{total_out_dim}",
        )
        fused_act.producer_id = fused_linear_op.id

        # 5. Create slice operations
        slice_ops = []
        curr_offset = 0
        for op, out_d in zip(cands, out_dims):
            slice_op = Operation(
                id=graph.new_op_id(),
                opcode=OpCode.SLICE,
                inputs=[fused_act.id],
                outputs=[op.outputs[0]],
                attributes={"dim": -1, "start": curr_offset, "end": curr_offset + out_d, "step": 1},
                name=f"slice_{op.name or tag}_{curr_offset}_{curr_offset + out_d}",
            )
            out_t = graph.get_tensor(op.outputs[0])
            if out_t is not None:
                out_t.producer_id = slice_op.id
            slice_ops.append(slice_op)
            curr_offset += out_d

        # 6. Schedule replacement at the earliest candidate operation's position
        earliest_op = min(cands, key=lambda op: node_index_map[op.id])
        nodes_to_replace[earliest_op.id] = [fused_linear_op] + slice_ops
        for op in cands:
            if op.id != earliest_op.id:
                ops_to_remove.add(op.id)

        return True

    for in_id, ops in by_input.items():
        remaining = list(ops)

        # 1. Match Attention Q + K + V
        if options.enable_horizontal_qkv and len(remaining) >= 3:
            q_op = next((op for op in remaining if _is_q(op)), None)
            k_op = next((op for op in remaining if _is_k(op)), None)
            v_op = next((op for op in remaining if _is_v(op)), None)
            qkv_group = None
            if q_op and k_op and v_op and len({q_op.id, k_op.id, v_op.id}) == 3:
                qkv_group = [q_op, k_op, v_op]
            elif len(remaining) == 3 and not any(_is_gate(op) or _is_up(op) for op in remaining):
                qkv_group = list(remaining)

            if qkv_group is not None and _fuse_subgroup(qkv_group, in_id, "qkv"):
                remaining = [op for op in remaining if op not in qkv_group]

        # 2. Match MLP Gate + Up
        if options.enable_horizontal_mlp and len(remaining) >= 2:
            gate_op = next((op for op in remaining if _is_gate(op)), None)
            up_op = next((op for op in remaining if _is_up(op)), None)
            mlp_group = None
            if gate_op and up_op and gate_op.id != up_op.id:
                mlp_group = [gate_op, up_op]
            elif len(remaining) == 2 and not any(
                _is_q(op) or _is_k(op) or _is_v(op) for op in remaining
            ):
                mlp_group = list(remaining)

            if mlp_group is not None and _fuse_subgroup(mlp_group, in_id, "gate_up"):
                remaining = [op for op in remaining if op not in mlp_group]

    if nodes_to_replace or ops_to_remove:
        new_nodes = []
        for op in graph.nodes:
            if op.id in nodes_to_replace:
                new_nodes.extend(nodes_to_replace[op.id])
            elif op.id in ops_to_remove:
                continue
            else:
                new_nodes.append(op)
        graph.nodes = new_nodes


def _fuse_swiglu_concatenated_patterns(graph: Graph) -> None:
    """Fuses SLICE(0..d, parent) and SLICE(d..2d, parent) feeding SWIGLU(gate, up)
    into a single-input SWIGLU(parent), eliminating the two slice operations."""
    producer_map: dict[int, Operation] = {}
    consumer_map: dict[int, list[Operation]] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_map.setdefault(in_id, []).append(op)

    ops_to_remove: set[int] = set()
    for op in graph.nodes:
        if op.opcode != OpCode.SWIGLU or len(op.inputs) != 2:
            continue
        gate_id, up_id = op.inputs[0], op.inputs[1]
        prod_gate = producer_map.get(gate_id)
        prod_up = producer_map.get(up_id)

        if not (prod_gate and prod_up):
            continue
        if prod_gate.opcode not in (OpCode.SLICE, OpCode.VIEW) or prod_up.opcode not in (
            OpCode.SLICE,
            OpCode.VIEW,
        ):
            continue
        if len(prod_gate.inputs) < 1 or len(prod_up.inputs) < 1:
            continue
        if prod_gate.inputs[0] != prod_up.inputs[0]:
            continue

        parent_id = prod_gate.inputs[0]
        # Gate and up must only be consumed by this SWIGLU op
        if len(consumer_map.get(gate_id, [])) != 1 or len(consumer_map.get(up_id, [])) != 1:
            continue

        start0 = prod_gate.attributes.get("start", 0)
        end0 = prod_gate.attributes.get("end", 0)
        start1 = prod_up.attributes.get("start", 0)
        end1 = prod_up.attributes.get("end", 0)

        if start0 == 0 and end0 > 0 and start1 == end0 and end1 == 2 * end0:
            # Gate is [0..d], Up is [d..2d] -> Standard non-swapped
            op.inputs = [parent_id]
            ops_to_remove.add(prod_gate.id)
            ops_to_remove.add(prod_up.id)
            graph.tensors.pop(gate_id, None)
            graph.tensors.pop(up_id, None)
        elif start1 == 0 and end1 > 0 and start0 == end1 and end0 == 2 * end1:
            # Up is [0..d], Gate is [d..2d] -> Swapped
            op.inputs = [parent_id]
            op.attributes["swapped"] = 1
            ops_to_remove.add(prod_gate.id)
            ops_to_remove.add(prod_up.id)
            graph.tensors.pop(gate_id, None)
            graph.tensors.pop(up_id, None)

    if ops_to_remove:
        graph.nodes = [op for op in graph.nodes if op.id not in ops_to_remove]


def _fuse_sdpa_transpose_patterns(graph: Graph) -> None:
    """Fuses transpose following scaled dot-product attention (SDPA):
    SDPA(...) -> (B, H, S, D) -> TRANSPOSE(dim0=1, dim1=2) -> (B, S, H, D)
    is fused by letting SDPA directly produce the (B, S, H, D) layout with
    fused_transpose=1 attribute, eliminating redundant permute/transpose steps.
    """
    consumers: dict[int, list[Operation]] = {}
    for n in graph.nodes:
        for in_id in n.inputs:
            consumers.setdefault(in_id, []).append(n)
    state_tids = {s.id for s in graph.states}
    graph_outs = set(graph.outputs) | state_tids

    nodes_to_remove = set()
    for n in graph.nodes:
        if n.opcode != OpCode.SDPA:
            continue
        out_id = n.outputs[0]
        if out_id in graph_outs:
            continue
        cons = consumers.get(out_id, [])
        if len(cons) != 1:
            continue
        n_trans = cons[0]
        if n_trans.opcode != OpCode.TRANSPOSE:
            continue
        d0 = n_trans.attributes.get("dim0", 0)
        d1 = n_trans.attributes.get("dim1", 1)
        trans_in_t = graph.tensors.get(out_id)
        if trans_in_t is not None:
            r = len(trans_in_t.shape.dims)
            if d0 < 0:
                d0 += r
            if d1 < 0:
                d1 += r
        if {d0, d1} != {1, 2}:
            continue

        trans_out_id = n_trans.outputs[0]
        n.outputs = [trans_out_id]
        n.attributes["fused_transpose"] = 1
        nodes_to_remove.add(n_trans.id)

    if nodes_to_remove:
        graph.nodes = [n for n in graph.nodes if n.id not in nodes_to_remove]
