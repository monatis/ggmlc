"""Compile-time const-affine folding into Linear / MatMul / Conv2D weights.

Absorbs static ``y = a ⊙ x + b`` into a consumer or producer GEMM/conv so the
runtime does not launch a separate MUL/ADD. No new kernels.

Preserves GGML CUDA fused chains:
  ``RMS_NORM/LAYER_NORM → MUL(γ) → ROPE``  (QK-Norm)
  ``RMS_NORM/LAYER_NORM → MUL(γ) → residual ADD``  (Gemma peri-norm)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass, Tensor

_STATIC = (StorageClass.PARAMETER, StorageClass.CONSTANT)
_LAYOUT_OPS = (
    OpCode.VIEW,
    OpCode.RESHAPE,
    OpCode.SQUEEZE,
    OpCode.UNSQUEEZE,
    OpCode.PERMUTE,
    OpCode.TRANSPOSE,
    OpCode.CONTIGUOUS,
)
_CONST_LAYOUT_OPS = _LAYOUT_OPS + (OpCode.EXPAND, OpCode.REPEAT)
_NORM_OPS = (OpCode.RMS_NORM, OpCode.LAYER_NORM)


@dataclass
class GemmSpec:
    op: Operation
    act_idx: int
    weight_id: int
    bias_idx: int | None
    in_features: int
    out_features: int
    in_axis: int
    out_axis: int
    is_addmm: bool


def bake_norm_affine_into_gemms(graph: Graph, opcode: OpCode) -> int:
    """Absorb fused RMS/LayerNorm γ (and LN β) into every GEMM consumer.

    Leaves a weightless norm. Skips RoPE / residual / tied-weight fanout.
    """
    _producers, consumers = _index(graph)
    baked = 0
    for op in list(graph.nodes):
        if op.opcode != opcode or len(op.inputs) < 2:
            continue
        x_id, gamma_id = op.inputs[0], op.inputs[1]
        gamma_t = graph.tensors.get(gamma_id)
        if not _has_static_data(gamma_t):
            continue
        gamma = np.asarray(gamma_t.data, dtype=np.float32).reshape(-1)
        if gamma.size == 0:
            continue

        beta = None
        if opcode == OpCode.LAYER_NORM and len(op.inputs) > 2:
            beta_t = graph.tensors.get(op.inputs[2])
            if _has_static_data(beta_t):
                beta = np.asarray(beta_t.data, dtype=np.float32).reshape(-1)
                if beta.size != gamma.size:
                    continue

        outs = consumers.get(op.outputs[0], [])
        if not outs:
            continue
        specs: list[GemmSpec] = []
        ok = True
        for c in outs:
            spec = _gemm_spec(c, graph)
            if spec is None or spec.op.inputs[spec.act_idx] != op.outputs[0]:
                ok = False
                break
            if spec.in_features != gamma.size:
                ok = False
                break
            specs.append(spec)
        if not ok or not specs:
            continue

        allowed = {s.op.id for s in specs}
        planned_w: dict[int, np.ndarray] = {}
        planned_b: list[tuple[GemmSpec, np.ndarray]] = []
        seen_w: set[int] = set()
        for spec in specs:
            if not _weight_exclusive(graph, spec.weight_id, allowed):
                ok = False
                break
            W = np.asarray(graph.get_tensor(spec.weight_id).data, dtype=np.float32)
            axis = _axis_matching(W, gamma.size, prefer_in=True, is_addmm=spec.is_addmm)
            if axis is None:
                ok = False
                break
            if spec.weight_id not in seen_w:
                seen_w.add(spec.weight_id)
                planned_w[spec.weight_id] = _scale_axis(W, gamma, axis)
            if beta is not None:
                extra = np.tensordot(W, beta, axes=(axis, 0)).astype(np.float32, copy=False)
                extra = np.ascontiguousarray(extra.reshape(-1))
                if extra.size != spec.out_features:
                    ok = False
                    break
                planned_b.append((spec, extra))
        if not ok:
            continue

        for spec, extra in planned_b:
            if spec.bias_idx is not None:
                b_t = graph.tensors.get(spec.op.inputs[spec.bias_idx])
                if not _has_static_data(b_t):
                    ok = False
                    break
                if np.asarray(b_t.data).reshape(-1).size != extra.size:
                    ok = False
                    break
        if not ok:
            continue

        for w_id, W_new in planned_w.items():
            graph.get_tensor(w_id).data = np.ascontiguousarray(W_new)
        for spec, extra in planned_b:
            _add_bias(graph, spec, extra)

        op.inputs = [x_id]
        baked += 1
        _, consumers = _index(graph)
    return baked


def bake_const_affine_into_gemms(graph: Graph) -> int:
    """Fold static MUL/DIV/ADD/SUB around Linear, MatMul, and Conv2D into weights."""
    folded = 0
    for _ in range(8):
        n = _bake_affine_once(graph)
        folded += n
        if n == 0:
            break
    return folded


def _bake_affine_once(graph: Graph) -> int:
    producers, consumers = _index(graph)
    # Post-op first so Conv→MUL→ADD (BatchNorm) collapses in two iterations.
    n = _fold_post_affine(graph, producers, consumers)
    if n:
        return n
    producers, consumers = _index(graph)
    return _fold_pre_affine(graph, producers, consumers)


def _fold_post_affine(
    graph: Graph, producers: dict[int, Operation], consumers: dict[int, list[Operation]]
) -> int:
    folded = 0
    for gemm in list(graph.nodes):
        spec = _gemm_spec(gemm, graph)
        if spec is None:
            continue
        if gemm.opcode == OpCode.CONV2D and (
            gemm.attributes.get("fused_relu") or gemm.attributes.get("fused_activation")
        ):
            continue
        chain, affine = _unique_forward_affine(gemm.outputs[0], consumers, producers, graph)
        if affine is None:
            continue
        if _norm_fusion_protected(affine, producers, consumers, graph):
            continue
        split = _split_binary(affine, graph, producers)
        if split is None:
            continue
        act_id, const_t, _const_tid = split
        expected_act = chain[-1].outputs[0] if chain else gemm.outputs[0]
        if act_id != expected_act:
            continue
        if not _weight_exclusive(graph, spec.weight_id, {gemm.id}):
            continue

        W_t = graph.get_tensor(spec.weight_id)
        W = np.asarray(W_t.data, dtype=np.float32)
        if affine.opcode in (OpCode.MUL, OpCode.DIV):
            vec = _affine_vector(const_t.data, spec.out_features)
            if vec is None:
                continue
            if affine.opcode == OpCode.DIV:
                if np.any(vec == 0):
                    continue
                vec = 1.0 / vec
            axis = _axis_matching(W, vec.size, prefer_in=False, is_addmm=spec.is_addmm)
            if axis is None:
                continue
            W_t.data = np.ascontiguousarray(_scale_axis(W, vec, axis))
            if spec.bias_idx is not None:
                b_t = graph.tensors.get(gemm.inputs[spec.bias_idx])
                if _has_static_data(b_t):
                    b = np.asarray(b_t.data, dtype=np.float32).reshape(-1)
                    if b.size == vec.size:
                        b_t.data = np.ascontiguousarray(b * vec)
        else:
            vec = _affine_vector(const_t.data, spec.out_features)
            if vec is None:
                continue
            if affine.opcode == OpCode.SUB:
                vec = -vec
            if not _add_bias(graph, spec, vec):
                continue

        _rewire_post(graph, gemm, chain, affine)
        folded += 1
        producers, consumers = _index(graph)
    return folded


def _fold_pre_affine(
    graph: Graph, producers: dict[int, Operation], consumers: dict[int, list[Operation]]
) -> int:
    folded = 0
    for gemm in list(graph.nodes):
        spec = _gemm_spec(gemm, graph)
        if spec is None:
            continue
        src_id, chain = _walk_layout_back(gemm.inputs[spec.act_idx], producers, consumers, graph)
        prod = producers.get(src_id)
        if prod is None or prod.opcode not in (OpCode.MUL, OpCode.DIV, OpCode.ADD, OpCode.SUB):
            continue
        if _norm_fusion_protected(prod, producers, consumers, graph):
            continue
        split = _split_binary(prod, graph, producers)
        if split is None:
            continue
        act_id, const_t, _const_tid = split
        if not _weight_exclusive(graph, spec.weight_id, {gemm.id}):
            continue

        W_t = graph.get_tensor(spec.weight_id)
        W = np.asarray(W_t.data, dtype=np.float32)
        if prod.opcode in (OpCode.MUL, OpCode.DIV):
            vec = _affine_vector(const_t.data, spec.in_features)
            if vec is None:
                continue
            if prod.opcode == OpCode.DIV:
                if np.any(vec == 0):
                    continue
                vec = 1.0 / vec
            axis = _axis_matching(W, vec.size, prefer_in=True, is_addmm=spec.is_addmm)
            if axis is None:
                continue
            W_t.data = np.ascontiguousarray(_scale_axis(W, vec, axis))
        else:
            vec = _affine_vector(const_t.data, spec.in_features)
            if vec is None:
                continue
            if prod.opcode == OpCode.SUB:
                vec = -vec
            axis = _axis_matching(W, vec.size, prefer_in=True, is_addmm=spec.is_addmm)
            if axis is None:
                continue
            extra = np.tensordot(W, vec, axes=(axis, 0)).astype(np.float32, copy=False)
            extra = np.ascontiguousarray(extra.reshape(-1))
            if extra.size != spec.out_features:
                continue
            if not _add_bias(graph, spec, extra):
                continue

        _rewire_pre(graph, gemm, spec.act_idx, chain, prod, act_id)
        folded += 1
        producers, consumers = _index(graph)
    return folded


def _gemm_spec(op: Operation, graph: Graph) -> GemmSpec | None:
    if op.opcode == OpCode.LINEAR:
        return _linear_spec(op, graph)
    if op.opcode == OpCode.MATMUL:
        return _matmul_spec(op, graph)
    if op.opcode == OpCode.CONV2D:
        return _conv_spec(op, graph)
    return None


def _linear_spec(op: Operation, graph: Graph) -> GemmSpec | None:
    if len(op.inputs) < 2:
        return None
    w_t = graph.tensors.get(op.inputs[1])
    if not _is_2d_param(w_t):
        return None
    W = np.asarray(w_t.data)
    is_addmm = op.attributes.get("is_addmm", 0) != 0
    k = _static_last_dim(graph.tensors.get(op.inputs[0]))
    if is_addmm or (k is not None and W.shape[0] == k and W.shape[1] != k):
        in_f, out_f, in_ax, out_ax = W.shape[0], W.shape[1], 0, 1
        is_addmm = True
    else:
        in_f, out_f, in_ax, out_ax = W.shape[1], W.shape[0], 1, 0
        is_addmm = False
    return GemmSpec(
        op=op,
        act_idx=0,
        weight_id=op.inputs[1],
        bias_idx=2 if len(op.inputs) > 2 else None,
        in_features=int(in_f),
        out_features=int(out_f),
        in_axis=in_ax,
        out_axis=out_ax,
        is_addmm=is_addmm,
    )


def _matmul_spec(op: Operation, graph: Graph) -> GemmSpec | None:
    if len(op.inputs) < 2:
        return None
    t0 = graph.tensors.get(op.inputs[0])
    t1 = graph.tensors.get(op.inputs[1])
    w0 = _is_2d_param(t0)
    w1 = _is_2d_param(t1)
    if w0 == w1:
        return None
    if w1:
        W = np.asarray(t1.data)
        trans = op.attributes.get("transpose_in1", 0) != 0
        if trans:
            in_f, out_f, in_ax, out_ax = W.shape[1], W.shape[0], 1, 0
            is_addmm = False
        else:
            in_f, out_f, in_ax, out_ax = W.shape[0], W.shape[1], 0, 1
            is_addmm = True
        return GemmSpec(
            op=op,
            act_idx=0,
            weight_id=op.inputs[1],
            bias_idx=2 if len(op.inputs) > 2 else None,
            in_features=int(in_f),
            out_features=int(out_f),
            in_axis=in_ax,
            out_axis=out_ax,
            is_addmm=is_addmm,
        )
    W = np.asarray(t0.data)
    trans = op.attributes.get("transpose_in0", 0) != 0
    # y = W @ x  → W (out, in); GGML transpose_in0 is a lowering hint, ignore here.
    if trans and W.shape[0] != W.shape[1]:
        in_f, out_f, in_ax, out_ax = W.shape[0], W.shape[1], 0, 1
        is_addmm = True
    else:
        in_f, out_f, in_ax, out_ax = W.shape[1], W.shape[0], 1, 0
        is_addmm = False
    return GemmSpec(
        op=op,
        act_idx=1,
        weight_id=op.inputs[0],
        bias_idx=2 if len(op.inputs) > 2 else None,
        in_features=int(in_f),
        out_features=int(out_f),
        in_axis=in_ax,
        out_axis=out_ax,
        is_addmm=is_addmm,
    )


def _conv_spec(op: Operation, graph: Graph) -> GemmSpec | None:
    if len(op.inputs) < 2:
        return None
    w_t = graph.tensors.get(op.inputs[1])
    if (
        w_t is None
        or w_t.data is None
        or not hasattr(w_t.data, "ndim")
        or w_t.data.ndim != 4
        or w_t.storage != StorageClass.PARAMETER
    ):
        return None
    W = np.asarray(w_t.data)
    return GemmSpec(
        op=op,
        act_idx=0,
        weight_id=op.inputs[1],
        bias_idx=2 if len(op.inputs) > 2 else None,
        in_features=int(W.shape[1]),
        out_features=int(W.shape[0]),
        in_axis=1,
        out_axis=0,
        is_addmm=False,
    )


def _axis_matching(W: np.ndarray, n: int, *, prefer_in: bool, is_addmm: bool) -> int | None:
    if W.ndim == 4:
        if prefer_in:
            return 1 if W.shape[1] == n else (0 if W.shape[0] == n else None)
        return 0 if W.shape[0] == n else (1 if W.shape[1] == n else None)
    if W.ndim != 2:
        return None
    if prefer_in:
        if is_addmm or (W.shape[0] == n and W.shape[1] != n):
            return 0 if W.shape[0] == n else None
        return 1 if W.shape[1] == n else None
    if is_addmm or (W.shape[1] == n and W.shape[0] != n):
        return 1 if W.shape[1] == n else None
    return 0 if W.shape[0] == n else None


def _scale_axis(W: np.ndarray, vec: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * W.ndim
    shape[axis] = -1
    return W * vec.reshape(shape)


def _add_bias(graph: Graph, spec: GemmSpec, extra: np.ndarray) -> bool:
    extra = np.ascontiguousarray(np.asarray(extra, dtype=np.float32).reshape(-1))
    op = spec.op
    if spec.bias_idx is not None:
        b_t = graph.tensors.get(op.inputs[spec.bias_idx])
        if not _has_static_data(b_t):
            return False
        b = np.asarray(b_t.data, dtype=np.float32).reshape(-1)
        if b.size != extra.size:
            return False
        b_t.data = np.ascontiguousarray(b + extra)
        return True
    name = f"{op.name or 'gemm'}_baked_bias"
    bt = graph.add_tensor(
        name=name,
        shape=Shape.from_tuple((int(extra.size),)),
        dtype=DType.F32,
        storage=StorageClass.PARAMETER,
        data=extra,
        role="bias",
    )
    graph.parameters.append(bt.id)
    op.inputs.append(bt.id)
    spec.bias_idx = len(op.inputs) - 1
    return True


def _rewire_post(
    graph: Graph,
    gemm: Operation,
    chain: list[Operation],
    affine: Operation,
) -> None:
    new_outs = list(affine.outputs)
    if chain:
        chain[-1].outputs = new_outs
        _set_producer(graph, new_outs, chain[-1].id)
    else:
        gemm.outputs = new_outs
        _set_producer(graph, new_outs, gemm.id)
    graph.nodes = [n for n in graph.nodes if n.id != affine.id]


def _rewire_pre(
    graph: Graph,
    gemm: Operation,
    act_idx: int,
    chain: list[Operation],
    affine: Operation,
    act_id: int,
) -> None:
    if chain:
        chain[0].inputs[0] = act_id
    else:
        gemm.inputs[act_idx] = act_id
    graph.nodes = [n for n in graph.nodes if n.id != affine.id]


def _unique_forward_affine(
    start_id: int,
    consumers: dict[int, list[Operation]],
    producers: dict[int, Operation],
    graph: Graph,
) -> tuple[list[Operation], Operation | None]:
    chain: list[Operation] = []
    tid = start_id
    for _ in range(6):
        if tid in graph.outputs:
            return chain, None
        outs = consumers.get(tid, [])
        if len(outs) != 1:
            return chain, None
        nxt = outs[0]
        if nxt.opcode in (OpCode.MUL, OpCode.DIV, OpCode.ADD, OpCode.SUB) and len(nxt.inputs) == 2:
            return chain, nxt
        if nxt.opcode in _LAYOUT_OPS and len(nxt.inputs) == 1:
            chain.append(nxt)
            tid = nxt.outputs[0]
            continue
        return chain, None
    return chain, None


def _walk_layout_back(
    tid: int,
    producers: dict[int, Operation],
    consumers: dict[int, list[Operation]],
    graph: Graph,
) -> tuple[int, list[Operation]]:
    chain: list[Operation] = []
    cur = tid
    for _ in range(6):
        prod = producers.get(cur)
        if prod is None or prod.opcode not in _LAYOUT_OPS or len(prod.inputs) != 1:
            break
        if cur in graph.outputs:
            break
        if len(consumers.get(cur, [])) != 1:
            break
        chain.insert(0, prod)
        cur = prod.inputs[0]
    return cur, chain


def _norm_fusion_protected(
    affine: Operation,
    producers: dict[int, Operation],
    consumers: dict[int, list[Operation]],
    graph: Graph,
) -> bool:
    """Keep RMS/LN → MUL(γ) → RoPE / residual ADD intact for CUDA fusion."""
    split = _split_binary(affine, graph, producers)
    if split is None:
        return False
    act_id, _const_t, _ = split
    src = act_id
    for _ in range(4):
        prod = producers.get(src)
        if prod is None:
            break
        if prod.opcode in _LAYOUT_OPS and prod.inputs:
            src = prod.inputs[0]
            continue
        if prod.opcode in _NORM_OPS:
            for c in consumers.get(affine.outputs[0], []):
                if c.opcode == OpCode.ROPE:
                    return True
                if c.opcode == OpCode.ADD and _is_residual_add(c, graph, producers):
                    return True
            return False
        break
    return False


def _is_residual_add(op: Operation, graph: Graph, producers: dict[int, Operation]) -> bool:
    if op.opcode != OpCode.ADD or len(op.inputs) != 2:
        return False
    s0 = _static_tensor(graph, op.inputs[0], producers)
    s1 = _static_tensor(graph, op.inputs[1], producers)
    return s0 is None and s1 is None


def _split_binary(
    op: Operation, graph: Graph, producers: dict[int, Operation]
) -> tuple[int, Tensor, int] | None:
    if len(op.inputs) != 2:
        return None
    s0 = _static_tensor(graph, op.inputs[0], producers)
    s1 = _static_tensor(graph, op.inputs[1], producers)
    if s1 is not None and s0 is None:
        if op.opcode in (OpCode.MUL, OpCode.DIV, OpCode.ADD, OpCode.SUB):
            return op.inputs[0], s1, op.inputs[1]
        return None
    if s0 is not None and s1 is None:
        if op.opcode in (OpCode.MUL, OpCode.ADD):
            return op.inputs[1], s0, op.inputs[0]
        return None
    return None


def _static_tensor(
    graph: Graph, tid: int, producers: dict[int, Operation], depth: int = 8
) -> Tensor | None:
    t = graph.tensors.get(tid)
    if _has_static_data(t):
        return t
    if depth <= 0:
        return None
    prod = producers.get(tid)
    if prod is not None and prod.opcode in _CONST_LAYOUT_OPS and len(prod.inputs) == 1:
        return _static_tensor(graph, prod.inputs[0], producers, depth - 1)
    return None


def _affine_vector(data: np.ndarray, n: int) -> np.ndarray | None:
    a = np.asarray(data, dtype=np.float32)
    if a.size == 1:
        return np.full((n,), float(a.reshape(-1)[0]), dtype=np.float32)
    sq = np.squeeze(a)
    if sq.ndim == 0:
        return np.full((n,), float(sq), dtype=np.float32)
    if sq.ndim == 1 and sq.size == n:
        return np.ascontiguousarray(sq.astype(np.float32))
    non1 = [i for i, s in enumerate(a.shape) if s != 1]
    if len(non1) == 1 and a.shape[non1[0]] == n:
        return np.ascontiguousarray(a.reshape(-1).astype(np.float32))
    if len(non1) == 0:
        return np.full((n,), float(a.reshape(-1)[0]), dtype=np.float32)
    return None


def _index(graph: Graph) -> tuple[dict[int, Operation], dict[int, list[Operation]]]:
    producers: dict[int, Operation] = {}
    consumers: dict[int, list[Operation]] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producers[out_id] = op
        for in_id in op.inputs:
            consumers.setdefault(in_id, []).append(op)
    return producers, consumers


def _weight_exclusive(graph: Graph, w_id: int, allowed: set[int]) -> bool:
    for n in graph.nodes:
        if w_id in n.inputs and n.id not in allowed:
            return False
    return True


def _has_static_data(t: Tensor | None) -> bool:
    return t is not None and t.data is not None and t.storage in _STATIC


def _is_2d_param(t: Tensor | None) -> bool:
    return (
        t is not None
        and t.data is not None
        and hasattr(t.data, "ndim")
        and t.data.ndim == 2
        and t.storage == StorageClass.PARAMETER
    )


def _static_last_dim(t: Tensor | None) -> int | None:
    if t is None or not t.shape.dims:
        return None
    d = t.shape.dims[-1]
    if d.is_static():
        return int(d.evaluate({}))
    return None


def _set_producer(graph: Graph, tids: list[int], op_id: int) -> None:
    for tid in tids:
        ten = graph.tensors.get(tid)
        if ten is not None:
            ten.producer_id = op_id
