"""Tests for compile-time RMSNorm gamma bake into Linear weights."""

from __future__ import annotations

import numpy as np

from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.dce import DeadCodeEliminationPass
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def _build_rms_linear_graph(
    *,
    hidden: int = 4,
    out: int = 8,
    is_addmm: bool = False,
    fanout: int = 1,
) -> tuple[Graph, np.ndarray, list[np.ndarray]]:
    g = Graph(name="bake_rms_test")
    t_x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    gamma = np.arange(1, hidden + 1, dtype=np.float32)
    t_gamma = g.add_tensor("gamma", Shape.from_tuple((hidden,)), DType.F32, StorageClass.PARAMETER)
    t_gamma.data = gamma.copy()

    t_rms = g.add_tensor("rms_out", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    g.inputs = [t_x.id]
    g.parameters = [t_gamma.id]
    g.add_op(OpCode.RMS_NORM, [t_x.id, t_gamma.id], [t_rms.id], attributes={"eps": 1e-5})

    weights: list[np.ndarray] = []
    outputs: list[int] = []
    for i in range(fanout):
        if is_addmm:
            W = np.arange(hidden * out, dtype=np.float32).reshape(hidden, out) + float(i)
        else:
            W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden) + float(i)
        t_w = g.add_tensor(
            f"w{i}",
            Shape.from_tuple(W.shape),
            DType.F32,
            StorageClass.PARAMETER,
        )
        t_w.data = W.copy()
        weights.append(W)
        g.parameters.append(t_w.id)
        t_y = g.add_tensor(
            f"y{i}",
            Shape.from_tuple((2, out)),
            DType.F32,
            StorageClass.OUTPUT if i == 0 else StorageClass.ACTIVATION,
        )
        attrs = {"is_addmm": 1} if is_addmm else {}
        g.add_op(OpCode.LINEAR, [t_rms.id, t_w.id], [t_y.id], attributes=attrs)
        outputs.append(t_y.id)

    g.outputs = outputs
    return g, gamma, weights


def test_bake_rms_scales_linear_columns():
    g, gamma, weights = _build_rms_linear_graph()
    fuse_operations(g, FusionOptions(enable_bake_rms_into_linear=True))

    rms = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM)
    assert len(rms.inputs) == 1

    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    w_t = g.get_tensor(lin.inputs[1])
    expected = weights[0] * gamma[np.newaxis, :]
    np.testing.assert_allclose(w_t.data, expected)

    result = DeadCodeEliminationPass().run(g)
    g = result.graph
    assert "gamma" not in {g.tensors[tid].name for tid in g.parameters}


def test_bake_rms_scales_addmm_rows():
    g, gamma, weights = _build_rms_linear_graph(is_addmm=True)
    fuse_operations(g, FusionOptions(enable_bake_rms_into_linear=True))

    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    w_t = g.get_tensor(lin.inputs[1])
    expected = weights[0] * gamma[:, np.newaxis]
    np.testing.assert_allclose(w_t.data, expected)


def test_bake_rms_multi_linear_fanout():
    g, gamma, weights = _build_rms_linear_graph(fanout=3, out=6)
    fuse_operations(
        g,
        FusionOptions(
            enable_bake_rms_into_linear=True,
            enable_horizontal_mlp=False,
            enable_horizontal_qkv=False,
        ),
    )

    rms = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM)
    assert len(rms.inputs) == 1
    linears = [n for n in g.nodes if n.opcode == OpCode.LINEAR]
    assert len(linears) == 3
    for lin, W0 in zip(linears, weights):
        w_t = g.get_tensor(lin.inputs[1])
        np.testing.assert_allclose(w_t.data, W0 * gamma[np.newaxis, :])


def test_bake_rms_skipped_when_disabled():
    g, _, weights = _build_rms_linear_graph()
    fuse_operations(g, FusionOptions(enable_bake_rms_into_linear=False))
    rms = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM)
    assert len(rms.inputs) == 2
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, weights[0])


def test_bake_rms_skipped_on_non_linear_consumer():
    g, _, weights = _build_rms_linear_graph()
    # Add a second consumer that is not LINEAR
    rms_out = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM).outputs[0]
    t_extra = g.add_tensor("extra", Shape.from_tuple((2, 4)), DType.F32, StorageClass.ACTIVATION)
    g.add_op(OpCode.NEG, [rms_out], [t_extra.id])
    fuse_operations(g, FusionOptions(enable_bake_rms_into_linear=True))
    rms = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM)
    assert len(rms.inputs) == 2
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, weights[0])


def test_bake_rms_skipped_on_tied_embedding_weight():
    """Tied embed/lm_head: same PARAMETER used by EMBEDDING and LINEAR — must not bake."""
    g, _, weights = _build_rms_linear_graph(hidden=4, out=4)
    # Reuse Linear weight as embedding table
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    w_id = lin.inputs[1]
    t_tok = g.add_tensor("tok", Shape.from_tuple((2,)), DType.I64, StorageClass.INPUT)
    t_emb = g.add_tensor("emb", Shape.from_tuple((2, 4)), DType.F32, StorageClass.ACTIVATION)
    g.inputs.append(t_tok.id)
    g.add_op(OpCode.EMBEDDING, [w_id, t_tok.id], [t_emb.id])

    fuse_operations(
        g,
        FusionOptions(
            enable_bake_rms_into_linear=True,
            enable_horizontal_mlp=False,
            enable_horizontal_qkv=False,
        ),
    )
    rms = next(n for n in g.nodes if n.opcode == OpCode.RMS_NORM)
    assert len(rms.inputs) == 2
    np.testing.assert_allclose(g.get_tensor(w_id).data, weights[0])
