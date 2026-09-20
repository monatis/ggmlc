"""Tests for generic const-affine folding into Linear / MatMul / Conv2D."""

from __future__ import annotations

import numpy as np
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.dce import DeadCodeEliminationPass
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def _opts(**kwargs) -> FusionOptions:
    defaults = {
        "enable_horizontal_mlp": False,
        "enable_horizontal_qkv": False,
        "enable_bake_rms_into_linear": False,
        "enable_bake_affine": True,
        "enable_bias_gelu": False,
        "enable_layer_norm": False,
        "enable_rms_norm": False,
        "enable_conv2d_relu": False,
    }
    defaults.update(kwargs)
    return FusionOptions(**defaults)


def _param(g: Graph, name: str, data: np.ndarray, storage=StorageClass.PARAMETER):
    t = g.add_tensor(name, Shape.from_tuple(data.shape), DType.F32, storage)
    t.data = data.copy()
    g.parameters.append(t.id)
    return t


def test_pre_mul_scales_linear_in_features():
    g = Graph(name="pre_mul")
    hidden, out = 4, 6
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    s = np.arange(1, hidden + 1, dtype=np.float32)
    t_s = _param(g, "s", s, StorageClass.CONSTANT)
    xs = g.add_tensor("xs", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden)
    t_w = _param(g, "w", W)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.MUL, [x.id, t_s.id], [xs.id])
    g.add_op(OpCode.LINEAR, [xs.id, t_w.id], [y.id])

    fuse_operations(g, _opts())
    g = DeadCodeEliminationPass().run(g).graph

    assert all(n.opcode != OpCode.MUL for n in g.nodes)
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    assert lin.inputs[0] == x.id
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, W * s[np.newaxis, :])


def test_post_mul_scales_linear_out_features_and_bias():
    g = Graph(name="post_mul")
    hidden, out = 3, 5
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden) + 1
    b = np.arange(out, dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    t_w = _param(g, "w", W)
    t_b = _param(g, "b", b)
    t_s = _param(g, "s", scale, StorageClass.CONSTANT)
    lin_out = g.add_tensor("lin", Shape.from_tuple((2, out)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.LINEAR, [x.id, t_w.id, t_b.id], [lin_out.id])
    g.add_op(OpCode.MUL, [lin_out.id, t_s.id], [y.id])

    fuse_operations(g, _opts())
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    assert lin.outputs[0] == y.id
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, W * scale[:, np.newaxis])
    np.testing.assert_allclose(g.get_tensor(lin.inputs[2]).data, b * scale)


def test_pre_add_bakes_into_linear_bias():
    g = Graph(name="pre_add")
    hidden, out = 4, 3
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    shift = np.arange(hidden, dtype=np.float32)
    W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden)
    t_sh = _param(g, "shift", shift, StorageClass.CONSTANT)
    t_w = _param(g, "w", W)
    xs = g.add_tensor("xs", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.ADD, [x.id, t_sh.id], [xs.id])
    g.add_op(OpCode.LINEAR, [xs.id, t_w.id], [y.id])

    fuse_operations(g, _opts())
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    assert lin.inputs[0] == x.id
    assert len(lin.inputs) == 3
    np.testing.assert_allclose(g.get_tensor(lin.inputs[2]).data, W @ shift)
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, W)


def test_layer_norm_gamma_beta_into_linear():
    g = Graph(name="ln_bake")
    hidden, out = 4, 8
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    gamma = np.arange(1, hidden + 1, dtype=np.float32)
    beta = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden)
    t_g = _param(g, "gamma", gamma)
    t_b = _param(g, "beta", beta)
    t_w = _param(g, "w", W)
    ln = g.add_tensor("ln", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.LAYER_NORM, [x.id, t_g.id, t_b.id], [ln.id], attributes={"eps": 1e-5})
    g.add_op(OpCode.LINEAR, [ln.id, t_w.id], [y.id])

    fuse_operations(g, _opts())
    g = DeadCodeEliminationPass().run(g).graph
    ln_op = next(n for n in g.nodes if n.opcode == OpCode.LAYER_NORM)
    assert ln_op.inputs == [x.id]
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, W * gamma[np.newaxis, :])
    np.testing.assert_allclose(g.get_tensor(lin.inputs[2]).data, W @ beta)


def test_conv_batchnorm_scale_shift_into_conv():
    g = Graph(name="conv_bn")
    oc, ic, k = 4, 3, 1
    x = g.add_tensor("x", Shape.from_tuple((1, ic, 2, 2)), DType.F32, StorageClass.INPUT)
    W = np.arange(oc * ic * k * k, dtype=np.float32).reshape(oc, ic, k, k) + 1
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    shift = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t_w = _param(g, "w", W)
    t_s = _param(g, "bn_s", scale.reshape(1, oc, 1, 1), StorageClass.CONSTANT)
    t_sh = _param(g, "bn_b", shift.reshape(1, oc, 1, 1), StorageClass.CONSTANT)
    conv_out = g.add_tensor(
        "c", Shape.from_tuple((1, oc, 2, 2)), DType.F32, StorageClass.ACTIVATION
    )
    scaled = g.add_tensor("sc", Shape.from_tuple((1, oc, 2, 2)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((1, oc, 2, 2)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.CONV2D, [x.id, t_w.id], [conv_out.id])
    g.add_op(OpCode.MUL, [conv_out.id, t_s.id], [scaled.id])
    g.add_op(OpCode.ADD, [scaled.id, t_sh.id], [y.id])

    fuse_operations(g, _opts())
    conv = next(n for n in g.nodes if n.opcode == OpCode.CONV2D)
    assert conv.outputs[0] == y.id
    assert all(n.opcode not in (OpCode.MUL, OpCode.ADD) for n in g.nodes)
    expected_w = W * scale.reshape(oc, 1, 1, 1)
    np.testing.assert_allclose(g.get_tensor(conv.inputs[1]).data, expected_w)
    np.testing.assert_allclose(g.get_tensor(conv.inputs[2]).data, shift)


def test_matmul_pre_mul_flax_layout():
    """Flax Dense is x @ W with W (in, out)."""
    g = Graph(name="matmul")
    hidden, out = 4, 5
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    s = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    W = np.arange(hidden * out, dtype=np.float32).reshape(hidden, out)
    t_s = _param(g, "s", s, StorageClass.CONSTANT)
    t_w = _param(g, "w", W)
    xs = g.add_tensor("xs", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.MUL, [x.id, t_s.id], [xs.id])
    g.add_op(OpCode.MATMUL, [xs.id, t_w.id], [y.id])

    fuse_operations(g, _opts())
    mm = next(n for n in g.nodes if n.opcode == OpCode.MATMUL)
    assert mm.inputs[0] == x.id
    np.testing.assert_allclose(g.get_tensor(mm.inputs[1]).data, W * s[:, np.newaxis])


def test_skip_qk_norm_mul_before_rope():
    g = Graph(name="qk_norm")
    hidden = 4
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    gamma = np.arange(1, hidden + 1, dtype=np.float32)
    pos = g.add_tensor("pos", Shape.from_tuple((2,)), DType.I32, StorageClass.INPUT)
    t_g = _param(g, "gamma", gamma)
    rms = g.add_tensor("rms", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    scaled = g.add_tensor("sc", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id, pos.id]
    g.outputs = [y.id]
    g.add_op(OpCode.RMS_NORM, [x.id], [rms.id], attributes={"eps": 1e-5})
    g.add_op(OpCode.MUL, [rms.id, t_g.id], [scaled.id])
    g.add_op(OpCode.ROPE, [scaled.id, pos.id], [y.id])

    fuse_operations(g, _opts())
    assert any(n.opcode == OpCode.MUL for n in g.nodes)
    np.testing.assert_allclose(g.get_tensor(t_g.id).data, gamma)


def test_skip_peri_norm_mul_before_residual_add():
    g = Graph(name="peri")
    hidden = 4
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    residual = g.add_tensor("res", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    gamma = np.ones((hidden,), dtype=np.float32) * 1.5
    t_g = _param(g, "gamma", gamma)
    rms = g.add_tensor("rms", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    scaled = g.add_tensor("sc", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id, residual.id]
    g.outputs = [y.id]
    g.add_op(OpCode.RMS_NORM, [x.id], [rms.id])
    g.add_op(OpCode.MUL, [rms.id, t_g.id], [scaled.id])
    g.add_op(OpCode.ADD, [scaled.id, residual.id], [y.id])

    fuse_operations(g, _opts())
    assert any(n.opcode == OpCode.MUL for n in g.nodes)


def test_skip_tied_embedding_weight():
    g = Graph(name="tied")
    hidden, out = 4, 4
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    tok = g.add_tensor("tok", Shape.from_tuple((2,)), DType.I64, StorageClass.INPUT)
    s = np.arange(1, hidden + 1, dtype=np.float32)
    W = np.eye(hidden, dtype=np.float32)
    t_s = _param(g, "s", s, StorageClass.CONSTANT)
    t_w = _param(g, "w", W)
    xs = g.add_tensor("xs", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    emb = g.add_tensor("emb", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    g.inputs = [x.id, tok.id]
    g.outputs = [y.id]
    g.add_op(OpCode.MUL, [x.id, t_s.id], [xs.id])
    g.add_op(OpCode.LINEAR, [xs.id, t_w.id], [y.id])
    g.add_op(OpCode.EMBEDDING, [t_w.id, tok.id], [emb.id])

    fuse_operations(g, _opts())
    np.testing.assert_allclose(g.get_tensor(t_w.id).data, W)
    assert any(n.opcode == OpCode.MUL for n in g.nodes)


def test_disabled_flag_is_noop():
    g = Graph(name="off")
    hidden, out = 4, 3
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    s = np.ones((hidden,), dtype=np.float32) * 2
    W = np.ones((out, hidden), dtype=np.float32)
    t_s = _param(g, "s", s, StorageClass.CONSTANT)
    t_w = _param(g, "w", W)
    xs = g.add_tensor("xs", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [y.id]
    g.add_op(OpCode.MUL, [x.id, t_s.id], [xs.id])
    g.add_op(OpCode.LINEAR, [xs.id, t_w.id], [y.id])

    fuse_operations(g, _opts(enable_bake_affine=False))
    assert any(n.opcode == OpCode.MUL for n in g.nodes)
    np.testing.assert_allclose(g.get_tensor(t_w.id).data, W)


def test_layerscale_post_mul_before_residual():
    """Linear → MUL(λ) → residual ADD is LayerScale; MUL should bake into W rows."""
    g = Graph(name="layerscale")
    hidden, out = 4, 4
    x = g.add_tensor("x", Shape.from_tuple((2, hidden)), DType.F32, StorageClass.INPUT)
    residual = g.add_tensor("res", Shape.from_tuple((2, out)), DType.F32, StorageClass.INPUT)
    lam = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    W = np.arange(out * hidden, dtype=np.float32).reshape(out, hidden) + 1
    t_l = _param(g, "lambda", lam)
    t_w = _param(g, "w", W)
    lin_out = g.add_tensor("lin", Shape.from_tuple((2, out)), DType.F32, StorageClass.ACTIVATION)
    scaled = g.add_tensor("sc", Shape.from_tuple((2, out)), DType.F32, StorageClass.ACTIVATION)
    y = g.add_tensor("y", Shape.from_tuple((2, out)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id, residual.id]
    g.outputs = [y.id]
    g.add_op(OpCode.LINEAR, [x.id, t_w.id], [lin_out.id])
    g.add_op(OpCode.MUL, [lin_out.id, t_l.id], [scaled.id])
    g.add_op(OpCode.ADD, [scaled.id, residual.id], [y.id])

    fuse_operations(g, _opts())
    lin = next(n for n in g.nodes if n.opcode == OpCode.LINEAR)
    assert all(n.opcode != OpCode.MUL for n in g.nodes)
    np.testing.assert_allclose(g.get_tensor(lin.inputs[1]).data, W * lam[:, np.newaxis])
    assert any(n.opcode == OpCode.ADD for n in g.nodes)
