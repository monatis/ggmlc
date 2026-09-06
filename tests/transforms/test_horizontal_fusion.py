"""Tests for horizontal operator fusion (Gate+Up and QKV) in ggmlc."""

from __future__ import annotations

import copy

import numpy as np

from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def _compare_outputs(out_a, out_b, atol=1e-5):
    if isinstance(out_a, dict):
        assert isinstance(out_b, dict)
        for k in out_a:
            diff = np.max(np.abs(out_a[k] - out_b[k]))
            assert diff < atol, f"Key {k} divergence: {diff}"
    elif isinstance(out_a, (list, tuple)):
        for u, f in zip(out_a, out_b):
            diff = np.max(np.abs(u - f))
            assert diff < atol, f"List item divergence: {diff}"
    else:
        diff = np.max(np.abs(out_a - out_b))
        assert diff < atol, f"Divergence: {diff}"


def test_horizontal_mlp_fusion():
    """Verify that Gate and Up linear projections are fused into a single linear op + slices."""
    g = Graph(name="mlp_fusion_test")
    d_in, d_ff = 32, 64
    x_data = np.random.randn(1, 4, d_in).astype(np.float32)
    w_gate_data = np.random.randn(d_ff, d_in).astype(np.float32)
    w_up_data = np.random.randn(d_ff, d_in).astype(np.float32)

    t_x = g.add_tensor("x", Shape.from_tuple((1, 4, d_in)), DType.F32, StorageClass.INPUT)
    t_w_gate = g.add_tensor("gate_proj.weight", Shape.from_tuple((d_ff, d_in)), DType.F32, StorageClass.PARAMETER, data=w_gate_data)
    t_w_up = g.add_tensor("up_proj.weight", Shape.from_tuple((d_ff, d_in)), DType.F32, StorageClass.PARAMETER, data=w_up_data)
    t_gate_out = g.add_tensor("gate_out", Shape.from_tuple((1, 4, d_ff)), DType.F32, StorageClass.OUTPUT)
    t_up_out = g.add_tensor("up_out", Shape.from_tuple((1, 4, d_ff)), DType.F32, StorageClass.OUTPUT)

    g.inputs = [t_x.id]
    g.parameters = [t_w_gate.id, t_w_up.id]
    g.outputs = [t_gate_out.id, t_up_out.id]

    g.add_op(OpCode.LINEAR, [t_x.id, t_w_gate.id], [t_gate_out.id], name="gate_linear")
    g.add_op(OpCode.LINEAR, [t_x.id, t_w_up.id], [t_up_out.id], name="up_linear")

    # 1. Unfused reference
    g_unfused = copy.deepcopy(g)
    fuse_operations(g_unfused, FusionOptions(enable_horizontal_mlp=False, enable_horizontal_qkv=False))
    linear_ops_unfused = [op for op in g_unfused.nodes if op.opcode == OpCode.LINEAR]
    slice_ops_unfused = [op for op in g_unfused.nodes if op.opcode == OpCode.SLICE]
    assert len(linear_ops_unfused) == 2
    assert len(slice_ops_unfused) == 0

    # 2. Fused
    g_fused = copy.deepcopy(g)
    fuse_operations(g_fused, FusionOptions(enable_horizontal_mlp=True, enable_horizontal_qkv=False))
    linear_ops_fused = [op for op in g_fused.nodes if op.opcode == OpCode.LINEAR]
    slice_ops_fused = [op for op in g_fused.nodes if op.opcode == OpCode.SLICE]
    assert len(linear_ops_fused) == 1
    assert len(slice_ops_fused) == 2

    # Verify weight concatenation shape: 2 * d_ff x d_in
    fused_w = g_fused.get_tensor(linear_ops_fused[0].inputs[1])
    assert fused_w.shape.dims[0].value == 2 * d_ff
    assert fused_w.shape.dims[1].value == d_in

    # 3. Numerical differential verification
    ggml_unfused = lower_to_ggml(g_unfused, enable_fusion=False)
    ggml_fused = lower_to_ggml(g_fused, enable_fusion=False)

    runner_unfused = ModelRunner(serialize_ggml_graph(ggml_unfused), device="cpu")
    runner_fused = ModelRunner(serialize_ggml_graph(ggml_fused), device="cpu")

    out_unfused = runner_unfused(x_data)
    out_fused = runner_fused(x_data)
    _compare_outputs(out_unfused, out_fused)


def test_horizontal_qkv_fusion():
    """Verify that Q, K, and V projections (including GQA different dimensions) are fused."""
    g = Graph(name="qkv_fusion_test")
    d_in, d_q, d_k, d_v = 32, 32, 16, 16
    x_data = np.random.randn(1, 4, d_in).astype(np.float32)
    w_q_data = np.random.randn(d_q, d_in).astype(np.float32)
    w_k_data = np.random.randn(d_k, d_in).astype(np.float32)
    w_v_data = np.random.randn(d_v, d_in).astype(np.float32)

    t_x = g.add_tensor("x", Shape.from_tuple((1, 4, d_in)), DType.F32, StorageClass.INPUT)
    t_w_q = g.add_tensor("self_attn.q_proj.weight", Shape.from_tuple((d_q, d_in)), DType.F32, StorageClass.PARAMETER, data=w_q_data)
    t_w_k = g.add_tensor("self_attn.k_proj.weight", Shape.from_tuple((d_k, d_in)), DType.F32, StorageClass.PARAMETER, data=w_k_data)
    t_w_v = g.add_tensor("self_attn.v_proj.weight", Shape.from_tuple((d_v, d_in)), DType.F32, StorageClass.PARAMETER, data=w_v_data)
    t_q_out = g.add_tensor("q_out", Shape.from_tuple((1, 4, d_q)), DType.F32, StorageClass.OUTPUT)
    t_k_out = g.add_tensor("k_out", Shape.from_tuple((1, 4, d_k)), DType.F32, StorageClass.OUTPUT)
    t_v_out = g.add_tensor("v_out", Shape.from_tuple((1, 4, d_v)), DType.F32, StorageClass.OUTPUT)

    g.inputs = [t_x.id]
    g.parameters = [t_w_q.id, t_w_k.id, t_w_v.id]
    g.outputs = [t_q_out.id, t_k_out.id, t_v_out.id]

    g.add_op(OpCode.LINEAR, [t_x.id, t_w_q.id], [t_q_out.id], name="q_linear")
    g.add_op(OpCode.LINEAR, [t_x.id, t_w_k.id], [t_k_out.id], name="k_linear")
    g.add_op(OpCode.LINEAR, [t_x.id, t_w_v.id], [t_v_out.id], name="v_linear")

    # 1. Fused QKV
    g_fused = copy.deepcopy(g)
    fuse_operations(g_fused, FusionOptions(enable_horizontal_mlp=False, enable_horizontal_qkv=True))
    linear_ops_fused = [op for op in g_fused.nodes if op.opcode == OpCode.LINEAR]
    slice_ops_fused = [op for op in g_fused.nodes if op.opcode == OpCode.SLICE]
    assert len(linear_ops_fused) == 1
    assert len(slice_ops_fused) == 3

    fused_w = g_fused.get_tensor(linear_ops_fused[0].inputs[1])
    assert fused_w.shape.dims[0].value == (d_q + d_k + d_v)
    assert fused_w.shape.dims[1].value == d_in

    # 2. Numerical differential verification
    ggml_unfused = lower_to_ggml(copy.deepcopy(g), enable_fusion=False)
    ggml_fused = lower_to_ggml(g_fused, enable_fusion=False)

    runner_unfused = ModelRunner(serialize_ggml_graph(ggml_unfused), device="cpu")
    runner_fused = ModelRunner(serialize_ggml_graph(ggml_fused), device="cpu")

    out_unfused = runner_unfused(x_data)
    out_fused = runner_fused(x_data)
    _compare_outputs(out_unfused, out_fused)


def test_horizontal_fusion_with_bias():
    """Verify horizontal fusion handles linear layers with biases correctly."""
    g = Graph(name="bias_fusion_test")
    d_in, d_out1, d_out2 = 16, 24, 32
    x_data = np.random.randn(2, 3, d_in).astype(np.float32)
    w1_data = np.random.randn(d_out1, d_in).astype(np.float32)
    b1_data = np.random.randn(d_out1).astype(np.float32)
    w2_data = np.random.randn(d_out2, d_in).astype(np.float32)
    b2_data = np.random.randn(d_out2).astype(np.float32)

    t_x = g.add_tensor("x", Shape.from_tuple((2, 3, d_in)), DType.F32, StorageClass.INPUT)
    t_w1 = g.add_tensor("mlp.gate_proj.weight", Shape.from_tuple((d_out1, d_in)), DType.F32, StorageClass.PARAMETER, data=w1_data)
    t_b1 = g.add_tensor("mlp.gate_proj.bias", Shape.from_tuple((d_out1,)), DType.F32, StorageClass.PARAMETER, data=b1_data)
    t_w2 = g.add_tensor("mlp.up_proj.weight", Shape.from_tuple((d_out2, d_in)), DType.F32, StorageClass.PARAMETER, data=w2_data)
    t_b2 = g.add_tensor("mlp.up_proj.bias", Shape.from_tuple((d_out2,)), DType.F32, StorageClass.PARAMETER, data=b2_data)
    t_out1 = g.add_tensor("out1", Shape.from_tuple((2, 3, d_out1)), DType.F32, StorageClass.OUTPUT)
    t_out2 = g.add_tensor("out2", Shape.from_tuple((2, 3, d_out2)), DType.F32, StorageClass.OUTPUT)

    g.inputs = [t_x.id]
    g.parameters = [t_w1.id, t_b1.id, t_w2.id, t_b2.id]
    g.outputs = [t_out1.id, t_out2.id]

    g.add_op(OpCode.LINEAR, [t_x.id, t_w1.id, t_b1.id], [t_out1.id], name="linear1")
    g.add_op(OpCode.LINEAR, [t_x.id, t_w2.id, t_b2.id], [t_out2.id], name="linear2")

    g_fused = copy.deepcopy(g)
    fuse_operations(g_fused, FusionOptions(enable_horizontal_mlp=True))
    linear_ops_fused = [op for op in g_fused.nodes if op.opcode == OpCode.LINEAR]
    assert len(linear_ops_fused) == 1
    # Check that fused op has 3 inputs: [x, fused_w, fused_b]
    assert len(linear_ops_fused[0].inputs) == 3

    # Verify numerical parity
    ggml_unfused = lower_to_ggml(copy.deepcopy(g), enable_fusion=False)
    ggml_fused = lower_to_ggml(g_fused, enable_fusion=False)

    runner_unfused = ModelRunner(serialize_ggml_graph(ggml_unfused), device="cpu")
    runner_fused = ModelRunner(serialize_ggml_graph(ggml_fused), device="cpu")

    out_unfused = runner_unfused(x_data)
    out_fused = runner_fused(x_data)
    _compare_outputs(out_unfused, out_fused)

