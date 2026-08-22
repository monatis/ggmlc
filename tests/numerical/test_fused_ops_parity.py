"""Differential numerical parity testing for fused operations (ggmlc-fused / ggmlc-stdlib) vs PyTorch."""

from __future__ import annotations

import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import run_compiled_model_wsl


def test_bias_gelu_parity():
    """Verify BiasGELU custom kernel numerical parity against PyTorch GELU(x + b)."""
    B, S, D = 2, 4, 64
    x_val = np.random.randn(B, S, D).astype(np.float32)
    b_val = np.random.randn(D).astype(np.float32)

    # 1. Reference PyTorch
    x_pt = torch.from_numpy(x_val)
    b_pt = torch.from_numpy(b_val)
    ref_out = torch.nn.functional.gelu(x_pt + b_pt).numpy()

    # 2. Canonical Graph with BIAS_GELU op
    g = Graph(name="bias_gelu_test")
    t_x = g.add_tensor("x", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.INPUT)
    t_b = g.add_tensor(
        "bias", Shape.from_tuple((D,)), DType.F32, StorageClass.PARAMETER, data=b_val
    )
    t_out = g.add_tensor("out", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [t_x.id]
    g.parameters = [t_b.id]
    g.outputs = [t_out.id]
    g.add_op(OpCode.BIAS_GELU, [t_x.id, t_b.id], [t_out.id])

    # 3. Lower and run
    ggml_graph = lower_to_ggml(g, enable_fusion=True)
    serialized = serialize_ggml_graph(ggml_graph)

    results = run_compiled_model_wsl(
        serialized_bytes=serialized,
        inputs={"x": x_val},
        output_tensor_ids=[t_out.id],
    )
    actual_out = results[t_out.id].reshape((B, S, D))

    max_diff = np.max(np.abs(ref_out - actual_out))
    cos_sim = np.dot(ref_out.flatten(), actual_out.flatten()) / (
        np.linalg.norm(ref_out) * np.linalg.norm(actual_out) + 1e-9
    )

    assert cos_sim > 0.99999, f"BiasGELU Cosine Sim too low: {cos_sim}"
    assert max_diff < 1e-3, f"BiasGELU Max Diff too high: {max_diff}"


def test_layer_norm_fused_parity():
    """Verify Fused LayerNorm numerical parity against PyTorch nn.LayerNorm."""
    B, S, D = 2, 8, 128
    eps = 1e-5
    x_val = np.random.randn(B, S, D).astype(np.float32)
    w_val = np.random.randn(D).astype(np.float32)
    b_val = np.random.randn(D).astype(np.float32)

    # 1. Reference PyTorch
    x_pt = torch.from_numpy(x_val)
    w_pt = torch.from_numpy(w_val)
    b_pt = torch.from_numpy(b_val)
    ref_out = torch.nn.functional.layer_norm(x_pt, (D,), weight=w_pt, bias=b_pt, eps=eps).numpy()

    # 2. Canonical Graph with LAYER_NORM op
    g = Graph(name="layer_norm_test")
    t_x = g.add_tensor("x", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.INPUT)
    t_w = g.add_tensor("w", Shape.from_tuple((D,)), DType.F32, StorageClass.PARAMETER, data=w_val)
    t_b = g.add_tensor("b", Shape.from_tuple((D,)), DType.F32, StorageClass.PARAMETER, data=b_val)
    t_out = g.add_tensor("out", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [t_x.id]
    g.parameters = [t_w.id, t_b.id]
    g.outputs = [t_out.id]
    g.add_op(OpCode.LAYER_NORM, [t_x.id, t_w.id, t_b.id], [t_out.id], attributes={"eps": eps})

    # 3. Lower and run
    ggml_graph = lower_to_ggml(g, enable_fusion=True)
    serialized = serialize_ggml_graph(ggml_graph)

    results = run_compiled_model_wsl(
        serialized_bytes=serialized,
        inputs={"x": x_val},
        output_tensor_ids=[t_out.id],
    )
    actual_out = results[t_out.id].reshape((B, S, D))

    max_diff = np.max(np.abs(ref_out - actual_out))
    cos_sim = np.dot(ref_out.flatten(), actual_out.flatten()) / (
        np.linalg.norm(ref_out) * np.linalg.norm(actual_out) + 1e-9
    )

    assert cos_sim > 0.99999, f"LayerNorm Cosine Sim too low: {cos_sim}"
    assert max_diff < 1e-3, f"LayerNorm Max Diff too high: {max_diff}"


def test_rms_norm_fused_parity():
    """Verify Fused RMSNorm numerical parity against PyTorch reference."""
    B, S, D = 2, 8, 128
    eps = 1e-6
    x_val = np.random.randn(B, S, D).astype(np.float32)
    w_val = np.random.randn(D).astype(np.float32)

    # 1. Reference PyTorch
    x_pt = torch.from_numpy(x_val)
    w_pt = torch.from_numpy(w_val)
    ms = x_pt.pow(2).mean(-1, keepdim=True)
    ref_out = (x_pt * torch.rsqrt(ms + eps) * w_pt).numpy()

    # 2. Canonical Graph with RMS_NORM op
    g = Graph(name="rms_norm_test")
    t_x = g.add_tensor("x", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.INPUT)
    t_w = g.add_tensor("w", Shape.from_tuple((D,)), DType.F32, StorageClass.PARAMETER, data=w_val)
    t_out = g.add_tensor("out", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [t_x.id]
    g.parameters = [t_w.id]
    g.outputs = [t_out.id]
    g.add_op(OpCode.RMS_NORM, [t_x.id, t_w.id], [t_out.id], attributes={"eps": eps})

    # 3. Lower and run
    ggml_graph = lower_to_ggml(g, enable_fusion=True)
    serialized = serialize_ggml_graph(ggml_graph)

    results = run_compiled_model_wsl(
        serialized_bytes=serialized,
        inputs={"x": x_val},
        output_tensor_ids=[t_out.id],
    )
    actual_out = results[t_out.id].reshape((B, S, D))

    max_diff = np.max(np.abs(ref_out - actual_out))
    cos_sim = np.dot(ref_out.flatten(), actual_out.flatten()) / (
        np.linalg.norm(ref_out) * np.linalg.norm(actual_out) + 1e-9
    )

    assert cos_sim > 0.99999, f"RMSNorm Cosine Sim too low: {cos_sim}"
    assert max_diff < 1e-3, f"RMSNorm Max Diff too high: {max_diff}"


def test_swiglu_fused_parity():
    """Verify Fused SwiGLU numerical parity against PyTorch silu(gate) * up."""
    B, S, D = 2, 8, 128
    gate_val = np.random.randn(B, S, D).astype(np.float32)
    up_val = np.random.randn(B, S, D).astype(np.float32)

    # 1. Reference PyTorch
    gate_pt = torch.from_numpy(gate_val)
    up_pt = torch.from_numpy(up_val)
    ref_out = (torch.nn.functional.silu(gate_pt) * up_pt).numpy()

    # 2. Canonical Graph with SWIGLU op
    g = Graph(name="swiglu_test")
    t_gate = g.add_tensor("gate", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.INPUT)
    t_up = g.add_tensor("up", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.INPUT)
    t_out = g.add_tensor("out", Shape.from_tuple((B, S, D)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [t_gate.id, t_up.id]
    g.outputs = [t_out.id]
    g.add_op(OpCode.SWIGLU, [t_gate.id, t_up.id], [t_out.id])

    # 3. Lower and run
    ggml_graph = lower_to_ggml(g, enable_fusion=True)
    serialized = serialize_ggml_graph(ggml_graph)

    results = run_compiled_model_wsl(
        serialized_bytes=serialized,
        inputs={"gate": gate_val, "up": up_val},
        output_tensor_ids=[t_out.id],
    )
    actual_out = results[t_out.id].reshape((B, S, D))

    max_diff = np.max(np.abs(ref_out - actual_out))
    cos_sim = np.dot(ref_out.flatten(), actual_out.flatten()) / (
        np.linalg.norm(ref_out) * np.linalg.norm(actual_out) + 1e-9
    )

    assert cos_sim > 0.99999, f"SwiGLU Cosine Sim too low: {cos_sim}"
    assert max_diff < 1e-4, f"SwiGLU Max Diff too high: {max_diff}"
