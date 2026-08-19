"""Unit tests for GGML quantization and dequantization algorithms."""

import numpy as np
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.quantization import (
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_graph_parameters,
    quantize_q4_0,
    quantize_q8_0,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


def test_quantize_dequantize_q8_0():
    """Verify Q8_0 packing, unpacking, byte size, and high numerical fidelity."""
    np.random.seed(42)
    shape = (64, 128)
    data = np.random.randn(*shape).astype(np.float32)

    raw_q8 = quantize_q8_0(data)
    expected_bytes = (64 * 128 // 32) * 34  # 34 bytes per 32 floats
    assert len(raw_q8) == expected_bytes

    reconstructed = dequantize_q8_0(raw_q8, shape)
    assert reconstructed.shape == shape

    cos_sim = cosine_similarity(data, reconstructed)
    max_err = float(np.max(np.abs(data - reconstructed)))

    assert cos_sim > 0.9999
    assert max_err < 0.05


def test_quantize_dequantize_q4_0():
    """Verify Q4_0 4-bit nibble packing, unpacking, byte size, and fidelity."""
    np.random.seed(42)
    shape = (64, 128)
    data = np.random.randn(*shape).astype(np.float32)

    raw_q4 = quantize_q4_0(data)
    expected_bytes = (64 * 128 // 32) * 18  # 18 bytes per 32 floats
    assert len(raw_q4) == expected_bytes

    reconstructed = dequantize_q4_0(raw_q4, shape)
    assert reconstructed.shape == shape

    cos_sim = cosine_similarity(data, reconstructed)
    max_err = float(np.max(np.abs(data - reconstructed)))

    assert cos_sim > 0.99
    assert max_err < 0.6


def test_quantize_graph_parameters_q8_0():
    """Verify graph-level parameter quantization to Q8_0."""
    g = Graph("mlp")
    x = g.add_tensor("x", Shape([1, 64]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor(
        "w",
        Shape([128, 64]),
        DType.F32,
        StorageClass.PARAMETER,
        data=np.random.randn(128, 64).astype(np.float32),
    )
    b = g.add_tensor(
        "b",
        Shape([128]),
        DType.F32,
        StorageClass.PARAMETER,
        data=np.random.randn(128).astype(np.float32),
    )
    out = g.add_tensor("out", Shape([1, 128]), DType.F32, StorageClass.ACTIVATION)

    g.add_node(OpCode.LINEAR, inputs=[x.id, w.id, b.id], outputs=[out.id])
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id, b.id]

    q_graph, stats = quantize_graph_parameters(g, target_dtype=DType.Q8_0)

    assert stats["tensors_quantized"] == 1  # only 2D w is quantized, 1D bias preserved as F32
    assert q_graph.tensors[w.id].dtype == DType.Q8_0
    assert q_graph.tensors[b.id].dtype == DType.F32
    assert stats["compression_ratio"] > 3.0


def test_quantize_graph_parameters_q4_0():
    """Verify graph-level parameter quantization to Q4_0."""
    g = Graph("mlp")
    x = g.add_tensor("x", Shape([1, 64]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor(
        "w",
        Shape([128, 64]),
        DType.F32,
        StorageClass.PARAMETER,
        data=np.random.randn(128, 64).astype(np.float32),
    )
    out = g.add_tensor("out", Shape([1, 128]), DType.F32, StorageClass.ACTIVATION)

    g.add_node(OpCode.MATMUL, inputs=[x.id, w.id], outputs=[out.id])
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id]

    q_graph, stats = quantize_graph_parameters(g, target_dtype=DType.Q4_0)

    assert stats["tensors_quantized"] == 1
    assert q_graph.tensors[w.id].dtype == DType.Q4_0
    assert stats["compression_ratio"] > 6.5


def test_quantized_model_e2e_cosine_similarity():
    """Verify end-to-end execution of Q8_0 and Q4_0 quantized models in C++ runtime."""
    import torch
    from ggmlc.dialect.ggml.lowering import lower_to_ggml
    from ggmlc.frontend.pytorch import export_torch_model
    from ggmlc.serialization.graph import serialize_ggml_graph
    from ggmlc.validation.numerical import run_compiled_model_wsl
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(42)
    model = MLP().eval()
    x = torch.randn(1, 64, dtype=torch.float32)
    ref_out = model(x).detach().numpy()

    # 1. Export FP32
    exp = export_torch_model(model, (x,), model_name="mlp_e2e")
    inp_name = exp.main_graph.tensors[exp.main_graph.inputs[0]].name
    g_fp32 = lower_to_ggml(exp.main_graph)
    ser_fp32 = serialize_ggml_graph(g_fp32)
    res_fp32 = run_compiled_model_wsl(ser_fp32, {inp_name: x.numpy()}, [exp.main_graph.outputs[0]])
    out_fp32 = res_fp32[exp.main_graph.outputs[0]].reshape(ref_out.shape)

    # FP32 parity
    cos_fp32 = cosine_similarity(ref_out, out_fp32)
    assert cos_fp32 > 0.9999

    # 2. Quantize to Q8_0 and execute
    g_q8, stats_q8 = quantize_graph_parameters(g_fp32, target_dtype=DType.Q8_0)
    ser_q8 = serialize_ggml_graph(g_q8)
    res_q8 = run_compiled_model_wsl(ser_q8, {inp_name: x.numpy()}, [exp.main_graph.outputs[0]])
    out_q8 = res_q8[exp.main_graph.outputs[0]].reshape(ref_out.shape)

    cos_q8 = cosine_similarity(ref_out, out_q8)
    assert cos_q8 > 0.999
    assert stats_q8["compression_ratio"] > 3.0

    # 3. Quantize to Q4_0 and execute
    g_q4, stats_q4 = quantize_graph_parameters(g_fp32, target_dtype=DType.Q4_0)
    ser_q4 = serialize_ggml_graph(g_q4)
    res_q4 = run_compiled_model_wsl(ser_q4, {inp_name: x.numpy()}, [exp.main_graph.outputs[0]])
    out_q4 = res_q4[exp.main_graph.outputs[0]].reshape(ref_out.shape)

    cos_q4 = cosine_similarity(ref_out, out_q4)
    assert cos_q4 > 0.98
    assert stats_q4["compression_ratio"] > 6.0
