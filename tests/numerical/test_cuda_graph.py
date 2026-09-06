import numpy as np
import pytest
import torch
from ggmlc._runtime import ModelExecutor, ModelLoader, get_available_devices
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, in_features=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_cuda_graph_execution_and_parity():
    devices = get_available_devices()
    cuda_devices = [d for d in devices if d.startswith("cuda")]
    if not cuda_devices:
        pytest.skip("CUDA device not available")

    torch.manual_seed(42)
    model = SimpleMLP(in_features=64, hidden=128).eval()
    x = torch.randn(1, 16, 64, dtype=torch.float32)

    # Compile model to GGML
    exported = export_torch_model(model, (x,), model_name="mlp")
    ggml_graph = lower_to_ggml(exported.main_graph)
    gguf_bytes = serialize_ggml_graph(ggml_graph)

    serialized_graph = ModelLoader.load_from_bytes(gguf_bytes)

    # 1. Direct Execution Baseline
    executor_direct = ModelExecutor(serialized_graph, device="cuda:0")
    executor_direct.prepare()
    executor_direct.set_input_by_name("x", x.numpy())
    executor_direct.run()
    out_direct = np.frombuffer(
        executor_direct.get_output_bytes(serialized_graph.outputs[0]), dtype=np.float32
    )

    # 2. CUDA Graph Execution
    executor_graph = ModelExecutor(serialized_graph, device="cuda:0")
    executor_graph.set_enable_cuda_graph(True)
    assert executor_graph.is_cuda_graph_enabled() is True
    assert executor_graph.is_cuda_graph_captured() is False

    executor_graph.prepare()
    executor_graph.set_input_by_name("x", x.numpy())

    # First run triggers stream capture and graph instantiation
    executor_graph.run()
    assert executor_graph.is_cuda_graph_captured() is True

    out_graph1 = np.frombuffer(
        executor_graph.get_output_bytes(serialized_graph.outputs[0]), dtype=np.float32
    )
    np.testing.assert_allclose(out_direct, out_graph1, atol=1e-4, rtol=1e-4)

    # Subsequent run executes from instantiated CUDA graph
    x2 = torch.randn(1, 16, 64, dtype=torch.float32)
    executor_direct.set_input_by_name("x", x2.numpy())
    executor_direct.run()
    out_direct2 = np.frombuffer(
        executor_direct.get_output_bytes(serialized_graph.outputs[0]), dtype=np.float32
    )

    executor_graph.set_input_by_name("x", x2.numpy())
    executor_graph.run()
    out_graph2 = np.frombuffer(
        executor_graph.get_output_bytes(serialized_graph.outputs[0]), dtype=np.float32
    )
    np.testing.assert_allclose(out_direct2, out_graph2, atol=1e-4, rtol=1e-4)
