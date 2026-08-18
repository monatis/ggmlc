import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch.exporter import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


class ArithmeticModel(nn.Module):
    def forward(self, x):
        return (x + 3.0) * 2.0


def test_e2e_elementwise_arithmetic():
    m = ArithmeticModel()
    x = torch.randn(4, 16, dtype=torch.float32)

    # 1. Reference output
    ref = m(x).detach().numpy()

    # 2. Export & Lower & Serialize
    model = export_torch_model(m, (x,), model_name="arithmetic")
    ggml_graph = lower_to_ggml(model.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Execute with generic C++ GGML runtime
    in_name = model.main_graph.get_tensor(model.main_graph.inputs[0]).name
    out_id = model.main_graph.outputs[0]

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={in_name: x.numpy()},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-4)
    assert res.passed, f"Numerical check failed: {res.message}"


class MatMulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        return self.fc(x)


def test_e2e_linear_matmul():
    m = MatMulModel()
    x = torch.randn(2, 32, dtype=torch.float32)

    # 1. Reference output
    ref = m(x).detach().numpy()

    # 2. Export & Lower & Serialize
    model = export_torch_model(m, (x,), model_name="linear")
    ggml_graph = lower_to_ggml(model.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Execute with generic C++ GGML runtime
    in_name = model.main_graph.get_tensor(model.main_graph.inputs[0]).name
    out_id = model.main_graph.outputs[0]

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={in_name: x.numpy()},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-4)
    assert res.passed, f"Numerical check failed: {res.message}"


class ReshapeReluModel(nn.Module):
    def forward(self, x):
        reshaped = torch.reshape(x, (8, 8))
        return torch.relu(reshaped)


def test_e2e_reshape_relu():
    m = ReshapeReluModel()
    x = torch.randn(2, 32, dtype=torch.float32)

    # 1. Reference output
    ref = m(x).detach().numpy()

    # 2. Export & Lower & Serialize
    model = export_torch_model(m, (x,), model_name="reshape_relu")
    ggml_graph = lower_to_ggml(model.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Execute with generic C++ GGML runtime
    in_name = model.main_graph.get_tensor(model.main_graph.inputs[0]).name
    out_id = model.main_graph.outputs[0]

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={in_name: x.numpy()},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-4)
    assert res.passed, f"Numerical check failed: {res.message}"
