import torch
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir import OpCode
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


def test_export_simple_model():
    model = SimpleModel()
    x = torch.randn(2, 32)
    m = export_torch_model(model, (x,), model_name="simple")

    assert m.name == "simple"
    g = m.main_graph
    assert len(g.inputs) == 1
    assert len(g.outputs) == 1
    assert len(g.parameters) == 2  # weight and bias

    # Check ops in graph
    opcodes = [op.opcode for op in g.nodes]
    assert OpCode.MATMUL in opcodes or OpCode.LINEAR in opcodes
    assert OpCode.RELU in opcodes


class DynamicModel(nn.Module):
    def forward(self, a, b):
        return (a + b) * 2.0


def test_dynamic_shape_export():
    m = DynamicModel()
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    dim_b = torch.export.Dim("batch", min=1, max=32)
    dynamic_shapes = {"a": {0: dim_b}, "b": {0: dim_b}}

    model = export_torch_model(m, (a, b), dynamic_shapes=dynamic_shapes, model_name="dynamic_add")
    g = model.main_graph
    assert len(g.inputs) == 2
    assert len(g.outputs) == 1

    in_a = g.get_tensor(g.inputs[0])
    assert not in_a.shape[0].is_static()
    assert len(in_a.shape[0].free_symbols()) == 1
