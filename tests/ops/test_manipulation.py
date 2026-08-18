import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


def _run_pt_model_e2e(model: nn.Module, example_args: tuple, model_name: str = "manip_model"):
    # 1. PyTorch reference
    with torch.no_grad():
        ref = model(*example_args).detach().numpy()

    # 2. Export & Lower & Serialize
    exported = export_torch_model(model, example_args, model_name=model_name)
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 3. Input mapping
    inputs = {}
    for i, inp_id in enumerate(exported.main_graph.inputs):
        name = exported.main_graph.get_tensor(inp_id).name
        inputs[name] = example_args[i].numpy()

    out_id = exported.main_graph.outputs[0]
    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs=inputs,
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=1e-5)
    assert res.passed, f"Differential test failed: {res.message}"


def test_op_transpose_2d():
    class TransposeModel(nn.Module):
        def forward(self, x):
            return torch.transpose(x, 0, 1)

    x = torch.randn(4, 8, dtype=torch.float32)
    _run_pt_model_e2e(TransposeModel(), (x,), "transpose_2d")


def test_op_permute_3d():
    class Permute3DModel(nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1)

    x = torch.randn(2, 6, 8, dtype=torch.float32)
    _run_pt_model_e2e(Permute3DModel(), (x,), "permute_3d")


def test_op_permute_4d():
    class Permute4DModel(nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1, 3)

    x = torch.randn(2, 4, 8, 16, dtype=torch.float32)
    _run_pt_model_e2e(Permute4DModel(), (x,), "permute_4d")


def test_op_slice_dim0():
    class SliceDim0Model(nn.Module):
        def forward(self, x):
            return x[2:6, :]

    x = torch.randn(8, 16, dtype=torch.float32)
    _run_pt_model_e2e(SliceDim0Model(), (x,), "slice_dim0")


def test_op_slice_dim1():
    class SliceDim1Model(nn.Module):
        def forward(self, x):
            return x[:, 4:12]

    x = torch.randn(8, 16, dtype=torch.float32)
    _run_pt_model_e2e(SliceDim1Model(), (x,), "slice_dim1")


def test_op_concat_dim0():
    class ConcatDim0Model(nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=0)

    a = torch.randn(3, 8, dtype=torch.float32)
    b = torch.randn(5, 8, dtype=torch.float32)
    _run_pt_model_e2e(ConcatDim0Model(), (a, b), "concat_dim0")


def test_op_concat_dim1():
    class ConcatDim1Model(nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=1)

    a = torch.randn(4, 6, dtype=torch.float32)
    b = torch.randn(4, 10, dtype=torch.float32)
    _run_pt_model_e2e(ConcatDim1Model(), (a, b), "concat_dim1")


def test_op_expand_broadcast():
    class ExpandModel(nn.Module):
        def forward(self, x):
            return x.expand(4, 16)

    x = torch.randn(1, 16, dtype=torch.float32)
    _run_pt_model_e2e(ExpandModel(), (x,), "expand_broadcast")


def test_op_squeeze_unsqueeze():
    class SqueezeUnsqueezeModel(nn.Module):
        def forward(self, x):
            return torch.unsqueeze(torch.squeeze(torch.unsqueeze(x, 0), 0), 1)

    x = torch.randn(4, 8, dtype=torch.float32)
    _run_pt_model_e2e(SqueezeUnsqueezeModel(), (x,), "squeeze_unsqueeze")
