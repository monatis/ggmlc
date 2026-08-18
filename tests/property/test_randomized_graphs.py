import numpy as np
import pytest
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


class RandomCompositeModel(nn.Module):
    def __init__(self, mode: int):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 0:
            # Reshape -> Linear combination -> ReLU -> Transpose
            b, c, h, w = x.shape
            flat = torch.reshape(x, (b, c * h * w))
            t = flat * 1.5 + 0.5
            act = torch.relu(t)
            return torch.transpose(act, 0, 1)

        elif self.mode == 1:
            # Slicing -> Permute -> SiLU
            # x shape: (B, H, W, C)
            sliced = x[:, :, :4, :]
            perm = sliced.permute(0, 3, 1, 2)
            return torch.nn.functional.silu(perm)

        elif self.mode == 2:
            # Split / Slices -> Concat -> GELU
            half = x.shape[1] // 2
            a = x[:, :half, ...]
            b = x[:, half:, ...]
            combined = torch.cat([b, a], dim=1)
            return torch.nn.functional.gelu(combined)

        elif self.mode == 3:
            # Arithmetic chain + Multi-axis Permute
            y = (x + 2.0) * 0.5 - 1.0
            return y.permute(3, 1, 0, 2)

        else:
            return x * 2.0


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [42, 123, 999, 2026])
def test_randomized_composite_graphs(mode: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mode == 0:
        x = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    elif mode == 1:
        x = torch.randn(2, 4, 8, 6, dtype=torch.float32)
    elif mode == 2:
        x = torch.randn(4, 6, 8, dtype=torch.float32)
    elif mode == 3:
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    else:
        x = torch.randn(2, 8, dtype=torch.float32)

    model = RandomCompositeModel(mode)

    with torch.no_grad():
        ref = model(x).detach().numpy()

    exported = export_torch_model(model, (x,), model_name=f"rand_graph_m{mode}_s{seed}")
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    in_name = exported.main_graph.get_tensor(exported.main_graph.inputs[0]).name
    out_id = exported.main_graph.outputs[0]

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={in_name: x.numpy()},
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    atol = 2e-3 if mode == 2 else 1e-4
    res = check_numerical_accuracy(ref, ggml_out, atol=atol)
    assert res.passed, f"Random graph mode {mode} seed {seed} failed: {res.message}"
