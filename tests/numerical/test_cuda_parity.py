import tempfile
from pathlib import Path

import ggmlc
import numpy as np
import pytest
import torch
from torch import nn


def is_cuda_available() -> bool:
    return any(d.startswith("cuda") for d in ggmlc.get_available_devices())


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA device required for CUDA parity tests")
def test_cuda_mlp_parity():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(64, 16)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = MLP().eval()
    x = torch.randn(2, 32)
    torch_out = model(x).detach().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "mlp.gguf"
        ggmlc.compile(model, (x,), output=model_path, model_name="MLP")

        cpu_runner = ggmlc.load(model_path, device="cpu")
        cuda_runner = ggmlc.load(model_path, device="cuda")

        cpu_out = cpu_runner(x.numpy())
        cuda_out = cuda_runner(x.numpy())

        assert np.max(np.abs(torch_out - cpu_out)) < 1e-4
        assert np.max(np.abs(torch_out - cuda_out)) < 1e-4
        assert np.max(np.abs(cpu_out - cuda_out)) < 1e-5


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA device required for CUDA parity tests")
def test_cuda_fused_transformer_layer_parity():
    class TransformerBlock(nn.Module):
        def __init__(self, d_model=32, n_heads=2):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

        def forward(self, x):
            norm_x = self.ln1(x)
            q = self.q_proj(norm_x)
            k = self.k_proj(norm_x)
            v = self.v_proj(norm_x)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_probs, v)
            h = x + self.out_proj(attn_out)
            return h + self.mlp(self.ln2(h))

    model = TransformerBlock().eval()
    x = torch.randn(1, 8, 32)
    torch_out = model(x).detach().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "trans.gguf"
        ggmlc.compile(model, (x,), output=model_path, model_name="TransformerBlock")

        cpu_runner = ggmlc.load(model_path, device="cpu")
        cuda_runner = ggmlc.load(model_path, device="cuda")

        cpu_out = cpu_runner(x.numpy())
        cuda_out = cuda_runner(x.numpy())

        assert np.max(np.abs(torch_out - cpu_out)) < 1e-3
        assert np.max(np.abs(torch_out - cuda_out)) < 1e-3
        assert np.max(np.abs(cpu_out - cuda_out)) < 1e-4


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA device required for CUDA parity tests")
def test_cuda_quantized_parity():
    class BigLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(128, 32)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = BigLinear().eval()
    x = torch.randn(2, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Q8_0 Quantization on CUDA
        q8_path = Path(tmpdir) / "model_q8.gguf"
        ggmlc.compile(model, (x,), output=q8_path, quantize="q8_0", model_name="BigLinearQ8")

        cpu_q8 = ggmlc.load(q8_path, device="cpu")
        cuda_q8 = ggmlc.load(q8_path, device="cuda")

        cpu_out = cpu_q8(x.numpy())
        cuda_out = cuda_q8(x.numpy())

        # Quantized CPU and GPU kernels should match closely
        diff = np.max(np.abs(cpu_out - cuda_out))
        assert diff < 1e-3
