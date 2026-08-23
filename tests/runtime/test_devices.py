import tempfile
from pathlib import Path

import ggmlc
import numpy as np
import pytest
import torch
from torch import nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_device_enumeration():
    devices = ggmlc.get_available_devices()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    assert "cpu" in devices


def test_cpu_device_execution():
    model = TinyModel().eval()
    x = torch.randn(1, 8)
    torch_out = model(x).detach().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tiny.gguf"
        ggmlc.compile(model, (x,), output=model_path, model_name="Tiny")

        runner = ggmlc.load(model_path, device="cpu")
        assert runner.device == "cpu"

        out = runner(x.numpy())
        diff = np.max(np.abs(torch_out - out))
        assert diff < 1e-4


def test_auto_device_execution():
    model = TinyModel().eval()
    x = torch.randn(1, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tiny.gguf"
        ggmlc.compile(model, (x,), output=model_path, model_name="Tiny")

        runner = ggmlc.load(model_path, device="auto")
        assert runner.device in ("cpu", "cuda:0", "cuda")
        out = runner(x.numpy())
        assert out.squeeze().shape == (4,)


def test_cuda_device_execution_if_available():
    devices = ggmlc.get_available_devices()
    cuda_available = any(d.startswith("cuda") for d in devices)

    if not cuda_available:
        pytest.skip("CUDA device not available in current environment")

    model = TinyModel().eval()
    x = torch.randn(2, 8)
    torch_out = model(x).detach().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tiny.gguf"
        ggmlc.compile(model, (x,), output=model_path, model_name="Tiny")

        cuda_runner = ggmlc.load(model_path, device="cuda")
        assert "cuda" in cuda_runner.device
        cuda_out = cuda_runner(x.numpy())

        diff = np.max(np.abs(torch_out - cuda_out))
        assert diff < 1e-4
