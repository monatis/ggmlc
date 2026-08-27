"""End-to-end tests for Vision Transformer (ViT) architectures."""

import pytest
import torch
from torchvision.models import vit_b_16

from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.runtime.runner import ModelRunner
from ggmlc.validation.numerical import check_numerical_accuracy


def test_vit_b_16_cpu():
    """Verifies Vision Transformer (ViT-B/16) on CPU backend."""
    model = vit_b_16(weights=None).eval()
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        ref = model(x).numpy()

    exported = export_torch_model(model, (x,), model_name="vit_b_16")
    assert len(exported.main_graph.nodes) > 0

    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser_bytes, device="cpu")
    act = runner(x.numpy())
    if isinstance(act, dict):
        act = next(iter(act.values()))
    act = act.reshape(ref.shape)

    result = check_numerical_accuracy(ref, act, atol=1e-3)
    assert result.passed, f"ViT-B/16 numerical parity failed: {result}"
