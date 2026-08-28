import pytest
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner, get_available_devices
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy
from torch import nn

from examples.models.hub_models import (
    load_gpt2_model,
    load_minilm_model,
    load_mobilenet_v3_model,
    load_resnet_model,
)

cuda_available = "cuda" in get_available_devices() or "cuda:0" in get_available_devices()
pytestmark = pytest.mark.skipif(not cuda_available, reason="Native CUDA device not available")


def _verify_model_cuda(
    model: nn.Module, inputs: tuple[torch.Tensor, ...], model_name: str, atol: float = 1e-3
):
    model.eval()
    with torch.no_grad():
        ref_out = model(*inputs)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        if hasattr(ref_out, "logits"):
            ref_out = ref_out.logits
        if hasattr(ref_out, "last_hidden_state"):
            ref_out = ref_out.last_hidden_state
        ref_np = ref_out.detach().cpu().numpy()

    exported = export_torch_model(model, inputs, model_name=model_name)
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser_bytes, device="cuda")
    inputs_np = [x.detach().cpu().numpy() for x in inputs]
    out = runner(*inputs_np)

    cmp = check_numerical_accuracy(ref_np, out.reshape(ref_np.shape), atol=atol)
    assert cmp.passed, f"CUDA numerical verification failed: {cmp.message}"


def test_cuda_mobilenet_v3():
    model, inputs, _ = load_mobilenet_v3_model("small")
    _verify_model_cuda(model, inputs, "mobilenet_v3_small", atol=1e-3)


def test_cuda_resnet18():
    model, inputs, _ = load_resnet_model("resnet18")
    _verify_model_cuda(model, inputs, "resnet18", atol=1e-3)


def test_cuda_minilm():
    model, inputs, _ = load_minilm_model()
    _verify_model_cuda(model, inputs, "minilm", atol=1e-2)


def test_cuda_gpt2():
    model, inputs, _ = load_gpt2_model()
    _verify_model_cuda(model, inputs, "gpt2", atol=1e-3)


def test_cuda_convnext():
    from examples.models.hub_models import load_convnext_model

    model, inputs, _ = load_convnext_model("tiny")
    _verify_model_cuda(model, inputs, "convnext_tiny", atol=2e-2)


def test_cuda_efficientnet():
    from examples.models.hub_models import load_efficientnet_model

    model, inputs, _ = load_efficientnet_model("b0")
    _verify_model_cuda(model, inputs, "efficientnet_b0", atol=1e-3)
