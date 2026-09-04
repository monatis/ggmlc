import pytest
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner, get_available_devices
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity

from examples.models.clip_model import (
    load_clip_full_model,
    load_clip_text_model,
    load_clip_vision_model,
)

cuda_available = "cuda" in get_available_devices() or "cuda:0" in get_available_devices()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not cuda_available, reason="CUDA not available")
        ),
    ],
)
def test_clip_vision_e2e(device: str):
    model, inputs, _ = load_clip_vision_model()
    model.eval()
    with torch.no_grad():
        ref_out = model(*inputs).detach().cpu().numpy()

    exp = export_torch_model(model, inputs, model_name="clip_vision")
    bytes_data = serialize_ggml_graph(lower_to_ggml(exp.main_graph))
    runner = ModelRunner(bytes_data, device=device)
    actual_out = runner(inputs[0].numpy()).reshape(ref_out.shape)

    cos = cosine_similarity(ref_out, actual_out)
    cmp = check_numerical_accuracy(ref_out, actual_out, atol=1e-3)
    assert cos > 0.9999, f"CLIP Vision cosine similarity too low on {device}: {cos}"
    assert cmp.passed, f"CLIP Vision numerical check failed on {device}: {cmp.message}"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not cuda_available, reason="CUDA not available")
        ),
    ],
)
def test_clip_text_e2e(device: str):
    model, inputs, _ = load_clip_text_model()
    model.eval()
    with torch.no_grad():
        ref_out = model(*inputs).detach().cpu().numpy()

    exp = export_torch_model(model, inputs, model_name="clip_text")
    bytes_data = serialize_ggml_graph(lower_to_ggml(exp.main_graph))
    runner = ModelRunner(bytes_data, device=device)
    actual_out = runner(inputs[0].numpy()).reshape(ref_out.shape)

    cos = cosine_similarity(ref_out, actual_out)
    cmp = check_numerical_accuracy(ref_out, actual_out, atol=1e-3)
    assert cos > 0.9999, f"CLIP Text cosine similarity too low on {device}: {cos}"
    assert cmp.passed, f"CLIP Text numerical check failed on {device}: {cmp.message}"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not cuda_available, reason="CUDA not available")
        ),
    ],
)
def test_clip_full_multimodal_e2e(device: str):
    model, inputs, _ = load_clip_full_model()
    model.eval()
    with torch.no_grad():
        ref_out = model(*inputs).detach().cpu().numpy()

    exp = export_torch_model(model, inputs, model_name="clip_full")
    bytes_data = serialize_ggml_graph(lower_to_ggml(exp.main_graph))
    runner = ModelRunner(bytes_data, device=device)
    actual_out = runner(*(x.numpy() for x in inputs)).reshape(ref_out.shape)

    cos = cosine_similarity(ref_out, actual_out)
    cmp = check_numerical_accuracy(ref_out, actual_out, atol=1e-3)
    assert cos > 0.9999, f"CLIP Full Multimodal cosine similarity too low on {device}: {cos}"
    assert cmp.passed, f"CLIP Full Multimodal numerical check failed on {device}: {cmp.message}"
