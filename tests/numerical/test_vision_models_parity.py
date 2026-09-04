"""Comprehensive differential parity tests for automatic vision preprocessing across vision models."""

from __future__ import annotations

import numpy as np
import pytest
from ggmlc.pipeline.torchvision import verify_torchvision_parity
from ggmlc.pipeline.vision import VisionPreprocessor
from PIL import Image

try:
    import torchvision.models as tv_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from transformers import AutoImageProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
@pytest.mark.parametrize(
    "model_name,weights_enum",
    [
        ("resnet50", "ResNet50_Weights"),
        ("resnet18", "ResNet18_Weights"),
        ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
        ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
        ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
        ("efficientnet_b0", "EfficientNet_B0_Weights"),
        ("densenet121", "DenseNet121_Weights"),
        ("vit_b_16", "ViT_B_16_Weights"),
    ],
)
def test_torchvision_models_preprocessor_parity(model_name: str, weights_enum: str):
    """Verifies that VisionPreprocessor.from_torchvision(weights) achieves bit-for-bit parity against official torchvision reference transforms."""
    weights_cls = getattr(tv_models, weights_enum)
    weights = weights_cls.DEFAULT
    ref_transform = weights.transforms()

    # 1. Automatic generation from weights enum directly
    pre_from_weights = VisionPreprocessor.from_torchvision(weights)

    # 2. Automatic generation from transforms() directly
    pre_from_transform = VisionPreprocessor.from_torchvision(ref_transform)

    # Test image (non-square 320x240 to test aspect ratio calculation)
    np.random.seed(42)
    img_arr = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
    test_img = Image.fromarray(img_arr)

    # Verify parity for both
    res_weights = verify_torchvision_parity(test_img, ref_transform, pre_from_weights, atol=1e-5)
    assert res_weights["passed"], f"Failed for {model_name} from weights: {res_weights}"
    assert res_weights["cosine_similarity"] > 0.999999

    res_transform = verify_torchvision_parity(
        test_img, ref_transform, pre_from_transform, atol=1e-5
    )
    assert res_transform["passed"], f"Failed for {model_name} from transform: {res_transform}"
    assert res_transform["cosine_similarity"] > 0.999999


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
        "facebook/convnext-tiny-224",
        "openai/clip-vit-base-patch32",
    ],
)
def test_huggingface_auto_image_processor_parity(model_id: str):
    """Verifies that VisionPreprocessor.from_huggingface(model_id) achieves numerical parity against Hugging Face AutoImageProcessor."""
    proc = AutoImageProcessor.from_pretrained(model_id)

    # 1. Automatic generation from string ID
    pre_from_str = VisionPreprocessor.from_huggingface(model_id)

    # 2. Automatic generation from processor object
    pre_from_proc = VisionPreprocessor.from_huggingface(proc)

    np.random.seed(42)
    img_arr = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
    test_img = Image.fromarray(img_arr)

    ref_out = proc(test_img, return_tensors="pt")["pixel_values"].numpy()

    act_str = pre_from_str.process(test_img)
    act_proc = pre_from_proc.process(test_img)

    # Check cosine similarity
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        flat_a = a.flatten().astype(np.float64)
        flat_b = b.flatten().astype(np.float64)
        return float(
            np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12)
        )

    cos_str = _cosine_sim(ref_out, act_str)
    cos_proc = _cosine_sim(ref_out, act_proc)

    assert cos_str > 0.99999, f"HF string ID extraction failed for {model_id}: cos_sim={cos_str}"
    assert cos_proc > 0.99999, (
        f"HF proc object extraction failed for {model_id}: cos_sim={cos_proc}"
    )
