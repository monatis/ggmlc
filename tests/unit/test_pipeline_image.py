import numpy as np
import pytest
from ggmlc.pipeline.spec import VisionPipelineSpec
from ggmlc.pipeline.torchvision import from_torchvision, verify_torchvision_parity
from ggmlc.pipeline.vision import VisionPreprocessor
from PIL import Image

try:
    import torchvision.transforms as T

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def test_vision_pipeline_spec():
    spec = VisionPipelineSpec(
        target_size=(224, 224),
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        interpolation="bicubic",
        crop_mode="center",
    )
    assert spec.target_size == (224, 224)
    assert len(spec.mean) == 3
    assert spec.interpolation == "bicubic"


def test_vision_preprocessor_numpy():
    spec = VisionPipelineSpec(
        target_size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        interpolation="bicubic",
        crop_mode="center",
    )
    preprocessor = VisionPreprocessor(spec)

    # Random test image 300x400
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    tensor = preprocessor.preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
def test_from_torchvision_introspection():
    transform = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pre = from_torchvision(transform)
    assert pre.target_size == (224, 224)
    assert pre.interpolation == "bicubic"
    assert pre.crop_mode == "center"
    assert pytest.approx(pre.mean[0], 1e-4) == 0.485


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not installed")
def test_torchvision_parity_verification():
    transform = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ]
    )
    pre = VisionPreprocessor.from_torchvision(transform)
    img = Image.fromarray(np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8))
    res = verify_torchvision_parity(img, transform, pre, atol=1e-2)
    assert res["passed"]
    assert res["max_abs_diff"] < 1e-2


def test_vision_preprocessor_from_huggingface():
    try:
        pre = VisionPreprocessor.from_huggingface("openai/clip-vit-base-patch32")
        assert pre.target_size == (224, 224)
        assert pre.interpolation == "bicubic"
        assert pre.crop_mode == "center"
    except ImportError:
        pass
