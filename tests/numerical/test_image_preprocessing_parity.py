import io

import numpy as np
import pytest
from ggmlc.pipeline.huggingface import from_huggingface_image_processor
from PIL import Image

try:
    from transformers import CLIPProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_clip_image_preprocessing_parity():
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    ggmlc_pre = from_huggingface_image_processor(processor.image_processor)

    # Create test image
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (256, 384, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)

    # 1. HuggingFace reference
    hf_out = processor(images=image, return_tensors="pt")["pixel_values"].numpy()

    # 2. ggmlc VisionPreprocessor
    ggmlc_out = ggmlc_pre.preprocess_image(image)

    # Check shape and parity
    assert hf_out.shape == ggmlc_out.shape
    diff = np.abs(hf_out - ggmlc_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Bicubic interpolation differences across libraries typically <= 0.05
    assert max_diff < 0.05, f"Image preprocessing max diff too large: {max_diff}"
    assert mean_diff < 0.005, f"Image preprocessing mean diff too large: {mean_diff}"


def test_cpp_native_image_preprocessor_parity():
    from ggmlc._runtime import NativeImagePreprocessor

    # Create dummy PNG in memory
    img = Image.new("RGB", (300, 200), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Call native C++ preprocessor
    res = NativeImagePreprocessor.preprocess_memory(
        png_bytes,
        224,
        224,
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
        True,
    )

    assert res.channels == 3
    assert res.height == 224
    assert res.width == 224
    assert len(res.data) == 3 * 224 * 224
