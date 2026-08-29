"""End-to-end integration and differential tests for Keras 3 models (JAX backend)."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import ggmlc
import numpy as np
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity

from examples.models.keras_models import (
    load_keras_convnext_tiny,
    load_keras_densenet121,
    load_keras_efficientnet_b0,
    load_keras_mobilenet_v3_large,
    load_keras_mobilenet_v3_small,
    load_keras_resnet50,
)


def test_keras_mobilenet_v3_small_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_mobilenet_v3_small(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_mobilenet_v3_small",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    res = check_numerical_accuracy(ref_out, out_np, atol=1e-3, rtol=1e-3)
    assert cos_sim > 0.999, f"MobileNetV3-Small cosine similarity low: {cos_sim}"
    assert res.passed or cos_sim > 0.9999


def test_keras_mobilenet_v3_large_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_mobilenet_v3_large(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_mobilenet_v3_large",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"MobileNetV3-Large cosine similarity low: {cos_sim}"


def test_keras_resnet50_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_resnet50(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_resnet50",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"ResNet50 cosine similarity low: {cos_sim}"


def test_keras_densenet121_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_densenet121(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_densenet121",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.99, f"DenseNet121 cosine similarity low: {cos_sim}"


def test_keras_convnext_tiny_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_convnext_tiny(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_convnext_tiny",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"ConvNeXtTiny cosine similarity low: {cos_sim}"


def test_keras_efficientnet_b0_e2e():
    forward_fn, (x_np,), _names, _ = load_keras_efficientnet_b0(resolution=224)
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="keras_efficientnet_b0",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"EfficientNetB0 cosine similarity low: {cos_sim}"
