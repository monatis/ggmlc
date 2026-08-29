"""Keras 3 model loader definitions targeting the JAX backend for ggmlc."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Ensure JAX backend is configured prior to importing Keras
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np


def load_keras_resnet50(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full production ResNet50 vision model in Keras 3 with JAX backend."""
    model = keras.applications.ResNet50(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"


def load_keras_mobilenet_v3_small(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full MobileNetV3Small vision model in Keras 3 with JAX backend."""
    model = keras.applications.MobileNetV3Small(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"


def load_keras_mobilenet_v3_large(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full MobileNetV3Large vision model in Keras 3 with JAX backend."""
    model = keras.applications.MobileNetV3Large(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"


def load_keras_convnext_tiny(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full ConvNeXtTiny vision model in Keras 3 with JAX backend."""
    model = keras.applications.ConvNeXtTiny(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"


def load_keras_densenet121(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full DenseNet121 vision model in Keras 3 with JAX backend."""
    model = keras.applications.DenseNet121(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"


def load_keras_efficientnet_b0(
    resolution: int = 224,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full EfficientNetB0 vision model in Keras 3 with JAX backend."""
    model = keras.applications.EfficientNetB0(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)

    def forward(inp: np.ndarray) -> np.ndarray:
        return model(inp)

    return forward, (x,), ["x"], "jax"
