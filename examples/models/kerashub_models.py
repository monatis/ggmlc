"""KerasHub model loaders for modern Transformer NLP and SLM architectures on JAX backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Ensure JAX backend is configured prior to importing Keras / KerasHub
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras_hub
import numpy as np


def load_kerashub_bert(
    seq_len: int = 128,
    vocabulary_size: int = 30522,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads BERT Transformer backbone from KerasHub."""
    model = keras_hub.models.BertBackbone(
        vocabulary_size=vocabulary_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        max_sequence_length=seq_len,
    )
    token_ids = np.random.randint(0, vocabulary_size - 1, size=(1, seq_len), dtype=np.int32)
    padding_mask = np.ones((1, seq_len), dtype=np.int32)
    segment_ids = np.zeros((1, seq_len), dtype=np.int32)

    def forward(t_ids: np.ndarray, p_mask: np.ndarray, s_ids: np.ndarray) -> np.ndarray:
        out = model({"token_ids": t_ids, "padding_mask": p_mask, "segment_ids": s_ids})
        if isinstance(out, dict):
            return out["sequence_output"]
        return out

    return (
        forward,
        (token_ids, padding_mask, segment_ids),
        ["token_ids", "padding_mask", "segment_ids"],
        "jax",
    )


def load_kerashub_distilbert(
    seq_len: int = 128,
    vocabulary_size: int = 30522,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads DistilBERT Transformer backbone from KerasHub."""
    model = keras_hub.models.DistilBertBackbone(
        vocabulary_size=vocabulary_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        max_sequence_length=seq_len,
    )
    token_ids = np.random.randint(0, vocabulary_size - 1, size=(1, seq_len), dtype=np.int32)
    padding_mask = np.ones((1, seq_len), dtype=np.int32)

    def forward(t_ids: np.ndarray, p_mask: np.ndarray) -> np.ndarray:
        out = model({"token_ids": t_ids, "padding_mask": p_mask})
        if isinstance(out, dict):
            return out["sequence_output"]
        return out

    return forward, (token_ids, padding_mask), ["token_ids", "padding_mask"], "jax"


def load_kerashub_gpt2(
    seq_len: int = 128,
    vocabulary_size: int = 50257,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads GPT-2 Autoregressive Causal Transformer backbone from KerasHub."""
    model = keras_hub.models.GPT2Backbone(
        vocabulary_size=vocabulary_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        max_sequence_length=seq_len,
    )
    token_ids = np.random.randint(0, vocabulary_size - 1, size=(1, seq_len), dtype=np.int32)
    padding_mask = np.ones((1, seq_len), dtype=np.int32)

    def forward(t_ids: np.ndarray, p_mask: np.ndarray) -> np.ndarray:
        out = model({"token_ids": t_ids, "padding_mask": p_mask})
        if isinstance(out, dict):
            return out["sequence_output"]
        return out

    return forward, (token_ids, padding_mask), ["token_ids", "padding_mask"], "jax"
