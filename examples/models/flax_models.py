"""Flax neural network model definitions for compilation and benchmarking."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class FlaxMLPClassifier(nn.Module):
    """Multi-Layer Perceptron Classifier with LayerNorm and GELU in Flax."""

    hidden_dim: int = 128
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.LayerNorm(name="ln1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes, name="fc3")(x)
        return x


class FlaxTransformerLayer(nn.Module):
    """Transformer Encoder Block in Flax with SelfAttention, LayerNorm, and MLP."""

    dim: int = 64
    num_heads: int = 4
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pre-LN Self-Attention
        norm1 = nn.LayerNorm(name="ln_1")(x)
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="self_attn",
        )(norm1)
        x = x + attn

        # Pre-LN Feed-Forward Network
        norm2 = nn.LayerNorm(name="ln_2")(x)
        mlp = nn.Dense(self.mlp_dim, name="mlp_dense1")(norm2)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.dim, name="mlp_dense2")(mlp)
        x = x + mlp
        return x


class FlaxFullTransformer(nn.Module):
    """Multi-layer Flax Transformer for sequence representations."""

    num_layers: int = 2
    dim: int = 64
    num_heads: int = 4
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.num_layers):
            x = FlaxTransformerLayer(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f"layer_{i}",
            )(x)
        x = nn.LayerNorm(name="final_ln")(x)
        return x
