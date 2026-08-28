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


class FlaxResNetBlock(nn.Module):
    """Residual convolutional block with LayerNorm and ReLU."""

    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.Conv(
            self.channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv1"
        )(x)
        y = nn.LayerNorm(name="norm1")(y)
        y = nn.relu(y)
        y = nn.Conv(
            self.channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv2"
        )(y)
        y = nn.LayerNorm(name="norm2")(y)
        return nn.relu(residual + y)


class FlaxResNet(nn.Module):
    """Flax Residual Convolutional Network for image classification."""

    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="init_conv")(x)
        x = nn.relu(x)
        x = FlaxResNetBlock(32, name="b1")(x)
        x = FlaxResNetBlock(32, name="b2")(x)
        B = x.shape[0]
        x = jnp.reshape(x, (B, -1))
        x = nn.Dense(self.num_classes, name="fc")(x)
        return x


class FlaxVisionTransformer(nn.Module):
    """Vision Transformer in Flax with patch tokenization and Self-Attention."""

    patch_size: int = 4
    embed_dim: int = 64
    num_heads: int = 4
    mlp_dim: int = 128
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jnp.reshape(
            x,
            (B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C),
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(
            x,
            (
                B,
                (H // self.patch_size) * (W // self.patch_size),
                self.patch_size * self.patch_size * C,
            ),
        )
        x = nn.Dense(self.embed_dim, name="patch_proj")(x)
        norm1 = nn.LayerNorm(name="ln1")(x)
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.embed_dim, name="attn")(
            norm1
        )
        x = x + attn
        norm2 = nn.LayerNorm(name="ln2")(x)
        mlp = nn.Dense(self.mlp_dim, name="mlp1")(norm2)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.embed_dim, name="mlp2")(mlp)
        x = x + mlp
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.num_classes, name="head")(x)
        return x
