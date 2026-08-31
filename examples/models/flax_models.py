"""Flax neural network model definitions for compilation and benchmarking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

try:
    if (
        hasattr(jax, "extend")
        and hasattr(jax.extend, "core")
        and not hasattr(jax.core, "get_opaque_trace_state")
    ):
        jax.core.get_opaque_trace_state = getattr(jax.extend.core, "get_opaque_trace_state", None)
except (AttributeError, ImportError):
    pass

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


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


class FlaxConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise 7x7 conv and inverted bottleneck."""

    dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.Conv(
            self.dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.dim,
            name="dwconv",
        )(x)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(self.dim * 4, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, name="pwconv2")(x)
        return residual + x


class FlaxConvNeXt(nn.Module):
    """Flax ConvNeXt vision architecture with stem, inverted blocks, and global pooling."""

    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, kernel_size=(4, 4), strides=(4, 4), padding="VALID", name="stem")(x)
        x = nn.LayerNorm(name="stem_norm")(x)
        x = FlaxConvNeXtBlock(32, name="block1")(x)
        x = FlaxConvNeXtBlock(32, name="block2")(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.LayerNorm(name="head_norm")(x)
        x = nn.Dense(self.num_classes, name="head")(x)
        return x


class FlaxCausalLM(nn.Module):
    """Causal Language Model in Flax with Embedding, Causal Masking, LayerNorm, and Dense."""

    vocab_size: int = 100
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="wte")(token_ids)
        mask = nn.make_causal_mask(token_ids)
        for i in range(self.num_layers):
            norm1 = nn.LayerNorm(name=f"ln1_{i}")(x)
            attn = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                name=f"attn_{i}",
            )(norm1, mask=mask)
            x = x + attn
            norm2 = nn.LayerNorm(name=f"ln2_{i}")(x)
            mlp = nn.Dense(self.embed_dim * 2, name=f"mlp_fc1_{i}")(norm2)
            mlp = nn.gelu(mlp)
            mlp = nn.Dense(self.embed_dim, name=f"mlp_fc2_{i}")(mlp)
            x = x + mlp
        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)
        return logits


class FlaxSelfAttention(nn.Module):
    """Multi-Head Self-Attention for Vision Transformer in Flax."""

    num_heads: int = 12
    qkv_features: int = 768

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, S, C = x.shape
        head_dim = self.qkv_features // self.num_heads
        qkv = nn.Dense(self.qkv_features * 3, use_bias=True, name="qkv")(x)
        qkv = qkv.reshape((B, S, 3, self.num_heads, head_dim))
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        scale = 1.0 / np.sqrt(head_dim)
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((B, S, C))
        return nn.Dense(C, use_bias=True, name="proj")(out)


class FlaxViTBlock(nn.Module):
    """Transformer Encoder Block in Flax."""

    dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm(name="ln1")(x)
        h = FlaxSelfAttention(num_heads=self.num_heads, qkv_features=self.dim, name="attn")(h)
        x = x + h
        h = nn.LayerNorm(name="ln2")(x)
        mlp_hidden = int(self.dim * self.mlp_ratio)
        h = nn.Dense(mlp_hidden, name="mlp_dense1")(h)
        h = jax.nn.gelu(h, approximate=False)
        h = nn.Dense(self.dim, name="mlp_dense2")(h)
        return x + h


class FlaxViTB16(nn.Module):
    """Full Vision Transformer ViT-B/16 (224x224, 12 layers, 768 dim, 86M params) in Flax."""

    num_classes: int = 1000
    num_layers: int = 12
    dim: int = 768
    num_heads: int = 12
    patch_size: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _C = x.shape
        P = self.patch_size
        patches = nn.Conv(
            self.dim, kernel_size=(P, P), strides=(P, P), padding="VALID", name="patch_embed"
        )(x)
        num_patches = (H // P) * (W // P)
        patches = patches.reshape((B, num_patches, self.dim))

        cls_token = self.param("cls", nn.initializers.zeros, (1, 1, self.dim))
        cls_tokens = jnp.tile(cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, patches], axis=1)

        pos_embed = self.param(
            "pos_embed", nn.initializers.normal(0.02), (1, num_patches + 1, self.dim)
        )
        x = x + pos_embed

        for i in range(self.num_layers):
            x = FlaxViTBlock(dim=self.dim, num_heads=self.num_heads, name=f"block_{i}")(x)

        x = nn.LayerNorm(name="norm")(x)
        cls_out = x[:, 0]
        return nn.Dense(self.num_classes, name="head")(cls_out)


def load_flax_vit_b16(
    resolution: int = 224,
    num_layers: int = 12,
    dim: int = 768,
    num_heads: int = 12,
) -> tuple[Callable[..., Any], tuple[np.ndarray, ...], list[str], str]:
    """Loads full production Vision Transformer ViT-B/16 in Flax Linen."""
    model = FlaxViTB16(num_layers=num_layers, dim=dim, num_heads=num_heads)
    x = np.random.randn(1, resolution, resolution, 3).astype(np.float32)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.asarray(x))

    def forward(inp: np.ndarray) -> np.ndarray:
        return model.apply(params, inp)

    return forward, (x,), ["x"], "jax"
