"""Tensor role classification for mixed-precision and dynamic quantization."""

from __future__ import annotations

import re
from enum import Enum, unique


@unique
class TensorRole(Enum):
    """Semantic role of a tensor in neural network architectures."""

    EMBEDDING = "embedding"              # Token embeddings, positional embeddings, codebooks
    OUTPUT_HEAD = "output_head"          # LM head, classification head, final linear projection
    ATTN_VALUE = "attn_value"            # Attention Value (v_proj, value, qkv)
    ATTN_OUT = "attn_out"                # Attention Output projection (o_proj, out_proj, wo)
    ATTN_QK = "attn_qk"                  # Attention Query & Key (q_proj, k_proj, query, key)
    FFN_DOWN = "ffn_down"                # Feed-Forward down-projection (down_proj, w2, fc2)
    FFN_GATE_UP = "ffn_gate_up"          # Feed-Forward gate & up projections (gate_proj, up_proj, w1, w3, fc1)
    NORMALIZATION_1D = "normalization_1d"# RMSNorm, LayerNorm weights, biases, 1D constants
    GENERIC_WEIGHT = "generic_weight"    # Other 2D/3D weights (convolutions, adapters, etc.)


def classify_tensor_role(name: str, dims: list[int] | tuple[int, ...] | None = None) -> TensorRole:
    """Classifies a tensor's architectural role based on its name and dimensionality.

    Args:
        name: Name of the tensor (from PyTorch, HuggingFace, Flax, Keras, or GGUF).
        dims: Dimension sizes of the tensor.

    Returns:
        TensorRole indicating the functional role of the tensor.
    """
    # 1. Check for 1D tensors (Strict 1D F32 Rule)
    if dims is not None:
        non_unit_dims = [d for d in dims if d > 1]
        if len(non_unit_dims) <= 1:
            return TensorRole.NORMALIZATION_1D

    name_lower = name.lower()

    # 1D normalization / bias patterns
    if any(k in name_lower for k in [
        "norm.weight", "norm.bias", "ln.weight", "ln.bias", "layernorm",
        "rmsnorm", "_norm", "gamma_embed", "timestep_mlp", "inv_freq", "bias"
    ]):
        if dims is not None and len([d for d in dims if d > 1]) <= 1:
            return TensorRole.NORMALIZATION_1D

    # Embeddings & Codebooks
    if any(k in name_lower for k in [
        "token_embd", "embed_tokens", "wte", "word_embeddings", "embedding_matrix",
        "emb.weight", "emb", "shared_embedding", "codebook"
    ]):
        return TensorRole.EMBEDDING

    # Output / LM Head
    if any(k in name_lower for k in [
        "output.weight", "output_linear", "lm_head", "classifier.weight",
        "head.weight", "shift_head", "final_linear"
    ]):
        return TensorRole.OUTPUT_HEAD

    # Attention Query / Key
    if any(k in name_lower for k in [
        "q_proj", "k_proj", "attn_q", "attn_k", "query.weight", "key.weight",
        "w_q", "w_k", "wq", "wk", "attention.q", "attention.k"
    ]):
        return TensorRole.ATTN_QK

    # Attention Value / QKV
    if any(k in name_lower for k in [
        "v_proj", "attn_v", "value.weight", "w_v", "wv", "attention.v", "attn.v", "qkv"
    ]):
        return TensorRole.ATTN_VALUE

    # Attention Output
    if any(k in name_lower for k in [
        "o_proj", "out_proj", "attn_output", "wo", "w_o", "attention.o", "attn.out"
    ]):
        return TensorRole.ATTN_OUT

    # FFN Down Projection
    if any(k in name_lower for k in [
        "down_proj", "ffn_down", "w2", "w_2", "mlp.down", "ffn.down", "fc2", "dense_4h_to_h"
    ]):
        return TensorRole.FFN_DOWN

    # FFN Gate / Up Projection
    if any(k in name_lower for k in [
        "gate_proj", "up_proj", "ffn_gate", "ffn_up", "w1", "w3", "w_1", "w_3",
        "mlp.gate", "mlp.up", "ffn.gate", "ffn.up", "fc1", "dense_h_to_4h", "gate_up_proj"
    ]):
        return TensorRole.FFN_GATE_UP

    return TensorRole.GENERIC_WEIGHT
