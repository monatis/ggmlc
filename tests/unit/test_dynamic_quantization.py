"""Unit tests for role-based dynamic quantization and Unsloth-style policies."""

import numpy as np
import pytest

from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph, GGMLTensorDef
from ggmlc.dialect.ggml.ops import GGMLType
from ggmlc.ir.dtype import DType
from ggmlc.ir.shape import StaticDim
from ggmlc.ir.tensor import StorageClass
from ggmlc.quantization import (
    QuantizationPolicy,
    TensorRole,
    classify_tensor_role,
    get_quantization_policy,
    quantize_graph_parameters,
)


def test_classify_tensor_roles():
    # 1D tensors
    assert classify_tensor_role("model.layers.0.input_layernorm.weight", (1024,)) == TensorRole.NORMALIZATION_1D
    assert classify_tensor_role("norm.bias", (512,)) == TensorRole.NORMALIZATION_1D
    assert classify_tensor_role("gamma", (1,)) == TensorRole.NORMALIZATION_1D

    # Embeddings
    assert classify_tensor_role("model.embed_tokens.weight", (151936, 1024)) == TensorRole.EMBEDDING
    assert classify_tensor_role("embedding_matrix", (151936, 16)) == TensorRole.EMBEDDING
    assert classify_tensor_role("transformer.wte.weight", (50257, 768)) == TensorRole.EMBEDDING

    # Output Heads
    assert classify_tensor_role("lm_head.weight", (151936, 1024)) == TensorRole.OUTPUT_HEAD
    assert classify_tensor_role("output_linear.weight", (151936, 1024)) == TensorRole.OUTPUT_HEAD

    # Attention Value & Out
    assert classify_tensor_role("model.layers.0.self_attn.v_proj.weight", (1024, 1024)) == TensorRole.ATTN_VALUE
    assert classify_tensor_role("model.layers.0.self_attn.o_proj.weight", (1024, 1024)) == TensorRole.ATTN_OUT

    # Attention Q & K
    assert classify_tensor_role("model.layers.0.self_attn.q_proj.weight", (1024, 1024)) == TensorRole.ATTN_QK
    assert classify_tensor_role("model.layers.0.self_attn.k_proj.weight", (1024, 1024)) == TensorRole.ATTN_QK

    # FFN Down & Gate/Up
    assert classify_tensor_role("model.layers.0.mlp.down_proj.weight", (1024, 4096)) == TensorRole.FFN_DOWN
    assert classify_tensor_role("model.layers.0.mlp.gate_proj.weight", (4096, 1024)) == TensorRole.FFN_GATE_UP
    assert classify_tensor_role("model.layers.0.mlp.up_proj.weight", (4096, 1024)) == TensorRole.FFN_GATE_UP


def test_quantization_policy_resolution():
    ud_policy = get_quantization_policy("unsloth_dynamic")
    assert ud_policy.name == "unsloth_dynamic"

    # Embeddings -> F16
    assert ud_policy.resolve_dtype("model.embed_tokens.weight", TensorRole.EMBEDDING, (151936, 1024)) == DType.F16

    # Output head -> Q8_0
    assert ud_policy.resolve_dtype("lm_head.weight", TensorRole.OUTPUT_HEAD, (151936, 1024)) == DType.Q8_0

    # Attn Value & FFN Down -> Q8_0
    assert ud_policy.resolve_dtype("v_proj.weight", TensorRole.ATTN_VALUE, (1024, 1024)) == DType.Q8_0
    assert ud_policy.resolve_dtype("down_proj.weight", TensorRole.FFN_DOWN, (1024, 4096)) == DType.Q8_0

    # Attn QK & FFN Gate/Up -> Q4_0
    assert ud_policy.resolve_dtype("q_proj.weight", TensorRole.ATTN_QK, (1024, 1024)) == DType.Q4_0
    assert ud_policy.resolve_dtype("gate_proj.weight", TensorRole.FFN_GATE_UP, (4096, 1024)) == DType.Q4_0

    # 1D Normalization -> F32
    assert ud_policy.resolve_dtype("norm.weight", TensorRole.NORMALIZATION_1D, (1024,)) == DType.F32


def test_dynamic_quantize_graph():
    # Build synthetic GGMLExecutionGraph with various tensor roles
    graph = GGMLExecutionGraph(name="test_transformer")

    # 1. 1D Norm (F32)
    norm_data = np.ones((64,), dtype=np.float32)
    graph.tensors[0] = GGMLTensorDef(
        id=0, name="input_layernorm.weight", ggml_type=GGMLType.GGML_TYPE_F32,
        ne=(StaticDim(64), StaticDim(1), StaticDim(1), StaticDim(1)),
        storage=StorageClass.PARAMETER, data=norm_data
    )

    # 2. Embedding Matrix (F16 in Unsloth Dynamic)
    emb_data = np.random.randn(100, 64).astype(np.float32)
    graph.tensors[1] = GGMLTensorDef(
        id=1, name="model.embed_tokens.weight", ggml_type=GGMLType.GGML_TYPE_F32,
        ne=(StaticDim(64), StaticDim(100), StaticDim(1), StaticDim(1)),
        storage=StorageClass.PARAMETER, data=emb_data
    )

    # 3. Attention V-Proj (Q8_0 in Unsloth Dynamic)
    v_data = np.random.randn(64, 64).astype(np.float32)
    graph.tensors[2] = GGMLTensorDef(
        id=2, name="self_attn.v_proj.weight", ggml_type=GGMLType.GGML_TYPE_F32,
        ne=(StaticDim(64), StaticDim(64), StaticDim(1), StaticDim(1)),
        storage=StorageClass.PARAMETER, data=v_data
    )

    # 4. Attention Q-Proj (Q4_0 in Unsloth Dynamic)
    q_data = np.random.randn(64, 64).astype(np.float32)
    graph.tensors[3] = GGMLTensorDef(
        id=3, name="self_attn.q_proj.weight", ggml_type=GGMLType.GGML_TYPE_F32,
        ne=(StaticDim(64), StaticDim(64), StaticDim(1), StaticDim(1)),
        storage=StorageClass.PARAMETER, data=q_data
    )

    # 5. Output Linear Head (Q8_0 in Unsloth Dynamic)
    head_data = np.random.randn(100, 64).astype(np.float32)
    graph.tensors[4] = GGMLTensorDef(
        id=4, name="lm_head.weight", ggml_type=GGMLType.GGML_TYPE_F32,
        ne=(StaticDim(64), StaticDim(100), StaticDim(1), StaticDim(1)),
        storage=StorageClass.PARAMETER, data=head_data
    )

    # Quantize under unsloth_dynamic policy
    q_graph, stats = quantize_graph_parameters(graph, target_dtype="unsloth_dynamic")

    assert stats["policy"] == "unsloth_dynamic"
    assert stats["tensors_quantized"] == 4  # 4 weights quantized, 1 norm kept in F32

    # Check resulting GGML types
    assert q_graph.tensors[0].ggml_type == GGMLType.GGML_TYPE_F32  # 1D norm
    assert q_graph.tensors[1].ggml_type == GGMLType.GGML_TYPE_F16  # Embedding
    assert q_graph.tensors[2].ggml_type == GGMLType.GGML_TYPE_Q8_0 # V-proj
    assert q_graph.tensors[3].ggml_type == GGMLType.GGML_TYPE_Q4_0 # Q-proj
    assert q_graph.tensors[4].ggml_type == GGMLType.GGML_TYPE_Q8_0 # LM head

    print("\nDynamic Quantization Stats:", stats)
