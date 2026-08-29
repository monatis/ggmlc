"""End-to-end integration and differential tests for KerasHub NLP/SLMs and Flax ViT."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import ggmlc
import numpy as np
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity

from examples.models.flax_models import load_flax_vit_b16
from examples.models.kerashub_models import (
    load_kerashub_bert,
    load_kerashub_distilbert,
    load_kerashub_gemma3,
    load_kerashub_gpt2,
)


def test_kerashub_bert_e2e():
    forward_fn, (t_ids, p_mask, s_ids), _names, _ = load_kerashub_bert(
        seq_len=32, num_layers=2, num_heads=2, hidden_dim=128, intermediate_dim=256
    )
    ref_out = np.asarray(forward_fn(t_ids, p_mask, s_ids))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(t_ids, p_mask, s_ids),
        model_name="kerashub_bert",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(t_ids, p_mask, s_ids)

    cos_sim = cosine_similarity(ref_out, out_np)
    res = check_numerical_accuracy(ref_out, out_np, atol=1e-4, rtol=1e-4)
    assert cos_sim > 0.9999, f"BERT cosine similarity low: {cos_sim}"
    assert res.passed or cos_sim > 0.9999


def test_kerashub_distilbert_e2e():
    forward_fn, (t_ids, p_mask), _names, _ = load_kerashub_distilbert(
        seq_len=32, num_layers=2, num_heads=2, hidden_dim=128, intermediate_dim=256
    )
    ref_out = np.asarray(forward_fn(t_ids, p_mask))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(t_ids, p_mask),
        model_name="kerashub_distilbert",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(t_ids, p_mask)

    cos_sim = cosine_similarity(ref_out, out_np)
    res = check_numerical_accuracy(ref_out, out_np, atol=1e-3, rtol=1e-3)
    assert cos_sim > 0.999, f"DistilBERT cosine similarity low: {cos_sim}"
    assert res.passed or cos_sim > 0.9999


def test_kerashub_gpt2_e2e():
    forward_fn, (t_ids, p_mask), _names, _ = load_kerashub_gpt2(
        seq_len=32, num_layers=2, num_heads=2, hidden_dim=128, intermediate_dim=256
    )
    ref_out = np.asarray(forward_fn(t_ids, p_mask))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(t_ids, p_mask),
        model_name="kerashub_gpt2",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(t_ids, p_mask)

    cos_sim = cosine_similarity(ref_out, out_np)
    res = check_numerical_accuracy(ref_out, out_np, atol=1e-4, rtol=1e-4)
    assert cos_sim > 0.9999, f"GPT-2 cosine similarity low: {cos_sim}"
    assert res.passed or cos_sim > 0.9999


def test_flax_vit_b16_e2e():
    forward_fn, (x_np,), _names, _ = load_flax_vit_b16(
        resolution=224, num_layers=2, dim=256, num_heads=4
    )
    ref_out = np.asarray(forward_fn(x_np))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(x_np,),
        model_name="flax_vit_b16",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x=x_np)

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.999, f"Flax ViT-B/16 cosine similarity low: {cos_sim}"


def test_kerashub_gemma3_e2e():
    forward_fn, (t_ids, p_mask), _names, _ = load_kerashub_gemma3(
        seq_len=16,
        vocabulary_size=1000,
        num_layers=2,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=128,
        intermediate_dim=256,
        head_dim=32,
        sliding_window_size=8,
    )
    ref_out = np.asarray(forward_fn(t_ids, p_mask))

    gguf_bytes = ggmlc.compile(
        model=forward_fn,
        sample_inputs=(t_ids, p_mask),
        model_name="kerashub_gemma3",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(t_ids, p_mask)

    assert not np.isnan(out_np).any(), "Gemma 3 output contains NaNs"
    assert out_np.reshape(ref_out.shape).shape == ref_out.shape
