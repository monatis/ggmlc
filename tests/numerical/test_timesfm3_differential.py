"""Differential numerical parity testing for Google TimesFM 3.0 foundation forecasting model vs PyTorch."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ggmlc.runtime.runner import ModelRunner

timesfm_available = False
try:
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

    timesfm_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not timesfm_available, reason="timesfm3 package not installed")


class TimesFM3CleanTrunk(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.pre_transformer_resblock = base_model.pre_transformer_resblock
        self.layers = base_model.transformer_stack.layers
        self.output_head = base_model.output_head
        self.num_layers = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_transformer_resblock(x)
        b, v, n, d = h.shape

        for layer in self.layers:
            residual = h
            h_seq = layer.pre_seq_attn_ln(h)
            h_seq_flat = h_seq.view(b * v, n, d)

            q = layer.seq_attn.query_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )
            k = layer.seq_attn.key_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )
            val = layer.seq_attn.value_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )

            pos = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(0)
            q = layer.seq_attn.rotary_position_embedding(q, pos)
            k = layer.seq_attn.rotary_position_embedding(k, pos)

            if layer.seq_attn.query_ln is not None:
                q = layer.seq_attn.query_ln(q)
            if layer.seq_attn.key_ln is not None:
                k = layer.seq_attn.key_ln(k)
            if layer.seq_attn.per_dim_scale is not None:
                q = layer.seq_attn.per_dim_scale(q)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            val = val.transpose(1, 2)

            scale = math.sqrt(layer.seq_attn.head_dim)
            attn_out = F.scaled_dot_product_attention(q, k, val, is_causal=True, scale=scale)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b * v, n, d)
            attn_out = layer.seq_attn.out_proj(attn_out).view(b, v, n, d)

            if layer.post_seq_attn_ln is not None:
                attn_out = layer.post_seq_attn_ln(attn_out)

            h = residual + attn_out

            if layer.var_attn is not None:
                residual = h
                h_var = layer.pre_var_attn_ln(h) if layer.pre_var_attn_ln is not None else h
                h_var_flat = h_var.permute(0, 2, 1, 3).contiguous().view(b * n, v, d)

                vq = layer.var_attn.query_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )
                vk = layer.var_attn.key_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )
                vval = layer.var_attn.value_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )

                if layer.var_attn.query_ln is not None:
                    vq = layer.var_attn.query_ln(vq)
                if layer.var_attn.key_ln is not None:
                    vk = layer.var_attn.key_ln(vk)
                if layer.var_attn.per_dim_scale is not None:
                    vq = layer.var_attn.per_dim_scale(vq)

                vq = vq.transpose(1, 2)
                vk = vk.transpose(1, 2)
                vval = vval.transpose(1, 2)

                var_scale = math.sqrt(layer.var_attn.head_dim)
                var_out = F.scaled_dot_product_attention(
                    vq, vk, vval, is_causal=False, scale=var_scale
                )
                var_out = var_out.transpose(1, 2).contiguous().view(b * n, v, d)
                var_out = (
                    layer.var_attn.out_proj(var_out)
                    .view(b, n, v, d)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

                if layer.post_var_attn_ln is not None:
                    var_out = layer.post_var_attn_ln(var_out)

                h = residual + var_out

            residual = h
            h_ff = layer.pre_ff_ln(h)
            mlp_out = layer.ff1(F.relu(layer.ff0(h_ff)))
            if layer.post_ff_ln is not None:
                mlp_out = layer.post_ff_ln(mlp_out)
            h = residual + mlp_out

        logits = self.output_head(h)
        return logits


@pytest.fixture(scope="module")
def timesfm_base_models():
    os.environ["HF_HUB_OFFLINE"] = "1"
    forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    orig_model = forecaster.model.eval()
    clean_trunk = TimesFM3CleanTrunk(orig_model).eval()
    return orig_model, clean_trunk


def test_timesfm3_clean_trunk_parity(timesfm_base_models):
    orig_model, clean_trunk = timesfm_base_models
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1, 16, 192, dtype=torch.float32)

    with torch.no_grad():
        patch_mask = torch.zeros(1, 1, 16, dtype=torch.bool)
        orig_res = orig_model.pre_transformer_resblock(dummy_input)
        orig_out, _, _ = orig_model.transformer_stack(orig_res, patch_mask)
        orig_logits = orig_model.output_head(orig_out)
        clean_logits = clean_trunk(dummy_input)

    max_diff = (orig_logits - clean_logits).abs().max().item()
    cos_sim = F.cosine_similarity(orig_logits.flatten(), clean_logits.flatten(), dim=0).item()

    assert max_diff < 1e-4, f"Trunk mismatch: max_diff={max_diff}"
    assert cos_sim > 0.999999, f"Cosine similarity too low: {cos_sim}"


def test_timesfm3_ggml_f16_parity(timesfm_base_models):
    _, clean_trunk = timesfm_base_models
    gguf_path = Path("scratch/timesfm3_f16.gguf")
    if not gguf_path.exists():
        pytest.skip("scratch/timesfm3_f16.gguf not found")

    runner = ModelRunner(str(gguf_path), device="cpu", n_threads=4)

    for batch_size in [1, 2, 4]:
        for n_patches in [4, 8, 16]:
            torch.manual_seed(42 + batch_size * 100 + n_patches)
            x = torch.randn(batch_size, 1, n_patches, 192, dtype=torch.float32)

            with torch.no_grad():
                py_logits = clean_trunk(x).numpy()

            ggml_res = runner(x.numpy())
            ggml_arr = (
                ggml_res[0]
                if isinstance(ggml_res, (list, tuple))
                else (next(iter(ggml_res.values())) if isinstance(ggml_res, dict) else ggml_res)
            )

            cos_sim = np.dot(py_logits.flatten(), ggml_arr.flatten()) / (
                np.linalg.norm(py_logits.flatten()) * np.linalg.norm(ggml_arr.flatten())
            )
            assert cos_sim > 0.9999, (
                f"Low cosine similarity at B={batch_size}, N={n_patches}: {cos_sim}"
            )


def test_timesfm3_domain_math():
    # 1. Linear Detrending Test
    t = np.arange(100, dtype=np.float32)
    slope_true = 2.5
    intercept_true = 10.0
    noise = np.random.randn(100).astype(np.float32) * 0.1
    series = intercept_true + slope_true * t + noise

    # Compute regression manually
    n = float(len(t))
    sum_t = float(np.sum(t))
    sum_y = float(np.sum(series))
    sum_tt = float(np.sum(t * t))
    sum_ty = float(np.sum(t * series))
    slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
    intercept = (sum_y - slope * sum_t) / n

    assert abs(slope - slope_true) < 0.05
    assert abs(intercept - intercept_true) < 0.5

    # 2. RevIN normalization & invertibility test
    mu = float(np.mean(series))
    sigma = float(np.std(series))
    normed = (series - mu) / sigma
    reconstructed = normed * sigma + mu
    max_recon_diff = np.max(np.abs(series - reconstructed))
    assert max_recon_diff < 1e-4
