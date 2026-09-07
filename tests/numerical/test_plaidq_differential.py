"""Differential numerical parity testing for PlaidQ continuous diffusion model vs PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy

plaidq_dir = Path(__file__).resolve().parent.parent.parent / "scratch" / "plaidq"
if str(plaidq_dir) not in sys.path:
    sys.path.insert(0, str(plaidq_dir))

import plaidq.models_plaid_q as mpq
import plaidq.qwen3_trunk as qt
import plaidq.schedule as ps


def _patch_plaidq_for_export():
    import contextlib

    torch.amp.autocast = lambda *args, **kwargs: contextlib.nullcontext()

    def custom_rotary_forward(self, seq_len: int, device, dtype, position_ids=None):
        if position_ids is not None:
            t = position_ids.to(device=device, dtype=torch.float32)
            freqs = t.unsqueeze(-1) * self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[:, :, None, :].to(dtype), emb.sin()[:, :, None, :].to(dtype)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t.unsqueeze(-1) * self.inv_freq.to(device).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :].to(dtype)
        sin = emb.sin()[None, :, None, :].to(dtype)
        return cos, sin

    qt.Qwen3Rotary.forward = custom_rotary_forward

    def custom_forward(self, x, cos, sin, cu_seqlens=None, cond=None, is_causal=False):
        b, s = x.shape[0], x.shape[1]
        shift_a, scale_a, shift_m, scale_m = self._cond_chunks(x, cond)
        q, k, v = self._qkv(x, cos, sin, shift_a, scale_a)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        attn_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal, enable_gqa=True)
        attn = attn_t.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self._attn_out(x, attn, shift_m, scale_m)

    qt.Qwen3DiffusionBlock.forward = custom_forward


def test_plaidq_mini_numerical_parity():
    """Verify 2-layer Mini-PlaidQ forward pass numerical parity against PyTorch reference."""
    _patch_plaidq_for_export()

    embed_dim = 16
    vocab_size = 256
    seq_len = 32
    hidden_size = 64
    num_layers = 2
    n_heads = 4
    n_kv = 2
    head_dim = 16
    intermediate_size = 128

    config = qt.qwen3_config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    )

    model = mpq.PlaidQDiffusionModel(config, embed_dim=embed_dim, shift_head=True).eval()
    embedding_matrix = ps.EmbeddingMatrix(vocab_size, embed_dim)()

    # Test inputs
    batch_size = 1
    torch.manual_seed(42)
    np.random.seed(42)

    z = torch.randn((batch_size, seq_len, embed_dim), dtype=torch.float32)
    gamma = torch.tensor([1.5], dtype=torch.float32)
    x_selfcond = torch.zeros_like(z)

    class DenoiserWrapper(torch.nn.Module):
        def __init__(self, m, emb):
            super().__init__()
            self.m = m
            self.register_buffer("emb", emb)

        def forward(self, z, gamma, x_selfcond):
            logits, _ = self.m(
                z=z,
                gamma=gamma,
                embedding_matrix=self.emb,
                clean_prefix_mask=None,
                clean_prefix_embeddings=None,
                x_selfcond=x_selfcond,
                return_logits=True,
                return_reconst=True,
            )
            return logits

    wrapper = DenoiserWrapper(model, embedding_matrix).eval()

    # 1. Reference PyTorch computation
    with torch.no_grad():
        ref_logits = wrapper(z, gamma, x_selfcond)
        ref_np = ref_logits.detach().cpu().numpy()

    # 2. Export to Canonical IR
    exported = export_torch_model(
        wrapper,
        example_args=(z, gamma, x_selfcond),
        model_name="plaidq_mini_parity",
        optimize=False,
    )
    assert len(exported.main_graph.nodes) > 0

    # 3. Lower to GGML dialect
    ggml_graph = lower_to_ggml(exported.main_graph)
    assert len(ggml_graph.nodes) > 0

    # 4. Serialize to GGUF format bytes
    ser_bytes = serialize_ggml_graph(ggml_graph)
    assert len(ser_bytes) > 0

    # 5. Execute in Generic C++ Runtime via ModelRunner
    inputs_dict = {
        "z": z.numpy(),
        "gamma": gamma.numpy(),
        "x_selfcond": x_selfcond.numpy(),
    }

    runner = ModelRunner(ser_bytes, device="cpu")
    out = runner(inputs_dict)
    actual_raw = next(iter(out.values())) if isinstance(out, dict) else out
    actual_np = actual_raw.reshape(ref_np.shape)

    # 6. Verify numerical parity
    from ggmlc.validation.numerical import cosine_similarity

    cos_sim = cosine_similarity(ref_np, actual_np)
    cmp = check_numerical_accuracy(ref_np, actual_np, atol=1e-3)
    assert cmp.passed, f"Numerical check failed: {cmp.message}, max_diff={cmp.max_abs_diff}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim}"
    print(
        f"\n[Differential Parity] max_abs_diff={cmp.max_abs_diff:.6e}, cosine_similarity={cos_sim:.6f}"
    )
