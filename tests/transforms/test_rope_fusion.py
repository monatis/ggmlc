"""Tests for RoPE operator pattern matching and fusion pass."""

from __future__ import annotations

import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.op import OpCode
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.transforms.fusion import FusionOptions, fuse_operations
from ggmlc.validation.numerical import cosine_similarity


def test_rope_pattern_fusion_synthetic():
    """Verify that synthetic unrolled RoPE subgraph is pattern-matched and fused."""
    head_dim = 64
    n_heads = 4
    seq_len = 8
    freq_base = 100000.0

    # Build unrolled PyTorch RoPE module
    class SyntheticRoPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inv_freq = 1.0 / (
                freq_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        def forward(self, x):
            _, seq_len, _, _ = x.shape
            pos = torch.arange(0, seq_len, dtype=torch.float32)
            freqs = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(1)
            sin = emb.sin().unsqueeze(0).unsqueeze(1)

            x_trans = x.transpose(1, 2)

            def rotate_half(t):
                x1 = t[..., : t.shape[-1] // 2]
                x2 = t[..., t.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            return (x_trans * cos) + (rotate_half(x_trans) * sin)

    mod = SyntheticRoPE().eval()
    inp = (torch.randn(1, seq_len, n_heads, head_dim, dtype=torch.float32),)

    with torch.no_grad():
        ref_out = mod(*inp).numpy()

    # Export (automatically runs default pipeline including OperatorFusionPass)
    exp = export_torch_model(mod, inp, model_name="synthetic_rope")
    g = exp.main_graph

    # Verify that RoPE is fused
    opcodes = [op.opcode for op in g.nodes]
    assert OpCode.ROPE in opcodes
    assert OpCode.NEG not in opcodes

    # Lower to GGML and execute
    ggml_g = lower_to_ggml(g)
    ggml_opcodes = [op.opcode for op in ggml_g.nodes]
    assert GGMLOpCode.GGML_OP_ROPE in ggml_opcodes

    ser = serialize_ggml_graph(ggml_g)
    runner = ModelRunner(ser, device="cpu", n_threads=1)
    act_out = runner(inp[0].numpy())

    sim = cosine_similarity(act_out, ref_out)
    diff = float(np.max(np.abs(act_out - ref_out)))
    assert sim > 0.99999, f"Cosine similarity too low: {sim}"
    assert diff < 1e-4, f"Max diff too high: {diff}"


def test_rope_fusion_toggle_disabled():
    """Verify that setting enable_rope=False leaves the unrolled ops intact."""
    head_dim = 64
    n_heads = 4
    seq_len = 8
    freq_base = 100000.0

    class SyntheticRoPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inv_freq = 1.0 / (
                freq_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        def forward(self, x):
            _, seq_len, _, _ = x.shape
            pos = torch.arange(0, seq_len, dtype=torch.float32)
            freqs = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(1)
            sin = emb.sin().unsqueeze(0).unsqueeze(1)

            x_trans = x.transpose(1, 2)

            def rotate_half(t):
                x1 = t[..., : t.shape[-1] // 2]
                x2 = t[..., t.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            return (x_trans * cos) + (rotate_half(x_trans) * sin)

    mod = SyntheticRoPE().eval()
    inp = (torch.randn(1, seq_len, n_heads, head_dim, dtype=torch.float32),)

    from ggmlc.frontend.pytorch.importer import import_exported_program

    exported_program = torch.export.export(mod, inp)
    raw_graph = import_exported_program(exported_program)

    # When enable_rope=False
    fuse_operations(raw_graph, FusionOptions(enable_rope=False))
    raw_opcodes = [op.opcode for op in raw_graph.nodes]
    assert OpCode.ROPE not in raw_opcodes
    assert OpCode.NEG in raw_opcodes
