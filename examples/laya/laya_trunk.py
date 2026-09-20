"""Exportable Laya DecisionModel trunk: ModernBERT + typed head, SDPA, no unpadding."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

MAX_LEN = 512
MAX_OPTS = 16
WINDOW = 64  # config.sliding_window; flash uses +1 inclusive → |i-j| <= 64


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, D]; cos/sin: [1, S, D] -> unsqueeze heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class LayaCleanTrunk(nn.Module):
    """TimesFM-style rewrite of Laya DecisionModel for torch.export / ggmlc.compile.

    Inputs (dynamic batch ``b`` and sequence ``s``, option axis static 16):
      input_ids      [B, S] int32
      attention_mask [B, S] float32   (1 = token, 0 = pad)
      marker_pos     [B, 16] int32    (index into flattened [B*S, H]; unused 0)
      marker_mask    [B, 16] float32  (1 = live option, 0 = pad)
      qtype          [B]     int32    (0=choice, 1=score, 2=noul)

    Host code (not this graph) must bake batch offsets into ``marker_pos``:
    ``marker_pos[b, k] = b * S + token_index``. CUDA binbcast cannot ADD I32.

    Outputs:
      logits         [B, 16]  float32   (padded slots ~ -1e4)
      act_logits     [B, 2]   float32
    """

    def __init__(self, agent: Any, max_len: int = MAX_LEN, max_opts: int = MAX_OPTS):
        super().__init__()
        model = agent.model
        enc = model.encoder
        cfg = enc.config
        d = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        head_dim = d // n_heads
        n_layers = cfg.num_hidden_layers
        eps = getattr(cfg, "norm_eps", 1e-5)

        self.max_len = int(max_len)
        self.max_opts = int(max_opts)
        # One extra RoPE/window slot so slicing [:s] is never a no-op at s=max_len.
        # torch.export would otherwise guard s != 512 and refuse Dim max=512.
        self.buf_len = int(max_len) + 1
        self.hidden = d
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.eps = eps

        self.tok_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_token_id)
        self.tok_emb.weight.data.copy_(enc.embeddings.tok_embeddings.weight.data)
        self.emb_norm_w = nn.Parameter(enc.embeddings.norm.weight.data.clone())

        self.final_norm_w = nn.Parameter(enc.final_norm.weight.data.clone())

        self.layer_types: list[str] = list(cfg.layer_types)
        self.qkv = nn.ModuleList()
        self.wo = nn.ModuleList()
        self.attn_norm_w = nn.ParameterList()
        self.attn_norm_is_id: list[bool] = []
        self.mlp_wi = nn.ModuleList()
        self.mlp_wo = nn.ModuleList()
        self.mlp_norm_w = nn.ParameterList()

        for i, layer in enumerate(enc.layers):
            qkv = nn.Linear(d, 3 * d, bias=False)
            qkv.weight.data.copy_(layer.attn.Wqkv.weight.data)
            wo = nn.Linear(d, d, bias=False)
            wo.weight.data.copy_(layer.attn.Wo.weight.data)
            self.qkv.append(qkv)
            self.wo.append(wo)

            is_id = isinstance(layer.attn_norm, nn.Identity)
            self.attn_norm_is_id.append(is_id)
            if is_id:
                self.attn_norm_w.append(nn.Parameter(torch.ones(d)))
            else:
                self.attn_norm_w.append(nn.Parameter(layer.attn_norm.weight.data.clone()))

            wi = nn.Linear(d, 2 * cfg.intermediate_size, bias=False)
            wi.weight.data.copy_(layer.mlp.Wi.weight.data)
            wo_m = nn.Linear(cfg.intermediate_size, d, bias=False)
            wo_m.weight.data.copy_(layer.mlp.Wo.weight.data)
            self.mlp_wi.append(wi)
            self.mlp_wo.append(wo_m)
            self.mlp_norm_w.append(nn.Parameter(layer.mlp_norm.weight.data.clone()))

        # Precomputed RoPE (fp32) for both thetas, positions 0..buf_len-1
        self.register_buffer(
            "cos_full",
            self._rope_cos_sin(enc.rotary_emb.full_attention_inv_freq, self.buf_len)[0],
            persistent=True,
        )
        self.register_buffer(
            "sin_full",
            self._rope_cos_sin(enc.rotary_emb.full_attention_inv_freq, self.buf_len)[1],
            persistent=True,
        )
        self.register_buffer(
            "cos_slide",
            self._rope_cos_sin(enc.rotary_emb.sliding_attention_inv_freq, self.buf_len)[0],
            persistent=True,
        )
        self.register_buffer(
            "sin_slide",
            self._rope_cos_sin(enc.rotary_emb.sliding_attention_inv_freq, self.buf_len)[1],
            persistent=True,
        )

        pos = torch.arange(self.buf_len)
        window = (pos[:, None] - pos[None, :]).abs() <= WINDOW
        slide_bias = torch.zeros(1, 1, self.buf_len, self.buf_len, dtype=torch.float32)
        slide_bias = slide_bias.masked_fill(~window, -1.0e4)
        self.register_buffer("slide_bias", slide_bias, persistent=True)

        # Decision head (2 x pre-LN TransformerEncoderLayer, ReLU FFN)
        self.type_emb = nn.Embedding(3, d)
        self.type_emb.weight.data.copy_(model.type_emb.weight.data)

        self.head_qkv = nn.ModuleList()
        self.head_wo = nn.ModuleList()
        self.head_wo_b = nn.ParameterList()
        self.head_qkv_b = nn.ParameterList()
        self.head_fc1 = nn.ModuleList()
        self.head_fc2 = nn.ModuleList()
        self.head_n1_w = nn.ParameterList()
        self.head_n1_b = nn.ParameterList()
        self.head_n2_w = nn.ParameterList()
        self.head_n2_b = nn.ParameterList()
        for layer in model.head.layers:
            attn = layer.self_attn
            qkv_l = nn.Linear(d, 3 * d, bias=True)
            qkv_l.weight.data.copy_(attn.in_proj_weight.data)
            qkv_l.bias.data.copy_(attn.in_proj_bias.data)
            wo_l = nn.Linear(d, d, bias=True)
            wo_l.weight.data.copy_(attn.out_proj.weight.data)
            wo_l.bias.data.copy_(attn.out_proj.bias.data)
            self.head_qkv.append(qkv_l)
            self.head_wo.append(wo_l)

            fc1 = nn.Linear(d, 4 * d, bias=True)
            fc1.weight.data.copy_(layer.linear1.weight.data)
            fc1.bias.data.copy_(layer.linear1.bias.data)
            fc2 = nn.Linear(4 * d, d, bias=True)
            fc2.weight.data.copy_(layer.linear2.weight.data)
            fc2.bias.data.copy_(layer.linear2.bias.data)
            self.head_fc1.append(fc1)
            self.head_fc2.append(fc2)
            self.head_n1_w.append(nn.Parameter(layer.norm1.weight.data.clone()))
            self.head_n1_b.append(nn.Parameter(layer.norm1.bias.data.clone()))
            self.head_n2_w.append(nn.Parameter(layer.norm2.weight.data.clone()))
            self.head_n2_b.append(nn.Parameter(layer.norm2.bias.data.clone()))

        self.scorer_ln = nn.LayerNorm(d)
        self.scorer_ln.weight.data.copy_(model.scorer[0].weight.data)
        self.scorer_ln.bias.data.copy_(model.scorer[0].bias.data)
        self.scorer_fc1 = nn.Linear(d, d)
        self.scorer_fc1.weight.data.copy_(model.scorer[1].weight.data)
        self.scorer_fc1.bias.data.copy_(model.scorer[1].bias.data)
        self.scorer_fc2 = nn.Linear(d, 1)
        self.scorer_fc2.weight.data.copy_(model.scorer[3].weight.data)
        self.scorer_fc2.bias.data.copy_(model.scorer[3].bias.data)

        self.act_fc1 = nn.Linear(d + 4, 256)
        self.act_fc1.weight.data.copy_(model.act_head[0].weight.data)
        self.act_fc1.bias.data.copy_(model.act_head[0].bias.data)
        self.act_fc2 = nn.Linear(256, 2)
        self.act_fc2.weight.data.copy_(model.act_head[2].weight.data)
        self.act_fc2.bias.data.copy_(model.act_head[2].bias.data)

        self.eval()

    @staticmethod
    def _rope_cos_sin(inv_freq: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.detach().cpu().float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)  # [1, S, D]

    def _ln_weight_only(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.hidden,), weight, None, self.eps)

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.hidden,), w, b, 1e-5)

    def _sdpa(
        self,
        x: torch.Tensor,
        qkv: nn.Linear,
        wo: nn.Linear,
        attn_bias: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        qkv_out = qkv(x)
        # Split Q/K/V on the last 4D axis (not a 5D view). The importer folds 5D
        # permutes by dropping the batch axis and that scrambles ModernBERT QKV.
        q, k, v = qkv_out.split(self.hidden, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        if cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False, scale=scale
        )
        # transpose+reshape (not permute+contiguous+view): the importer treats
        # aten.contiguous as identity, and fuse_sdpa_transpose matches TRANSPOSE(1,2).
        y = y.transpose(1, 2).reshape(b, s, self.hidden)
        return wo(y)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s = input_ids.shape
        h = self.tok_emb(input_ids.to(dtype=torch.long))
        h = self._ln_weight_only(h, self.emb_norm_w)

        pad_bias = (attention_mask.to(dtype=h.dtype) - 1.0) * 1.0e4
        pad_bias = pad_bias[:, None, None, :]  # [B,1,1,S]
        # [B,1,S,S] zeros from the live pad vector so S stays a dynamic symbol.
        full_zeros = attention_mask[:, None, :, None] * (attention_mask[:, None, None, :] * 0.0)
        full_bias = full_zeros.to(dtype=h.dtype) + pad_bias
        # Slice the static window buffer. Example export uses S<512 so torch.export
        # does not specialize s; runtime VIEW uses the concrete output shape.
        slide_bias = self.slide_bias[:, :, :s, :s] + pad_bias

        for i in range(len(self.qkv)):
            x = h if self.attn_norm_is_id[i] else self._ln_weight_only(h, self.attn_norm_w[i])
            if self.layer_types[i] == "full_attention":
                attn = self._sdpa(
                    x,
                    self.qkv[i],
                    self.wo[i],
                    full_bias,
                    self.cos_full[:, :s],
                    self.sin_full[:, :s],
                )
            else:
                attn = self._sdpa(
                    x,
                    self.qkv[i],
                    self.wo[i],
                    slide_bias,
                    self.cos_slide[:, :s],
                    self.sin_slide[:, :s],
                )
            h = h + attn
            m = self._ln_weight_only(h, self.mlp_norm_w[i])
            up, gate = self.mlp_wi[i](m).chunk(2, dim=-1)
            h = h + self.mlp_wo[i](F.gelu(up) * gate)

        h = self._ln_weight_only(h, self.final_norm_w)
        h = h + self.type_emb(qtype.to(dtype=torch.long))[:, None, :]

        for i in range(len(self.head_qkv)):
            x = self._ln(h, self.head_n1_w[i], self.head_n1_b[i])
            attn = self._sdpa(x, self.head_qkv[i], self.head_wo[i], full_bias)
            h = h + attn
            y = self._ln(h, self.head_n2_w[i], self.head_n2_b[i])
            h = h + self.head_fc2[i](F.relu(self.head_fc1[i](y)))

        # Gather option markers via embedding/GET_ROWS (not aten.gather → prefix slice).
        # marker_pos is a host-flattened index into [B*S, H]. Do not ADD an I32
        # batch offset in-graph: CUDA binbcast only accepts F32/F16 src1.
        flat = h.reshape(b * s, self.hidden)
        idx = marker_pos.reshape(-1)
        gathered = F.embedding(idx, flat).view(b, self.max_opts, self.hidden)

        logits = self.scorer_fc2(F.gelu(self.scorer_fc1(self.scorer_ln(gathered)))).squeeze(-1)
        logits = logits + (1.0 - marker_mask) * (-1.0e4)

        # Act/escalate features must stay float32. The importer treats aten.to.dtype as
        # identity, so argmax (I32) - arange (F32) would hit CUDA binbcast asserts.
        p = torch.softmax(logits, dim=-1)
        k = marker_mask.sum(dim=-1, keepdim=True).clamp(min=2.0)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1, keepdim=True) / torch.log(k)
        top1 = p[:, 0:1]
        for i in range(1, self.max_opts):
            slot = p[:, i : i + 1]
            top1 = F.relu(top1 - slot) + slot
        keep = (top1 - p) / ((top1 - p) + 1.0e-6)
        scaled = p * keep
        top2 = scaled[:, 0:1]
        for i in range(1, self.max_opts):
            slot = scaled[:, i : i + 1]
            top2 = F.relu(top2 - slot) + slot
        feats = torch.cat([top1, top1 - top2, ent, k / 255.0], dim=-1)
        pooled = h[:, 0]
        act_logits = self.act_fc2(F.gelu(self.act_fc1(torch.cat([pooled, feats], dim=-1))))
        return logits, act_logits


def flatten_marker_index(marker_pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Bake ``b * S`` into marker indices so GET_ROWS hits [B*S, H]. Host-side only."""
    b = int(marker_pos.shape[0])
    if b <= 1:
        return marker_pos
    off = torch.arange(b, dtype=marker_pos.dtype, device=marker_pos.device).unsqueeze(1) * int(
        seq_len
    )
    return marker_pos + off


def pad_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    max_len: int = MAX_LEN,
    max_opts: int = MAX_OPTS,
    pad_id: int = 50283,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Pad a collated Laya batch to ``[B, seq_len]`` / ``[B, 16]`` and flatten markers."""
    b = input_ids.shape[0]
    s = input_ids.shape[1]
    k = marker_pos.shape[1]
    tgt = int(seq_len) if seq_len is not None else int(max_len)
    if tgt < s:
        raise ValueError(f"seq_len {tgt} < live tokens {s}")
    ids = torch.full((b, tgt), pad_id, dtype=torch.int32)
    att = torch.zeros((b, tgt), dtype=torch.float32)
    mpos = torch.zeros((b, max_opts), dtype=torch.int32)
    mmask = torch.zeros((b, max_opts), dtype=torch.float32)
    ids[:, :s] = input_ids.to(dtype=torch.int32)
    att[:, :s] = attention_mask.to(dtype=torch.float32)
    mpos[:, :k] = marker_pos.to(dtype=torch.int32)
    mmask[:, :k] = marker_mask.to(dtype=torch.float32)
    mpos = flatten_marker_index(mpos, tgt)
    qt = qtype.to(dtype=torch.int32).reshape(b)
    return ids, att, mpos, mmask, qt
