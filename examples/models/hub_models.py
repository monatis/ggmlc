"""Real checkpoint loaders for vision, language, and embedding architectures."""

import torch
from torch import nn


def load_resnet_model(
    variant: str = "resnet18",
    resolution: int = 224,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real torchvision pretrained ResNet checkpoint."""
    from torchvision import models

    variant = variant.lower()
    if variant == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

    example_input = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    input_names = ["x"]
    return model, example_input, input_names


def load_minilm_model(seq_len: int = 16) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real sentence-transformers all-MiniLM-L6-v2 checkpoint."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int32)
    example_input = (input_ids,)
    input_names = ["input_ids"]

    # Wrap to output only dense sequence output
    class MiniLMWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids):
            out = self.base_model(input_ids=input_ids)
            return out.last_hidden_state

    return MiniLMWrapper(model), example_input, input_names


def load_gpt2_model(seq_len: int = 8) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real Hugging Face GPT-2 checkpoint."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    base_ids = [542, 67, 876, 414, 26, 335, 620, 924]
    if seq_len <= len(base_ids):
        input_ids = torch.tensor([base_ids[:seq_len]], dtype=torch.int32)
    else:
        # Repeat or generate tokens
        repeated = (base_ids * ((seq_len // len(base_ids)) + 1))[:seq_len]
        input_ids = torch.tensor([repeated], dtype=torch.int32)
    example_input = (input_ids,)
    input_names = ["input_ids"]

    class GPT2Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.wte = base.transformer.wte
            self.wpe = base.transformer.wpe
            self.blocks = nn.ModuleList([base.transformer.h[i] for i in range(12)])
            self.ln_f = base.transformer.ln_f
            self.lm_head = base.lm_head

        def forward(self, input_ids):
            pos_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int32).unsqueeze(0)
            h = self.wte(input_ids) + self.wpe(pos_ids)
            for blk in self.blocks:
                if h.ndim == 2:
                    h = h.unsqueeze(0)
                h = blk(h)[0]
            if h.ndim == 2:
                h = h.unsqueeze(0)
            h = self.ln_f(h)
            return self.lm_head(h)

    return GPT2Wrapper(model), example_input, input_names


def load_qwen_model(
    variant: str = "Qwen/Qwen2.5-0.5B",
    seq_len: int = 8,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real Hugging Face Qwen checkpoint."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(variant, torch_dtype=torch.float32).eval()
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int32)
    example_input = (input_ids,)
    input_names = ["input_ids"]

    class QwenWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.embed_tokens = base_model.model.embed_tokens
            self.layers = base_model.model.layers
            self.norm = base_model.model.norm
            self.lm_head = base_model.lm_head
            self.rotary_emb = base_model.model.rotary_emb
            self.num_heads = base_model.config.num_attention_heads
            self.num_kv_heads = base_model.config.num_key_value_heads
            self.head_dim = base_model.config.hidden_size // self.num_heads
            self.kv_groups = self.num_heads // self.num_kv_heads

        def forward(self, input_ids):
            h = self.embed_tokens(input_ids)
            bsz, seq_len, _ = h.shape
            pos = torch.arange(0, seq_len, dtype=torch.float32, device=h.device)
            inv_freq = self.rotary_emb.inv_freq
            freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)

            # Apply RoPE helper inline
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            def apply_rope(x):
                c = cos.unsqueeze(1)
                s = sin.unsqueeze(1)
                return (x * c) + (rotate_half(x) * s)

            for layer in self.layers:
                residual = h
                h_norm = layer.input_layernorm(h)

                q = (
                    layer.self_attn.q_proj(h_norm)
                    .view(bsz, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    layer.self_attn.k_proj(h_norm)
                    .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    layer.self_attn.v_proj(h_norm)
                    .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )

                q = apply_rope(q)
                k = apply_rope(k)

                def repeat_kv(x, kv_groups):
                    if kv_groups == 1:
                        return x
                    heads = [
                        x[:, i : i + 1, :, :].expand(-1, kv_groups, -1, -1)
                        for i in range(x.shape[1])
                    ]
                    return torch.cat(heads, dim=1)

                k = repeat_kv(k, self.kv_groups)
                v = repeat_kv(v, self.kv_groups)

                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                attn_out = layer.self_attn.o_proj(attn_out)
                h = residual + attn_out

                residual = h
                h_norm = layer.post_attention_layernorm(h)
                mlp_out = layer.mlp.down_proj(
                    torch.nn.functional.silu(layer.mlp.gate_proj(h_norm))
                    * layer.mlp.up_proj(h_norm)
                )
                h = residual + mlp_out

            h = self.norm(h)
            return self.lm_head(h)

    return QwenWrapper(model), example_input, input_names


def load_bge_m3_distill_model(
    seq_len: int = 16,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real altaidevorg/bge-m3-distill-8l checkpoint via sentence-transformers/transformers."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("altaidevorg/bge-m3-distill-8l").eval()
    input_ids = torch.randint(10, 1000, (1, seq_len), dtype=torch.int32)
    example_input = (input_ids,)
    input_names = ["input_ids"]

    class BGEM3Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.word_embeddings = base.embeddings.word_embeddings
            self.position_embeddings = base.embeddings.position_embeddings
            self.token_type_embeddings = base.embeddings.token_type_embeddings
            self.LayerNorm = base.embeddings.LayerNorm
            self.encoder = base.encoder
            self.padding_idx = base.embeddings.padding_idx

        def forward(self, input_ids):
            cur_seq = input_ids.shape[-1]
            pos_ids = torch.arange(
                self.padding_idx + 1, self.padding_idx + 1 + cur_seq, dtype=torch.int32
            ).unsqueeze(0)
            token_type_ids = torch.zeros((1, cur_seq), dtype=torch.int32)
            words_emb = self.word_embeddings(input_ids)
            pos_emb = self.position_embeddings(pos_ids)
            type_emb = self.token_type_embeddings(token_type_ids)
            emb = self.LayerNorm(words_emb + pos_emb + type_emb)
            out = self.encoder(emb)
            return out.last_hidden_state

    return BGEM3Wrapper(model), example_input, input_names


def load_mobilenet_v3_model(
    variant: str = "small",
    resolution: int = 224,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads torchvision MobileNetV3 checkpoint (small or large)."""
    from torchvision import models

    variant = variant.lower()
    if variant == "large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).eval()
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).eval()

    example_input = (torch.randn(1, 3, resolution, resolution, dtype=torch.float32),)
    input_names = ["x"]
    return model, example_input, input_names


def load_ssdlite320_mobilenet_v3_model() -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads torchvision SSDLite320-MobileNetV3 object detection backbone + head checkpoint."""
    from torchvision import models

    model = models.detection.ssdlite320_mobilenet_v3_large(
        weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()
    example_input = (torch.randn(1, 3, 320, 320, dtype=torch.float32),)
    input_names = ["images"]

    class SSDLitePredictor(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.backbone = base.backbone
            self.head = base.head

        def forward(self, images):
            features = self.backbone(images)
            feature_list = list(features.values())
            head_outputs = self.head(feature_list)
            return head_outputs["bbox_regression"], head_outputs["cls_logits"]

    return SSDLitePredictor(model), example_input, input_names
