"""Real checkpoint loaders for OpenAI CLIP multimodal vision and text architectures."""

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPModel, CLIPTextModel, CLIPVisionModel


def load_clip_vision_model(
    variant: str = "openai/clip-vit-base-patch32",
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real pretrained CLIP Vision Transformer."""
    model = CLIPVisionModel.from_pretrained(variant).eval()
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    example_input = (pixel_values,)
    input_names = ["pixel_values"]

    class CLIPVisionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, pixel_values):
            out = self.base(pixel_values=pixel_values)
            return out.pooler_output

    return CLIPVisionWrapper(model), example_input, input_names


def load_clip_text_model(
    variant: str = "openai/clip-vit-base-patch32",
    seq_len: int = 77,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads real pretrained CLIP Text Transformer."""
    model = CLIPTextModel.from_pretrained(variant).eval()
    base_ids = [49406, 320, 1125, 539, 320, 2368, 49407]
    padded_ids = (base_ids + [49407] * (seq_len - len(base_ids)))[:seq_len]
    input_ids = torch.tensor([padded_ids], dtype=torch.int32)
    example_input = (input_ids,)
    input_names = ["input_ids"]

    class CLIPTextWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids):
            out = self.base(input_ids=input_ids)
            return out.pooler_output

    return CLIPTextWrapper(model), example_input, input_names


def load_clip_full_model(
    variant: str = "openai/clip-vit-base-patch32",
    seq_len: int = 77,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    """Loads complete end-to-end multimodal CLIP model computing image-text similarity logits."""
    model = CLIPModel.from_pretrained(variant).eval()
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    base_ids = [49406, 320, 1125, 539, 320, 2368, 49407]
    padded_ids = (base_ids + [49407] * (seq_len - len(base_ids)))[:seq_len]
    input_ids = torch.tensor([padded_ids], dtype=torch.int32)

    example_input = (pixel_values, input_ids)
    input_names = ["pixel_values", "input_ids"]

    class CLIPMultimodalWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, pixel_values, input_ids):
            # Compute image embeds
            vis_out = self.base.vision_model(pixel_values=pixel_values)
            img_embeds = self.base.visual_projection(vis_out.pooler_output)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            # Compute text embeds
            txt_out = self.base.text_model(input_ids=input_ids)
            txt_embeds = self.base.text_projection(txt_out.pooler_output)
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

            # Scaled cosine similarity logits
            logit_scale = self.base.logit_scale.exp()
            logits_per_image = torch.matmul(img_embeds, txt_embeds.t()) * logit_scale
            return logits_per_image

    return CLIPMultimodalWrapper(model), example_input, input_names
