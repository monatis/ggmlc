"""Automatic extraction of VisionPreprocessor and Tokenizers from Hugging Face."""

from __future__ import annotations

from typing import Any

from ggmlc.pipeline.spec import PipelineSpec, PostprocessSpec
from ggmlc.pipeline.tokenizer import BPETokenizer, WordPieceTokenizer
from ggmlc.pipeline.vision import VisionPreprocessor


def from_huggingface_image_processor(image_proc: Any) -> VisionPreprocessor:
    """Extracts a VisionPreprocessor from Hugging Face ImageProcessor / FeatureExtractor."""
    # Target size
    size = getattr(image_proc, "size", {"shortest_edge": 224})
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            target_size = (int(size["height"]), int(size["width"]))
        elif "shortest_edge" in size:
            target_size = (int(size["shortest_edge"]), int(size["shortest_edge"]))
        else:
            target_size = (224, 224)
    elif isinstance(size, (list, tuple)):
        target_size = (
            (int(size[0]), int(size[1])) if len(size) > 1 else (int(size[0]), int(size[0]))
        )
    elif isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = (224, 224)

    # Crop size
    crop_size = getattr(image_proc, "crop_size", None)
    if isinstance(crop_size, dict) and "height" in crop_size and "width" in crop_size:
        target_size = (int(crop_size["height"]), int(crop_size["width"]))

    # Mean & Std
    mean = getattr(image_proc, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_proc, "image_std", [0.26862954, 0.26130258, 0.27577711])
    rescale = getattr(image_proc, "rescale_factor", 1.0 / 255.0)

    # Resample / Interpolation
    resample = getattr(image_proc, "resample", 3)
    if resample in (3, "bicubic", "BICUBIC"):
        interpolation = "bicubic"
    elif resample in (2, "bilinear", "BILINEAR"):
        interpolation = "bilinear"
    elif resample in (0, "nearest", "NEAREST"):
        interpolation = "nearest"
    else:
        interpolation = "bicubic"

    crop_mode = "center" if getattr(image_proc, "do_center_crop", True) else "stretch"

    return VisionPreprocessor(
        target_size=target_size,
        interpolation=interpolation,
        crop_mode=crop_mode,
        mean=mean,
        std=std,
        rescale_factor=rescale,
        channel_format="NCHW",
    )


def from_huggingface_tokenizer(tokenizer: Any, context_length: int | None = None) -> Any:
    """Extracts a BPETokenizer or WordPieceTokenizer from Hugging Face PreTrainedTokenizer."""
    cls_name = tokenizer.__class__.__name__.lower()
    if (
        "bert" in cls_name
        and "clip" not in cls_name
        and "gpt" not in cls_name
        and "roberta" not in cls_name
    ):
        return WordPieceTokenizer.from_huggingface(tokenizer, context_length=context_length)
    return BPETokenizer.from_huggingface(tokenizer, context_length=context_length)


def from_huggingface(processor: Any) -> PipelineSpec:
    """Extracts a unified PipelineSpec from a Hugging Face Processor (e.g. CLIPProcessor)."""
    vision_specs = {}
    text_specs = {}

    # Check for image processor
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        v_pre = from_huggingface_image_processor(processor.image_processor)
        vision_specs["pixel_values"] = v_pre.spec
    elif hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
        v_pre = from_huggingface_image_processor(processor.feature_extractor)
        vision_specs["pixel_values"] = v_pre.spec

    # Check for tokenizer
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        tok = from_huggingface_tokenizer(processor.tokenizer)
        text_specs["input_ids"] = tok.spec

    return PipelineSpec(
        name="hf_extracted_pipeline",
        vision=vision_specs,
        text=text_specs,
        postprocess=PostprocessSpec(task="similarity", metric="cosine"),
    )
