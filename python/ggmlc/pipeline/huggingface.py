"""Automatic extraction of VisionPreprocessor and Tokenizers from Hugging Face."""

from __future__ import annotations

from typing import Any

from ggmlc.pipeline.spec import PipelineSpec, PostprocessSpec
from ggmlc.pipeline.tokenizer import BPETokenizer, WordPieceTokenizer
from ggmlc.pipeline.vision import VisionPreprocessor


def from_huggingface_image_processor(image_proc: Any) -> VisionPreprocessor:
    """Extracts a VisionPreprocessor from Hugging Face ImageProcessor, FeatureExtractor, or model ID."""
    if isinstance(image_proc, str):
        try:
            from transformers import AutoImageProcessor

            image_proc = AutoImageProcessor.from_pretrained(image_proc)
        except Exception:  # noqa: BLE001
            from transformers import AutoFeatureExtractor

            image_proc = AutoFeatureExtractor.from_pretrained(image_proc)

    d = image_proc.to_dict() if hasattr(image_proc, "to_dict") else {}

    # 1. Target crop size
    target_size = (224, 224)
    if "crop_size" in d and d["crop_size"] is not None:
        cs = d["crop_size"]
        if isinstance(cs, dict):
            target_size = (int(cs.get("height", 224)), int(cs.get("width", 224)))
        elif isinstance(cs, (list, tuple)):
            target_size = (int(cs[0]), int(cs[1])) if len(cs) > 1 else (int(cs[0]), int(cs[0]))
        elif isinstance(cs, int):
            target_size = (cs, cs)
    elif "size" in d and d["size"] is not None:
        sz = d["size"]
        if isinstance(sz, dict):
            if "height" in sz and "width" in sz:
                target_size = (int(sz["height"]), int(sz["width"]))
            elif "shortest_edge" in sz:
                target_size = (int(sz["shortest_edge"]), int(sz["shortest_edge"]))
        elif isinstance(sz, (list, tuple)):
            target_size = (int(sz[0]), int(sz[1])) if len(sz) > 1 else (int(sz[0]), int(sz[0]))
        elif isinstance(sz, int):
            target_size = (sz, sz)

    # 2. Resize size
    resize_size: tuple[int, int] | None = None
    if "size" in d and d["size"] is not None:
        sz = d["size"]
        if isinstance(sz, dict):
            if "shortest_edge" in sz:
                se = int(sz["shortest_edge"])
                resize_size = (se, se)
            elif "height" in sz and "width" in sz:
                resize_size = (int(sz["height"]), int(sz["width"]))
        elif isinstance(sz, (list, tuple)):
            resize_size = (int(sz[0]), int(sz[1])) if len(sz) > 1 else (int(sz[0]), int(sz[0]))
        elif isinstance(sz, int):
            resize_size = (sz, sz)

    if "crop_pct" in d and d["crop_pct"] is not None and float(d["crop_pct"]) > 0:
        crop_pct = float(d["crop_pct"])
        se = int(target_size[0] / crop_pct)
        resize_size = (se, se)

    # 3. Mean & Std
    mean = d.get("image_mean", getattr(image_proc, "image_mean", [0.485, 0.456, 0.406]))
    if hasattr(mean, "__iter__"):
        mean = [float(x) for x in mean]
    std = d.get("image_std", getattr(image_proc, "image_std", [0.229, 0.224, 0.225]))
    if hasattr(std, "__iter__"):
        std = [float(x) for x in std]
    rescale = float(d.get("rescale_factor", getattr(image_proc, "rescale_factor", 1.0 / 255.0)))

    # 4. Resample / Interpolation
    resample = d.get("resample", getattr(image_proc, "resample", 3))
    if hasattr(resample, "value"):
        resample = resample.value
    if resample in (3, "bicubic", "BICUBIC"):
        interpolation = "bicubic"
    elif resample in (2, "bilinear", "BILINEAR"):
        interpolation = "bilinear"
    elif resample in (0, "nearest", "NEAREST"):
        interpolation = "nearest"
    else:
        interpolation = "bilinear"

    # 5. Crop mode
    if (
        "crop_size" in d
        and d["crop_size"] is not None
        or d.get("crop_pct") is not None
        or d.get("do_center_crop", False)
        or isinstance(d.get("size"), dict)
        and "shortest_edge" in d.get("size", {})
    ):
        crop_mode = "center"
    else:
        crop_mode = "stretch"

    return VisionPreprocessor(
        target_size=target_size,
        resize_size=resize_size,
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
