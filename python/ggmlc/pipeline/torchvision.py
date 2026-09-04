"""Torchvision transformation introspection and parity verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from ggmlc.pipeline.vision import VisionPreprocessor


def from_torchvision(transforms: Any) -> VisionPreprocessor:
    """Introspects torchvision transforms, Weights enums, or models to create a VisionPreprocessor.

    Extracts:
    - Resize size (shortest edge or pre-crop size) & interpolation
    - CenterCrop target size
    - Normalize mean & std
    - Channel ordering (NCHW)
    """
    # Check if transforms is a Weights enum or has callable transforms()
    if hasattr(transforms, "transforms") and callable(transforms.transforms):
        transforms = transforms.transforms()

    crop_size = (224, 224)
    resize_size: tuple[int, int] | None = None
    interpolation = "bilinear"
    crop_mode = "center"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Check ImageClassification or similar transform objects (torchvision v0.13+)
    if hasattr(transforms, "crop_size"):
        cs = transforms.crop_size
        crop_size = (
            (int(cs[0]), int(cs[1]))
            if isinstance(cs, (list, tuple)) and len(cs) > 1
            else (int(cs[0]), int(cs[0]))
        )
    if hasattr(transforms, "resize_size"):
        rs = transforms.resize_size
        resize_size = (
            (int(rs[0]), int(rs[1]))
            if isinstance(rs, (list, tuple)) and len(rs) > 1
            else (int(rs[0]), int(rs[0]))
        )
    if hasattr(transforms, "mean"):
        mean = [float(x) for x in transforms.mean]
    if hasattr(transforms, "std"):
        std = [float(x) for x in transforms.std]
    if hasattr(transforms, "interpolation"):
        interp = str(transforms.interpolation).lower()
        if "bicubic" in interp:
            interpolation = "bicubic"
        elif "bilinear" in interp:
            interpolation = "bilinear"
        elif "nearest" in interp:
            interpolation = "nearest"

    # Unwrap Compose or iterable
    transform_list = []
    if hasattr(transforms, "transforms") and isinstance(transforms.transforms, (list, tuple)):
        transform_list = list(transforms.transforms)
    elif isinstance(transforms, (list, tuple)):
        transform_list = list(transforms)
    elif not hasattr(transforms, "crop_size"):
        transform_list = [transforms]

    for t in transform_list:
        cls_name = t.__class__.__name__

        if "Resize" in cls_name:
            if hasattr(t, "size"):
                sz = t.size
                if isinstance(sz, (list, tuple)):
                    resize_size = (
                        (int(sz[0]), int(sz[1])) if len(sz) > 1 else (int(sz[0]), int(sz[0]))
                    )
                else:
                    resize_size = (int(sz), int(sz))
            if hasattr(t, "interpolation"):
                interp = str(t.interpolation).lower()
                if "bicubic" in interp:
                    interpolation = "bicubic"
                elif "bilinear" in interp:
                    interpolation = "bilinear"
                elif "nearest" in interp:
                    interpolation = "nearest"

        elif "CenterCrop" in cls_name:
            crop_mode = "center"
            if hasattr(t, "size"):
                sz = t.size
                if isinstance(sz, (list, tuple)):
                    crop_size = (
                        (int(sz[0]), int(sz[1])) if len(sz) > 1 else (int(sz[0]), int(sz[0]))
                    )
                else:
                    crop_size = (int(sz), int(sz))

        elif "Normalize" in cls_name:
            if hasattr(t, "mean"):
                mean = [float(x) for x in t.mean]
            if hasattr(t, "std"):
                std = [float(x) for x in t.std]

    return VisionPreprocessor(
        target_size=crop_size,
        resize_size=resize_size,
        interpolation=interpolation,
        crop_mode=crop_mode,
        mean=mean,
        std=std,
        rescale_factor=1.0 / 255.0,
        channel_format="NCHW",
    )


def verify_torchvision_parity(
    image: Image.Image | np.ndarray,
    torchvision_transform: Any,
    vision_preprocessor: VisionPreprocessor,
    atol: float = 1e-2,
    min_cosine_similarity: float = 0.9999,
) -> dict[str, Any]:
    """Verifies numerical parity between torchvision transform and ggmlc VisionPreprocessor.

    Returns dictionary with max_abs_diff, mean_abs_diff, cosine_similarity, and passed status.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # 1. Run reference torchvision transform
    ref_out = torchvision_transform(pil_img)
    if isinstance(ref_out, torch.Tensor):
        ref_np = ref_out.detach().cpu().numpy()
        if ref_np.ndim == 3:
            ref_np = np.expand_dims(ref_np, 0)
    else:
        ref_np = np.array(ref_out, dtype=np.float32)

    # 2. Run ggmlc VisionPreprocessor
    actual_np = vision_preprocessor.process(pil_img)

    # 3. Compare metrics
    diff = np.abs(ref_np - actual_np)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    flat_ref = ref_np.flatten().astype(np.float64)
    flat_act = actual_np.flatten().astype(np.float64)
    norm_ref = np.linalg.norm(flat_ref)
    norm_act = np.linalg.norm(flat_act)
    cosine_sim = float(np.dot(flat_ref, flat_act) / (norm_ref * norm_act + 1e-12))

    passed = cosine_sim >= min_cosine_similarity and max_diff <= atol

    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "cosine_similarity": cosine_sim,
        "passed": passed,
    }
