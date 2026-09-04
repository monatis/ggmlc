"""Vision preprocessing routines matching PIL / torchvision and C++ runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ggmlc.pipeline.spec import VisionPipelineSpec


class VisionPreprocessor:
    """Preprocesses raw images into normalized float32 tensor buffers.

    Matches standard torchvision / PIL pipelines (OpenAI CLIP, ImageNet, etc.)
    with bit-exact mathematical parameter parity.
    """

    def __init__(
        self,
        target_size: tuple[int, int] | int | VisionPipelineSpec = (224, 224),
        interpolation: str = "bicubic",
        crop_mode: str = "center",
        mean: list[float] | tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: list[float] | tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        rescale_factor: float = 1.0 / 255.0,
        channel_format: str = "NCHW",
    ):
        if isinstance(target_size, VisionPipelineSpec):
            spec = target_size
            self.target_size = spec.target_size
            self.interpolation = spec.interpolation.lower()
            self.crop_mode = spec.crop_mode.lower()
            self.mean = np.array(spec.mean, dtype=np.float32)
            self.std = np.array(spec.std, dtype=np.float32)
            self.rescale_factor = float(spec.rescale_factor)
            self.channel_format = spec.channel_format.upper()
        else:
            if isinstance(target_size, int):
                self.target_size = (target_size, target_size)
            else:
                self.target_size = tuple(target_size)

            self.interpolation = interpolation.lower()
            self.crop_mode = crop_mode.lower()
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
            self.rescale_factor = float(rescale_factor)
            self.channel_format = channel_format.upper()

        self._pil_resample = {
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
        }.get(self.interpolation, Image.Resampling.BICUBIC)

    def preprocess_image(self, image_input: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        """Alias for process()."""
        return self.process(image_input)

    def __call__(self, image_input: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        return self.process(image_input)

    @property
    def spec(self) -> VisionPipelineSpec:
        return VisionPipelineSpec(
            target_size=self.target_size,
            interpolation=self.interpolation,
            crop_mode=self.crop_mode,
            mean=self.mean.tolist(),
            std=self.std.tolist(),
            rescale_factor=self.rescale_factor,
            channel_format=self.channel_format,
        )

    def process(self, image_input: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        """Processes an image from a filepath, PIL Image, or numpy array.

        Returns:
            np.ndarray of shape (1, 3, H, W) for NCHW or (1, H, W, 3) for NHWC.
        """
        # 1. Load image to PIL RGB
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = (np.clip(image_input, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        target_h, target_w = self.target_size

        # 2. Resize and Crop
        if self.crop_mode == "center":
            # Scale shortest edge to target size, then center crop
            w, h = pil_img.size
            if w < h:
                new_w = target_w
                new_h = int(h * target_w / w)
            else:
                new_h = target_h
                new_w = int(w * target_h / h)

            resized = pil_img.resize((new_w, new_h), resample=self._pil_resample)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))
        elif self.crop_mode == "letterbox":
            # Maintain aspect ratio by scaling longest edge and padding
            w, h = pil_img.size
            scale = min(target_w / w, target_h / h)
            new_w = round(w * scale)
            new_h = round(h * scale)
            resized = pil_img.resize((new_w, new_h), resample=self._pil_resample)

            cropped = Image.new("RGB", (target_w, target_h), (114, 114, 114))
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2
            cropped.paste(resized, (pad_left, pad_top))
        else:
            # Direct resize (stretch)
            cropped = pil_img.resize((target_w, target_h), resample=self._pil_resample)

        # 3. Convert to Float32 & Rescale
        arr = np.array(cropped, dtype=np.float32) * self.rescale_factor

        # 4. Normalize: (x - mean) / std
        arr = (arr - self.mean) / self.std

        # 5. Format channel layout
        if self.channel_format == "NCHW":
            # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, 0)
        else:
            # (H, W, C) -> (1, H, W, C)
            arr = np.expand_dims(arr, 0)

        return np.ascontiguousarray(arr, dtype=np.float32)

    def to_gguf_metadata(self) -> dict[str, Any]:
        """Generates GGUF key-value metadata pairs."""
        return {
            "clip.vision.image_size": int(self.target_size[0]),
            "clip.vision.image_mean": self.mean.tolist(),
            "clip.vision.image_std": self.std.tolist(),
            "preprocessor.image.interpolation": self.interpolation,
            "preprocessor.image.crop_mode": self.crop_mode,
            "preprocessor.image.channel_format": self.channel_format,
        }
