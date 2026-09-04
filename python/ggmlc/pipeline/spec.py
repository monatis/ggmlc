"""Pipeline specifications for preprocessing and postprocessing in ggmlc."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VisionPipelineSpec:
    """Specification for image preprocessing."""

    target_size: tuple[int, int] = (224, 224)  # (height, width)
    interpolation: str = "bicubic"  # "bicubic", "bilinear", "nearest"
    crop_mode: str = "center"  # "center", "letterbox", "stretch", "none"
    mean: list[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    std: list[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    rescale_factor: float = 1.0 / 255.0
    channel_format: str = "NCHW"  # "NCHW", "NHWC"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VisionPipelineSpec:
        size = d.get("target_size", (224, 224))
        if isinstance(size, list):
            size = tuple(size)
        return cls(
            target_size=size,
            interpolation=d.get("interpolation", "bicubic"),
            crop_mode=d.get("crop_mode", "center"),
            mean=d.get("mean", [0.48145466, 0.4578275, 0.40821073]),
            std=d.get("std", [0.26862954, 0.26130258, 0.27577711]),
            rescale_factor=d.get("rescale_factor", 1.0 / 255.0),
            channel_format=d.get("channel_format", "NCHW"),
        )


@dataclass
class TokenizerSpec:
    """Specification for text tokenization."""

    model_type: str = "bpe"  # "bpe", "wordpiece", "sentencepiece"
    pre_tokenizer: str = "clip"  # "clip", "gpt2", "llama3", "bert", "default"
    vocab: dict[str, int] = field(default_factory=dict)
    merges: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    context_length: int = 77
    bos_token_id: int | None = 49406
    eos_token_id: int | None = 49407
    pad_token_id: int | None = 49407
    unk_token_id: int | None = 49407
    add_bos: bool = True
    add_eos: bool = True
    clean_spaces: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "pre_tokenizer": self.pre_tokenizer,
            "vocab_size": len(self.vocab),
            "n_merges": len(self.merges),
            "context_length": self.context_length,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "clean_spaces": self.clean_spaces,
        }


@dataclass
class PostprocessSpec:
    """Specification for model output postprocessing."""

    task: str = "similarity"  # "similarity", "classification", "generation", "detection"
    metric: str = "cosine"  # "cosine", "softmax", "sigmoid", "identity"
    top_k: int = 5
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSpec:
    """Container specification linking preprocessors, model, and postprocessor."""

    name: str = "pipeline"
    vision: dict[str, VisionPipelineSpec] = field(default_factory=dict)
    text: dict[str, TokenizerSpec] = field(default_factory=dict)
    postprocess: PostprocessSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vision": {k: v.to_dict() for k, v in self.vision.items()},
            "text": {k: v.to_dict() for k, v in self.text.items()},
            "postprocess": self.postprocess.to_dict() if self.postprocess else None,
        }
