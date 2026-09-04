"""High-level Pipeline orchestrator bridging preprocessors, model runner, and postprocessors."""

from __future__ import annotations

from typing import Any

from ggmlc.pipeline.spec import PipelineSpec, PostprocessSpec
from ggmlc.pipeline.vision import VisionPreprocessor


class Pipeline:
    """Unified Pipeline containing input preprocessors, model execution, and postprocessing."""

    def __init__(
        self,
        vision_preprocessors: dict[str, VisionPreprocessor] | None = None,
        tokenizers: dict[str, Any] | None = None,
        postprocess: PostprocessSpec | dict[str, Any] | None = None,
        name: str = "pipeline",
    ):
        self.name = name
        self.vision_preprocessors = vision_preprocessors or {}
        self.tokenizers = tokenizers or {}
        if isinstance(postprocess, dict):
            self.postprocess = PostprocessSpec(**postprocess)
        else:
            self.postprocess = postprocess

    @property
    def spec(self) -> PipelineSpec:
        return PipelineSpec(
            name=self.name,
            vision={k: v.spec for k, v in self.vision_preprocessors.items()},
            text={k: v.spec for k, v in self.tokenizers.items()},
            postprocess=self.postprocess,
        )

    def to_gguf_metadata(self) -> dict[str, Any]:
        """Combines all metadata keys for GGUF serialization."""
        meta: dict[str, Any] = {
            "ggmlc.pipeline_name": self.name,
        }
        for v_proc in self.vision_preprocessors.values():
            meta.update(v_proc.to_gguf_metadata())
        for tok in self.tokenizers.values():
            meta.update(tok.to_gguf_metadata())
        return meta
