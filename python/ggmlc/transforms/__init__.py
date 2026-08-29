"""Optimization and transformation pass pipeline for ggmlc graphs."""

from __future__ import annotations

from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats
from ggmlc.transforms.constant_folding import ConstantFoldingPass
from ggmlc.transforms.dce import DeadCodeEliminationPass
from ggmlc.transforms.fusion import FusionOptions, OperatorFusionPass
from ggmlc.transforms.manager import PassManager
from ggmlc.transforms.redundant import RedundantCastPruner


def create_standard_optimization_pipeline(
    enable_fusion: bool = True,
    options: FusionOptions | None = None,
) -> PassManager:
    """Creates a standard optimization pipeline containing ConstantFolding, OperatorFusion, RedundantCast, and DCE."""
    pm = PassManager()
    pm.add_pass(ConstantFoldingPass())
    if enable_fusion:
        pm.add_pass(OperatorFusionPass(options=options))
    pm.add_pass(RedundantCastPruner())
    pm.add_pass(DeadCodeEliminationPass())
    return pm


__all__ = [
    "ConstantFoldingPass",
    "DeadCodeEliminationPass",
    "FusionOptions",
    "GraphTransformResult",
    "OperatorFusionPass",
    "Pass",
    "PassManager",
    "PassStats",
    "RedundantCastPruner",
    "create_standard_optimization_pipeline",
]
