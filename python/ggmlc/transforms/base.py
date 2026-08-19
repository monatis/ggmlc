"""Base classes for Canonical IR and dialect transformation passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ggmlc.ir.graph import Graph


@dataclass
class PassStats:
    """Statistics collected during pass execution."""

    nodes_before: int = 0
    nodes_after: int = 0
    tensors_before: int = 0
    tensors_after: int = 0
    constants_folded: int = 0
    fusions_applied: int = 0
    dead_nodes_pruned: int = 0
    duration_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def node_reduction_pct(self) -> float:
        if self.nodes_before == 0:
            return 0.0
        return ((self.nodes_before - self.nodes_after) / self.nodes_before) * 100.0


@dataclass
class GraphTransformResult:
    """Result returned by a graph transformation pass."""

    graph: Graph
    modified: bool = False
    stats: PassStats = field(default_factory=PassStats)


class Pass(ABC):
    """Abstract base class for all compiler transformation passes."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, graph: Graph) -> GraphTransformResult:
        """Executes the pass on the provided graph and returns the transformed graph."""
        ...

    def __call__(self, graph: Graph) -> Graph:
        """Convenience method to execute the pass and return the transformed graph directly."""
        result = self.run(graph)
        return result.graph
