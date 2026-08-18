from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ggmlc.ir.graph import Graph


@dataclass
class Model:
    """Represents a compiled neural-network model with one or more subgraphs."""

    name: str
    graphs: Dict[str, Graph] = field(default_factory=dict)
    main_graph_name: str = "main"
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def main_graph(self) -> Graph:
        if self.main_graph_name not in self.graphs:
            raise KeyError(f"Main graph '{self.main_graph_name}' not found in model '{self.name}'")
        return self.graphs[self.main_graph_name]

    def add_graph(self, graph: Graph, is_main: bool = False) -> None:
        self.graphs[graph.name] = graph
        if is_main or len(self.graphs) == 1:
            self.main_graph_name = graph.name

    def validate(self) -> None:
        for g in self.graphs.values():
            g.validate_invariants()
