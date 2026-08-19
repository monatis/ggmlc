"""Pipeline coordinator and manager for compiler passes."""

from __future__ import annotations

import time
from collections.abc import Sequence

from ggmlc.ir.graph import Graph
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class PassManager:
    """Manages and executes an ordered pipeline of graph transformation passes."""

    def __init__(self, passes: Sequence[Pass] | None = None, validate_each: bool = True):
        self.passes: list[Pass] = list(passes) if passes else []
        self.validate_each = validate_each

    def add_pass(self, p: Pass) -> PassManager:
        """Adds a transformation pass to the pipeline."""
        self.passes.append(p)
        return self

    def run(self, graph: Graph) -> GraphTransformResult:
        """Executes all passes in order on the graph and returns the final transformed graph and aggregate statistics."""
        current_graph = graph
        total_modified = False
        start_nodes = len(graph.nodes)
        start_tensors = len(graph.tensors)

        aggregate_stats = PassStats(
            nodes_before=start_nodes,
            tensors_before=start_tensors,
        )

        t0 = time.perf_counter()
        for p in self.passes:
            pass_start = time.perf_counter()
            res = p.run(current_graph)
            pass_dur = (time.perf_counter() - pass_start) * 1000.0
            res.stats.duration_ms = pass_dur

            if res.modified:
                total_modified = True
                current_graph = res.graph

            # Accumulate stats
            aggregate_stats.constants_folded += res.stats.constants_folded
            aggregate_stats.fusions_applied += res.stats.fusions_applied
            aggregate_stats.dead_nodes_pruned += res.stats.dead_nodes_pruned

            if self.validate_each:
                self._validate_graph(current_graph, pass_name=p.name)

        total_dur = (time.perf_counter() - t0) * 1000.0
        aggregate_stats.nodes_after = len(current_graph.nodes)
        aggregate_stats.tensors_after = len(current_graph.tensors)
        aggregate_stats.duration_ms = total_dur

        return GraphTransformResult(
            graph=current_graph,
            modified=total_modified,
            stats=aggregate_stats,
        )

    def __call__(self, graph: Graph) -> Graph:
        return self.run(graph).graph

    @staticmethod
    def _validate_graph(graph: Graph, pass_name: str) -> None:
        """Validates internal consistency of the graph after a pass."""
        # Ensure all tensor IDs referenced by nodes exist
        for node in graph.nodes:
            for in_id in node.inputs:
                if in_id not in graph.tensors:
                    msg = f"Pass '{pass_name}' corrupted graph: input tensor {in_id} in node {node.opcode.name} does not exist."
                    raise RuntimeError(msg)
            for out_id in node.outputs:
                if out_id not in graph.tensors:
                    msg = f"Pass '{pass_name}' corrupted graph: output tensor {out_id} in node {node.opcode.name} does not exist."
                    raise RuntimeError(msg)
        # Ensure outputs and states exist
        for out_id in graph.outputs:
            if out_id not in graph.tensors:
                msg = f"Pass '{pass_name}' corrupted graph: output tensor {out_id} does not exist."
                raise RuntimeError(msg)
        for s in graph.states:
            if s.id not in graph.tensors:
                msg = f"Pass '{pass_name}' corrupted graph: state tensor {s.id} does not exist."
                raise RuntimeError(msg)
