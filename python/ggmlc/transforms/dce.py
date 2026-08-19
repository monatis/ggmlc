"""Dead Code Elimination (DCE) Pass."""

from __future__ import annotations

from ggmlc.ir.graph import Graph
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class DeadCodeEliminationPass(Pass):
    """Prunes unreachable operations and unreferenced intermediate tensors from the graph."""

    def __init__(self):
        super().__init__(name="DeadCodeElimination")

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        # 1. Identify live root tensors: outputs and persistent states
        live_tensors: set[int] = set(graph.outputs)
        state_tids = {s.id for s in graph.states}
        live_tensors.update(state_tids)

        # Map each tensor ID to the node that produces it
        producer_map: dict[int, int] = {}  # tensor_id -> node_idx
        for idx, node in enumerate(graph.nodes):
            for out_id in node.outputs:
                producer_map[out_id] = idx

        # 2. Backward reachability search
        worklist: list[int] = list(live_tensors)
        visited_tensors: set[int] = set(live_tensors)
        live_node_indices: set[int] = set()

        while worklist:
            tid = worklist.pop()
            # If this tensor is produced by an operation node
            if tid in producer_map:
                node_idx = producer_map[tid]
                if node_idx not in live_node_indices:
                    live_node_indices.add(node_idx)
                    node = graph.nodes[node_idx]
                    for in_id in node.inputs:
                        if in_id not in visited_tensors:
                            visited_tensors.add(in_id)
                            worklist.append(in_id)

        # Always preserve inputs and parameters
        for tid, tensor in graph.tensors.items():
            if tensor.storage in (StorageClass.INPUT, StorageClass.PARAMETER, StorageClass.STATE):
                visited_tensors.add(tid)

        # 3. Construct clean pruned graph
        new_graph = Graph(name=graph.name)

        # Copy live tensors
        for tid in sorted(visited_tensors):
            if tid in graph.tensors:
                new_graph.tensors[tid] = graph.tensors[tid]

        # Copy live nodes in original topological order
        for idx, node in enumerate(graph.nodes):
            if idx in live_node_indices:
                new_graph.nodes.append(node)

        # Preserve metadata
        new_graph.inputs = [tid for tid in graph.inputs if tid in visited_tensors]
        new_graph.outputs = list(graph.outputs)
        new_graph.parameters = [tid for tid in graph.parameters if tid in visited_tensors]
        new_graph.states = [s for s in graph.states if s.id in visited_tensors]

        nodes_after = len(new_graph.nodes)
        tensors_after = len(new_graph.tensors)
        pruned_nodes = nodes_before - nodes_after
        modified = pruned_nodes > 0 or (tensors_before > tensors_after)

        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            tensors_before=tensors_before,
            tensors_after=tensors_after,
            dead_nodes_pruned=pruned_nodes,
        )

        return GraphTransformResult(graph=new_graph, modified=modified, stats=stats)
