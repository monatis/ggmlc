"""Redundant Cast and Identity Operation Pruning Pass."""

from __future__ import annotations

from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class RedundantCastPruner(Pass):
    """Eliminates redundant identity operations such as self-transpositions and identical contiguous buffers."""

    def __init__(self):
        super().__init__(name="RedundantCastPruner")

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        remap: dict[int, int] = {}
        nodes_to_keep = []
        eliminated = 0

        for node in graph.nodes:
            is_identity = False

            # Pattern 1: Transpose dim0 == dim1
            if node.opcode == OpCode.TRANSPOSE:
                d0 = node.attributes.get("dim0", 0)
                d1 = node.attributes.get("dim1", 1)
                if d0 == d1 and len(node.inputs) == 1 and len(node.outputs) == 1:
                    is_identity = True

            # Pattern 2: Permute with [0, 1, 2, ...]
            elif node.opcode == OpCode.PERMUTE:
                dims = node.attributes.get("dims") or node.attributes.get("permutation")
                if (
                    dims is not None
                    and len(dims) > 0
                    and list(dims) == list(range(len(dims)))
                    and len(node.inputs) == 1
                    and len(node.outputs) == 1
                ):
                    is_identity = True

            if is_identity:
                in_id = node.inputs[0]
                out_id = node.outputs[0]
                state_tids = {s.id for s in graph.states}
                # If output is not an externally visible output or state
                if out_id not in graph.outputs and out_id not in state_tids:
                    remap[out_id] = remap.get(in_id, in_id)
                    eliminated += 1
                    continue

            nodes_to_keep.append(node)

        # Build output graph
        new_graph = Graph(name=graph.name)
        new_graph.tensors = dict(graph.tensors)
        new_graph.inputs = list(graph.inputs)
        new_graph.outputs = list(graph.outputs)
        new_graph.parameters = list(graph.parameters)
        new_graph.states = list(graph.states)

        for node in nodes_to_keep:
            node.inputs = [remap.get(i, i) for i in node.inputs]
            new_graph.nodes.append(node)

        modified = eliminated > 0
        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=len(new_graph.nodes),
            tensors_before=tensors_before,
            tensors_after=len(new_graph.tensors),
            extra={"redundant_casts_eliminated": eliminated},
        )

        return GraphTransformResult(graph=new_graph, modified=modified, stats=stats)
