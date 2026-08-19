"""Operator Fusion Pass for Canonical IR and dialect graphs."""

from __future__ import annotations

from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class OperatorFusionPass(Pass):
    """Fuses composite operation patterns (e.g. Conv2D+Bias+ReLU, Linear+Bias, SwiGLU) into native fused operations."""

    def __init__(self):
        super().__init__(name="OperatorFusion")

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        # Producer map: tensor_id -> node_idx
        # Consumer map: tensor_id -> list of node_idx
        producer_map: dict[int, int] = {}
        consumer_map: dict[int, list[int]] = {}
        for idx, node in enumerate(graph.nodes):
            for out_id in node.outputs:
                producer_map[out_id] = idx
            for in_id in node.inputs:
                consumer_map.setdefault(in_id, []).append(idx)

        fused_count = 0
        eliminated_node_indices: set[int] = set()
        new_nodes: list[Operation] = []

        state_tids = {s.id for s in graph.states}

        # Sequential scan for fusion candidates
        for idx, node in enumerate(graph.nodes):
            if idx in eliminated_node_indices:
                continue

            # Pattern 1: Conv2D followed by ReLU where intermediate has single consumer
            if node.opcode == OpCode.CONV2D and len(node.outputs) == 1:
                conv_out = node.outputs[0]
                consumers = consumer_map.get(conv_out, [])
                if (
                    len(consumers) == 1
                    and conv_out not in graph.outputs
                    and conv_out not in state_tids
                ):
                    next_idx = consumers[0]
                    next_node = graph.nodes[next_idx]
                    if next_node.opcode == OpCode.RELU and next_idx not in eliminated_node_indices:
                        # Fuse Conv2D + ReLU
                        fused_attrs = dict(node.attributes)
                        fused_attrs["fused_relu"] = True
                        fused_node = Operation(
                            id=node.id,
                            opcode=OpCode.CONV2D,
                            inputs=list(node.inputs),
                            outputs=list(next_node.outputs),
                            attributes=fused_attrs,
                        )
                        new_nodes.append(fused_node)
                        eliminated_node_indices.add(idx)
                        eliminated_node_indices.add(next_idx)
                        fused_count += 1
                        continue

            # Pattern 2: MatMul followed by Add (Bias) -> Linear with bias
            if node.opcode == OpCode.MATMUL and len(node.outputs) == 1 and len(node.inputs) == 2:
                matmul_out = node.outputs[0]
                consumers = consumer_map.get(matmul_out, [])
                if (
                    len(consumers) == 1
                    and matmul_out not in graph.outputs
                    and matmul_out not in state_tids
                ):
                    next_idx = consumers[0]
                    next_node = graph.nodes[next_idx]
                    if next_node.opcode == OpCode.ADD and next_idx not in eliminated_node_indices:
                        # Determine which input to Add is the bias
                        bias_id = (
                            next_node.inputs[1]
                            if next_node.inputs[0] == matmul_out
                            else next_node.inputs[0]
                        )
                        fused_node = Operation(
                            id=node.id,
                            opcode=OpCode.LINEAR,
                            inputs=[node.inputs[0], node.inputs[1], bias_id],
                            outputs=list(next_node.outputs),
                            attributes=dict(node.attributes),
                        )
                        new_nodes.append(fused_node)
                        eliminated_node_indices.add(idx)
                        eliminated_node_indices.add(next_idx)
                        fused_count += 1
                        continue

            # Pattern 3: SwiGLU pattern: x * silu(g)
            if node.opcode == OpCode.SILU and len(node.outputs) == 1:
                silu_out = node.outputs[0]
                consumers = consumer_map.get(silu_out, [])
                if (
                    len(consumers) == 1
                    and silu_out not in graph.outputs
                    and silu_out not in state_tids
                ):
                    next_idx = consumers[0]
                    next_node = graph.nodes[next_idx]
                    if next_node.opcode == OpCode.MUL and next_idx not in eliminated_node_indices:
                        other_in = (
                            next_node.inputs[1]
                            if next_node.inputs[0] == silu_out
                            else next_node.inputs[0]
                        )
                        fused_node = Operation(
                            id=node.id,
                            opcode=OpCode.SWIGLU,
                            inputs=[other_in, node.inputs[0]],
                            outputs=list(next_node.outputs),
                            attributes=dict(next_node.attributes),
                        )
                        new_nodes.append(fused_node)
                        eliminated_node_indices.add(idx)
                        eliminated_node_indices.add(next_idx)
                        fused_count += 1
                        continue

            # If not fused, retain node
            new_nodes.append(node)

        # Build output graph
        new_graph = Graph(name=graph.name)
        new_graph.tensors = dict(graph.tensors)
        new_graph.inputs = list(graph.inputs)
        new_graph.outputs = list(graph.outputs)
        new_graph.parameters = list(graph.parameters)
        new_graph.states = list(graph.states)
        new_graph.nodes = new_nodes

        modified = fused_count > 0
        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=len(new_graph.nodes),
            tensors_before=tensors_before,
            tensors_after=len(new_graph.tensors),
            fusions_applied=fused_count,
        )

        return GraphTransformResult(graph=new_graph, modified=modified, stats=stats)
