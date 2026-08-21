"""Constant Folding Pass."""

from __future__ import annotations

import numpy as np

from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class ConstantFoldingPass(Pass):
    """Identifies and evaluates subgraphs composed entirely of static constants and parameters at compile time."""

    def __init__(self):
        super().__init__(name="ConstantFolding")

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        # Track known constant data arrays: tensor_id -> np.ndarray
        constant_values: dict[int, np.ndarray] = {}
        for tid, tensor in graph.tensors.items():
            if tensor.data is not None and tensor.storage == StorageClass.CONSTANT:
                constant_values[tid] = np.array(tensor.data)

        folded_count = 0
        remap_tensors: dict[int, int] = {}  # old_out_tid -> new_const_tid
        nodes_to_keep: list = []

        # Iterate through nodes
        for node in graph.nodes:
            # Check if all inputs have known constant arrays and no dynamic symbols
            all_const = (
                len(node.inputs) > 0
                and all(in_id in constant_values for in_id in node.inputs)
                and all(
                    not graph.tensors[in_id].shape.is_dynamic
                    for in_id in node.inputs
                    if in_id in graph.tensors
                )
            )

            # Check if opcode is foldable
            folded_val = None
            if all_const:
                in_arrays = [constant_values[in_id] for in_id in node.inputs]
                folded_val = self._evaluate_node(node.opcode, in_arrays, node.attributes)

            if folded_val is not None and len(node.outputs) == 1:
                out_id = node.outputs[0]
                state_tids = {s.id for s in graph.states}
                # If output is not a graph-level output or state
                if out_id not in graph.outputs and out_id not in state_tids:
                    # Record the folded array
                    constant_values[out_id] = folded_val
                    # Update the tensor in the graph to be a constant
                    t = graph.tensors[out_id]
                    t.storage = StorageClass.CONSTANT
                    t.data = (
                        folded_val.tolist() if isinstance(folded_val, np.ndarray) else [folded_val]
                    )
                    folded_count += 1
                    continue

            # Keep node if not folded
            nodes_to_keep.append(node)

        # Update remaining nodes if any tensor was remapped
        new_graph = Graph(name=graph.name)
        new_graph.tensors = dict(graph.tensors)
        new_graph.inputs = list(graph.inputs)
        new_graph.outputs = list(graph.outputs)
        new_graph.parameters = list(graph.parameters)
        new_graph.states = list(graph.states)

        for node in nodes_to_keep:
            new_inputs = [remap_tensors.get(i, i) for i in node.inputs]
            node.inputs = new_inputs
            new_graph.nodes.append(node)

        modified = folded_count > 0
        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=len(new_graph.nodes),
            tensors_before=tensors_before,
            tensors_after=len(new_graph.tensors),
            constants_folded=folded_count,
        )

        return GraphTransformResult(graph=new_graph, modified=modified, stats=stats)

    @staticmethod
    def _evaluate_node(opcode: OpCode, inputs: list[np.ndarray], attrs: dict) -> np.ndarray | None:
        """Evaluates an operation on constant NumPy arrays."""
        try:
            if opcode == OpCode.ADD:
                return inputs[0] + inputs[1]
            elif opcode == OpCode.SUB:
                return inputs[0] - inputs[1]
            elif opcode == OpCode.MUL:
                return inputs[0] * inputs[1]
            elif opcode == OpCode.DIV:
                return inputs[0] / inputs[1]
            elif opcode == OpCode.NEG:
                return -inputs[0]
            elif opcode == OpCode.POW:
                exp = attrs.get("exponent", inputs[1] if len(inputs) > 1 else 2)
                return np.power(inputs[0], exp)
            elif opcode == OpCode.SQRT:
                return np.sqrt(inputs[0])
            elif opcode == OpCode.RSQRT:
                return 1.0 / np.sqrt(inputs[0])
            elif opcode == OpCode.RESHAPE:
                target_shape = attrs.get("shape")
                if target_shape:
                    return inputs[0].reshape(target_shape)
            elif opcode == OpCode.TRANSPOSE:
                d0 = attrs.get("dim0", 0)
                d1 = attrs.get("dim1", 1)
                axes = list(range(inputs[0].ndim))
                axes[d0], axes[d1] = axes[d1], axes[d0]
                return np.transpose(inputs[0], axes)
            elif opcode == OpCode.PERMUTE:
                dims = attrs.get("dims")
                if dims:
                    return np.transpose(inputs[0], dims)
            elif opcode == OpCode.CONCAT:
                axis = attrs.get("axis", 0)
                return np.concatenate(inputs, axis=axis)
        except (ValueError, TypeError, ZeroDivisionError, FloatingPointError, IndexError):
            return None
        return None
