"""Graph transformation passes for targeted operator fusion (ggmlc-fused)."""

from __future__ import annotations

from dataclasses import dataclass

from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


@dataclass
class FusionOptions:
    """Configuration options for enabling or disabling individual fusion passes."""

    enable_bias_gelu: bool = True
    enable_layer_norm: bool = True
    enable_rms_norm: bool = True
    enable_swiglu: bool = True
    enable_conv2d_relu: bool = True
    enable_softmax: bool = True


class OperatorFusionPass(Pass):
    """Compiler transformation pass that fuses eligible operator patterns."""

    def __init__(self, options: FusionOptions | None = None, name: str = "OperatorFusionPass"):
        super().__init__(name)
        self.options = options or FusionOptions()

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        fuse_operations(graph, self.options)

        nodes_after = len(graph.nodes)
        tensors_after = len(graph.tensors)
        fusions_applied = max(0, nodes_before - nodes_after)

        stats = PassStats(
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            tensors_before=tensors_before,
            tensors_after=tensors_after,
            fusions_applied=fusions_applied,
        )
        return GraphTransformResult(
            graph=graph,
            modified=(nodes_before != nodes_after),
            stats=stats,
        )


def fuse_operations(graph: Graph, options: FusionOptions | None = None) -> Graph:
    """Applies pattern-matching fusion rewrites to a Canonical IR Graph.

    Returns a new or modified Graph with fused operations where applicable.
    """
    if options is None:
        options = FusionOptions()

    if options.enable_softmax:
        _fuse_softmax_patterns(graph)

    if options.enable_conv2d_relu:
        _fuse_conv2d_relu_patterns(graph)

    if options.enable_swiglu:
        _fuse_swiglu_patterns(graph)

    if options.enable_bias_gelu:
        _fuse_bias_gelu_patterns(graph)

    return graph


def _fuse_conv2d_relu_patterns(graph: Graph) -> None:
    """Fuses CONV2D followed by RELU into a single CONV2D with fused_activation attribute."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()

    for op in graph.nodes:
        if op.opcode == OpCode.RELU and len(op.inputs) == 1:
            inp_id = op.inputs[0]
            prod = producer_map.get(inp_id)
            if (
                prod
                and prod.opcode == OpCode.CONV2D
                and consumer_counts.get(prod.outputs[0], 0) <= 1
            ):
                prod.attributes["fused_relu"] = True
                prod.attributes["fused_activation"] = "relu"
                prod.outputs = list(op.outputs)
                if op.outputs[0] in graph.tensors:
                    graph.tensors[op.outputs[0]].producer_id = prod.id
                ops_to_remove.add(op.id)

    graph.nodes = [n for n in graph.nodes if n.id not in ops_to_remove]


def _fuse_swiglu_patterns(graph: Graph) -> None:
    """Matches Silu(gate) * up and replaces with SWIGLU(gate, up)."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.MUL and len(op.inputs) == 2:
            in0_id, in1_id = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0_id)
            prod1 = producer_map.get(in1_id)

            gate_id = None
            up_id = None
            silu_op = None

            if prod0 and prod0.opcode == OpCode.SILU:
                silu_op = prod0
                gate_id = prod0.inputs[0]
                up_id = in1_id
            elif prod1 and prod1.opcode == OpCode.SILU:
                silu_op = prod1
                gate_id = prod1.inputs[0]
                up_id = in0_id

            if silu_op is not None and gate_id is not None and up_id is not None:
                # If the intermediate silu output is only consumed by this mul, we can prune it
                if consumer_counts.get(silu_op.outputs[0], 0) <= 1:
                    ops_to_remove.add(silu_op.id)

                fused_op = Operation(
                    id=op.id,
                    opcode=OpCode.SWIGLU,
                    inputs=[gate_id, up_id],
                    outputs=list(op.outputs),
                    attributes=dict(op.attributes),
                    name=f"{op.name or 'swiglu'}_fused",
                )
                new_nodes.append(fused_op)
                continue

        new_nodes.append(op)

    # Filter out nodes marked for removal
    final_nodes = [n for n in new_nodes if n.id not in ops_to_remove]
    graph.nodes = final_nodes


def _fuse_bias_gelu_patterns(graph: Graph) -> None:
    """Matches Linear(x, w, bias) -> GELU or Add(x, bias) -> GELU and fuses into BIAS_GELU."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()
    new_nodes: list[Operation] = []

    for op in graph.nodes:
        if op.id in ops_to_remove:
            continue

        if op.opcode == OpCode.GELU and len(op.inputs) == 1:
            inp_id = op.inputs[0]
            prod = producer_map.get(inp_id)

            if prod and prod.opcode == OpCode.LINEAR and len(prod.inputs) >= 3:
                # Linear(x, w, bias) -> GELU(out)
                # Split into Linear(x, w) and BiasGELU(linear_out, bias)
                x_id = prod.inputs[0]
                w_id = prod.inputs[1]
                b_id = prod.inputs[2]
                prod.inputs = [x_id, w_id]

                fused_op = Operation(
                    id=op.id,
                    opcode=OpCode.BIAS_GELU,
                    inputs=[prod.outputs[0], b_id],
                    outputs=list(op.outputs),
                    attributes=dict(op.attributes),
                    name=f"{op.name or 'bias_gelu'}_fused",
                )
                new_nodes.append(fused_op)
                continue

            elif prod and prod.opcode == OpCode.ADD and len(prod.inputs) == 2:
                # Check if one input is an activation and the other is a bias/parameter
                in0 = graph.get_tensor(prod.inputs[0])
                in1 = graph.get_tensor(prod.inputs[1])

                act_id = None
                bias_id = None
                if (
                    in1.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                    or len(in1.shape.dims) == 1
                ):
                    act_id = in0.id
                    bias_id = in1.id
                elif (
                    in0.storage in (StorageClass.PARAMETER, StorageClass.CONSTANT)
                    or len(in0.shape.dims) == 1
                ):
                    act_id = in1.id
                    bias_id = in0.id

                if act_id is not None and bias_id is not None:
                    if consumer_counts.get(prod.outputs[0], 0) <= 1:
                        ops_to_remove.add(prod.id)

                    fused_op = Operation(
                        id=op.id,
                        opcode=OpCode.BIAS_GELU,
                        inputs=[act_id, bias_id],
                        outputs=list(op.outputs),
                        attributes=dict(op.attributes),
                        name=f"{op.name or 'bias_gelu'}_fused",
                    )
                    new_nodes.append(fused_op)
                    continue

        new_nodes.append(op)

    final_nodes = [n for n in new_nodes if n.id not in ops_to_remove]
    graph.nodes = final_nodes


def _fuse_softmax_patterns(graph: Graph) -> None:
    """Matches decomposed softmax: div(exp(x - max(x)), sum(exp(x - max(x)))) -> SOFTMAX(x)."""
    producer_map: dict[int, Operation] = {}
    consumer_counts: dict[int, int] = {}
    for op in graph.nodes:
        for out_id in op.outputs:
            producer_map[out_id] = op
        for in_id in op.inputs:
            consumer_counts[in_id] = consumer_counts.get(in_id, 0) + 1

    ops_to_remove: set[int] = set()

    for op in graph.nodes:
        if op.opcode == OpCode.DIV and len(op.inputs) == 2:
            num_id, denom_id = op.inputs[0], op.inputs[1]
            exp_op = producer_map.get(num_id)
            if not exp_op or exp_op.opcode != OpCode.EXP:
                continue

            sub_op = producer_map.get(exp_op.inputs[0])
            if sub_op and sub_op.opcode == OpCode.SUB:
                x_id = sub_op.inputs[0]
            else:
                x_id = exp_op.inputs[0]

            denom_op = producer_map.get(denom_id)
            if denom_op and denom_op.opcode in (
                OpCode.RESHAPE,
                OpCode.VIEW,
                OpCode.SQUEEZE,
                OpCode.UNSQUEEZE,
            ):
                sum_op = producer_map.get(denom_op.inputs[0])
            else:
                sum_op = denom_op

            if sum_op and sum_op.opcode == OpCode.SUM:
                if sum_op.inputs[0] == exp_op.outputs[0]:
                    op.opcode = OpCode.SOFTMAX
                    op.inputs = [x_id]
                    op.attributes["dim"] = -1
                    ops_to_remove.add(exp_op.id)
                    if sub_op:
                        ops_to_remove.add(sub_op.id)
                        max_in = sub_op.inputs[1]
                        max_op = producer_map.get(max_in)
                        if max_op:
                            if max_op.opcode in (
                                OpCode.RESHAPE,
                                OpCode.VIEW,
                                OpCode.SQUEEZE,
                                OpCode.UNSQUEEZE,
                            ):
                                inner_max = producer_map.get(max_op.inputs[0])
                                if inner_max:
                                    ops_to_remove.add(inner_max.id)
                                ops_to_remove.add(max_op.id)
                            else:
                                ops_to_remove.add(max_op.id)
                    if denom_op and denom_op != sum_op:
                        ops_to_remove.add(denom_op.id)
                    ops_to_remove.add(sum_op.id)

    if ops_to_remove:
        graph.nodes = [n for n in graph.nodes if n.id not in ops_to_remove]
