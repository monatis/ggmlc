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

    if options.enable_layer_norm:
        _fuse_layer_norm_patterns(graph)

    if options.enable_rms_norm:
        _fuse_rms_norm_patterns(graph)

    if options.enable_softmax:
        _fuse_softmax_patterns(graph)

    if options.enable_conv2d_relu:
        _fuse_conv2d_relu_patterns(graph)

    if options.enable_swiglu:
        _fuse_swiglu_patterns(graph)

    if options.enable_bias_gelu:
        _fuse_bias_gelu_patterns(graph)

    return graph


def _fuse_layer_norm_patterns(graph: Graph) -> None:
    """Matches decomposed LayerNorm subgraphs (e.g. from JAX/XLA) and fuses into LAYER_NORM."""
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

        if op.opcode == OpCode.ADD and len(op.inputs) == 2:
            in0, in1 = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0)
            prod1 = producer_map.get(in1)

            mul_x_op = None
            bias_term_op = None
            if prod0 and prod0.opcode == OpCode.MUL:
                mul_x_op = prod0
                bias_term_op = prod1
            elif prod1 and prod1.opcode == OpCode.MUL:
                mul_x_op = prod1
                bias_term_op = prod0

            if mul_x_op is not None and len(mul_x_op.inputs) == 2:
                cand_x_0, cand_rstd_gamma_0 = mul_x_op.inputs[0], mul_x_op.inputs[1]
                prod_rg = producer_map.get(cand_rstd_gamma_0)
                x_id = cand_x_0
                if prod_rg is None or prod_rg.opcode != OpCode.MUL:
                    prod_rg = producer_map.get(cand_x_0)
                    x_id = cand_rstd_gamma_0

                # If x_id is SUB(x, mean), extract true input x
                prod_x = producer_map.get(x_id)
                if prod_x and prod_x.opcode == OpCode.SUB and len(prod_x.inputs) == 2:
                    x_id = prod_x.inputs[0]
                    prod_x = producer_map.get(x_id)

                # Ensure x_id is valid and rank <= 3
                x_t = graph.get_tensor(x_id) if x_id in graph.tensors else None
                is_valid_x = prod_x is None or prod_x.opcode not in (
                    OpCode.NEG,
                    OpCode.DIV,
                    OpCode.SUM,
                    OpCode.RSQRT,
                )
                is_valid_rank = x_t is not None and len(x_t.shape.dims) <= 3

                if (
                    is_valid_x
                    and is_valid_rank
                    and prod_rg is not None
                    and prod_rg.opcode == OpCode.MUL
                ):
                    rg_in0, rg_in1 = prod_rg.inputs[0], prod_rg.inputs[1]
                    rsqrt_op = producer_map.get(rg_in0)
                    gamma_id = rg_in1
                    if rsqrt_op is None or rsqrt_op.opcode != OpCode.RSQRT:
                        rsqrt_op = producer_map.get(rg_in1)
                        gamma_id = rg_in0

                    if rsqrt_op is not None and rsqrt_op.opcode == OpCode.RSQRT:
                        var_add_op = producer_map.get(rsqrt_op.inputs[0])
                        eps = 1e-5
                        if var_add_op and var_add_op.opcode == OpCode.ADD:
                            for inp_t_id in var_add_op.inputs:
                                t = graph.get_tensor(inp_t_id)
                                if (
                                    t
                                    and t.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)
                                    and t.data is not None
                                    and float(t.data) > 0.0
                                ):
                                    eps = float(t.data)

                        # Verify that var_add_op is a direct reduction of x_id (not across conv/matmul layers)
                        is_direct_norm = False
                        curr = rsqrt_op.inputs[0] if rsqrt_op.inputs else None
                        for _ in range(6):
                            if curr == x_id:
                                is_direct_norm = True
                                break
                            p = producer_map.get(curr)
                            if not p or p.opcode in (OpCode.CONV2D, OpCode.MATMUL, OpCode.LINEAR):
                                break
                            if x_id in p.inputs:
                                is_direct_norm = True
                                break
                            curr = p.inputs[0] if p.inputs else None

                        if is_direct_norm:
                            beta_id = None
                            if bias_term_op and bias_term_op.opcode == OpCode.ADD:
                                b_in0, b_in1 = bias_term_op.inputs[0], bias_term_op.inputs[1]
                                t0 = graph.get_tensor(b_in0)
                                t1 = graph.get_tensor(b_in1)
                                if t1 and t1.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = b_in1
                                elif t0 and t0.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = b_in0
                            elif bias_term_op is None:
                                # Direct beta parameter (e.g. in Flax)
                                other_in = in1 if mul_x_op == prod0 else in0
                                t_other = graph.get_tensor(other_in)
                                if t_other and t_other.storage in (
                                    StorageClass.PARAMETER,
                                    StorageClass.CONSTANT,
                                ):
                                    beta_id = other_in

                            # Squeeze/match gamma and beta shapes to 1D if needed
                            for param_cand_id in (gamma_id, beta_id):
                                if param_cand_id is not None:
                                    p_t = graph.get_tensor(param_cand_id)
                                    if p_t and len(p_t.shape.dims) > 1:
                                        # Reshape to 1D
                                        from ggmlc.ir.shape import Shape

                                        last_d = p_t.shape.dims[-1]
                                        p_t.shape = Shape([last_d])
                                        if p_t.data is not None:
                                            p_t.data = p_t.data.reshape(-1)

                            ops_to_remove.add(op.id)

                            ln_inputs = [x_id]
                            if gamma_id is not None:
                                ln_inputs.append(gamma_id)
                            if beta_id is not None:
                                ln_inputs.append(beta_id)

                            fused_id = graph.new_op_id()
                            fused_op = Operation(
                                id=fused_id,
                                opcode=OpCode.LAYER_NORM,
                                inputs=ln_inputs,
                                outputs=list(op.outputs),
                                attributes={"eps": eps},
                                name=f"{op.name or 'layer_norm'}_fused",
                            )
                            new_nodes.append(fused_op)
                            continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


def _fuse_rms_norm_patterns(graph: Graph) -> None:
    """Matches decomposed RMSNorm subgraphs (e.g. from JAX/XLA) and fuses into RMS_NORM."""
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
            in0, in1 = op.inputs[0], op.inputs[1]
            prod0 = producer_map.get(in0)
            prod1 = producer_map.get(in1)

            # RMSNorm output is MUL(MUL(x, rstd), gamma) or MUL(x, MUL(rstd, gamma))
            rstd_op = None
            x_id = None
            gamma_id = None

            if prod0 and prod0.opcode == OpCode.MUL:
                # in0 is MUL(x, rstd) or MUL(rstd, gamma)
                gamma_cand = graph.get_tensor(in1)
                if gamma_cand and gamma_cand.storage in (
                    StorageClass.PARAMETER,
                    StorageClass.CONSTANT,
                ):
                    gamma_id = in1
                    sub_prod0 = producer_map.get(prod0.inputs[0])
                    sub_prod1 = producer_map.get(prod0.inputs[1])
                    if sub_prod0 and sub_prod0.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod0
                        x_id = prod0.inputs[1]
                    elif sub_prod1 and sub_prod1.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod1
                        x_id = prod0.inputs[0]
            elif prod1 and prod1.opcode == OpCode.MUL:
                gamma_cand = graph.get_tensor(in0)
                if gamma_cand and gamma_cand.storage in (
                    StorageClass.PARAMETER,
                    StorageClass.CONSTANT,
                ):
                    gamma_id = in0
                    sub_prod0 = producer_map.get(prod1.inputs[0])
                    sub_prod1 = producer_map.get(prod1.inputs[1])
                    if sub_prod0 and sub_prod0.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod0
                        x_id = prod1.inputs[1]
                    elif sub_prod1 and sub_prod1.opcode == OpCode.RSQRT:
                        rstd_op = sub_prod1
                        x_id = prod1.inputs[0]

            if rstd_op is not None and x_id is not None and gamma_id is not None:
                # Ensure tensor is 1D, 2D, or 3D where innermost dimension is ne0
                x_t = graph.get_tensor(x_id)
                if x_t and len(x_t.shape.dims) <= 3:
                    var_add_op = producer_map.get(rstd_op.inputs[0])
                    eps = 1e-5
                    if var_add_op and var_add_op.opcode == OpCode.ADD:
                        for inp_t_id in var_add_op.inputs:
                            t = graph.get_tensor(inp_t_id)
                            if (
                                t
                                and t.storage in (StorageClass.CONSTANT, StorageClass.PARAMETER)
                                and t.data is not None
                                and float(t.data) > 0.0
                            ):
                                eps = float(t.data)

                    # Verify that var_add_op is a direct reduction of x_id
                    is_direct_norm = False
                    curr = rstd_op.inputs[0] if rstd_op.inputs else None
                    for _ in range(6):
                        if curr == x_id:
                            is_direct_norm = True
                            break
                        p = producer_map.get(curr)
                        if not p or p.opcode in (OpCode.CONV2D, OpCode.MATMUL, OpCode.LINEAR):
                            break
                        if x_id in p.inputs:
                            is_direct_norm = True
                            break
                        curr = p.inputs[0] if p.inputs else None

                    if is_direct_norm:
                        # Reshape gamma to 1D if needed
                        p_t = graph.get_tensor(gamma_id)
                        if p_t and len(p_t.shape.dims) > 1:
                            from ggmlc.ir.shape import Shape

                            last_d = p_t.shape.dims[-1]
                            p_t.shape = Shape([last_d])
                            if p_t.data is not None:
                                p_t.data = p_t.data.reshape(-1)

                        ops_to_remove.add(op.id)

                        fused_id = graph.new_op_id()
                        fused_op = Operation(
                            id=fused_id,
                            opcode=OpCode.RMS_NORM,
                            inputs=[x_id, gamma_id],
                            outputs=list(op.outputs),
                            attributes={"eps": eps},
                            name=f"{op.name or 'rms_norm'}_fused",
                        )
                        new_nodes.append(fused_op)
                        continue

        new_nodes.append(op)

    graph.nodes = [n for n in new_nodes if n.id not in ops_to_remove]


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
                gate_t = graph.get_tensor(gate_id)
                up_t = graph.get_tensor(up_id)
                # SwiGLU is only valid for identical shape MLP projections (not Squeeze-and-Excitation or broadcasted gates)
                if (
                    gate_t is not None
                    and up_t is not None
                    and gate_t.shape == up_t.shape
                    and (prod0 is None or prod0.opcode != OpCode.SIGMOID)
                    and (prod1 is None or prod1.opcode != OpCode.SIGMOID)
                ):
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

            denom_chain: list[int] = []
            curr_denom = producer_map.get(denom_id)
            sum_op = None
            while curr_denom and curr_denom.opcode in (
                OpCode.RESHAPE,
                OpCode.VIEW,
                OpCode.SQUEEZE,
                OpCode.UNSQUEEZE,
                OpCode.EXPAND,
                OpCode.REPEAT,
            ):
                denom_chain.append(curr_denom.id)
                next_op = producer_map.get(curr_denom.inputs[0])
                if next_op and next_op.opcode == OpCode.SUM:
                    sum_op = next_op
                    break
                curr_denom = next_op
            if not sum_op and curr_denom and curr_denom.opcode == OpCode.SUM:
                sum_op = curr_denom

            if sum_op and sum_op.inputs[0] == exp_op.outputs[0]:
                op.opcode = OpCode.SOFTMAX
                op.inputs = [x_id]
                op.attributes["dim"] = -1
                ops_to_remove.add(exp_op.id)
                if sub_op:
                    ops_to_remove.add(sub_op.id)
                    max_in = sub_op.inputs[1]
                    curr_max = producer_map.get(max_in)
                    while curr_max:
                        ops_to_remove.add(curr_max.id)
                        if curr_max.opcode in (
                            OpCode.RESHAPE,
                            OpCode.VIEW,
                            OpCode.SQUEEZE,
                            OpCode.UNSQUEEZE,
                            OpCode.EXPAND,
                            OpCode.REPEAT,
                        ):
                            curr_max = producer_map.get(curr_max.inputs[0])
                        elif curr_max.opcode in (OpCode.AMAX, OpCode.MAXIMUM):
                            break
                        else:
                            break
                for d_id in denom_chain:
                    ops_to_remove.add(d_id)
                ops_to_remove.add(sum_op.id)

    if ops_to_remove:
        graph.nodes = [n for n in graph.nodes if n.id not in ops_to_remove]
