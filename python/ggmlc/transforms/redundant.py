"""Redundant Cast and Identity Operation Pruning Pass."""

from __future__ import annotations

from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode, Operation
from ggmlc.transforms.base import GraphTransformResult, Pass, PassStats


class RedundantCastPruner(Pass):
    """Eliminates redundant identity operations such as self-transpositions and collapses unsqueeze-transpose-squeeze chains."""

    def __init__(self):
        super().__init__(name="RedundantCastPruner")

    def run(self, graph: Graph) -> GraphTransformResult:
        nodes_before = len(graph.nodes)
        tensors_before = len(graph.tensors)

        # 1. Collapse unsqueeze -> transpose/permute -> squeeze patterns to direct permute
        fused_shape_ops = self._fuse_unsqueeze_transpose_squeeze(graph)

        # 2. Prune identity transpositions and permutations
        remap: dict[int, int] = {}
        nodes_to_keep = []
        eliminated = fused_shape_ops

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

            # Pattern 3: CAST same dtype or integer input adaptation
            elif node.opcode == OpCode.CAST:
                in_id = node.inputs[0]
                out_id = node.outputs[0]
                in_t = graph.get_tensor(in_id)
                out_t = graph.get_tensor(out_id)
                if in_t and out_t:
                    if in_t.dtype == out_t.dtype:
                        is_identity = True
                    elif (
                        in_id in graph.inputs
                        and in_t.dtype.name.startswith("I")
                        and out_t.dtype.name.startswith("I")
                    ):
                        in_t.dtype = out_t.dtype
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

    @staticmethod
    def _fuse_unsqueeze_transpose_squeeze(graph: Graph) -> int:
        consumers: dict[int, list[Operation]] = {}
        for n in graph.nodes:
            for in_id in n.inputs:
                consumers.setdefault(in_id, []).append(n)

        eliminated_node_ids: set[int] = set()
        new_permute_nodes: list[tuple[int, Operation]] = []
        eliminated_count = 0

        for n_u in graph.nodes:
            if n_u.opcode != OpCode.UNSQUEEZE or n_u.id in eliminated_node_ids:
                continue
            if len(n_u.inputs) != 1 or len(n_u.outputs) != 1:
                continue
            in_t = graph.get_tensor(n_u.inputs[0])
            R = len(in_t.shape.dims)
            u_dim = n_u.attributes.get("dim", 0)
            if u_dim < 0:
                u_dim += R + 1
            A = list(range(R))
            A.insert(u_dim, -1)

            out_u = n_u.outputs[0]
            cons_u = consumers.get(out_u, [])
            if len(cons_u) != 1:
                continue
            n_t = cons_u[0]
            if n_t.opcode not in (OpCode.TRANSPOSE, OpCode.PERMUTE):
                continue
            if n_t.opcode == OpCode.TRANSPOSE:
                d0 = n_t.attributes.get("dim0", 0)
                d1 = n_t.attributes.get("dim1", 1)
                if d0 < 0:
                    d0 += len(A)
                if d1 < 0:
                    d1 += len(A)
                if d0 < len(A) and d1 < len(A):
                    A[d0], A[d1] = A[d1], A[d0]
                else:
                    continue
            else:
                p = n_t.attributes.get("dims") or n_t.attributes.get("permutation")
                if not p or len(p) != len(A):
                    continue
                A = [A[i] for i in p]

            out_t = n_t.outputs[0]
            cons_t = consumers.get(out_t, [])
            if len(cons_t) != 1:
                continue
            n_s = cons_t[0]
            if n_s.opcode != OpCode.SQUEEZE:
                continue
            s_dim = n_s.attributes.get("dim", 0)
            if s_dim < 0:
                s_dim += len(A)
            if 0 <= s_dim < len(A) and A[s_dim] == -1:
                del A[s_dim]
                eliminated_node_ids.add(n_u.id)
                eliminated_node_ids.add(n_t.id)
                eliminated_node_ids.add(n_s.id)
                eliminated_count += 2

                out_s = n_s.outputs[0]
                perm_node = Operation(
                    id=n_u.id,
                    name=f"{n_u.name}_fused_permute",
                    opcode=OpCode.PERMUTE,
                    inputs=[n_u.inputs[0]],
                    outputs=[out_s],
                    attributes={"dims": A},
                )
                new_permute_nodes.append((n_u.id, perm_node))

        if eliminated_node_ids:
            perm_dict = dict(new_permute_nodes)
            new_nodes = []
            for n in graph.nodes:
                if n.id in perm_dict:
                    new_nodes.append(perm_dict[n.id])
                elif n.id not in eliminated_node_ids:
                    new_nodes.append(n)
            graph.nodes = new_nodes

        return eliminated_count
