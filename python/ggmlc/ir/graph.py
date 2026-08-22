from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ggmlc.ir.dtype import DType
from ggmlc.ir.op import OpCode, Operation
from ggmlc.ir.shape import Shape
from ggmlc.ir.state import StateDeclaration
from ggmlc.ir.tensor import StorageClass, Tensor


@dataclass
class Graph:
    """Canonical Intermediate Representation Graph."""

    name: str = "main"
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    parameters: list[int] = field(default_factory=list)
    states: list[StateDeclaration] = field(default_factory=list)
    nodes: list[Operation] = field(default_factory=list)
    tensors: dict[int, Tensor] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    _next_tensor_id: int = 0
    _next_op_id: int = 0

    def new_tensor_id(self) -> int:
        tid = self._next_tensor_id
        self._next_tensor_id += 1
        return tid

    def new_op_id(self) -> int:
        oid = self._next_op_id
        self._next_op_id += 1
        return oid

    def add_tensor(
        self,
        name: str,
        shape: Shape,
        dtype: DType,
        storage: StorageClass,
        producer_id: int | None = None,
        data: any | None = None,
        role: str | None = None,
        tensor_id: int | None = None,
    ) -> Tensor:
        if tensor_id is None:
            tensor_id = self.new_tensor_id()
        else:
            self._next_tensor_id = max(self._next_tensor_id, tensor_id + 1)

        t = Tensor(
            id=tensor_id,
            name=name,
            shape=shape,
            dtype=dtype,
            storage=storage,
            producer_id=producer_id,
            data=data,
            role=role,
        )
        self.tensors[tensor_id] = t
        return t

    def add_op(
        self,
        opcode: OpCode,
        inputs: list[int],
        outputs: list[int],
        attributes: dict[str, any] | None = None,
        name: str | None = None,
        op_id: int | None = None,
    ) -> Operation:
        if op_id is None:
            op_id = self.new_op_id()
        else:
            self._next_op_id = max(self._next_op_id, op_id + 1)

        op = Operation(
            id=op_id,
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            attributes=attributes or {},
            name=name,
        )
        for out_id in outputs:
            if out_id in self.tensors:
                self.tensors[out_id].producer_id = op_id

        self.nodes.append(op)
        return op

    add_node = add_op

    def get_tensor(self, tid: int) -> Tensor:
        if tid not in self.tensors:
            raise KeyError(f"Tensor with ID {tid} does not exist in graph '{self.name}'")
        return self.tensors[tid]

    def validate_invariants(self) -> None:
        """Validate topological ordering, connectivity, and storage class invariants."""
        # 1. Inputs, outputs, parameters exist
        for tid in self.inputs:
            t = self.get_tensor(tid)
            if t.storage != StorageClass.INPUT:
                raise ValueError(f"Input tensor {t} must have StorageClass.INPUT")

        for tid in self.parameters:
            t = self.get_tensor(tid)
            if t.storage not in (StorageClass.PARAMETER, StorageClass.CONSTANT):
                raise ValueError(f"Parameter tensor {t} must have PARAMETER or CONSTANT storage")

        for tid in self.outputs:
            self.get_tensor(tid)

        # 2. Check SSA and topological ordering
        defined_tensors: set[int] = set(self.inputs) | set(self.parameters)
        for state in self.states:
            defined_tensors.add(state.id)

        for op in self.nodes:
            for in_id in op.inputs:
                if in_id not in defined_tensors:
                    raise ValueError(
                        f"Op {op} references tensor {in_id} before it is defined or declared."
                    )
            for out_id in op.outputs:
                if out_id in defined_tensors:
                    raise ValueError(f"Op {op} redefines existing tensor {out_id} (violates SSA).")
                defined_tensors.add(out_id)

    def summary(self) -> str:
        lines = [
            f"Graph '{self.name}':",
            f"  Inputs ({len(self.inputs)}): {[self.get_tensor(i).name for i in self.inputs]}",
            f"  Outputs ({len(self.outputs)}): {[self.get_tensor(i).name for i in self.outputs]}",
            f"  Parameters ({len(self.parameters)}): {len(self.parameters)} tensors",
            f"  States ({len(self.states)}): {[s.name for s in self.states]}",
            f"  Operations ({len(self.nodes)}):",
        ]
        for op in self.nodes:
            in_names = [self.get_tensor(i).name for i in op.inputs]
            out_names = [self.get_tensor(i).name for i in op.outputs]
            lines.append(f"    %{','.join(out_names)} = {op.opcode.name}(%{','.join(in_names)})")
        return "\n".join(lines)

    def to_mermaid(self, title: str | None = None, direction: str = "TD") -> str:
        """Exports the graph to Mermaid flowchart diagram code."""
        from ggmlc.visualization.mermaid import graph_to_mermaid

        return graph_to_mermaid(self, title=title or self.name, direction=direction)

    def visualize(self, output_path: str = "graph.html", format: str = "auto") -> Path:
        """Visualizes the graph to HTML, Mermaid markdown, or image."""
        from ggmlc.visualization.mermaid import visualize

        return visualize(self, output_path=output_path, format=format, title=self.name)
