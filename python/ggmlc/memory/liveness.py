"""Tensor liveness analysis for static and dynamic execution graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ggmlc.ir.dtype import DType
from ggmlc.ir.tensor import StorageClass

if TYPE_CHECKING:
    from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph
    from ggmlc.ir.graph import Graph


@dataclass
class TensorLiveness:
    """Liveness interval and metadata for a single tensor."""

    tensor_id: int
    name: str
    storage: StorageClass
    dtype: DType
    first_use: int
    last_use: int
    size_bytes: int
    is_reclaimable: bool


def compute_tensor_byte_size(tensor, symbol_env: dict[str, int] | None = None) -> int:
    """Calculates concrete byte size of a tensor given an optional symbol environment."""
    env = symbol_env or {}
    numel = 1

    if hasattr(tensor, "shape"):
        for d in tensor.shape.dims:
            numel *= d.evaluate(env)
    elif hasattr(tensor, "ne"):
        for d in tensor.ne:
            numel *= d.evaluate(env)

    # Calculate bytes based on dtype or ggml_type
    dt = getattr(tensor, "dtype", None)
    if dt is not None and isinstance(dt, DType):
        if dt == DType.Q8_0:
            return (numel // 32) * 34
        elif dt == DType.Q4_0:
            return (numel // 32) * 18
        return int(numel * dt.itemsize)

    # Fallback to 4 bytes per element
    return numel * 4


def analyze_liveness(
    graph: Graph | GGMLExecutionGraph,
    symbol_env: dict[str, int] | None = None,
) -> dict[int, TensorLiveness]:
    """Analyzes the lifetime of all tensors in topological operation execution order.

    Returns:
        Mapping of tensor_id -> TensorLiveness.
    """
    env = symbol_env or {}
    liveness_map: dict[int, TensorLiveness] = {}

    nodes = graph.nodes if hasattr(graph, "nodes") else getattr(graph, "ops", [])
    num_ops = len(nodes)

    # Initialize all tensors
    for tid, tensor in graph.tensors.items():
        is_reclaimable = tensor.storage in (StorageClass.ACTIVATION, StorageClass.CONSTANT)
        # Graph inputs and outputs must remain valid across the entire invocation
        if tid in graph.inputs or tid in graph.outputs:
            is_reclaimable = False
        # Parameters and persistent states are never part of ephemeral activation reuse
        if tensor.storage in (StorageClass.PARAMETER, StorageClass.STATE):
            is_reclaimable = False

        size_bytes = compute_tensor_byte_size(tensor, env)

        # Default intervals
        first_use = (
            0
            if tensor.storage in (StorageClass.INPUT, StorageClass.PARAMETER, StorageClass.STATE)
            else num_ops
        )
        last_use = num_ops if not is_reclaimable else 0

        dtype = getattr(tensor, "dtype", DType.F32)
        liveness_map[tid] = TensorLiveness(
            tensor_id=tid,
            name=tensor.name,
            storage=tensor.storage,
            dtype=dtype,
            first_use=first_use,
            last_use=last_use,
            size_bytes=size_bytes,
            is_reclaimable=is_reclaimable,
        )

    # Scan operations in topological order
    for op_idx, op in enumerate(nodes):
        # Update produced outputs
        for out_id in op.outputs:
            if out_id in liveness_map:
                liveness_map[out_id].first_use = min(liveness_map[out_id].first_use, op_idx)
                liveness_map[out_id].last_use = max(liveness_map[out_id].last_use, op_idx)

        # Update consumed inputs
        for in_id in op.inputs:
            if in_id in liveness_map:
                liveness_map[in_id].last_use = max(liveness_map[in_id].last_use, op_idx)

    # Graph outputs must live until the end of execution
    for out_id in graph.outputs:
        if out_id in liveness_map:
            liveness_map[out_id].last_use = num_ops

    return liveness_map
