"""Static memory arena planner for activation buffer reuse and zero-allocation inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ggmlc.ir.tensor import StorageClass
from ggmlc.memory.liveness import analyze_liveness, compute_tensor_byte_size

if TYPE_CHECKING:
    from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph
    from ggmlc.ir.graph import Graph


@dataclass
class MemoryPlan:
    """Static memory allocation plan with byte offsets for graph execution."""

    peak_activation_bytes: int
    unplanned_activation_bytes: int
    reuse_ratio: float
    persistent_bytes: int
    tensor_offsets: dict[int, int] = field(default_factory=dict)
    alignment: int = 32

    def summary(self) -> str:
        saved_mb = (self.unplanned_activation_bytes - self.peak_activation_bytes) / (1024 * 1024)
        pct = 0.0
        if self.unplanned_activation_bytes > 0:
            pct = (1.0 - (self.peak_activation_bytes / self.unplanned_activation_bytes)) * 100.0
        return (
            f"MemoryPlan: {self.unplanned_activation_bytes / (1024 * 1024):.2f} MB -> "
            f"{self.peak_activation_bytes / (1024 * 1024):.2f} MB activation arena "
            f"({self.reuse_ratio:.2f}x reuse, -{pct:.1f}%, saved {saved_mb:.2f} MB), "
            f"persistent: {self.persistent_bytes / (1024 * 1024):.2f} MB"
        )


def _align_offset(offset: int, alignment: int) -> int:
    """Rounds up offset to the nearest alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def plan_memory_arena(
    graph: Graph | GGMLExecutionGraph,
    symbol_env: dict[str, int] | None = None,
    alignment: int = 32,
) -> MemoryPlan:
    """Computes an optimal static memory allocation plan for intermediate activation tensors.

    Uses an interval-coloring first-fit greedy allocation strategy with strict byte alignment.
    """
    env = symbol_env or {}
    liveness_map = analyze_liveness(graph, env)

    persistent_bytes = 0
    unplanned_activation_bytes = 0

    # Calculate persistent tensor sizes and sum of unplanned activation sizes
    for tid, tensor in graph.tensors.items():
        sz = compute_tensor_byte_size(tensor, env)
        if tensor.storage in (StorageClass.PARAMETER, StorageClass.STATE):
            persistent_bytes += sz
        elif (
            tensor.storage == StorageClass.ACTIVATION or tid in graph.inputs or tid in graph.outputs
        ):
            unplanned_activation_bytes += sz

    # Collect reclaimable activation tensors
    reclaimable_tensors = [
        info for info in liveness_map.values() if info.is_reclaimable and info.size_bytes > 0
    ]

    # Sort by first use (start of lifetime), then descending by size
    reclaimable_tensors.sort(key=lambda t: (t.first_use, -t.size_bytes, t.tensor_id))

    # Active intervals: list of (start_offset, end_offset, last_use_op, tensor_id)
    active_intervals: list[tuple[int, int, int, int]] = []
    tensor_offsets: dict[int, int] = {}
    peak_offset = 0

    for item in reclaimable_tensors:
        curr_op = item.first_use

        # Free all intervals whose lifetime expired before curr_op
        active_intervals = [
            (start, end, last_use, tid)
            for start, end, last_use, tid in active_intervals
            if last_use >= curr_op
        ]

        # Find first fit offset that does not collide with any currently active intervals
        # Sort active intervals by start offset
        active_intervals.sort(key=lambda x: x[0])

        candidate_offset = 0
        allocated = False

        for start, end, _, _ in active_intervals:
            aligned_candidate = _align_offset(candidate_offset, alignment)
            if aligned_candidate + item.size_bytes <= start:
                # Found gap before current active interval
                tensor_offsets[item.tensor_id] = aligned_candidate
                active_intervals.append(
                    (
                        aligned_candidate,
                        aligned_candidate + item.size_bytes,
                        item.last_use,
                        item.tensor_id,
                    )
                )
                peak_offset = max(peak_offset, aligned_candidate + item.size_bytes)
                allocated = True
                break
            else:
                candidate_offset = max(candidate_offset, end)

        if not allocated:
            aligned_candidate = _align_offset(candidate_offset, alignment)
            tensor_offsets[item.tensor_id] = aligned_candidate
            active_intervals.append(
                (
                    aligned_candidate,
                    aligned_candidate + item.size_bytes,
                    item.last_use,
                    item.tensor_id,
                )
            )
            peak_offset = max(peak_offset, aligned_candidate + item.size_bytes)

    peak_activation_bytes = _align_offset(peak_offset, alignment)
    reuse_ratio = (
        unplanned_activation_bytes / max(1, peak_activation_bytes)
        if peak_activation_bytes > 0
        else 1.0
    )

    return MemoryPlan(
        peak_activation_bytes=peak_activation_bytes,
        unplanned_activation_bytes=unplanned_activation_bytes,
        reuse_ratio=reuse_ratio,
        persistent_bytes=persistent_bytes,
        tensor_offsets=tensor_offsets,
        alignment=alignment,
    )
