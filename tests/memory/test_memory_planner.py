"""Unit tests for liveness analysis and static memory arena planner."""

from __future__ import annotations

import torch
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape, StaticDim
from ggmlc.ir.tensor import StorageClass
from ggmlc.memory.liveness import analyze_liveness
from ggmlc.memory.planner import plan_memory_arena
from torch import nn


def test_liveness_basic_chain():
    """Verify first_use and last_use intervals in a sequential graph."""
    g = Graph("test_chain")
    x = g.add_tensor("x", Shape([StaticDim(1), StaticDim(64)]), DType.F32, StorageClass.INPUT)
    t1 = g.add_tensor(
        "t1", Shape([StaticDim(1), StaticDim(64)]), DType.F32, StorageClass.ACTIVATION
    )
    t2 = g.add_tensor(
        "t2", Shape([StaticDim(1), StaticDim(64)]), DType.F32, StorageClass.ACTIVATION
    )
    out = g.add_tensor("out", Shape([StaticDim(1), StaticDim(64)]), DType.F32, StorageClass.OUTPUT)

    g.inputs = [x.id]
    g.outputs = [out.id]

    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[t1.id])  # op 0
    g.add_node(OpCode.RELU, inputs=[t1.id], outputs=[t2.id])  # op 1
    g.add_node(OpCode.RELU, inputs=[t2.id], outputs=[out.id])  # op 2

    liveness = analyze_liveness(g)

    # t1 is produced at op 0, consumed at op 1 -> lifetime [0, 1]
    assert liveness[t1.id].first_use == 0
    assert liveness[t1.id].last_use == 1

    # t2 is produced at op 1, consumed at op 2 -> lifetime [1, 2]
    assert liveness[t2.id].first_use == 1
    assert liveness[t2.id].last_use == 2

    # out is graph output -> lives until end of execution (op 3)
    assert liveness[out.id].last_use == 3


def test_memory_arena_reuse_disjoint_tensors():
    """Verify that non-overlapping activation buffers reuse the same memory offset."""
    g = Graph("test_reuse")
    x = g.add_tensor("x", Shape([StaticDim(1), StaticDim(1024)]), DType.F32, StorageClass.INPUT)
    # Stage 1: t1, t2
    t1 = g.add_tensor(
        "t1", Shape([StaticDim(1), StaticDim(1024)]), DType.F32, StorageClass.ACTIVATION
    )
    # Stage 2: t2 uses t1, so t1 dies after stage 2
    t2 = g.add_tensor(
        "t2", Shape([StaticDim(1), StaticDim(1024)]), DType.F32, StorageClass.ACTIVATION
    )
    # Stage 3: t3 uses t2, so t1's memory is free and can be reused by t3!
    t3 = g.add_tensor(
        "t3", Shape([StaticDim(1), StaticDim(1024)]), DType.F32, StorageClass.ACTIVATION
    )
    out = g.add_tensor(
        "out", Shape([StaticDim(1), StaticDim(1024)]), DType.F32, StorageClass.OUTPUT
    )

    g.inputs = [x.id]
    g.outputs = [out.id]

    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[t1.id])  # op 0
    g.add_node(OpCode.RELU, inputs=[t1.id], outputs=[t2.id])  # op 1: t1 dies
    g.add_node(
        OpCode.RELU, inputs=[t2.id], outputs=[t3.id]
    )  # op 2: t2 dies, t3 can reuse t1's offset
    g.add_node(OpCode.RELU, inputs=[t3.id], outputs=[out.id])  # op 3

    plan = plan_memory_arena(g, alignment=32)

    assert t1.id in plan.tensor_offsets
    assert t3.id in plan.tensor_offsets
    # t1 and t3 have disjoint lifetimes ([0, 1] vs [2, 3]), so they should share the exact same offset!
    assert plan.tensor_offsets[t1.id] == plan.tensor_offsets[t3.id]
    assert plan.reuse_ratio > 1.0


def test_memory_arena_no_collision_overlapping():
    """Verify that simultaneously live activation buffers never collide in memory."""
    g = Graph("test_overlap")
    x = g.add_tensor("x", Shape([StaticDim(1), StaticDim(512)]), DType.F32, StorageClass.INPUT)
    # Branch 1
    b1 = g.add_tensor(
        "b1", Shape([StaticDim(1), StaticDim(512)]), DType.F32, StorageClass.ACTIVATION
    )
    # Branch 2
    b2 = g.add_tensor(
        "b2", Shape([StaticDim(1), StaticDim(512)]), DType.F32, StorageClass.ACTIVATION
    )
    # Join
    out = g.add_tensor("out", Shape([StaticDim(1), StaticDim(512)]), DType.F32, StorageClass.OUTPUT)

    g.inputs = [x.id]
    g.outputs = [out.id]

    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[b1.id])  # op 0: b1 born
    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[b2.id])  # op 1: b2 born, b1 still alive
    g.add_node(OpCode.ADD, inputs=[b1.id, b2.id], outputs=[out.id])  # op 2: b1, b2 both consumed

    plan = plan_memory_arena(g, alignment=32)

    # b1 and b2 are simultaneously alive at op 2, so their intervals must NOT overlap
    off1 = plan.tensor_offsets[b1.id]
    off2 = plan.tensor_offsets[b2.id]
    size1 = 512 * 4
    size2 = 512 * 4

    assert (off1 + size1 <= off2) or (off2 + size2 <= off1)


def test_memory_planner_alignment():
    """Verify that all planned offsets are strictly 32-byte aligned."""
    g = Graph("test_align")
    x = g.add_tensor("x", Shape([StaticDim(1), StaticDim(33)]), DType.F32, StorageClass.INPUT)
    t1 = g.add_tensor(
        "t1", Shape([StaticDim(1), StaticDim(33)]), DType.F32, StorageClass.ACTIVATION
    )
    t2 = g.add_tensor(
        "t2", Shape([StaticDim(1), StaticDim(33)]), DType.F32, StorageClass.ACTIVATION
    )
    out = g.add_tensor("out", Shape([StaticDim(1), StaticDim(33)]), DType.F32, StorageClass.OUTPUT)

    g.inputs = [x.id]
    g.outputs = [out.id]

    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[t1.id])
    g.add_node(OpCode.RELU, inputs=[x.id], outputs=[t2.id])
    g.add_node(OpCode.ADD, inputs=[t1.id, t2.id], outputs=[out.id])

    plan = plan_memory_arena(g, alignment=32)

    for tid, offset in plan.tensor_offsets.items():
        assert offset % 32 == 0, f"Tensor {tid} offset {offset} is not 32-byte aligned"


def test_memory_planner_on_real_model():
    """Verify memory planning on a multi-layer deep network and assert substantial memory reuse."""

    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(8)])
            self.act = nn.GELU()

        def forward(self, x):
            for layer in self.layers:
                x = self.act(layer(x))
            return x

    model = DeepMLP().eval()
    x = torch.randn(1, 256)
    exported = export_torch_model(model, (x,), model_name="deep_mlp")

    plan = plan_memory_arena(exported.main_graph, alignment=32)

    print(plan.summary())
    assert plan.peak_activation_bytes < plan.unplanned_activation_bytes
    assert plan.reuse_ratio > 1.5
