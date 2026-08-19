"""Unit and integration tests for compiler optimization passes."""

import numpy as np
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms import (
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    OperatorFusionPass,
    RedundantCastPruner,
    create_standard_optimization_pipeline,
)


def test_dead_code_elimination():
    """Verify that unused nodes and intermediate tensors are removed by DCE."""
    g = Graph("test_dce")
    t0 = g.add_tensor("x", Shape([4, 4]), DType.F32, StorageClass.INPUT)
    t1 = g.add_tensor("t1", Shape([4, 4]), DType.F32, StorageClass.ACTIVATION)
    t2 = g.add_tensor("t2", Shape([4, 4]), DType.F32, StorageClass.ACTIVATION)
    t_dead = g.add_tensor("dead", Shape([4, 4]), DType.F32, StorageClass.ACTIVATION)

    # Live path: t0 -> t1 -> t2 (output)
    g.add_node(OpCode.RELU, inputs=[t0.id], outputs=[t1.id])
    g.add_node(OpCode.GELU, inputs=[t1.id], outputs=[t2.id])
    # Dead path: t0 -> t_dead (unused)
    g.add_node(OpCode.SIGMOID, inputs=[t0.id], outputs=[t_dead.id])

    g.inputs = [t0.id]
    g.outputs = [t2.id]

    assert len(g.nodes) == 3
    assert len(g.tensors) == 4

    dce = DeadCodeEliminationPass()
    res = dce.run(g)

    assert res.modified is True
    assert len(res.graph.nodes) == 2
    assert len(res.graph.tensors) == 3
    assert t_dead.id not in res.graph.tensors
    assert res.stats.dead_nodes_pruned == 1


def test_constant_folding():
    """Verify compile-time evaluation of static math operations."""
    g = Graph("test_const_folding")
    c1 = g.add_tensor(
        "c1",
        Shape([2, 2]),
        DType.F32,
        StorageClass.CONSTANT,
        data=[[1.0, 2.0], [3.0, 4.0]],
    )
    c2 = g.add_tensor(
        "c2",
        Shape([2, 2]),
        DType.F32,
        StorageClass.CONSTANT,
        data=[[10.0, 20.0], [30.0, 40.0]],
    )
    c3 = g.add_tensor("c3", Shape([2, 2]), DType.F32, StorageClass.ACTIVATION)
    x = g.add_tensor("x", Shape([2, 2]), DType.F32, StorageClass.INPUT)
    out = g.add_tensor("out", Shape([2, 2]), DType.F32, StorageClass.ACTIVATION)

    # Static: c1 + c2 -> c3
    g.add_node(OpCode.ADD, inputs=[c1.id, c2.id], outputs=[c3.id])
    # Dynamic: x * c3 -> out
    g.add_node(OpCode.MUL, inputs=[x.id, c3.id], outputs=[out.id])

    g.inputs = [x.id]
    g.outputs = [out.id]

    assert len(g.nodes) == 2

    cf = ConstantFoldingPass()
    res = cf.run(g)

    assert res.modified is True
    assert len(res.graph.nodes) == 1
    assert res.stats.constants_folded == 1
    folded_t = res.graph.tensors[c3.id]
    assert folded_t.storage == StorageClass.CONSTANT
    assert np.allclose(folded_t.data, [[11.0, 22.0], [33.0, 44.0]])


def test_operator_fusion_conv2d_relu():
    """Verify fusion of Conv2D + ReLU into fused Conv2D."""
    g = Graph("test_fusion")
    x = g.add_tensor("x", Shape([1, 3, 32, 32]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 3, 3, 3]), DType.F32, StorageClass.PARAMETER)
    conv_out = g.add_tensor("conv_out", Shape([1, 16, 30, 30]), DType.F32, StorageClass.ACTIVATION)
    relu_out = g.add_tensor("relu_out", Shape([1, 16, 30, 30]), DType.F32, StorageClass.ACTIVATION)

    g.add_node(OpCode.CONV2D, inputs=[x.id, w.id], outputs=[conv_out.id], attributes={"stride": 1})
    g.add_node(OpCode.RELU, inputs=[conv_out.id], outputs=[relu_out.id])

    g.inputs = [x.id]
    g.outputs = [relu_out.id]

    fusion = OperatorFusionPass()
    res = fusion.run(g)

    assert res.modified is True
    assert len(res.graph.nodes) == 1
    fused_node = res.graph.nodes[0]
    assert fused_node.opcode == OpCode.CONV2D
    assert fused_node.attributes.get("fused_relu") is True
    assert fused_node.outputs == [relu_out.id]


def test_operator_fusion_swiglu():
    """Verify fusion of x * silu(g) -> SWIGLU."""
    g = Graph("test_swiglu")
    x = g.add_tensor("x", Shape([1, 8, 64]), DType.F32, StorageClass.INPUT)
    gate = g.add_tensor("gate", Shape([1, 8, 64]), DType.F32, StorageClass.INPUT)
    silu_out = g.add_tensor("silu_out", Shape([1, 8, 64]), DType.F32, StorageClass.ACTIVATION)
    mul_out = g.add_tensor("mul_out", Shape([1, 8, 64]), DType.F32, StorageClass.ACTIVATION)

    g.add_node(OpCode.SILU, inputs=[gate.id], outputs=[silu_out.id])
    g.add_node(OpCode.MUL, inputs=[x.id, silu_out.id], outputs=[mul_out.id])

    g.inputs = [x.id, gate.id]
    g.outputs = [mul_out.id]

    fusion = OperatorFusionPass()
    res = fusion.run(g)

    assert res.modified is True
    assert len(res.graph.nodes) == 1
    assert res.graph.nodes[0].opcode == OpCode.SWIGLU


def test_redundant_cast_pruner():
    """Verify elimination of identity transpose."""
    g = Graph("test_redundant")
    x = g.add_tensor("x", Shape([2, 4]), DType.F32, StorageClass.INPUT)
    t1 = g.add_tensor("t1", Shape([2, 4]), DType.F32, StorageClass.ACTIVATION)
    t2 = g.add_tensor("t2", Shape([2, 4]), DType.F32, StorageClass.ACTIVATION)

    # Identity transpose (dim0=0, dim1=0)
    g.add_node(OpCode.TRANSPOSE, inputs=[x.id], outputs=[t1.id], attributes={"dim0": 0, "dim1": 0})
    g.add_node(OpCode.RELU, inputs=[t1.id], outputs=[t2.id])

    g.inputs = [x.id]
    g.outputs = [t2.id]

    pruner = RedundantCastPruner()
    res = pruner.run(g)

    assert res.modified is True
    assert len(res.graph.nodes) == 1
    assert res.graph.nodes[0].opcode == OpCode.RELU
    assert res.graph.nodes[0].inputs == [x.id]


def test_standard_optimization_pipeline_e2e():
    """Verify full optimization pipeline reduces nodes on a composite neural graph."""
    pipeline = create_standard_optimization_pipeline()

    g = Graph("composite")
    x = g.add_tensor("x", Shape([1, 3, 32, 32]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([8, 3, 3, 3]), DType.F32, StorageClass.PARAMETER)
    c1 = g.add_tensor("c1", Shape([1]), DType.F32, StorageClass.CONSTANT, data=[2.0])
    c2 = g.add_tensor("c2", Shape([1]), DType.F32, StorageClass.CONSTANT, data=[3.0])
    c3 = g.add_tensor("c3", Shape([1]), DType.F32, StorageClass.ACTIVATION)

    conv_out = g.add_tensor("conv_out", Shape([1, 8, 30, 30]), DType.F32, StorageClass.ACTIVATION)
    relu_out = g.add_tensor("relu_out", Shape([1, 8, 30, 30]), DType.F32, StorageClass.ACTIVATION)
    scale_out = g.add_tensor("scale_out", Shape([1, 8, 30, 30]), DType.F32, StorageClass.ACTIVATION)
    dead_out = g.add_tensor("dead_out", Shape([1, 8, 30, 30]), DType.F32, StorageClass.ACTIVATION)

    # Static: c1 * c2 -> c3 (foldable)
    g.add_node(OpCode.MUL, inputs=[c1.id, c2.id], outputs=[c3.id])
    # Conv2D + ReLU (fusable)
    g.add_node(OpCode.CONV2D, inputs=[x.id, w.id], outputs=[conv_out.id])
    g.add_node(OpCode.RELU, inputs=[conv_out.id], outputs=[relu_out.id])
    # Scale: relu_out * c3
    g.add_node(OpCode.MUL, inputs=[relu_out.id, c3.id], outputs=[scale_out.id])
    # Dead node
    g.add_node(OpCode.SIGMOID, inputs=[x.id], outputs=[dead_out.id])

    g.inputs = [x.id]
    g.outputs = [scale_out.id]

    res = pipeline.run(g)
    assert res.modified is True
    # 5 nodes -> Conv2D(fused_relu) + MUL(folded_c3) = 2 nodes!
    assert len(res.graph.nodes) == 2
    assert res.stats.node_reduction_pct >= 50.0
