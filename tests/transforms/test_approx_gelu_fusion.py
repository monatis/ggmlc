import numpy as np
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def test_approx_gelu_fusion():
    """Verify that polynomial NewGELU subgraphs are automatically fused into OpCode.GELU."""
    graph = Graph(name="test_gelu")

    # Inputs: x
    t_x = graph.add_tensor(
        "x", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )

    # Constants:
    t_half = graph.add_tensor(
        "c_half",
        shape=Shape([1]),
        dtype=DType.F32,
        storage=StorageClass.CONSTANT,
        data=np.array(0.5, dtype=np.float32),
    )
    t_sqrt = graph.add_tensor(
        "c_sqrt",
        shape=Shape([1]),
        dtype=DType.F32,
        storage=StorageClass.CONSTANT,
        data=np.array(0.7978845608, dtype=np.float32),
    )
    t_coeff = graph.add_tensor(
        "c_coeff",
        shape=Shape([1]),
        dtype=DType.F32,
        storage=StorageClass.CONSTANT,
        data=np.array(0.044715, dtype=np.float32),
    )
    t_one = graph.add_tensor(
        "c_one",
        shape=Shape([1]),
        dtype=DType.F32,
        storage=StorageClass.CONSTANT,
        data=np.array(1.0, dtype=np.float32),
    )
    t_three = graph.add_tensor(
        "c_three",
        shape=Shape([1]),
        dtype=DType.F32,
        storage=StorageClass.CONSTANT,
        data=np.array(3.0, dtype=np.float32),
    )

    t_half_x = graph.add_tensor(
        "half_x", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_cube = graph.add_tensor(
        "cube", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_coeff_cube = graph.add_tensor(
        "coeff_cube", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_poly = graph.add_tensor(
        "poly", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_scaled = graph.add_tensor(
        "scaled", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_tanh = graph.add_tensor(
        "tanh", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_add_one = graph.add_tensor(
        "add_one", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )
    t_out = graph.add_tensor(
        "out", shape=Shape([1, 768]), dtype=DType.F32, storage=StorageClass.ACTIVATION
    )

    graph.add_node(OpCode.MUL, inputs=[t_x.id, t_half.id], outputs=[t_half_x.id])
    graph.add_node(OpCode.POW, inputs=[t_x.id, t_three.id], outputs=[t_cube.id])
    graph.add_node(OpCode.MUL, inputs=[t_cube.id, t_coeff.id], outputs=[t_coeff_cube.id])
    graph.add_node(OpCode.ADD, inputs=[t_x.id, t_coeff_cube.id], outputs=[t_poly.id])
    graph.add_node(OpCode.MUL, inputs=[t_poly.id, t_sqrt.id], outputs=[t_scaled.id])
    graph.add_node(OpCode.TANH, inputs=[t_scaled.id], outputs=[t_tanh.id])
    graph.add_node(OpCode.ADD, inputs=[t_tanh.id, t_one.id], outputs=[t_add_one.id])
    graph.add_node(OpCode.MUL, inputs=[t_half_x.id, t_add_one.id], outputs=[t_out.id])

    graph.inputs = [t_x.id]
    graph.outputs = [t_out.id]

    assert len(graph.nodes) == 8

    fuse_operations(graph, FusionOptions(enable_approx_gelu=True))

    assert len(graph.nodes) == 1
    fused_op = graph.nodes[0]
    assert fused_op.opcode == OpCode.GELU
    assert fused_op.inputs == [t_x.id]
    assert fused_op.outputs == [t_out.id]
    assert fused_op.attributes.get("approximate") == "tanh"
