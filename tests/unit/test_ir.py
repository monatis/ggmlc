from ggmlc.ir import (
    DType,
    Graph,
    Model,
    OpCode,
    Shape,
    StaticDim,
    StorageClass,
    SymbolDim,
)


def test_shape_arithmetic():
    b = SymbolDim("B")
    t = SymbolDim("T")
    d = StaticDim(4096)

    s = Shape([b, t, d])
    assert not s.is_static()
    assert s.free_symbols() == {"B", "T"}
    assert s.evaluate({"B": 2, "T": 128}) == (2, 128, 4096)
    assert s.numel({"B": 2, "T": 128}) == 2 * 128 * 4096

    s_derived = Shape([b, t + 1, d // 2])
    assert s_derived.evaluate({"B": 2, "T": 127}) == (2, 128, 2048)


def test_graph_construction_and_validation():
    g = Graph(name="test_mlp")
    in_t = g.add_tensor("x", Shape([1, 128]), DType.F32, StorageClass.INPUT)
    w_t = g.add_tensor("w", Shape([128, 256]), DType.F32, StorageClass.PARAMETER)
    b_t = g.add_tensor("b", Shape([256]), DType.F32, StorageClass.PARAMETER)

    g.inputs.append(in_t.id)
    g.parameters.extend([w_t.id, b_t.id])

    mm_out = g.add_tensor("mm_out", Shape([1, 256]), DType.F32, StorageClass.ACTIVATION)
    g.add_op(OpCode.MATMUL, [in_t.id, w_t.id], [mm_out.id])

    add_out = g.add_tensor("out", Shape([1, 256]), DType.F32, StorageClass.OUTPUT)
    g.add_op(OpCode.ADD, [mm_out.id, b_t.id], [add_out.id])
    g.outputs.append(add_out.id)

    # Invariants should pass
    g.validate_invariants()

    model = Model(name="simple_mlp")
    model.add_graph(g)
    model.validate()
    assert len(model.main_graph.nodes) == 2
