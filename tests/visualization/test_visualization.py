"""Unit tests for Mermaid graph visualization and HTML generation."""

import tempfile
from pathlib import Path

import ggmlc
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass


def test_canonical_graph_mermaid_generation():
    g = Graph(name="TestMLP")
    t_in = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    t_w = g.add_tensor("w", Shape([16, 32]), DType.F32, StorageClass.PARAMETER)
    t_matmul = g.add_tensor("linear_out", Shape([1, 32]), DType.F32, StorageClass.ACTIVATION)
    t_relu = g.add_tensor("relu_out", Shape([1, 32]), DType.F32, StorageClass.OUTPUT)

    g.add_node(OpCode.MATMUL, [t_in.id, t_w.id], [t_matmul.id], name="fc1")
    g.add_node(OpCode.RELU, [t_matmul.id], [t_relu.id], name="act1")

    mmd = g.to_mermaid()
    assert "graph TD" in mmd
    assert "TestMLP" in mmd
    assert "MATMUL" in mmd
    assert "RELU" in mmd
    assert "Input: x" in mmd


def test_ggml_graph_mermaid_generation():
    g = Graph(name="TestGGML")
    t_in = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    t_w = g.add_tensor("w", Shape([16, 32]), DType.F32, StorageClass.PARAMETER)
    t_out = g.add_tensor("out", Shape([1, 32]), DType.F32, StorageClass.OUTPUT)
    g.add_node(OpCode.MATMUL, [t_in.id, t_w.id], [t_out.id])

    ggml_g = lower_to_ggml(g)
    mmd = ggml_g.to_mermaid()

    assert "graph TD" in mmd
    assert "GGML_OP_MUL_MAT" in mmd


def test_visualize_html_export():
    g = Graph(name="TestExport")
    t_in = g.add_tensor("x", Shape([2, 8]), DType.F32, StorageClass.INPUT)
    t_out = g.add_tensor("out", Shape([2, 8]), DType.F32, StorageClass.OUTPUT)
    g.add_node(OpCode.RELU, [t_in.id], [t_out.id])

    with tempfile.TemporaryDirectory() as tmpdir:
        html_file = Path(tmpdir) / "model.html"
        out_path = ggmlc.visualize(g, output_path=html_file, format="html")
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "mermaid.min.js" in content
        assert "TestExport" in content

        mmd_file = Path(tmpdir) / "model.mmd"
        out_mmd = g.visualize(output_path=str(mmd_file), format="mmd")
        assert out_mmd.exists()
        assert "graph TD" in out_mmd.read_text(encoding="utf-8")
