import tempfile
from pathlib import Path

import numpy as np
import pytest

from ggmlc.codegen import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import DType, StorageClass


def test_cpp_codegen_project_structure():
    """Verify that generate_cpp_project generates model.h, ggmlc_main.cpp, and CMakeLists.txt."""
    g = Graph("simple_mlp")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 16]), DType.F32, StorageClass.PARAMETER)
    b = g.add_tensor("b", Shape([16]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 16]), DType.F32, StorageClass.ACTIVATION)

    w.data = np.eye(16, dtype=np.float32)
    b.data = np.zeros((16,), dtype=np.float32)

    g.add_node(OpCode.LINEAR, inputs=[x.id, w.id, b.id], outputs=[out.id], name="linear_layer_0")
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id, b.id]

    ggml_graph = lower_to_ggml(g)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_cpp_project(ggml_graph, tmpdir, model_name="SimpleMLP")

        header_p = paths["header"]
        main_p = paths["main"]
        cmake_p = paths["cmake"]

        assert header_p.exists()
        assert main_p.exists()
        assert cmake_p.exists()

        header_content = header_p.read_text(encoding="utf-8")
        assert "namespace SimpleMLP" in header_content
        assert "struct Weights" in header_content
        assert "build_graph" in header_content
        assert "ggml_mul_mat" in header_content

        main_content = main_p.read_text(encoding="utf-8")
        assert '#include "SimpleMLP.h"' in main_content
        assert "ggml_graph_compute_with_ctx" in main_content

        cmake_content = cmake_p.read_text(encoding="utf-8")
        assert "project(SimpleMLP_standalone" in cmake_content
