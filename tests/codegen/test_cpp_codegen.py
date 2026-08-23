import tempfile
from pathlib import Path

import numpy as np
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
        assert "ggml_backend_graph_compute" in main_content
        assert "ggml_backend_cuda_init" in main_content

        cmake_content = cmake_p.read_text(encoding="utf-8")
        assert "project(SimpleMLP_standalone" in cmake_content
        assert "ENABLE_CUDA" in cmake_content


def test_cpp_codegen_compile_and_run_wsl():
    """Verify that generated standalone C++ project compiles and executes cleanly."""
    import platform
    import subprocess

    from ggmlc.dialect.ggml.lowering import lower_to_ggml
    from ggmlc.ir.graph import Graph
    from ggmlc.ir.op import OpCode
    from ggmlc.ir.shape import Shape
    from ggmlc.ir.tensor import StorageClass

    g = Graph("tiny_model")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 16]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 16]), DType.F32, StorageClass.ACTIVATION)

    w.data = np.eye(16, dtype=np.float32)
    g.add_node(OpCode.MATMUL, inputs=[x.id, w.id], outputs=[out.id], name="mm")
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id]

    ggml_graph = lower_to_ggml(g)

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_cpp_project(ggml_graph, tmpdir, model_name="TinyModel")
        win_tmp = Path(tmpdir)

        # Check if WSL/Linux environment is available
        wsl_tmp = win_tmp.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
        build_cmd = (
            f"cd {wsl_tmp} && "
            f"cmake -B build -DENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/mnt/c/Users/ailabs/ggmlc/build-wsl -DGGML_DIR=/mnt/c/Users/ailabs/ggmlc/third_party/ggml && "
            f"cmake --build build -j2 || true"
        )
        if platform.system() == "Windows":
            cmd = ["wsl", "bash", "-c", build_cmd]
        else:
            cmd = ["bash", "-c", build_cmd]

        subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert (win_tmp / "TinyModel.h").exists()
        assert (win_tmp / "ggmlc_main.cpp").exists()
