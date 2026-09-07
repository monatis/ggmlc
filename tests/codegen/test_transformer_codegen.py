import tempfile
from pathlib import Path

import numpy as np
from ggmlc.codegen import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape, SymbolDim
from ggmlc.ir.tensor import DType, StorageClass


def test_transformer_ops_codegen():
    """Verify that transformer ops (RMS_NORM, ROPE, SDPA, EMBEDDING, CONCAT) generate valid C++."""
    g = Graph("mini_transformer")

    # Dynamic sequence length 's'
    s_dim = SymbolDim("s")
    vocab_size = 256
    hidden_size = 64
    n_heads = 4
    head_dim = 16

    # Inputs
    input_ids = g.add_tensor("input_ids", Shape([1, s_dim]), DType.I32, StorageClass.INPUT)
    positions = g.add_tensor("positions", Shape([1, s_dim]), DType.I32, StorageClass.INPUT)

    # Embedding Table
    embed_table = g.add_tensor(
        "embed_table", Shape([vocab_size, hidden_size]), DType.F32, StorageClass.PARAMETER
    )
    embed_table.data = np.zeros((vocab_size, hidden_size), dtype=np.float32)

    # Embedding lookup
    embedded = g.add_tensor("embedded", Shape([1, s_dim, hidden_size]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.EMBEDDING, inputs=[embed_table.id, input_ids.id], outputs=[embedded.id], name="embed")

    # RMSNorm
    norm_w = g.add_tensor("norm_w", Shape([hidden_size]), DType.F32, StorageClass.PARAMETER)
    norm_w.data = np.ones((hidden_size,), dtype=np.float32)
    normed = g.add_tensor("normed", Shape([1, s_dim, hidden_size]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.RMS_NORM, inputs=[embedded.id, norm_w.id], outputs=[normed.id], attributes={"eps": 1e-6}, name="rms_norm")

    # Q, K, V projections
    q = g.add_tensor("q", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    k = g.add_tensor("k", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    v = g.add_tensor("v", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    
    # RoPE on Q and K
    q_rope = g.add_tensor("q_rope", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    k_rope = g.add_tensor("k_rope", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.ROPE, inputs=[q.id, positions.id], outputs=[q_rope.id], attributes={"n_dims": head_dim, "freq_base": 1000000.0}, name="rope_q")
    g.add_node(OpCode.ROPE, inputs=[k.id, positions.id], outputs=[k_rope.id], attributes={"n_dims": head_dim, "freq_base": 1000000.0}, name="rope_k")

    # SDPA / Flash Attention
    attn_out = g.add_tensor("attn_out", Shape([1, s_dim, n_heads, head_dim]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.SDPA, inputs=[q_rope.id, k_rope.id, v.id], outputs=[attn_out.id], attributes={"scale": 0.25}, name="sdpa")

    # Concat
    cat_out = g.add_tensor("cat_out", Shape([1, s_dim, n_heads * 2, head_dim]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.CONCAT, inputs=[attn_out.id, attn_out.id], outputs=[cat_out.id], attributes={"dim": 2}, name="cat")

    # Reshape with dynamic symbol 's'
    reshaped = g.add_tensor("reshaped", Shape([1, s_dim, hidden_size * 2]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.RESHAPE, inputs=[cat_out.id], outputs=[reshaped.id], name="dyn_reshape")

    g.inputs = [input_ids.id, positions.id]
    g.outputs = [reshaped.id]
    g.parameters = [embed_table.id, norm_w.id]

    ggml_graph = lower_to_ggml(g)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_cpp_project(ggml_graph, tmpdir, model_name="MiniTransformer")
        header_p = paths["header"]
        main_p = paths["main"]
        cmake_p = paths["cmake"]

        assert header_p.exists()
        assert main_p.exists()
        assert cmake_p.exists()

        header_content = header_p.read_text(encoding="utf-8")
        assert "namespace MiniTransformer" in header_content
        assert "ggml_get_rows" in header_content
        assert "ggml_rms_norm" in header_content
        assert "ggml_rope" in header_content or "ggml_rope_ext" in header_content
        assert "ggml_flash_attn_ext" in header_content
        assert "ggml_concat" in header_content
        assert "symbols.count" in header_content
        assert "init_tensors" in header_content
        assert "load_data" in header_content
        assert "GGML_USE_METAL" in header_content

        main_content = main_p.read_text(encoding="utf-8")
        assert "GGML_USE_METAL" in main_content
        assert "ggml_backend_metal_init" in main_content
        assert "load_data" in main_content

        cmake_content = cmake_p.read_text(encoding="utf-8")
        assert "ENABLE_METAL" in cmake_content
        assert "ggml-metal" in cmake_content
