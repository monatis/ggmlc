import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir import DType, Graph, OpCode, Shape, StorageClass
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


def test_stateful_accumulator_persistence():
    """Verifies that StorageClass.STATE preserves memory across sequential runs."""
    g = Graph(name="stateful_acc")

    # In: x (16,)
    in_x = g.add_tensor(
        name="x",
        shape=Shape.from_tuple((16,)),
        dtype=DType.F32,
        storage=StorageClass.INPUT,
        role="input",
    )
    g.inputs.append(in_x.id)

    # State: state (16,)
    state_t = g.add_tensor(
        name="state",
        shape=Shape.from_tuple((16,)),
        dtype=DType.F32,
        storage=StorageClass.STATE,
        role="state",
    )

    # Out: out = x + state
    out_t = g.add_tensor(
        name="out",
        shape=Shape.from_tuple((16,)),
        dtype=DType.F32,
        storage=StorageClass.ACTIVATION,
        role="output",
    )
    g.outputs.append(out_t.id)

    # Add op: out = x + state
    g.add_op(
        opcode=OpCode.ADD,
        inputs=[in_x.id, state_t.id],
        outputs=[out_t.id],
        name="add_op",
    )

    # Lower and serialize
    ggml_graph = lower_to_ggml(g)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # Step 1: Initial state = 0. x = [1, 1, ..., 1]. Expected out = [1, ..., 1].
    x1 = np.ones((16,), dtype=np.float32)
    init_state = np.zeros((16,), dtype=np.float32)
    results, _ = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={"x": x1},
        output_tensor_ids=[out_t.id],
        states_in={"state": init_state},
        states_out=["state"],
    )

    out1 = results[out_t.id]
    np.testing.assert_allclose(out1, np.ones(16, dtype=np.float32), atol=1e-5)

    # Step 2: Next step: pass updated state (now set state = out1). x = [2, 2, ..., 2]. Expected out = 2 + 1 = 3.
    x2 = np.full((16,), 2.0, dtype=np.float32)
    results, _ = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={"x": x2},
        output_tensor_ids=[out_t.id],
        states_in={"state": out1},
        states_out=["state"],
    )

    out2 = results[out_t.id]
    np.testing.assert_allclose(out2, np.full(16, 3.0, dtype=np.float32), atol=1e-5)

    # Step 3: Pass state = out2. x = [5, 5, ..., 5]. Expected out = 5 + 3 = 8.
    x3 = np.full((16,), 5.0, dtype=np.float32)
    results, _ = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={"x": x3},
        output_tensor_ids=[out_t.id],
        states_in={"state": out2},
        states_out=["state"],
    )

    out3 = results[out_t.id]
    np.testing.assert_allclose(out3, np.full(16, 8.0, dtype=np.float32), atol=1e-5)


class KVCacheSelfAttentionModel(nn.Module):
    def __init__(self, d_model: int = 16, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_tok, k_all, v_all):
        """Computes attention between current query token(s) and full active KV history.

        q_tok: (B, 1, D) or (B, S, D)
        k_all: (B, S_total, D)
        v_all: (B, S_total, D)
        """
        B, S_q, D = q_tok.shape
        _, S_kv, _ = k_all.shape

        q = self.q_proj(q_tok).view(B, S_q, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(k_all).view(B, S_kv, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(v_all).view(B, S_kv, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S_q, D)
        return self.out_proj(out)


def test_kv_cache_autoregressive_generation():
    """Simulates multi-step autoregressive generation with dynamic KV cache vs full reference."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = KVCacheSelfAttentionModel(d_model=16, n_heads=2)
    model.eval()

    # Dynamic shapes for S_q and S_kv
    dim_sq = torch.export.Dim("s_q", min=1, max=16)
    dim_skv = torch.export.Dim("s_kv", min=1, max=64)

    example_q = torch.randn(1, 4, 16, dtype=torch.float32)
    example_k = torch.randn(1, 4, 16, dtype=torch.float32)
    example_v = torch.randn(1, 4, 16, dtype=torch.float32)

    dynamic_shapes = {
        "q_tok": {1: dim_sq},
        "k_all": {1: dim_skv},
        "v_all": {1: dim_skv},
    }

    # Compile the model ONCE
    exported = export_torch_model(
        model,
        (example_q, example_k, example_v),
        dynamic_shapes=dynamic_shapes,
        model_name="kv_cache_attn",
    )
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)
    out_id = exported.main_graph.outputs[0]

    # Generation loop:
    # 1. Prefill prompt: 4 tokens
    # 2. Decode step 1: 1 token (total 5)
    # 3. Decode step 2: 1 token (total 6)
    # 4. Decode step 3: 1 token (total 7)

    prompt = torch.randn(1, 4, 16, dtype=torch.float32)
    kv_history = prompt.clone()

    # Step 0: Prefill
    with torch.no_grad():
        ref_prefill = model(prompt, prompt, prompt).numpy()

    # WSL inference for prefill
    symbols = {}
    for sym in ggml_graph.symbol_table:
        symbols[sym] = 4  # s_q=4, s_kv=4

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs={
            "q_tok": prompt.numpy(),
            "k_all": kv_history.numpy(),
            "v_all": kv_history.numpy(),
        },
        output_tensor_ids=[out_id],
        symbols=symbols,
    )
    res = check_numerical_accuracy(ref_prefill, results[out_id], atol=1e-4)
    assert res.passed, f"Prefill verification failed: {res.message}"

    # Autoregressive Decode steps
    for step in range(1, 4):
        # Generate new token
        new_tok = torch.randn(1, 1, 16, dtype=torch.float32)
        # Update full KV history
        kv_history = torch.cat([kv_history, new_tok], dim=1)
        s_kv_len = kv_history.shape[1]

        # Full reference computation
        with torch.no_grad():
            ref_step = model(new_tok, kv_history, kv_history).numpy()

        # WSL execution using compiled artifact
        # Map dynamic symbols
        step_symbols = {}
        for s_idx, sym in enumerate(ggml_graph.symbol_table):
            # Check if this symbol represents s_q (size 1) or s_kv (size s_kv_len)
            # Both symbols in symbol_table
            if "s_q" in sym or s_idx == 0:
                step_symbols[sym] = (
                    1 if len(ggml_graph.symbol_table) > 1 and s_idx == 0 else s_kv_len
                )
            else:
                step_symbols[sym] = s_kv_len

        # More robust symbol resolution: resolve from tensor shapes
        resolved = {}
        for sym_name in ggml_graph.symbol_table:
            # Resolve from q_tok shape (dim 1 is 1) and k_all shape (dim 1 is s_kv_len)
            t_q = exported.main_graph.get_tensor(exported.main_graph.inputs[0])
            t_k = exported.main_graph.get_tensor(exported.main_graph.inputs[1])
            if sym_name in t_q.shape.dims[1].free_symbols():
                resolved[sym_name] = 1
            elif sym_name in t_k.shape.dims[1].free_symbols():
                resolved[sym_name] = s_kv_len
            else:
                resolved[sym_name] = s_kv_len

        res_decode = run_compiled_model_wsl(
            serialized_bytes=ser_bytes,
            inputs={
                "q_tok": new_tok.numpy(),
                "k_all": kv_history.numpy(),
                "v_all": kv_history.numpy(),
            },
            output_tensor_ids=[out_id],
            symbols=resolved,
        )

        cmp = check_numerical_accuracy(ref_step, res_decode[out_id], atol=1e-4)
        assert cmp.passed, f"Decode step {step} (history length {s_kv_len}) failed: {cmp.message}"
