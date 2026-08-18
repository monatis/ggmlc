import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


def _run_dynamic_model_e2e(
    model: nn.Module,
    example_args: tuple,
    dynamic_shapes: dict,
    test_cases: list[tuple[tuple, dict[str, int]]],
    model_name: str,
    atol: float = 1e-4,
):
    model.eval()

    # 1. Export ONCE with dynamic shapes
    exported = export_torch_model(
        model,
        example_args,
        dynamic_shapes=dynamic_shapes,
        model_name=model_name,
    )
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    # 2. Execute multiple times with different dynamic shapes on the SAME binary artifact
    for actual_args, symbol_bindings in test_cases:
        with torch.no_grad():
            ref = model(*actual_args).detach().numpy()

        inputs = {}
        for i, inp_id in enumerate(exported.main_graph.inputs):
            name = exported.main_graph.get_tensor(inp_id).name
            inputs[name] = actual_args[i].detach().numpy()

        out_id = exported.main_graph.outputs[0]

        # Bind all symbols from the graph's symbol table
        # If symbol_table has generated names (e.g. s0, s1), map them using the input tensor shapes
        resolved_symbols = {}
        for sym_name in ggml_graph.symbol_table:
            if sym_name in symbol_bindings:
                resolved_symbols[sym_name] = symbol_bindings[sym_name]
            else:
                # Try to resolve symbol from input shapes
                for i, inp_id in enumerate(exported.main_graph.inputs):
                    t = exported.main_graph.get_tensor(inp_id)
                    actual_shape = actual_args[i].shape
                    for d_idx, d in enumerate(t.shape.dims):
                        if sym_name in d.free_symbols():
                            resolved_symbols[sym_name] = actual_shape[d_idx]

        results = run_compiled_model_wsl(
            serialized_bytes=ser_bytes,
            inputs=inputs,
            output_tensor_ids=[out_id],
            symbols=resolved_symbols,
        )

        ggml_out = results[out_id]
        res = check_numerical_accuracy(ref, ggml_out, atol=atol)
        assert res.passed, (
            f"Dynamic test for {model_name} failed with symbols {resolved_symbols}: {res.message}"
        )


class DynamicAddMul(nn.Module):
    def forward(self, a, b):
        return (a + b) * 0.5


def test_dynamic_elementwise_batch():
    torch.manual_seed(42)
    model = DynamicAddMul()
    dim_b = torch.export.Dim("batch", min=1, max=32)

    example_a = torch.randn(2, 16, dtype=torch.float32)
    example_b = torch.randn(2, 16, dtype=torch.float32)
    dynamic_shapes = {"a": {0: dim_b}, "b": {0: dim_b}}

    test_cases = [
        ((torch.randn(1, 16), torch.randn(1, 16)), {"batch": 1}),
        ((torch.randn(4, 16), torch.randn(4, 16)), {"batch": 4}),
        ((torch.randn(8, 16), torch.randn(8, 16)), {"batch": 8}),
        ((torch.randn(13, 16), torch.randn(13, 16)), {"batch": 13}),
    ]

    _run_dynamic_model_e2e(
        model,
        (example_a, example_b),
        dynamic_shapes,
        test_cases,
        "dyn_add_mul",
    )


class DynamicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_dynamic_mlp_batch():
    torch.manual_seed(42)
    model = DynamicMLP()
    dim_b = torch.export.Dim("batch", min=1, max=32)

    example_x = torch.randn(2, 16, dtype=torch.float32)
    dynamic_shapes = {"x": {0: dim_b}}

    test_cases = [
        ((torch.randn(1, 16),), {"batch": 1}),
        ((torch.randn(3, 16),), {"batch": 3}),
        ((torch.randn(7, 16),), {"batch": 7}),
        ((torch.randn(16, 16),), {"batch": 16}),
    ]

    _run_dynamic_model_e2e(
        model,
        (example_x,),
        dynamic_shapes,
        test_cases,
        "dyn_mlp",
        atol=2e-3,
    )


class DynamicAttention(nn.Module):
    def __init__(self, d_model: int = 16, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.out_proj(out)


def test_dynamic_seq_len_attention():
    torch.manual_seed(42)
    model = DynamicAttention(d_model=16, n_heads=2)
    dim_s = torch.export.Dim("seq", min=1, max=64)

    example_x = torch.randn(2, 4, 16, dtype=torch.float32)
    dynamic_shapes = {"x": {1: dim_s}}

    test_cases = [
        ((torch.randn(2, 2, 16),), {"seq": 2}),
        ((torch.randn(2, 8, 16),), {"seq": 8}),
        ((torch.randn(2, 12, 16),), {"seq": 12}),
        ((torch.randn(2, 24, 16),), {"seq": 24}),
    ]

    _run_dynamic_model_e2e(
        model,
        (example_x,),
        dynamic_shapes,
        test_cases,
        "dyn_attention",
        atol=1e-4,
    )


class DynamicTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ffn: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DynamicAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        out = h + self.fc2(self.act(self.fc1(self.norm2(h))))
        return out


def test_dynamic_batch_and_seq_transformer():
    torch.manual_seed(42)
    model = DynamicTransformerBlock(d_model=16, n_heads=2, d_ffn=32)
    dim_b = torch.export.Dim("batch", min=1, max=16)
    dim_s = torch.export.Dim("seq", min=1, max=64)

    example_x = torch.randn(2, 4, 16, dtype=torch.float32)
    dynamic_shapes = {"x": {0: dim_b, 1: dim_s}}

    test_cases = [
        ((torch.randn(1, 2, 16),), {"batch": 1, "seq": 2}),
        ((torch.randn(3, 6, 16),), {"batch": 3, "seq": 6}),
        ((torch.randn(4, 10, 16),), {"batch": 4, "seq": 10}),
        ((torch.randn(2, 16, 16),), {"batch": 2, "seq": 16}),
    ]

    _run_dynamic_model_e2e(
        model,
        (example_x,),
        dynamic_shapes,
        test_cases,
        "dyn_transformer",
        atol=2e-3,
    )
