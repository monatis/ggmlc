import math

import torch
import torch.nn.functional as F
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn


def _verify_torch_model(model: nn.Module, example_args: tuple, model_name: str, atol: float = 1e-4):
    model.eval()
    with torch.no_grad():
        ref = model(*example_args).detach().numpy()

    exported = export_torch_model(model, example_args, model_name=model_name)
    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    inputs = {}
    for i, inp_id in enumerate(exported.main_graph.inputs):
        name = exported.main_graph.get_tensor(inp_id).name
        val = example_args[i]
        if isinstance(val, torch.Tensor):
            inputs[name] = val.detach().numpy()

    out_id = exported.main_graph.outputs[0]
    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs=inputs,
        output_tensor_ids=[out_id],
    )

    ggml_out = results[out_id]
    res = check_numerical_accuracy(ref, ggml_out, atol=atol)
    assert res.passed, f"Golden test for {model_name} failed: {res.message}"


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.ln(self.fc1(x))))


class TinyRMSNorm(nn.Module):
    def __init__(self, dim: int = 32, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class TinyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self.emb(indices)


class TinySwiGLUFFN(nn.Module):
    def __init__(self, d_model: int = 16, d_ffn: int = 32):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class TinySelfAttention(nn.Module):
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
        # x: (B, S, D)
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, S, d_head)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.out_proj(out)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ffn: int = 32):
        super().__init__()
        self.norm1 = TinyRMSNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_heads)
        self.norm2 = TinyRMSNorm(d_model)
        self.ffn = TinySwiGLUFFN(d_model, d_ffn)

    def forward(self, x):
        # Attention with residual
        h = x + self.attn(self.norm1(x))
        # FFN with residual
        out = h + self.ffn(self.norm2(h))
        return out


def test_golden_tiny_mlp():
    torch.manual_seed(42)
    model = TinyMLP(d_in=16, d_hidden=32, d_out=8)
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    _verify_torch_model(model, (x,), "golden_tiny_mlp", atol=2e-3)


def test_golden_rms_norm():
    torch.manual_seed(42)
    model = TinyRMSNorm(dim=32)
    x = torch.randn(2, 6, 32, dtype=torch.float32)
    _verify_torch_model(model, (x,), "golden_rms_norm", atol=1e-4)


def test_golden_swiglu_ffn():
    torch.manual_seed(42)
    model = TinySwiGLUFFN(d_model=16, d_ffn=32)
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    _verify_torch_model(model, (x,), "golden_swiglu_ffn", atol=1e-4)


def test_golden_tiny_attention():
    torch.manual_seed(42)
    model = TinySelfAttention(d_model=16, n_heads=2)
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    _verify_torch_model(model, (x,), "golden_tiny_attention", atol=1e-4)


def test_golden_tiny_transformer():
    torch.manual_seed(42)
    model = TinyTransformerBlock(d_model=16, n_heads=2, d_ffn=32)
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    _verify_torch_model(model, (x,), "golden_tiny_transformer", atol=1e-4)


def test_golden_embedding():
    torch.manual_seed(42)
    model = TinyEmbedding(num_embeddings=64, embedding_dim=16)
    indices = torch.tensor([[1, 5, 2, 8], [0, 3, 4, 7]], dtype=torch.int32)
    _verify_torch_model(model, (indices,), "golden_embedding", atol=1e-5)
