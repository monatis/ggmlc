import ggmlc
import torch
from ggmlc.validation.numerical import check_numerical_accuracy, cosine_similarity
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_python_nanobind_runtime_e2e():
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    x = torch.randn(2, 32, dtype=torch.float32)

    # 1. Compile directly in Python to GGUF bytes
    gguf_bytes = ggmlc.compile(
        model=model,
        sample_inputs=(x,),
        model_name="simple_mlp",
    )
    assert len(gguf_bytes) > 0
    assert gguf_bytes.startswith(b"GGUF")

    # 2. Load model into native nanobind runtime
    runner = ggmlc.load(gguf_bytes, n_threads=2)
    assert runner.name == "simple_mlp"
    assert len(runner.inputs) == 1
    assert len(runner.outputs) == 1

    # 3. Execute with numpy inputs (Zero PyTorch runtime dependency)
    x_np = x.numpy()
    out_np = runner(x=x_np)

    # 4. Numerical verification against PyTorch golden truth
    with torch.no_grad():
        ref_out = model(x).numpy()

    res = check_numerical_accuracy(ref_out, out_np, atol=1e-4, rtol=1e-3)
    cos_sim = cosine_similarity(ref_out, out_np)
    assert res.passed, f"Numerical check failed: {res.message}, cos_sim={cos_sim}"
    assert cos_sim > 0.9999


def test_python_nanobind_quantized_q4_0():
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    x = torch.randn(1, 32, dtype=torch.float32)

    # Compile with Q4_0 quantization
    gguf_bytes = ggmlc.compile(
        model=model,
        sample_inputs=(x,),
        quantize="q4_0",
        model_name="mlp_q4",
    )

    runner = ggmlc.load(gguf_bytes)
    out_np = runner(x.numpy())

    with torch.no_grad():
        ref_out = model(x).numpy()

    cos_sim = cosine_similarity(ref_out, out_np)
    assert cos_sim > 0.985, f"Q4_0 cosine similarity too low: {cos_sim}"


def test_python_nanobind_dynamic_shapes():
    class DynamicLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(42)
    model = DynamicLinear().eval()
    x_sample = torch.randn(1, 4, 16)

    dim_s = torch.export.Dim("seq_len", min=1, max=64)
    dynamic_shapes = ({1: dim_s},)

    gguf_bytes = ggmlc.compile(
        model=model,
        sample_inputs=(x_sample,),
        dynamic_shapes=dynamic_shapes,
        model_name="dynamic_linear",
    )

    runner = ggmlc.load(gguf_bytes)

    # Test varying sequence lengths (S=4, S=8, S=16)
    for seq_len in [4, 8, 16]:
        x_test = torch.randn(1, seq_len, 16)
        out_np = runner(x_test.numpy(), symbols={"seq_len": seq_len})

        with torch.no_grad():
            ref_out = model(x_test).numpy()

        res = check_numerical_accuracy(ref_out, out_np, atol=1e-4)
        cos_sim = cosine_similarity(ref_out, out_np)
        assert res.passed, f"Failed at seq_len={seq_len}: {res.message}, cos_sim={cos_sim}"
