import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.generator import verify_generation_parity_with_pytorch
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy, run_compiled_model_wsl
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from examples.models.hub_models import (
    load_bert_model,
    load_bge_m3_distill_model,
    load_convnext_model,
    load_densenet_model,
    load_efficientnet_model,
    load_gpt2_model,
    load_minilm_model,
    load_qwen_model,
    load_regnet_model,
    load_resnet_model,
)


def _verify_full_model_e2e(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    model_name: str,
    atol: float = 1e-4,
):
    model.eval()

    # 1. Reference PyTorch computation
    with torch.no_grad():
        ref_out = model(*inputs)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        if hasattr(ref_out, "logits"):
            ref_out = ref_out.logits
        if hasattr(ref_out, "last_hidden_state"):
            ref_out = ref_out.last_hidden_state
        ref_np = ref_out.detach().cpu().numpy()

    # 2. Export to Canonical IR
    exported = export_torch_model(model, inputs, model_name=model_name)
    assert len(exported.main_graph.nodes) > 0

    # 3. Lower to GGML dialect
    ggml_graph = lower_to_ggml(exported.main_graph)
    assert len(ggml_graph.nodes) > 0

    # 4. Serialize to GGUF v3 binary format
    ser_bytes = serialize_ggml_graph(ggml_graph)
    assert len(ser_bytes) > 0

    # 5. Execute in C++ Generic Runtime via WSL
    inputs_dict = {
        name: tensor_val.numpy() for name, tensor_val in zip(input_names, inputs, strict=False)
    }
    out_id = exported.main_graph.outputs[0]

    results = run_compiled_model_wsl(
        serialized_bytes=ser_bytes,
        inputs=inputs_dict,
        output_tensor_ids=[out_id],
    )

    actual_np = results[out_id].reshape(ref_np.shape)
    cmp = check_numerical_accuracy(ref_np, actual_np, atol=atol)
    assert cmp.passed, f"Hub model verification failed for {model_name}: {cmp.message}"


def test_resnet_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_resnet_model("resnet18")
    _verify_full_model_e2e(model, inputs, names, "resnet18", atol=1e-2)


def test_minilm_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_minilm_model()
    _verify_full_model_e2e(model, inputs, names, "all_minilm_l6_v2", atol=1e-2)


def test_gpt2_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_gpt2_model()
    _verify_full_model_e2e(model, inputs, names, "gpt2", atol=1e-3)


def test_qwen_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_qwen_model("Qwen/Qwen2.5-0.5B")
    _verify_full_model_e2e(model, inputs, names, "qwen2.5_0.5b", atol=1e-3)


def test_bge_m3_distill_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_bge_m3_distill_model()
    _verify_full_model_e2e(model, inputs, names, "bge_m3_distill_8l", atol=0.2)


def test_gpt2_autoregressive_generation_parity():
    """Validates multi-token autoregressive generation parity between ggmlc and PyTorch model.generate."""
    torch.manual_seed(42)
    model_id = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).eval()

    prompt = "The capital of France is"
    passed, ref_text, actual_text = verify_generation_parity_with_pytorch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=6,
    )
    assert passed, f"Generation mismatch!\nPyTorch: '{ref_text}'\nggmlc:   '{actual_text}'"


def test_convnext_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_convnext_model("tiny")
    _verify_full_model_e2e(model, inputs, names, "convnext_tiny", atol=2e-2)


def test_efficientnet_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_efficientnet_model("b0")
    _verify_full_model_e2e(model, inputs, names, "efficientnet_b0", atol=1e-3)


def test_densenet_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_densenet_model("densenet121")
    _verify_full_model_e2e(model, inputs, names, "densenet121", atol=1e-3)


def test_regnet_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_regnet_model("regnet_y_400mf")
    _verify_full_model_e2e(model, inputs, names, "regnet_y_400mf", atol=1e-3)


def test_bert_hub_compilation_and_execution():
    torch.manual_seed(42)
    model, inputs, names = load_bert_model(seq_len=16)
    _verify_full_model_e2e(model, inputs, names, "bert_base_uncased", atol=0.05)
