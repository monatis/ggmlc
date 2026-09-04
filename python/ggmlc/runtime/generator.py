"""High-level text generation engine for autoregressive language models compiled with ggmlc."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.gguf import serialize_ggml_graph
from torch import nn


class GGMLCGenerator:
    """End-to-end text generation pipeline wrapping compiled ggmlc models and tokenizers."""

    def __init__(
        self,
        model: nn.Module | bytes | ModelRunner,
        tokenizer: Any,
        model_name: str = "llm",
        max_seq_len: int = 256,
        device: str = "auto",
        enable_fusion: bool = True,
        fusion_options: Any = None,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device
        self.enable_fusion = enable_fusion
        self.fusion_options = fusion_options
        self.compiled_bytes: bytes | None = None
        self.runner: ModelRunner | None = None

        if isinstance(model, ModelRunner):
            self.runner = model
        elif isinstance(model, (bytes, bytearray)):
            self.compiled_bytes = bytes(model)
            self.runner = ModelRunner(self.compiled_bytes, device=self.device)
        else:
            self.model = model
            self._compile()

    def _compile(self):
        """Compiles the PyTorch model graph into a serialized GGUF artifact and initializes ModelRunner."""
        self.model.eval()
        dummy_input = (torch.randint(0, 1000, (1, 8), dtype=torch.int32),)
        try:
            dim_s = torch.export.Dim("s", min=1, max=self.max_seq_len)
            dynamic_shapes = ({1: dim_s},)
            exported = export_torch_model(
                self.model, dummy_input, dynamic_shapes=dynamic_shapes, model_name=self.model_name
            )
        except Exception:  # noqa: BLE001
            exported = export_torch_model(self.model, dummy_input, model_name=self.model_name)

        ggml_graph = lower_to_ggml(
            exported.main_graph,
            enable_fusion=self.enable_fusion,
            fusion_options=self.fusion_options,
        )
        self.compiled_bytes = serialize_ggml_graph(ggml_graph)
        self.runner = ModelRunner(self.compiled_bytes, device=self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        greedy: bool = True,
        add_special_tokens: bool = False,
    ) -> str:
        """Generates text autoregressively given a prompt string."""
        # Handle tokenizer encoding
        if hasattr(self.tokenizer, "encode"):
            generated_tokens = list(
                self.tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens, pad_to_max=False
                )
            )
        else:
            encoded = self.tokenizer(
                prompt, return_tensors="np", add_special_tokens=add_special_tokens
            )
            generated_tokens = list(encoded["input_ids"][0])

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for _ in range(max_new_tokens):
            curr_input = np.array([generated_tokens], dtype=np.int32)
            out = self.runner(curr_input)
            out_tensor = next(iter(out.values())) if isinstance(out, dict) else out

            S = len(generated_tokens)
            vocab_size = out_tensor.size // S
            logits = out_tensor.reshape((1, S, vocab_size))
            next_token_logits = logits[0, -1, :]

            if greedy or temperature <= 0:
                next_token = int(np.argmax(next_token_logits))
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / np.sum(exp_logits)

                if top_p < 1.0:
                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_indices]
                    cumulative_probs = np.cumsum(sorted_probs)
                    valid_mask = cumulative_probs <= top_p
                    valid_mask[0] = True
                    filtered_indices = sorted_indices[valid_mask]
                    filtered_probs = probs[filtered_indices]
                    filtered_probs = filtered_probs / np.sum(filtered_probs)
                    next_token = int(np.random.choice(filtered_indices, p=filtered_probs))
                else:
                    next_token = int(np.random.choice(len(probs), p=probs))

            generated_tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        if hasattr(self.tokenizer, "decode"):
            decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            decoded_text = self.tokenizer.decode(generated_tokens)
        return decoded_text


def verify_generation_parity_with_pytorch(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 8,
    model_name: str = "gpt2",
    device: str = "auto",
    ref_model: nn.Module | None = None,
    add_special_tokens: bool = False,
) -> tuple[bool, str, str]:
    """Verifies greedy generation parity between PyTorch model and ggmlc compiled execution."""
    model.eval()
    ref = ref_model if ref_model is not None else model
    ref.eval()

    # 1. Reference PyTorch generation
    with torch.no_grad():
        if hasattr(tokenizer, "encode"):
            prompt_tokens = list(
                tokenizer.encode(prompt, add_special_tokens=add_special_tokens, pad_to_max=False)
            )
        else:
            prompt_tokens = list(
                tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
            )
            if isinstance(prompt_tokens[0], list):
                prompt_tokens = prompt_tokens[0]

        curr = list(prompt_tokens)
        for _ in range(max_new_tokens):
            t_in = torch.tensor([curr], dtype=torch.long)
            out = ref(t_in)
            logits = out.logits if hasattr(out, "logits") else out
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            curr.append(nxt)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and nxt == eos:
                break

        if hasattr(tokenizer, "decode"):
            ref_text = tokenizer.decode(curr, skip_special_tokens=True)
        else:
            ref_text = str(curr)

    # 2. ggmlc compiled generation
    generator = GGMLCGenerator(model, tokenizer, model_name=model_name, device=device)
    actual_text = generator.generate(
        prompt, max_new_tokens=max_new_tokens, greedy=True, add_special_tokens=add_special_tokens
    )

    passed = ref_text.strip() == actual_text.strip()
    return passed, ref_text, actual_text
