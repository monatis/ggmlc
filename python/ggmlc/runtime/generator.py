"""High-level text generation engine for autoregressive language models compiled with ggmlc."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import run_compiled_model_wsl
from torch import nn


class GGMLCGenerator:
    """End-to-end text generation pipeline wrapping compiled ggmlc models and tokenizers."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_name: str = "llm",
        max_seq_len: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.compiled_bytes: bytes | None = None
        self.out_tensor_id: int | None = None

        # Build and compile model
        self._compile()

    def _compile(self):
        """Compiles the PyTorch model graph into a serialized .ggmlc artifact."""
        self.model.eval()
        dim_s = torch.export.Dim("s", min=1, max=self.max_seq_len)
        dummy_input = (torch.randint(0, 1000, (1, 8), dtype=torch.int32),)
        dynamic_shapes = ({1: dim_s},)
        try:
            exported = export_torch_model(
                self.model, dummy_input, dynamic_shapes=dynamic_shapes, model_name=self.model_name
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(
                f"[GGMLCGenerator] Warning: dynamic export failed ({e}), falling back to static export."
            )
            exported = export_torch_model(self.model, dummy_input, model_name=self.model_name)
        ggml_graph = lower_to_ggml(exported.main_graph)
        self.symbol_table = list(ggml_graph.symbol_table)
        print(f"[GGMLCGenerator] GGML symbol table: {self.symbol_table}")
        self.compiled_bytes = serialize_ggml_graph(ggml_graph)
        self.out_tensor_id = exported.main_graph.outputs[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        greedy: bool = True,
    ) -> str:
        """Generates text autoregressively given a prompt string."""
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int32)
        generated_tokens = list(input_ids[0])

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for _ in range(max_new_tokens):
            curr_input = np.array([generated_tokens], dtype=np.int32)
            S = len(generated_tokens)
            sym_dict = {s: S for s in self.symbol_table}
            results = run_compiled_model_wsl(
                serialized_bytes=self.compiled_bytes,
                inputs={"input_ids": curr_input},
                symbols=sym_dict,
                output_tensor_ids=[self.out_tensor_id],
            )
            raw_logits = results[self.out_tensor_id]
            S = len(generated_tokens)
            vocab_size = raw_logits.size // S
            logits = raw_logits.reshape((1, S, vocab_size))
            next_token_logits = logits[0, -1, :]

            if greedy or temperature <= 0:
                next_token = int(np.argmax(next_token_logits))
            else:
                # Temperature scaling & softmax
                scaled_logits = next_token_logits / max(temperature, 1e-5)
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / np.sum(exp_logits)

                if top_p < 1.0:
                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_indices]
                    cumulative_probs = np.cumsum(sorted_probs)
                    valid_mask = cumulative_probs <= top_p
                    valid_mask[0] = True  # Ensure at least one token is kept
                    filtered_indices = sorted_indices[valid_mask]
                    filtered_probs = probs[filtered_indices]
                    filtered_probs = filtered_probs / np.sum(filtered_probs)
                    next_token = int(np.random.choice(filtered_indices, p=filtered_probs))
                else:
                    next_token = int(np.random.choice(len(probs), p=probs))

            generated_tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded_text


def verify_generation_parity_with_pytorch(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 8,
) -> tuple[bool, str, str]:
    """Verifies greedy generation parity between PyTorch model.generate and ggmlc compiled execution."""
    model.eval()

    # 1. Reference PyTorch generation
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        ref_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        ref_text = tokenizer.decode(ref_tokens[0], skip_special_tokens=True)

    # 2. ggmlc compiled generation
    from examples.models.hub_models import load_gpt2_model

    wrapped, _, _ = load_gpt2_model()
    generator = GGMLCGenerator(wrapped, tokenizer, model_name="gpt2")
    actual_text = generator.generate(prompt, max_new_tokens=max_new_tokens, greedy=True)

    passed = ref_text.strip() == actual_text.strip()
    return passed, ref_text, actual_text
