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
        chunk_size: int = 128,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device
        self.enable_fusion = enable_fusion
        self.fusion_options = fusion_options
        self.chunk_size = chunk_size
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

    def _sample_token(
        self,
        next_token_logits: np.ndarray,
        greedy: bool,
        temperature: float,
        top_p: float,
    ) -> int:
        """Samples the next token from logit distribution using greedy or top-p nucleus sampling."""
        if greedy or temperature <= 0:
            return int(np.argmax(next_token_logits))

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
            return int(np.random.choice(filtered_indices, p=filtered_probs))
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        greedy: bool = True,
        add_special_tokens: bool = False,
        chunk_size: int | None = None,
    ) -> str:
        """Generates text autoregressively given a prompt string using chunked prompt prefill."""
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
        use_kv_cache = False
        if hasattr(self.runner, "init_kv_cache") and hasattr(self.runner, "has_kv_cache"):
            self.runner.init_kv_cache(len(generated_tokens) + max_new_tokens + 256)
            use_kv_cache = self.runner.has_kv_cache()

        prompt_len = len(generated_tokens)
        c_size = chunk_size if chunk_size is not None else self.chunk_size
        effective_chunk_size = c_size if (c_size > 0 and use_kv_cache) else prompt_len
        n_chunks = (prompt_len + effective_chunk_size - 1) // effective_chunk_size

        pos = 0
        last_token = 0
        stopped = False

        # Phase 1: Prompt Prefill (Chunked or Full)
        for chunk_idx in range(n_chunks):
            c_start = chunk_idx * effective_chunk_size
            c_end = min(c_start + effective_chunk_size, prompt_len)
            c_len = c_end - c_start
            c_tokens = generated_tokens[c_start:c_end]

            curr_input = np.array([c_tokens], dtype=np.int32)
            symbols = {"pos": c_start, "s": c_len} if use_kv_cache else None

            out = self.runner(curr_input, symbols=symbols)

            if chunk_idx == n_chunks - 1 and max_new_tokens > 0:
                out_tensor = next(iter(out.values())) if isinstance(out, dict) else out
                vocab_size = out_tensor.size // c_len
                logits = out_tensor.reshape((1, c_len, vocab_size))
                next_token_logits = logits[0, -1, :]

                next_token = self._sample_token(next_token_logits, greedy, temperature, top_p)
                last_token = next_token
                generated_tokens.append(next_token)
                pos = prompt_len

                if eos_token_id is not None and next_token == eos_token_id:
                    stopped = True
                    break

        # Phase 2: Token-by-Token Decode (S = 1)
        if not stopped:
            for _ in range(1, max_new_tokens):
                if use_kv_cache:
                    curr_input = np.array([[last_token]], dtype=np.int32)
                    symbols = {"pos": pos, "s": 1}
                else:
                    curr_input = np.array([generated_tokens], dtype=np.int32)
                    symbols = None

                out = self.runner(curr_input, symbols=symbols)
                out_tensor = next(iter(out.values())) if isinstance(out, dict) else out

                S = curr_input.shape[1]
                vocab_size = out_tensor.size // S
                logits = out_tensor.reshape((1, S, vocab_size))
                next_token_logits = logits[0, -1, :]

                next_token = self._sample_token(next_token_logits, greedy, temperature, top_p)
                last_token = next_token
                generated_tokens.append(next_token)
                if use_kv_cache:
                    pos += 1

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
