"""Tokenizer specifications and extraction utilities for GGUF metadata."""

from __future__ import annotations

from typing import Any

from ggmlc.pipeline.spec import TokenizerSpec


class BPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer specification and helper."""

    def __init__(
        self,
        vocab: dict[str, int] | list[str] | TokenizerSpec,
        merges: list[str] | None = None,
        pre_tokenizer: str = "clip",
        context_length: int = 77,
        bos_token_id: int | None = 49406,
        eos_token_id: int | None = 49407,
        pad_token_id: int | None = 49407,
        unk_token_id: int | None = 49407,
        chat_template: str | None = None,
    ):
        if isinstance(vocab, TokenizerSpec):
            spec = vocab
            if isinstance(spec.vocab, list):
                self.vocab = {tok: idx for idx, tok in enumerate(spec.vocab)}
            else:
                self.vocab = dict(spec.vocab)
            self.merges = list(spec.merges)
            self.pre_tokenizer = spec.pre_tokenizer
            self.context_length = spec.context_length
            self.bos_token_id = spec.bos_token_id
            self.eos_token_id = spec.eos_token_id
            self.pad_token_id = spec.pad_token_id
            self.unk_token_id = spec.unk_token_id
        else:
            if isinstance(vocab, (list, tuple)):
                self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
            else:
                self.vocab = dict(vocab)
            self.merges = list(merges) if merges is not None else []
            self.pre_tokenizer = pre_tokenizer
            self.context_length = context_length
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.unk_token_id = unk_token_id

        self.chat_template = chat_template
        self._hf_tokenizer: Any = None

        # Initialize native C++ tokenizer
        try:
            from ggmlc._runtime import NativeBPETokenizer

            token_list = [""] * len(self.vocab)
            for tok, idx in self.vocab.items():
                if idx < len(token_list):
                    token_list[idx] = tok
            self._native = NativeBPETokenizer()
            self._native.init(
                tokens=token_list,
                merges=self.merges,
                bos_id=self.bos_token_id if self.bos_token_id is not None else -1,
                eos_id=self.eos_token_id if self.eos_token_id is not None else -1,
                pad_id=self.pad_token_id if self.pad_token_id is not None else -1,
                unk_id=self.unk_token_id if self.unk_token_id is not None else 0,
                pre_tokenizer=self.pre_tokenizer,
            )
        except Exception:  # noqa: BLE001
            self._native = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        pad_to_max: bool = False,
    ) -> list[int]:
        max_len = max_length or self.context_length
        if self._hf_tokenizer is not None:
            ids = self._hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if max_len > 0:
                if len(ids) > max_len:
                    ids = ids[:max_len]
                elif pad_to_max and self.pad_token_id is not None:
                    ids.extend([self.pad_token_id] * (max_len - len(ids)))
            return ids

        if self._native is not None:
            return self._native.encode(text, max_len, add_special_tokens, pad_to_max)

        # Fallback pure-python encoder
        if self.pre_tokenizer == "clip":
            import re

            pat = re.compile(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]""",
                re.IGNORECASE,
            )
            bpe_ranks = {}
            for rank, line in enumerate(self.merges):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    bpe_ranks[(parts[0], parts[1])] = rank

            def _bpe_word(word: str) -> list[str]:
                if not word:
                    return []
                symbols = (
                    [word + "</w>"] if len(word) == 1 else list(word[:-1]) + [word[-1] + "</w>"]
                )
                while len(symbols) > 1:
                    min_r = 100000000
                    best_idx = -1
                    for j in range(len(symbols) - 1):
                        p = (symbols[j], symbols[j + 1])
                        if p in bpe_ranks and bpe_ranks[p] < min_r:
                            min_r = bpe_ranks[p]
                            best_idx = j
                    if best_idx == -1:
                        break
                    new_s = []
                    j = 0
                    while j < len(symbols):
                        if j == best_idx:
                            new_s.append(symbols[j] + symbols[j + 1])
                            j += 2
                        else:
                            new_s.append(symbols[j])
                            j += 1
                    symbols = new_s
                return symbols

            tokens = (
                [self.bos_token_id or 49406]
                if add_special_tokens and self.bos_token_id is not None
                else []
            )
            for m in pat.findall(text.lower()):
                if m in ("<|startoftext|>", "<|endoftext|>"):
                    continue
                for piece in _bpe_word(m):
                    tokens.append(self.vocab.get(piece, self.unk_token_id or 49407))
            if add_special_tokens and self.eos_token_id is not None:
                tokens.append(self.eos_token_id)
            if max_len > 0:
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                    if add_special_tokens and self.eos_token_id is not None:
                        tokens[-1] = self.eos_token_id
                elif pad_to_max:
                    tokens.extend([self.pad_token_id or 49407] * (max_len - len(tokens)))
            return tokens

        tokens = []
        if add_special_tokens and self.bos_token_id is not None:
            tokens.append(self.bos_token_id)
        for w in text.split():
            tokens.append(self.vocab.get(w, self.unk_token_id or 0))
        if add_special_tokens and self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        if max_len > 0:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            elif pad_to_max and self.pad_token_id is not None:
                tokens.extend([self.pad_token_id] * (max_len - len(tokens)))
        return tokens

    def decode(self, ids: list[int] | Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]

        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        if self._native is not None:
            return self._native.decode(ids, skip_special_tokens)

        special_set = set()
        if skip_special_tokens:
            for s in (self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id):
                if s is not None:
                    special_set.add(s)

        rev = {v: k for k, v in self.vocab.items()}
        pieces = []
        for i in ids:
            if skip_special_tokens and i in special_set:
                continue
            tok_str = rev.get(i, "")
            if skip_special_tokens and tok_str.startswith("<|") and tok_str.endswith("|>"):
                continue
            pieces.append(tok_str)

        text = "".join(pieces)
        # Byte-level / BPE space replacements
        text = text.replace("Ġ", " ").replace("</w>", " ").replace(" ", " ")
        return text.strip()

    @property
    def spec(self) -> TokenizerSpec:
        return TokenizerSpec(
            model_type="bpe",
            pre_tokenizer=self.pre_tokenizer,
            vocab=self.vocab,
            merges=self.merges,
            context_length=self.context_length,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            unk_token_id=self.unk_token_id,
        )

    @classmethod
    def from_huggingface(cls, tokenizer: Any, context_length: int | None = None) -> BPETokenizer:
        """Extracts BPE vocab and merges from Hugging Face PreTrainedTokenizer or Model ID."""
        hf_tok = None
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained(tokenizer)
            tokenizer = hf_tok
        elif hasattr(tokenizer, "get_vocab"):
            hf_tok = tokenizer

        vocab = tokenizer.get_vocab()

        # Extract merges
        merges: list[str] = []
        if hasattr(tokenizer, "_merges") and tokenizer._merges:
            for pair in tokenizer._merges:
                if isinstance(pair, (list, tuple)):
                    merges.append(f"{pair[0]} {pair[1]}")
                else:
                    merges.append(str(pair))
        elif hasattr(tokenizer, "bpe_ranks") and tokenizer.bpe_ranks:
            # Sort by rank
            sorted_merges = sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])
            for pair, _ in sorted_merges:
                merges.append(f"{pair[0]} {pair[1]}")
        elif hasattr(tokenizer, "_tokenizer"):
            import json

            try:
                tok_data = json.loads(tokenizer._tokenizer.to_str())
                if "model" in tok_data and "merges" in tok_data["model"]:
                    for pair in tok_data["model"]["merges"]:
                        if isinstance(pair, (list, tuple)):
                            merges.append(f"{pair[0]} {pair[1]}")
                        else:
                            merges.append(str(pair))
            except Exception:  # noqa: BLE001, S110
                pass

        # Context length
        max_len = context_length or getattr(tokenizer, "model_max_length", 2048)
        if max_len is None or max_len > 100000:
            max_len = 2048

        # Pre-tokenizer detection
        cls_name = tokenizer.__class__.__name__.lower()
        model_name = str(getattr(tokenizer, "name_or_path", "")).lower()
        if "clip" in cls_name or "clip" in model_name:
            pre_tok = "clip"
        elif "llama" in cls_name or "smol" in model_name or "llama" in model_name:
            pre_tok = "llama"
        elif "gemma" in cls_name or "gemma" in model_name:
            pre_tok = "gemma"
        else:
            pre_tok = "gpt2"

        chat_template = getattr(tokenizer, "chat_template", None)

        inst = cls(
            vocab=vocab,
            merges=merges,
            pre_tokenizer=pre_tok,
            context_length=int(max_len),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            unk_token_id=getattr(tokenizer, "unk_token_id", None),
            chat_template=chat_template,
        )
        inst._hf_tokenizer = hf_tok
        return inst

    def apply_chat_template(
        self,
        messages: list[dict[str, str]] | str,
        system: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Applies chat template formatting for instruction-tuned models."""
        if isinstance(messages, str):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": messages})
        else:
            msgs = list(messages)

        if (
            self._hf_tokenizer is not None
            and hasattr(self._hf_tokenizer, "apply_chat_template")
            and getattr(self._hf_tokenizer, "chat_template", None)
        ):
            try:
                return self._hf_tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception:  # noqa: BLE001, S110
                pass

        # Built-in chat template formatting fallbacks
        is_chatml = (
            self.pre_tokenizer in ("llama", "smol")
            or (self.chat_template and "<|im_start|>" in self.chat_template)
            or (self.vocab and "<|im_start|>" in self.vocab)
        )
        is_gemma = (
            self.pre_tokenizer == "gemma"
            or (self.chat_template and "<start_of_turn>" in self.chat_template)
            or (self.vocab and "<start_of_turn>" in self.vocab)
        )
        is_llama3 = (self.chat_template and "<|start_header_id|>" in self.chat_template) or (
            self.vocab and "<|start_header_id|>" in self.vocab
        )

        formatted = ""
        if is_chatml:
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            if add_generation_prompt:
                formatted += "<|im_start|>assistant\n"
        elif is_gemma:
            for m in msgs:
                role = m.get("role", "user")
                if role == "assistant":
                    role = "model"
                content = m.get("content", "")
                formatted += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
            if add_generation_prompt:
                formatted += "<start_of_turn>model\n"
        elif is_llama3:
            formatted += "<|begin_of_text|>"
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            if add_generation_prompt:
                formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            for m in msgs:
                role = m.get("role", "user").capitalize()
                content = m.get("content", "")
                formatted += f"{role}: {content}\n\n"
            if add_generation_prompt:
                formatted += "Assistant: "

        return formatted

    def to_gguf_metadata(self) -> dict[str, Any]:
        """Generates GGUF key-value metadata pairs for GGUF tokenization specification."""
        token_list = [""] * len(self.vocab)
        for tok, idx in self.vocab.items():
            if idx < len(token_list):
                token_list[idx] = tok

        meta: dict[str, Any] = {
            "tokenizer.ggml.model": "gpt2" if self.pre_tokenizer != "clip" else "clip",
            "tokenizer.ggml.pre": self.pre_tokenizer,
            "tokenizer.ggml.tokens": token_list,
            "tokenizer.ggml.merges": self.merges,
        }
        if self.bos_token_id is not None:
            meta["tokenizer.ggml.bos_token_id"] = int(self.bos_token_id)
        if self.eos_token_id is not None:
            meta["tokenizer.ggml.eos_token_id"] = int(self.eos_token_id)
        if self.pad_token_id is not None:
            meta["tokenizer.ggml.padding_token_id"] = int(self.pad_token_id)
        if self.unk_token_id is not None:
            meta["tokenizer.ggml.unknown_token_id"] = int(self.unk_token_id)
        if self.chat_template:
            meta["tokenizer.chat_template"] = str(self.chat_template)

        return meta


class WordPieceTokenizer:
    """WordPiece Tokenizer specification and helper."""

    def __init__(
        self,
        vocab: dict[str, int] | list[str] | TokenizerSpec,
        context_length: int = 512,
        cls_token_id: int | None = 101,
        sep_token_id: int | None = 102,
        pad_token_id: int | None = 0,
        unk_token_id: int | None = 100,
    ):
        if isinstance(vocab, TokenizerSpec):
            spec = vocab
            if isinstance(spec.vocab, list):
                self.vocab = {tok: idx for idx, tok in enumerate(spec.vocab)}
            else:
                self.vocab = dict(spec.vocab)
            self.context_length = spec.context_length
            self.cls_token_id = spec.bos_token_id
            self.sep_token_id = spec.eos_token_id
            self.pad_token_id = spec.pad_token_id or 0
            self.unk_token_id = spec.unk_token_id or 100
        else:
            if isinstance(vocab, (list, tuple)):
                self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
            else:
                self.vocab = dict(vocab)
            self.context_length = context_length
            self.cls_token_id = cls_token_id
            self.sep_token_id = sep_token_id
            self.pad_token_id = pad_token_id
            self.unk_token_id = unk_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        pad_to_max: bool = False,
    ) -> list[int]:
        tokens: list[int] = []
        if add_special_tokens and self.cls_token_id is not None:
            tokens.append(self.cls_token_id)
        for word in text.strip().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.unk_token_id or 0)
        if add_special_tokens and self.sep_token_id is not None:
            tokens.append(self.sep_token_id)
        if pad_to_max and self.context_length:
            while len(tokens) < self.context_length:
                tokens.append(self.pad_token_id or 0)
        return tokens

    @property
    def spec(self) -> TokenizerSpec:
        return TokenizerSpec(
            model_type="wordpiece",
            pre_tokenizer="bert",
            vocab=self.vocab,
            context_length=self.context_length,
            bos_token_id=self.cls_token_id,
            eos_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id,
            unk_token_id=self.unk_token_id,
        )

    @classmethod
    def from_huggingface(
        cls, tokenizer: Any, context_length: int | None = None
    ) -> WordPieceTokenizer:
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        vocab = tokenizer.get_vocab()
        max_len = context_length or getattr(tokenizer, "model_max_length", 512)
        if max_len is None or max_len > 100000:
            max_len = 512

        return cls(
            vocab=vocab,
            context_length=int(max_len),
            cls_token_id=getattr(tokenizer, "cls_token_id", 101),
            sep_token_id=getattr(tokenizer, "sep_token_id", 102),
            pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
            unk_token_id=getattr(tokenizer, "unk_token_id", 100) or 100,
        )

    def to_gguf_metadata(self) -> dict[str, Any]:
        token_list = [""] * len(self.vocab)
        for tok, idx in self.vocab.items():
            if idx < len(token_list):
                token_list[idx] = tok

        meta = {
            "tokenizer.ggml.model": "bert",
            "tokenizer.ggml.pre": "bert",
            "tokenizer.ggml.tokens": token_list,
        }
        if self.cls_token_id is not None:
            meta["tokenizer.ggml.bos_token_id"] = int(self.cls_token_id)
        if self.sep_token_id is not None:
            meta["tokenizer.ggml.eos_token_id"] = int(self.sep_token_id)
        if self.pad_token_id is not None:
            meta["tokenizer.ggml.padding_token_id"] = int(self.pad_token_id)
        if self.unk_token_id is not None:
            meta["tokenizer.ggml.unknown_token_id"] = int(self.unk_token_id)

        return meta
