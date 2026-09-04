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
        pad_to_max: bool = True,
    ) -> list[int]:
        max_len = max_length or self.context_length
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

        return (
            [self.bos_token_id or 0]
            + [self.vocab.get(w, self.unk_token_id or 0) for w in text.split()]
            + [self.eos_token_id or 0]
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if self._native is not None:
            return self._native.decode(ids, skip_special_tokens)
        rev = {v: k for k, v in self.vocab.items()}
        return "".join([rev.get(i, "") for i in ids])

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
        """Extracts BPE vocab and merges from Hugging Face PreTrainedTokenizer."""
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
        max_len = context_length or getattr(tokenizer, "model_max_length", 77)
        if max_len is None or max_len > 100000:
            max_len = 77

        return cls(
            vocab=vocab,
            merges=merges,
            pre_tokenizer="clip",
            context_length=int(max_len),
            bos_token_id=getattr(tokenizer, "bos_token_id", 49406),
            eos_token_id=getattr(tokenizer, "eos_token_id", 49407),
            pad_token_id=getattr(tokenizer, "pad_token_id", 49407),
            unk_token_id=getattr(tokenizer, "unk_token_id", 49407),
        )

    def to_gguf_metadata(self) -> dict[str, Any]:
        """Generates GGUF key-value metadata pairs for GGUF tokenization specification."""
        token_list = [""] * len(self.vocab)
        for tok, idx in self.vocab.items():
            if idx < len(token_list):
                token_list[idx] = tok

        meta = {
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
        pad_to_max: bool = True,
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
        vocab = tokenizer.get_vocab()
        max_len = context_length or getattr(tokenizer, "model_max_length", 512)
        if max_len is None or max_len > 100000:
            max_len = 512

        return cls(
            vocab=vocab,
            context_length=int(max_len),
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
            unk_token_id=tokenizer.unk_token_id or 100,
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
