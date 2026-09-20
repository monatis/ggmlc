"""Compile Laya family checkpoints (English / multilingual / typed-decisions) to GGUF."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "laya"))

import ggmlc
import laya
import torch
from ggmlc.pipeline.tokenizer import BPETokenizer
from ggmlc.transforms.fusion import FusionOptions
from laya.common import QTYPES, build_sequence, collate_items
from laya_trunk import MAX_OPTS, LayaCleanTrunk, pad_batch

CHECKPOINTS = {
    "english": {
        "repo": "convaiinnovations/laya",
        "family": "english",
        "model_name": "laya",
        "stem": "laya_english",
        "pre_tokenizer": "gpt2",
    },
    "multilingual": {
        "repo": "convaiinnovations/laya-multilingual",
        "family": "multilingual",
        "model_name": "laya-multilingual",
        "stem": "laya_multilingual",
        "pre_tokenizer": "gemma",
    },
    "typed-decisions": {
        "repo": "convaiinnovations/laya-typed-decisions",
        "family": "typed-decisions",
        "model_name": "laya-typed-decisions",
        "stem": "laya_typed_decisions",
        "pre_tokenizer": "gpt2",
    },
}
ALIASES = {
    "laya": "english",
    "en": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
}
QUANT_CHOICES = ["f32", "f16", "q8_0", "q4_0", "q4_k_m", "ud_q4_k_m"]


def _normalize_family(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in CHECKPOINTS:
        raise SystemExit(f"unknown family {name!r}; choose {list(CHECKPOINTS)}")
    return key


def _length_buckets(max_len: int) -> list[int]:
    return [b for b in (64, 128, 256, 512, 1024) if b <= max_len] or [max_len]


def _example_batch(agent, seq_len: int = 128) -> tuple[torch.Tensor, ...]:
    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    }
    qdefs = [
        {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        {"type": "noul", "instructions": "Does the customer ask for money back?"},
    ]
    items = []
    for qdef in qdefs:
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    return pad_batch(
        batch["input_ids"],
        batch["attention_mask"],
        batch["marker_pos"],
        batch["marker_mask"],
        batch["qtype"],
        pad_id=int(agent.tok.pad_token_id),
        seq_len=seq_len,
    )


def _pipeline_tokenizer(agent, pre_tokenizer: str, max_len: int) -> BPETokenizer:
    base = BPETokenizer.from_huggingface(agent.tok, context_length=max_len)
    tok = BPETokenizer(
        vocab=base.vocab,
        merges=base.merges,
        pre_tokenizer=pre_tokenizer,
        context_length=max_len,
        bos_token_id=base.bos_token_id,
        eos_token_id=base.eos_token_id,
        pad_token_id=base.pad_token_id,
        unk_token_id=base.unk_token_id,
        chat_template=base.chat_template,
    )
    if tok.bos_token_id is None:
        tok.bos_token_id = int(agent.tok.cls_token_id)
    if tok.eos_token_id is None:
        tok.eos_token_id = int(agent.tok.sep_token_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = int(agent.tok.pad_token_id)
    if tok.unk_token_id is None and agent.tok.unk_token_id is not None:
        tok.unk_token_id = int(agent.tok.unk_token_id)
    print(
        f"tokenizer pre={tok.pre_tokenizer} vocab={tok.vocab_size} merges={len(tok.merges)} "
        f"cls={agent.tok.cls_token_id} sep={agent.tok.sep_token_id} "
        f"pad={agent.tok.pad_token_id} mask={agent.tok.mask_token_id}"
    )
    if pre_tokenizer == "gemma" and len(tok.merges) < 1000:
        raise SystemExit("multilingual tokenizer expected Gemma BPE merges; got too few")
    return tok


def compile_one(family: str, quantize: str, output: Path | None, max_batch: int, min_seq: int) -> Path:
    spec = CHECKPOINTS[family]
    print(f"loading {spec['repo']} …")
    agent = laya.load(spec["repo"], device="cpu")
    agent.model.cpu().float().eval()
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    min_seq = min(min_seq, max_len)
    min_seq = max(min_seq, 1)

    print(f"building clean trunk  max_len={max_len} head_max_len={head_max_len}")
    trunk = LayaCleanTrunk(agent, max_len=max_len).eval()
    example = _example_batch(agent, seq_len=min(128, max_len))
    print("example shapes", [tuple(t.shape) for t in example], [t.dtype for t in example])

    tok = _pipeline_tokenizer(agent, spec["pre_tokenizer"], max_len)
    extra = {
        "laya.max_len": max_len,
        "laya.head_max_len": head_max_len,
        "laya.max_opts": MAX_OPTS,
        "laya.max_batch": int(max_batch),
        "laya.min_seq": int(min_seq),
        "laya.length_buckets": json.dumps(_length_buckets(max_len)),
        "laya.mask_token_id": int(agent.tok.mask_token_id),
        "laya.cls_token_id": int(agent.tok.cls_token_id),
        "laya.sep_token_id": int(agent.tok.sep_token_id),
        "laya.pad_token_id": int(agent.tok.pad_token_id),
        "laya.temperature": json.dumps(agent.temperature),
        "laya.temperature_by_options": json.dumps(agent.temperature_by_options),
        "laya.model_name": spec["model_name"],
        "laya.family": spec["family"],
        "laya.checkpoint": spec["repo"],
    }

    if output is None:
        suffix = quantize.replace("-", "_")
        output = ROOT / "scratch" / f"{spec['stem']}_{suffix}.gguf"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fusion = FusionOptions()
    fusion.enable_rope = False

    dim_b = torch.export.Dim("b", min=1, max=int(max_batch))
    dim_s = torch.export.Dim("s", min=int(min_seq), max=max_len)
    dynamic_shapes = (
        {0: dim_b, 1: dim_s},
        {0: dim_b, 1: dim_s},
        {0: dim_b},
        {0: dim_b},
        {0: dim_b},
    )
    print(f"compiling -> {output} family={family} quantize={quantize} b=1..{max_batch} s={min_seq}..{max_len}")
    ggmlc.compile(
        trunk,
        example,
        output=str(output),
        model_name=spec["stem"],
        quantize=quantize,
        pipeline=tok,
        tasks=["classification"],
        extra_metadata=extra,
        fusion_options=fusion,
        dynamic_shapes=dynamic_shapes,
    )
    print("wrote", output, "bytes", output.stat().st_size)
    del agent, trunk
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Laya checkpoints to GGUF (English, multilingual, typed-decisions)."
    )
    parser.add_argument("--family", default="english", help="english | multilingual | typed-decisions | all")
    parser.add_argument("--checkpoint", default=None, help="Alias for --family (HF repo or short name)")
    parser.add_argument("--output", default=None, help="Output GGUF path (single family only)")
    parser.add_argument("--quantize", default="f16", choices=QUANT_CHOICES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--min-seq", type=int, default=64)
    args = parser.parse_args()

    fam_arg = args.checkpoint or args.family
    families = list(CHECKPOINTS) if fam_arg.strip().lower() == "all" else [_normalize_family(fam_arg)]
    if args.output and len(families) > 1:
        raise SystemExit("--output can only be used with a single --family")

    for fam in families:
        compile_one(fam, args.quantize, Path(args.output) if args.output else None, args.max_batch, args.min_seq)


if __name__ == "__main__":
    main()
