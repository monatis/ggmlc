"""Compile Laya English checkpoint (ModernBERT-large DecisionModel) to GGUF via ggmlc."""

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

import torch
import laya
from laya.common import QTYPES, build_sequence, collate_items
from laya_trunk import MAX_LEN, MAX_OPTS, LayaCleanTrunk, pad_batch

import ggmlc
from ggmlc.pipeline.tokenizer import BPETokenizer


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "scratch" / "laya_english_f16.gguf"))
    parser.add_argument("--quantize", default="f16", choices=["f32", "f16", "q8_0", "q4_0"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--min-seq", type=int, default=64)
    args = parser.parse_args()

    print("loading laya…")
    agent = laya.load("convaiinnovations/laya", device="cpu")
    agent.model.cpu().float().eval()

    print("building clean trunk…")
    trunk = LayaCleanTrunk(agent).eval()
    example = _example_batch(agent)
    print("example shapes", [tuple(t.shape) for t in example], [t.dtype for t in example])

    tok = BPETokenizer.from_huggingface(agent.tok, context_length=int(agent.cfg.get("max_len", MAX_LEN)))
    # ModernBERT uses [CLS]/[SEP] rather than BOS/EOS.
    if tok.bos_token_id is None:
        tok.bos_token_id = int(agent.tok.cls_token_id)
    if tok.eos_token_id is None:
        tok.eos_token_id = int(agent.tok.sep_token_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = int(agent.tok.pad_token_id)
    if tok.unk_token_id is None and agent.tok.unk_token_id is not None:
        tok.unk_token_id = int(agent.tok.unk_token_id)
    extra = {
        "laya.max_len": int(agent.cfg.get("max_len", MAX_LEN)),
        "laya.head_max_len": int(agent.cfg.get("head_max_len", 192)),
        "laya.max_opts": MAX_OPTS,
        "laya.max_batch": int(args.max_batch),
        "laya.min_seq": int(args.min_seq),
        "laya.length_buckets": json.dumps([64, 128, 256, 512]),
        "laya.mask_token_id": int(agent.tok.mask_token_id),
        "laya.cls_token_id": int(agent.tok.cls_token_id),
        "laya.sep_token_id": int(agent.tok.sep_token_id),
        "laya.pad_token_id": int(agent.tok.pad_token_id),
        "laya.temperature": json.dumps(agent.temperature),
        "laya.temperature_by_options": json.dumps(agent.temperature_by_options),
        "laya.model_name": "laya",
        "laya.checkpoint": "convaiinnovations/laya",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"compiling -> {out} quantize={args.quantize}")
    from ggmlc.transforms.fusion import FusionOptions

    fusion = FusionOptions()
    # Precomputed cos/sin RoPE (two thetas: full vs sliding) must stay as MUL/ADD.
    # The LLaMA-style ROPE fusion pass would rewrite it to GGML_OP_ROPE with a
    # bogus position tensor and abort at ggml_rope (ne[2] != n_pos).
    fusion.enable_rope = False

    dim_b = torch.export.Dim("b", min=1, max=int(args.max_batch))
    dim_s = torch.export.Dim("s", min=int(args.min_seq), max=MAX_LEN)
    dynamic_shapes = (
        {0: dim_b, 1: dim_s},
        {0: dim_b, 1: dim_s},
        {0: dim_b},
        {0: dim_b},
        {0: dim_b},
    )
    print(f"dynamic dims b=1..{args.max_batch}  s={args.min_seq}..{MAX_LEN}")

    ggmlc.compile(
        trunk,
        example,
        output=str(out),
        model_name="laya_english",
        quantize=args.quantize,
        pipeline=tok,
        tasks=["classification"],
        extra_metadata=extra,
        fusion_options=fusion,
        dynamic_shapes=dynamic_shapes,
    )
    print("wrote", out, "bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
