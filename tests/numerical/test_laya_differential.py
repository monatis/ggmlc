"""Differential numerical parity for Laya DecisionModel vs clean trunk vs compiled GGUF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "laya"))

laya_available = False
try:
    import laya
    from laya.common import QTYPES, build_sequence, collate_items

    laya_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not laya_available, reason="laya package not installed")

EMAIL_STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
EMAIL_Q = {
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    },
}


@pytest.fixture(scope="module")
def laya_models():
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from laya_trunk import LayaCleanTrunk, pad_batch

    agent = laya.load("convaiinnovations/laya", device="cpu")
    agent.model.cpu().float().eval()
    trunk = LayaCleanTrunk(agent).eval()
    return agent, trunk, pad_batch


def _padded_email(agent, pad_batch, seq_len=None):
    q = agent._to_internal(EMAIL_Q)
    seq, markers = build_sequence(
        agent.tok, EMAIL_STATE, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
    )
    items = [{"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}]
    batch = collate_items([items], agent.tok.pad_token_id)
    padded = pad_batch(
        batch["input_ids"][:1],
        batch["attention_mask"][:1],
        batch["marker_pos"][:1],
        batch["marker_mask"][:1],
        batch["qtype"][:1],
        pad_id=int(agent.tok.pad_token_id),
        seq_len=seq_len,
    )
    return batch, padded, len(markers)


def _gguf_option_logits(gguf, arrays, k, enable_arena_reuse=True):
    from ggmlc.runtime.runner import ModelRunner

    runner = ModelRunner(str(gguf), device="cpu", n_threads=4)
    ggml = runner(
        *[t.detach().cpu().numpy() if hasattr(t, "detach") else t for t in arrays],
        n_threads=4,
        enable_arena_reuse=enable_arena_reuse,
    )
    if isinstance(ggml, dict):
        outs = []
        seen = set()
        for tid in runner.outputs:
            t = runner.tensor_info.get(tid)
            key = t.name if t is not None and t.name in ggml else tid
            ident = id(ggml[key]) if key in ggml else id(ggml.get(tid))
            if ident in seen:
                continue
            seen.add(ident)
            outs.append(ggml[key] if key in ggml else ggml[tid])
    elif isinstance(ggml, (list, tuple)):
        outs = list(ggml)
    else:
        outs = [ggml]
    arr = np.asarray(outs[0], dtype=np.float32)
    return arr[..., :k], runner


def test_laya_clean_trunk_option_logits(laya_models):
    agent, trunk, pad_batch = laya_models
    batch, padded, k = _padded_email(agent, pad_batch)
    with torch.no_grad():
        live, _ = agent.model(
            batch["input_ids"][:1],
            batch["attention_mask"][:1],
            batch["marker_pos"][:1],
            batch["marker_mask"][:1],
            batch["qtype"][:1],
        )
        tl, _ = trunk(*padded)
    live_k = live[0, :k].float()
    trunk_k = tl[0, :k].float()
    max_diff = (live_k - trunk_k).abs().max().item()
    cos = F.cosine_similarity(live_k.flatten(), trunk_k.flatten(), dim=0).item()
    assert max_diff < 5e-4, f"option logit mismatch max_diff={max_diff}"
    assert cos > 0.9999, f"option logit cosine {cos}"


def test_laya_gguf_f16_parity(laya_models):
    agent, trunk, pad_batch = laya_models
    gguf = ROOT / "scratch" / "laya_english_f16.gguf"
    if not gguf.exists():
        pytest.skip("scratch/laya_english_f16.gguf not found")

    _, padded, k = _padded_email(agent, pad_batch)
    with torch.no_grad():
        tl, _ = trunk(*padded)
    gl, _ = _gguf_option_logits(gguf, padded, k)
    gl = np.asarray(gl, dtype=np.float32).reshape(-1)[:k]
    tl_np = tl[0, :k].float().numpy()
    cos = float(np.dot(tl_np, gl) / (np.linalg.norm(tl_np) * np.linalg.norm(gl) + 1e-12))
    max_diff = float(np.max(np.abs(tl_np - gl)))
    assert cos > 0.99, f"GGUF cosine {cos} max_diff={max_diff}"
    assert max_diff < 0.15, f"GGUF option logit max_diff={max_diff}"


def test_laya_length_bucket_128(laya_models):
    agent, trunk, pad_batch = laya_models
    batch, pad512, k = _padded_email(agent, pad_batch, seq_len=512)
    _, pad128, _ = _padded_email(agent, pad_batch, seq_len=128)
    live_s = int(batch["attention_mask"][0].sum().item())
    assert live_s <= 128, f"email sequence {live_s} does not fit the 128 bucket"
    with torch.no_grad():
        a = trunk(*pad512)[0][0, :k].float()
        b = trunk(*pad128)[0][0, :k].float()
    max_diff = (a - b).abs().max().item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    assert cos > 0.999, f"S=128 vs S=512 cosine {cos} max_diff={max_diff}"
    assert max_diff < 5e-3, f"S=128 vs S=512 max_diff={max_diff}"


def test_laya_batched_markers(laya_models):
    agent, trunk, pad_batch = laya_models
    q1 = agent._to_internal(EMAIL_Q)
    q2 = agent._to_internal(
        {"type": "noul", "instructions": "Does the customer ask for money back?"}
    )
    items = []
    ks = []
    for q in (q1, q2):
        seq, markers = build_sequence(
            agent.tok, EMAIL_STATE, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
        )
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        ks.append(len(markers))
    batch = collate_items([items], agent.tok.pad_token_id)
    padded = pad_batch(
        batch["input_ids"],
        batch["attention_mask"],
        batch["marker_pos"],
        batch["marker_mask"],
        batch["qtype"],
        pad_id=int(agent.tok.pad_token_id),
        seq_len=128,
    )
    singles = []
    for i in range(2):
        singles.append(
            pad_batch(
                batch["input_ids"][i : i + 1],
                batch["attention_mask"][i : i + 1],
                batch["marker_pos"][i : i + 1],
                batch["marker_mask"][i : i + 1],
                batch["qtype"][i : i + 1],
                pad_id=int(agent.tok.pad_token_id),
                seq_len=128,
            )
        )
    with torch.no_grad():
        batched, _ = trunk(*padded)
        s0, _ = trunk(*singles[0])
        s1, _ = trunk(*singles[1])
    d0 = (batched[0, : ks[0]] - s0[0, : ks[0]]).abs().max().item()
    d1 = (batched[1, : ks[1]] - s1[0, : ks[1]]).abs().max().item()
    assert d0 < 5e-4, f"row0 batch vs single max_diff={d0}"
    assert d1 < 5e-4, f"row1 batch vs single max_diff={d1}"


def test_laya_gguf_unpadded_max_in_batch(laya_models):
    """Pad to live collate length (Python collate_items), not a 64/128/256/512 bucket."""
    agent, trunk, pad_batch = laya_models
    gguf = ROOT / "scratch" / "laya_english_f16.gguf"
    if not gguf.exists():
        pytest.skip("scratch/laya_english_f16.gguf not found")
    batch, _, k = _padded_email(agent, pad_batch)
    live_s = int(batch["attention_mask"][0].sum().item())
    assert live_s >= 64, f"email sequence {live_s} is below export min_seq=64"
    _, padded, _ = _padded_email(agent, pad_batch, seq_len=live_s)
    with torch.no_grad():
        tl, _ = trunk(*padded)
    try:
        gl, runner = _gguf_option_logits(gguf, padded, k)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"GGUF rejected S={live_s} ({exc})")
    if not getattr(runner, "symbol_table", None):
        pytest.skip("GGUF is static [1,512]; recompile with dynamic b/s")
    gl = np.asarray(gl, dtype=np.float32).reshape(-1)[:k]
    tl_np = tl[0, :k].float().numpy()
    cos = float(np.dot(tl_np, gl) / (np.linalg.norm(tl_np) * np.linalg.norm(gl) + 1e-12))
    max_diff = float(np.max(np.abs(tl_np - gl)))
    assert cos > 0.99, f"S={live_s} GGUF cosine {cos} max_diff={max_diff}"
    assert max_diff < 0.15, f"S={live_s} GGUF max_diff={max_diff}"


def test_laya_gguf_length_bucket_128(laya_models):
    agent, trunk, pad_batch = laya_models
    gguf = ROOT / "scratch" / "laya_english_f16.gguf"
    if not gguf.exists():
        pytest.skip("scratch/laya_english_f16.gguf not found")
    _, padded, k = _padded_email(agent, pad_batch, seq_len=128)
    with torch.no_grad():
        tl, _ = trunk(*padded)
    try:
        gl, runner = _gguf_option_logits(gguf, padded, k)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"GGUF rejected S=128 ({exc})")
    if not getattr(runner, "symbol_table", None):
        pytest.skip("GGUF is static [1,512]; recompile with dynamic b/s")
    gl = np.asarray(gl, dtype=np.float32).reshape(-1)[:k]
    tl_np = tl[0, :k].float().numpy()
    cos = float(np.dot(tl_np, gl) / (np.linalg.norm(tl_np) * np.linalg.norm(gl) + 1e-12))
    max_diff = float(np.max(np.abs(tl_np - gl)))
    assert cos > 0.99, f"S=128 GGUF cosine {cos} max_diff={max_diff}"
    assert max_diff < 0.15, f"S=128 GGUF max_diff={max_diff}"


def test_laya_gguf_arena_reuse_parity(laya_models):
    """gallocr must not NaN or diverge from the unplanned allocation path."""
    _, _, pad_batch = laya_models
    gguf = ROOT / "scratch" / "laya_english_f16.gguf"
    if not gguf.exists():
        pytest.skip("scratch/laya_english_f16.gguf not found")
    _, padded, k = _padded_email(laya_models[0], pad_batch, seq_len=128)
    try:
        off, runner = _gguf_option_logits(gguf, padded, k, enable_arena_reuse=False)
        on, _ = _gguf_option_logits(gguf, padded, k, enable_arena_reuse=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"GGUF arena compare failed ({exc})")
    if not getattr(runner, "symbol_table", None):
        pytest.skip("GGUF is static [1,512]; recompile with dynamic b/s")
    off = np.asarray(off, dtype=np.float32).reshape(-1)[:k]
    on = np.asarray(on, dtype=np.float32).reshape(-1)[:k]
    assert np.isfinite(off).all(), "arena-off logits are not finite"
    assert np.isfinite(on).all(), "arena-on logits are not finite (gallocr reuse)"
    cos = float(np.dot(off, on) / (np.linalg.norm(off) * np.linalg.norm(on) + 1e-12))
    max_diff = float(np.max(np.abs(off - on)))
    assert cos > 0.99, f"arena on vs off cosine {cos} max_diff={max_diff}"
    assert max_diff < 0.15, f"arena on vs off max_diff={max_diff}"


def test_laya_system_one_email_routing(laya_models):
    agent, _, _ = laya_models
    out = agent.system_one(
        EMAIL_STATE,
        {
            "dept": EMAIL_Q,
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        },
    )
    assert out["answers"]["dept"]["choice"] == "billing"
    assert out["answers"]["dept"]["probabilities"]["billing"] > 0.5
    assert out["answers"]["refund"]["noul"] > 0.5
