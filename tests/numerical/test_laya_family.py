"""Language routing + extra Laya GGUF families / quants (skip if artifacts missing)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

LAYA_EXE_CANDIDATES = [
    ROOT / "build-win-cuda" / "examples" / "laya" / "laya.exe",
    ROOT / "build-win" / "examples" / "laya" / "Release" / "laya.exe",
    ROOT / "build" / "examples" / "laya" / "laya",
]


def _laya_exe() -> Path | None:
    for p in LAYA_EXE_CANDIDATES:
        if p.exists():
            return p
    return None


ROUTE_CASES = [
    ("I was charged twice for this invoice, please refund today.", "english"),
    ("The billing department should handle refunds and invoices.", "english"),
    ("मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया।", "multilingual"),
    ("二重に請求されました。今日返金してください。", "multilingual"),
    ("Bitte erstatten Sie die doppelte Abbuchung auf meinem Konto.", "multilingual"),
]


@pytest.mark.parametrize("text,family", ROUTE_CASES)
def test_detect_lang_cli(text, family):
    exe = _laya_exe()
    if exe is None:
        pytest.skip("laya.exe not built")
    proc = subprocess.run(
        [str(exe), "detect-lang", "--text", text],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert f"route: {family}" in out, out


def test_detect_lang_short_latin_defaults_english():
    exe = _laya_exe()
    if exe is None:
        pytest.skip("laya.exe not built")
    proc = subprocess.run(
        [str(exe), "detect-lang", "--text", "INV-4411"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "route: english" in proc.stdout


def _gguf_cosine(gguf: Path, repo: str, min_cos: float) -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import sys

    import laya
    import torch
    from laya.common import QTYPES, build_sequence, collate_items

    sys.path.insert(0, str(ROOT / "examples" / "laya"))
    from laya_trunk import LayaCleanTrunk, pad_batch
    from test_laya_differential import EMAIL_Q, EMAIL_STATE, _gguf_option_logits

    agent = laya.load(repo, device="cpu")
    agent.model.cpu().float().eval()
    trunk = LayaCleanTrunk(agent, max_len=int(agent.cfg.get("max_len", 512))).eval()
    q = agent._to_internal(EMAIL_Q)
    seq, markers = build_sequence(agent.tok, EMAIL_STATE, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
    items = [{"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}]
    batch = collate_items([items], agent.tok.pad_token_id)
    padded = pad_batch(
        batch["input_ids"][:1],
        batch["attention_mask"][:1],
        batch["marker_pos"][:1],
        batch["marker_mask"][:1],
        batch["qtype"][:1],
        pad_id=int(agent.tok.pad_token_id),
        seq_len=min(128, int(agent.cfg["max_len"])),
    )
    k = len(markers)
    with torch.no_grad():
        tl, _ = trunk(*padded)
    gl, _ = _gguf_option_logits(gguf, padded, k)
    gl = np.asarray(gl, dtype=np.float32).reshape(-1)[:k]
    tl_np = tl[0, :k].float().numpy()
    cos = float(np.dot(tl_np, gl) / (np.linalg.norm(tl_np) * np.linalg.norm(gl) + 1e-12))
    assert cos > min_cos, f"{gguf.name} cosine {cos}"


FAMILY_GGUFS = [
    (ROOT / "scratch" / "laya_english_q8_0.gguf", "convaiinnovations/laya", 0.97),
    (ROOT / "scratch" / "laya_english_ud_q4_k_m.gguf", "convaiinnovations/laya", 0.94),
    (ROOT / "scratch" / "laya_multilingual_f16.gguf", "convaiinnovations/laya-multilingual", 0.99),
    (ROOT / "scratch" / "laya_multilingual_q8_0.gguf", "convaiinnovations/laya-multilingual", 0.97),
    (ROOT / "scratch" / "laya_typed_decisions_f16.gguf", "convaiinnovations/laya-typed-decisions", 0.99),
]


@pytest.mark.parametrize("gguf,repo,min_cos", FAMILY_GGUFS)
def test_family_or_quant_gguf_parity(gguf, repo, min_cos):
    if not Path(gguf).exists():
        pytest.skip(f"{gguf} not compiled")
    _gguf_cosine(Path(gguf), repo, min_cos)
