"""Unit tests for Hugging Face Hub download / from_pretrained helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ggmlc.hub import download, list_gguf_files, parse_hub_ref


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("mys/laya-GGUF", ("mys/laya-GGUF", None)),
        ("mys/laya-GGUF:laya_english_q8_0.gguf", ("mys/laya-GGUF", "laya_english_q8_0.gguf")),
        ("mys/laya-GGUF:subdir/model.gguf", ("mys/laya-GGUF", "subdir/model.gguf")),
        ("hf://mys/laya-GGUF/laya_english_q8_0.gguf", ("mys/laya-GGUF", "laya_english_q8_0.gguf")),
    ],
)
def test_parse_hub_ref(src: str, expected: tuple[str, str | None]):
    assert parse_hub_ref(src) == expected


@pytest.mark.parametrize("src", ["", "not-a-repo", "hf://mys/only-two"])
def test_parse_hub_ref_invalid(src: str):
    with pytest.raises(ValueError):
        parse_hub_ref(src)


@patch("ggmlc.hub._hub")
def test_list_and_download(mock_hub_fn: MagicMock, tmp_path: Path):
    cached = tmp_path / "only.gguf"
    cached.write_bytes(b"GGUF")
    hub = MagicMock()
    hub.list_repo_files.return_value = ["README.md", "only.gguf"]
    hub.hf_hub_download.return_value = str(cached)
    mock_hub_fn.return_value = hub

    assert list_gguf_files("org/single") == ["only.gguf"]
    assert download("org/single") == cached.resolve()
    assert hub.hf_hub_download.call_args.kwargs["filename"] == "only.gguf"

    hub.list_repo_files.return_value = ["a.gguf", "b.gguf"]
    with pytest.raises(ValueError, match="Multiple"):
        download("org/multi")

    hub.list_repo_files.return_value = ["weights/model_f16.gguf", "weights/model_q8_0.gguf"]
    download("org/repo", filename="*q8_0.gguf")
    assert hub.hf_hub_download.call_args.kwargs["filename"] == "weights/model_q8_0.gguf"

    download("mys/laya-GGUF:laya_english_q8_0.gguf")
    kw = hub.hf_hub_download.call_args.kwargs
    assert kw["repo_id"] == "mys/laya-GGUF"
    assert kw["filename"] == "laya_english_q8_0.gguf"

    # Explicit -f wins over embedded filename in combined ref
    download("mys/laya-GGUF:ignored.gguf", filename="picked.gguf")
    kw = hub.hf_hub_download.call_args.kwargs
    assert kw["repo_id"] == "mys/laya-GGUF"
    assert kw["filename"] == "picked.gguf"


@patch("ggmlc.hub.download")
@patch("ggmlc.runtime.runner.load")
def test_from_pretrained(mock_load: MagicMock, mock_download: MagicMock, tmp_path: Path):
    from ggmlc.hub import from_pretrained

    path = tmp_path / "m.gguf"
    mock_download.return_value = path
    mock_load.return_value = MagicMock()

    runner = from_pretrained("mys/laya-GGUF", filename="m.gguf", device="cuda", n_threads=4)
    mock_load.assert_called_once_with(path, n_threads=4, device="cuda")
    assert runner is mock_load.return_value


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from ggmlc.cli import main

    cached = tmp_path / "m.gguf"
    cached.write_bytes(b"GGUF")
    with patch("ggmlc.hub.download", return_value=cached.resolve()):
        assert main(["download", "mys/laya-GGUF", "-f", "m.gguf"]) == 0
    assert str(cached.resolve()) in capsys.readouterr().out

    with patch("ggmlc.hub.list_gguf_files", return_value=["a.gguf", "b.gguf"]):
        assert main(["list", "mys/laya-GGUF"]) == 0
    out = capsys.readouterr().out
    assert "a.gguf" in out and "b.gguf" in out
