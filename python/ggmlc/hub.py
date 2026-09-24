"""Download and load pre-compiled ggmlc GGUF models from the Hugging Face Hub."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ggmlc.runtime.runner import ModelRunner


def _hub():
    try:
        import huggingface_hub as hub
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Hub support requires huggingface_hub. "
            "Install with: pip install 'ggmlc[hub]'"
        ) from exc
    return hub


def _is_combined_ref(text: str) -> bool:
    """True for ``hf://…`` or ``org/repo:file`` (not Windows ``C:…``)."""
    return text.startswith("hf://") or (":" in text and not (len(text) > 1 and text[1] == ":"))


def parse_hub_ref(source: str) -> tuple[str, str | None]:
    """Parse ``org/repo``, ``org/repo:file.gguf``, or ``hf://org/repo/file.gguf``."""
    text = source.strip()
    if not text:
        raise ValueError("Hub reference must be a non-empty string")

    if text.startswith("hf://"):
        parts = text[5:].lstrip("/").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid hf:// reference '{source}'")
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])

    if _is_combined_ref(text):
        repo_id, filename = (p.strip() for p in text.split(":", 1))
        if "/" not in repo_id:
            raise ValueError(f"Invalid Hub reference '{source}' (expected org/repo[:file])")
        return repo_id, filename or None

    if "/" not in text:
        raise ValueError(f"Invalid Hub reference '{source}' (expected org/repo[:file])")
    return text, None


def list_gguf_files(
    repo_id: str, *, revision: str | None = None, token: str | None = None
) -> list[str]:
    """List ``.gguf`` files in a Hugging Face model repository."""
    files = _hub().list_repo_files(repo_id, revision=revision, token=token)
    return sorted(f for f in files if f.lower().endswith(".gguf"))


def _resolve_gguf(
    repo_id: str, filename: str | None, *, revision: str | None, token: str | None
) -> str:
    """Resolve to one repo-relative ``.gguf`` path (exact, glob, or sole file)."""
    if filename and not any(c in filename for c in "*?[]"):
        return filename

    ggufs = list_gguf_files(repo_id, revision=revision, token=token)
    matches = [f for f in ggufs if fnmatch.fnmatch(f, filename)] if filename else ggufs
    label = f"matching '{filename}'" if filename else "in repository"

    if not matches:
        available = "\n".join(f"  - {f}" for f in ggufs) or "  (none)"
        raise FileNotFoundError(f"No .gguf {label} in '{repo_id}'.\nAvailable:\n{available}")
    if len(matches) > 1:
        listing = "\n".join(f"  - {f}" for f in matches)
        raise ValueError(
            f"Multiple .gguf files {label} in '{repo_id}'. Pass filename= to select one:\n{listing}"
        )
    return matches[0]


def download(
    repo_id: str,
    filename: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download a Hub ggmlc ``.gguf`` into the local cache; return its path.

    ``repo_id`` may be ``org/name``, ``org/name:file.gguf``, or ``hf://org/name/file.gguf``.
    ``filename`` is optional when the repo has exactly one ``.gguf``; may be a glob.
    """
    if _is_combined_ref(repo_id):
        repo_id, embedded = parse_hub_ref(repo_id)
        if filename is None:
            filename = embedded

    resolved = _resolve_gguf(repo_id, filename, revision=revision, token=token)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": resolved,
        "revision": revision,
        "token": token,
        "force_download": force_download,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    return Path(_hub().hf_hub_download(**kwargs)).resolve()


def from_pretrained(
    repo_id: str,
    filename: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
    n_threads: int = 1,
    device: str = "cpu",
) -> ModelRunner:
    """Download (cached) a Hub ggmlc GGUF and open it as a ``ModelRunner``."""
    from ggmlc.runtime.runner import load

    return load(
        download(
            repo_id,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            token=token,
            force_download=force_download,
        ),
        n_threads=n_threads,
        device=device,
    )
