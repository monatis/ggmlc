"""CLI tools for ggmlc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _hub_kwargs(args: argparse.Namespace) -> dict:
    return {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_dir": args.local_dir,
        "token": args.token,
        "force_download": args.force_download,
    }


def _add_hub_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-f", "--filename", help="GGUF file, path, or glob (e.g. '*q8_0.gguf')")
    parser.add_argument("--revision", help="Hub revision (branch, tag, or commit)")
    parser.add_argument("--cache-dir", help="Override Hub cache directory")
    parser.add_argument("--local-dir", help="Download into this directory instead of Hub cache")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN)")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if cached")


def _cmd_download(args: argparse.Namespace) -> int:
    from ggmlc.hub import download

    print(download(args.repo, args.filename, **_hub_kwargs(args)))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from ggmlc.hub import list_gguf_files, parse_hub_ref

    repo_id, _ = parse_hub_ref(args.repo)
    files = list_gguf_files(repo_id, revision=args.revision, token=args.token)
    if not files:
        print(f"No .gguf files found in '{repo_id}'", file=sys.stderr)
        return 1
    print("\n".join(files))
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    from ggmlc.hub import download
    from ggmlc.runtime.runner import load

    source = Path(args.source)
    path = (
        source.resolve()
        if source.is_file()
        else download(args.source, args.filename, **_hub_kwargs(args))
    )
    print(f"Loading {path} on device={args.device} ...", file=sys.stderr)
    runner = load(path, n_threads=args.threads, device=args.device)
    print(f"name={runner.name}")
    print(f"device={runner.device}")
    print(f"inputs={runner.inputs}")
    print(f"outputs={runner.outputs}")
    print(f"path={path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggmlc", description="ggmlc — Hub download and GGUF tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download a Hub ggmlc GGUF into cache; print local path")
    p_dl.add_argument("repo", help="org/name, org/name:file.gguf, or hf://org/name/file.gguf")
    _add_hub_args(p_dl)
    p_dl.set_defaults(func=_cmd_download)

    p_list = sub.add_parser("list", help="List .gguf files in a Hub repo")
    p_list.add_argument("repo", help="org/name")
    p_list.add_argument("--revision")
    p_list.add_argument("--token")
    p_list.set_defaults(func=_cmd_list)

    p_load = sub.add_parser(
        "load", aliases=["from-pretrained"], help="Cache (if Hub) and open a ggmlc GGUF"
    )
    p_load.add_argument("source", help="Local .gguf path or Hub ref")
    _add_hub_args(p_load)
    p_load.add_argument("--device", default="auto", help="cpu | cuda | cuda:0 | metal | auto")
    p_load.add_argument("-t", "--threads", type=int, default=1)
    p_load.set_defaults(func=_cmd_load)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 — CLI boundary (incl. huggingface_hub errors)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
