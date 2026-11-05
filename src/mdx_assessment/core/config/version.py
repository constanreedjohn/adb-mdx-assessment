"""Config version: one hash over everything in `config/` (PDF §3.4 versioned configuration).

Every session records this version, so a run can be tied to the exact playbooks,
thresholds, profiles and prompts it used, and replayed.

How it's computed:
1. List every regular file under `config/`, skipping hidden files and folders
   (e.g. `.DS_Store`, editor swap files) so they can't change the version.
2. For each file, take its path relative to `config/` (POSIX form) and the SHA-256
   of its bytes.
3. Sort by path and hash the lines `<path>\\t<file sha256>\\n`.

So editing, adding, removing or renaming any file changes the version; nothing else does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

VERSION_PREFIX = "sha256:"


def _is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def config_files(config_dir: Path) -> list[Path]:
    """Versioned files under `config_dir`, as relative paths, sorted."""
    if not config_dir.is_dir():
        raise FileNotFoundError(f"config directory not found: {config_dir}")
    files = (p.relative_to(config_dir) for p in config_dir.rglob("*") if p.is_file())
    return sorted((f for f in files if not _is_hidden(f)), key=lambda p: p.as_posix())


def config_manifest(config_dir: Path) -> dict[str, str]:
    """`{relative path: sha256 of its bytes}` for every versioned file. Useful for diffs."""
    return {
        rel.as_posix(): hashlib.sha256((config_dir / rel).read_bytes()).hexdigest()
        for rel in config_files(config_dir)
    }


def config_version(config_dir: Path) -> str:
    """`sha256:<hex>` over the manifest (see module docstring)."""
    lines = "".join(f"{path}\t{digest}\n" for path, digest in config_manifest(config_dir).items())
    return VERSION_PREFIX + hashlib.sha256(lines.encode("utf-8")).hexdigest()
