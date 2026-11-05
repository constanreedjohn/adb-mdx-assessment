"""Prompt files: `config/prompts/<prompt_id>/<version>.md`.

- `prompt_id`: lowercase letters, digits and underscores, e.g. `master_plan`.
- `version`: `v` + a positive integer, e.g. `v1`, `v2`. Versions are never edited
  in place once used; a change is a new file, so every run records which wording it used.

The prompt registry used by the LLM gateway arrives in SP2 (E5-02); this module
owns the naming convention and lookup.
"""

from __future__ import annotations

import re
from pathlib import Path

from mdx_assessment.core.config.loader import ConfigError

PROMPTS_DIR = "prompts"
_PROMPT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^v[1-9][0-9]*$")


def discover_prompts(config_dir: Path) -> dict[str, list[str]]:
    """`{prompt_id: [versions, ascending]}`. Raises ConfigError on any misnamed entry."""
    root = config_dir / PROMPTS_DIR
    if not root.is_dir():
        return {}
    problems: list[str] = []
    found: dict[str, list[str]] = {}
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        rel = f"{PROMPTS_DIR}/{entry.name}"
        if not entry.is_dir() or not _PROMPT_ID.match(entry.name):
            problems.append(f"{rel}: expected a folder named like 'master_plan' (a prompt ID)")
            continue
        versions = []
        for file in sorted(entry.iterdir()):
            if file.name.startswith("."):
                continue
            if not (file.is_file() and file.suffix == ".md" and _VERSION.match(file.stem)):
                problems.append(f"{rel}/{file.name}: expected a file named like 'v1.md'")
                continue
            if not file.read_text(encoding="utf-8").strip():
                problems.append(f"{rel}/{file.name}: prompt is empty")
                continue
            versions.append(file.stem)
        if not versions and not any(p.startswith(rel + "/") for p in problems):
            problems.append(f"{rel}: no prompt versions (add v1.md)")
        found[entry.name] = sorted(versions, key=lambda v: int(v[1:]))
    if problems:
        raise ConfigError(problems)
    return found


def load_prompt(config_dir: Path, prompt_id: str, version: str) -> str:
    path = config_dir / PROMPTS_DIR / prompt_id / f"{version}.md"
    if not (_PROMPT_ID.match(prompt_id) and _VERSION.match(version)) or not path.is_file():
        raise ConfigError([f"{PROMPTS_DIR}/{prompt_id}/{version}.md: prompt not found"])
    return path.read_text(encoding="utf-8")
