"""Config schema registry: who owns which YAML file, and validation of the whole tree.

Core and each use-case pack register the files they own (an exact relative path
or a glob such as `playbooks/*.yaml`) with a Pydantic model. `load_config`
then validates every YAML file under `config/` and fails if:

- a file doesn't match its model,
- a required file is missing,
- a YAML file isn't owned by any registered schema (catches typos in file names
  and leftovers; a file nobody reads is a config change with no effect).

All problems are collected and reported together.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from mdx_assessment.core.config.loader import ConfigError, load_model
from mdx_assessment.core.config.prompts import discover_prompts
from mdx_assessment.core.config.version import config_files, config_version

YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class ConfigSchema:
    pattern: str  # relative POSIX path or glob, e.g. "expected_files.yaml", "playbooks/*.yaml"
    model: type[BaseModel]
    owner: str  # "core" or a use-case ID, for error messages
    required: bool = True  # at least one matching file must exist

    def matches(self, relative: str) -> bool:
        return fnmatch.fnmatchcase(relative, self.pattern)


@dataclass(frozen=True)
class LoadedConfig:
    """A validated config tree. `documents` maps each YAML file's relative path to its model."""

    config_dir: Path
    version: str
    documents: dict[str, BaseModel]
    prompts: dict[str, list[str]]

    def get[M: BaseModel](self, relative: str, model: type[M]) -> M:
        doc = self.documents.get(relative)
        if not isinstance(doc, model):
            raise KeyError(f"no {model.__name__} config at {relative!r}")
        return doc

    def all_of[M: BaseModel](self, model: type[M]) -> dict[str, M]:
        return {k: v for k, v in self.documents.items() if isinstance(v, model)}


@dataclass
class ConfigSchemaRegistry:
    _schemas: list[ConfigSchema] = field(default_factory=list)

    def register(
        self, pattern: str, model: type[BaseModel], *, owner: str, required: bool = True
    ) -> None:
        if any(s.pattern == pattern for s in self._schemas):
            raise ValueError(f"config pattern {pattern!r} is already registered")
        self._schemas.append(ConfigSchema(pattern, model, owner, required))

    @property
    def schemas(self) -> tuple[ConfigSchema, ...]:
        return tuple(self._schemas)

    def load_config(self, config_dir: Path) -> LoadedConfig:
        """Validate every YAML file and the prompts folder. Raises ConfigError with all problems."""
        problems: list[str] = []
        documents: dict[str, BaseModel] = {}
        yaml_files = [f.as_posix() for f in config_files(config_dir) if f.suffix in YAML_SUFFIXES]

        for rel in yaml_files:
            owners = [s for s in self._schemas if s.matches(rel)]
            if not owners:
                problems.append(f"{rel}: no registered schema owns this file")
                continue
            if len(owners) > 1:
                names = ", ".join(f"{s.pattern} ({s.owner})" for s in owners)
                problems.append(f"{rel}: claimed by more than one schema: {names}")
                continue
            try:
                documents[rel] = load_model(config_dir / rel, owners[0].model, display_name=rel)
            except ConfigError as exc:
                problems.extend(exc.problems)

        for schema in self._schemas:
            if schema.required and not any(schema.matches(rel) for rel in yaml_files):
                problems.append(f"{schema.pattern}: required by {schema.owner} but missing")

        prompts: dict[str, list[str]] = {}
        try:
            prompts = discover_prompts(config_dir)
        except ConfigError as exc:
            problems.extend(exc.problems)

        if problems:
            raise ConfigError(problems)
        return LoadedConfig(config_dir, config_version(config_dir), documents, prompts)
