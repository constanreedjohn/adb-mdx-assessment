"""YAML → Pydantic loading with readable errors.

- `yaml.safe_load` semantics (no arbitrary Python objects).
- Duplicate keys are errors. Plain PyYAML keeps the last one silently, which hides
  config mistakes such as a threshold written twice.
- Errors name the file and, for schema problems, the field path and line-free
  location, e.g. `profiles/data_types.yaml: profiles.rate.near_zero_floor: Input
  should be greater than 0`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class ConfigError(Exception):
    """One or more configuration files are invalid. `problems` lists each one."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"invalid configuration ({len(problems)} problem(s)):\n{lines}")


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_yaml(path: Path, *, display_name: str | None = None) -> Any:
    """Parse one YAML file. Raises ConfigError with the file name and line on failure."""
    name = display_name or str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError([f"{name}: cannot read file ({exc.strerror})"]) from exc
    try:
        return yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 (safe loader subclass)
    except yaml.YAMLError as exc:
        raise ConfigError([f"{name}: invalid YAML: {_one_line(str(exc))}"]) from exc


def format_validation_error(name: str, exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "(document)"
        problems.append(f"{name}: {location}: {err['msg']}")
    return problems


def load_model[M: BaseModel](path: Path, model: type[M], *, display_name: str | None = None) -> M:
    """Read `path` and validate it as `model`. Raises ConfigError listing every problem."""
    name = display_name or str(path)
    data = read_yaml(path, display_name=name)
    if data is None:
        raise ConfigError([f"{name}: file is empty"])
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(name, exc)) from exc


def _one_line(text: str) -> str:
    return " ".join(text.split())
