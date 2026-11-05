"""Architecture test: `core/` never depends on use-case packs or the composition root.

Rule (plan §2.7, DoD 7): modules under `mdx_assessment.core` must not import
`mdx_assessment.usecases` or `mdx_assessment.app`. Use-case packs register
themselves with core; core never names them.

The check parses source with `ast` (nothing is imported or executed), resolves
relative imports, and also rejects dynamic imports written as string literals
(e.g. `importlib.import_module("mdx_assessment.usecases.mdx")`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = "mdx_assessment"
CORE = f"{PACKAGE}.core"
FORBIDDEN = (f"{PACKAGE}.usecases", f"{PACKAGE}.app")


@dataclass(frozen=True)
class Violation:
    file: Path
    line: int
    target: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} imports {self.target}"


def _module_name(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (level - 1)] if level > 1 else package
    return ".".join([*base, target] if target else base)


def _is_forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def find_violations(src_root: Path, package_prefix: str = CORE) -> list[Violation]:
    """Every forbidden import inside `package_prefix` under `src_root`."""
    violations: list[Violation] = []
    pkg_dir = src_root.joinpath(*package_prefix.split("."))
    for path in sorted(pkg_dir.rglob("*.py")):
        module = _module_name(path, src_root)
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(module, is_package, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                # `from mdx_assessment import usecases` imports a submodule too.
                targets = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                targets = [node.value]
            for target in targets:
                if _is_forbidden(target):
                    violations.append(Violation(path, getattr(node, "lineno", 0), target))
                    break
    return violations


def test_core_does_not_import_usecases_or_app() -> None:
    violations = find_violations(SRC)
    assert not violations, "core/ must not depend on use cases:\n" + "\n".join(map(str, violations))


# --- The checker itself: prove it catches each way of breaking the rule. ---------


@pytest.fixture
def fake_src(tmp_path: Path) -> Path:
    for pkg in ("mdx_assessment", "mdx_assessment/core", "mdx_assessment/core/api"):
        (tmp_path / pkg).mkdir(parents=True, exist_ok=True)
        (tmp_path / pkg / "__init__.py").write_text("")
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        "import mdx_assessment.usecases",
        "import mdx_assessment.usecases.mdx.register as r",
        "from mdx_assessment.usecases.mdx import register",
        "from mdx_assessment import usecases",
        "from mdx_assessment.app import app",
        "from ...usecases.mdx import register",  # relative, from core/api/bad.py
        "from ... import usecases",
        "import importlib\nimportlib.import_module('mdx_assessment.usecases.mdx')",
    ],
)
def test_checker_flags_forbidden_import(fake_src: Path, source: str) -> None:
    (fake_src / "mdx_assessment/core/api/bad.py").write_text(source + "\n")
    assert len(find_violations(fake_src)) == 1


@pytest.mark.parametrize(
    "source",
    [
        "import mdx_assessment.core.registry",
        "from mdx_assessment.settings import get_settings",
        "from ..registry import specs",
        "from . import app",
        "import mdx_assessment_usecases_lookalike",
    ],
)
def test_checker_allows_core_and_settings_imports(fake_src: Path, source: str) -> None:
    (fake_src / "mdx_assessment/core/api/ok.py").write_text(source + "\n")
    assert find_violations(fake_src) == []
