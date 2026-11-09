"""Architecture test (E6-02): core's code never names a use-case task.

Tasks are registered by packs and looked up by ID at run time (plan §2.7, DoD 7).
This scans every string literal in `core/` (docstrings excluded: examples are fine
there) for any task ID a pack registers.
"""

from __future__ import annotations

import ast
from pathlib import Path

from mdx_assessment.app import build_registries

CORE = Path(__file__).resolve().parents[2] / "src" / "mdx_assessment" / "core"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def string_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_core_contains_no_registered_task_ids() -> None:
    task_ids = {t.id for t in build_registries().tasks}
    assert task_ids, "the MDX pack should register tasks"
    hits = [
        f"{path.relative_to(CORE.parent)}:{line}: {task}"
        for path in sorted(CORE.rglob("*.py"))
        for line, text in string_literals(path)
        for task in task_ids
        if task in text
    ]
    assert hits == [], "core names use-case tasks:\n" + "\n".join(hits)


def test_the_scan_would_catch_one(tmp_path: Path) -> None:
    sample = tmp_path / "bad.py"
    sample.write_text(
        '"""Docstring: input_validation.check_files_and_intake is fine here."""\nTASK = "input_validation.check_files_and_intake"\n'
    )
    assert [text for _, text in string_literals(sample)] == [
        "input_validation.check_files_and_intake"
    ]
