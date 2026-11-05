"""Composition root: the only module that wires core and the use-case packs together.

`core/` never imports `usecases/`. Instead this module imports both, lets each
pack register itself, and builds the API. Run with:

    uv run uvicorn mdx_assessment.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from mdx_assessment.core.api import create_app
from mdx_assessment.usecases.mdx import register as mdx_pack


def build_app() -> FastAPI:
    mdx_pack.register()
    return create_app()


app = build_app()
