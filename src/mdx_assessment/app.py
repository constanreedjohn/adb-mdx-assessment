"""Composition root: the only module that wires core and the use-case packs together.

`core/` never imports `usecases/`. Instead this module imports both, lets each
pack register itself, validates the configuration and builds the API. Run with:

    uv run uvicorn mdx_assessment.app:app --reload

Startup fails fast (with every problem listed) if `config/` is invalid.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from mdx_assessment.core.api import create_app
from mdx_assessment.core.config import LoadedConfig
from mdx_assessment.core.registry import PlatformRegistries, create_platform_registries
from mdx_assessment.settings import get_settings
from mdx_assessment.usecases.mdx import register as mdx_pack


def build_registries() -> PlatformRegistries:
    registries = create_platform_registries()
    mdx_pack.register(registries)
    return registries


def load_config(config_dir: Path | None = None) -> LoadedConfig:
    """Validate `config/` against every registered schema. Raises ConfigError."""
    directory = config_dir or get_settings().app.config_dir
    return build_registries().config.load_config(directory)


def build_app(config_dir: Path | None = None) -> FastAPI:
    config = load_config(config_dir)
    return create_app(config_version=config.version)


app = build_app()
