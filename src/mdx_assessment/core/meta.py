"""Application identity: distribution name and installed version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DIST_NAME = "mdx-assessment"


def app_version() -> str:
    try:
        return version(DIST_NAME)
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"
