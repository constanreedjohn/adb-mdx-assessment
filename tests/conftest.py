"""Suite-wide test configuration.

Tests marked `live` call real external services (Azure OpenAI, Langfuse Cloud).
They are skipped unless `MDX_RUN_LIVE=1`, so `make test` stays offline and
deterministic; run them with `make test-live`. CI excludes them as well.
"""

from __future__ import annotations

import os

import pytest

RUN_LIVE = os.environ.get("MDX_RUN_LIVE") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if RUN_LIVE:
        return
    skip_live = pytest.mark.skip(reason="live test: set MDX_RUN_LIVE=1 (make test-live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
