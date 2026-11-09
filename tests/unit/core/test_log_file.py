"""One log file per app lifespan."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mdx_assessment.app import configure_logging
from mdx_assessment.core.api import create_app
from mdx_assessment.core.observability.log_file import file_log_lifespan

LOGGERS = ("mdx_assessment", "uvicorn", "uvicorn.access")


def handler_counts() -> list[int]:
    return [len(logging.getLogger(name).handlers) for name in LOGGERS]


def test_the_lifespan_writes_one_file_and_closes_it(tmp_path: Path) -> None:
    configure_logging()
    before = handler_counts()
    log_dir = tmp_path / "log"
    with TestClient(create_app(lifespan=file_log_lifespan(log_dir))):
        logging.getLogger("mdx_assessment.core.something").info("from the app")
        logging.getLogger("uvicorn.access").warning("from the server")
    logging.getLogger("mdx_assessment.core.something").info("after shutdown")

    [path] = list(log_dir.iterdir())
    assert path.name.startswith("app-") and path.suffix == ".log"
    text = path.read_text()
    for expected in ("app started", "from the app", "from the server", "app stopping"):
        assert expected in text
    assert "after shutdown" not in text
    assert handler_counts() == before


def test_an_unusable_folder_does_not_stop_the_app(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    configure_logging()
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    with (
        caplog.at_level(logging.WARNING, logger="mdx_assessment.core.observability.log_file"),
        TestClient(create_app(lifespan=file_log_lifespan(blocked))) as client,
    ):
        assert client.get("/health").status_code == 200
    assert "not writing a log file" in caplog.text
