"""One log file per app lifespan: `<log dir>/app-<UTC start time>.log`.

The file opens when the app starts and closes when it stops. It takes the application's
own logs plus uvicorn's server and request logs. A log folder that can't be written
costs the file only: the app still starts, and the terminal still has the logs.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOGGERS = ("mdx_assessment", "uvicorn", "uvicorn.access")  # uvicorn.access doesn't propagate

logger = logging.getLogger(__name__)
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _open_log_file(log_dir: Path) -> logging.FileHandler | None:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / f"app-{stamp}.log", encoding="utf-8")
    except OSError as exc:
        logger.warning("not writing a log file: cannot use %s (%s)", log_dir, exc.strerror)
        return None
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def file_log_lifespan(log_dir: Path) -> Lifespan:
    """A FastAPI lifespan that writes the logs of this app run to a new file in `log_dir`."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        handler = _open_log_file(log_dir)
        if handler is None:
            yield
            return
        for name in _LOGGERS:
            logging.getLogger(name).addHandler(handler)
        logger.info("app started; logging to %s", handler.baseFilename)
        try:
            yield
        finally:
            logger.info("app stopping")
            for name in _LOGGERS:
                logging.getLogger(name).removeHandler(handler)
            handler.close()

    return lifespan
