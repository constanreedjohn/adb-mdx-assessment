# The API image. Secrets come from the environment at run time (compose: `env_file`),
# and `.dockerignore` keeps `.env` and local data out of the image.
FROM python:3.12-slim

ENV UV_VERSION=0.13.0 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY alembic.ini ./
COPY config ./config
COPY src ./src
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /app/log && chown -R app /app/data /app/log
USER app

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn mdx_assessment.app:app --host 0.0.0.0 --port 8000"]
