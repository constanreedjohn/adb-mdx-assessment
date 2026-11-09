# MDX Daily Validation Agent: developer commands.
# Every Python command runs through `uv run`, so the locked environment is always used.

.DEFAULT_GOAL := help
.PHONY: help install up logs sim-docker dev down dev-reset test test-live lint fmt typecheck migrate migration api sim smoke-trace smoke-llm

UV ?= uv

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and pre-commit hooks
	$(UV) sync --locked
	$(UV) run pre-commit install

COMPOSE ?= docker compose

up: ## Build and start the whole stack in Docker (API and Postgres)
	@test -f .env || (echo "Missing .env: cp .env.example .env and fill it in." && exit 1)
	$(COMPOSE) up -d --build --wait --wait-timeout 180
	@echo "API      : http://127.0.0.1:$${API_PORT:-8000}/docs"

logs: ## Follow the API's log
	$(COMPOSE) logs -f api

sim-docker: ## Generate a simulated MDX day from inside the API image, e.g. make sim-docker DATE=2026-10-15
	@test -n "$(DATE)" || (echo "Usage: make sim-docker DATE=YYYY-MM-DD [SEED=n]" && exit 1)
	$(COMPOSE) run --rm --no-deps api python -m mdx_assessment.usecases.mdx.sim --date $(DATE) $(if $(SEED),--seed $(SEED),)

dev: ## Start Postgres only and wait until healthy
	@test -f .env || (echo "Missing .env: cp .env.example .env and fill it in." && exit 1)
	$(COMPOSE) up -d --wait --wait-timeout 120 postgres
	@$(COMPOSE) ps --format 'table {{.Service}}\t{{.Status}}'
	@echo ""
	@echo "Postgres : localhost:$${POSTGRES_PORT:-5432}"
	@echo "Langfuse : Langfuse Cloud (LANGFUSE_HOST in .env)"

down: ## Stop local services (data is kept)
	$(COMPOSE) down

dev-reset: ## Stop local services AND delete their data volume
	$(COMPOSE) down --volumes

test: ## Run the test suite
	$(UV) run pytest

test-live: ## Also run tests that call live services (Azure OpenAI, Langfuse)
	MDX_RUN_LIVE=1 $(UV) run pytest -m live

lint: ## Lint, check formatting and type-check
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

fmt: ## Auto-format and apply safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## Type-check only (mypy --strict on src/)
	$(UV) run mypy

migrate: ## Apply database migrations (Alembic upgrade head)
	$(UV) run alembic upgrade head

migration: ## Create a migration from model changes, e.g. make migration MSG="add decisions"
	@test -n "$(MSG)" || (echo "Usage: make migration MSG=\"what changed\"" && exit 1)
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

API_PORT ?= 8000

api: ## Run the API with reload on API_PORT (default 8000)
	$(UV) run uvicorn mdx_assessment.app:app --reload --host 127.0.0.1 --port $(API_PORT)

sim: ## Generate a simulated MDX day, e.g. make sim DATE=2026-10-15 [SEED=7]
	@test -n "$(DATE)" || (echo "Usage: make sim DATE=YYYY-MM-DD [SEED=n]" && exit 1)
	$(UV) run python -m mdx_assessment.usecases.mdx.sim --date $(DATE) $(if $(SEED),--seed $(SEED),)

smoke-trace: ## Send a test trace to Langfuse and confirm it was stored
	$(UV) run python -m mdx_assessment.core.observability.smoke

smoke-llm: ## One structured-output call to the Azure OpenAI deployment
	$(UV) run python -m mdx_assessment.core.gateway.smoke
