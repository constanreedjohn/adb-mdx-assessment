# MDX Daily Validation Agent: developer commands.
# Every Python command runs through `uv run`, so the locked environment is always used.

.DEFAULT_GOAL := help
.PHONY: help install dev down test lint fmt typecheck migrate api sim smoke-llm

UV ?= uv

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and pre-commit hooks
	$(UV) sync --locked
	$(UV) run pre-commit install

dev: ## Start local services (Postgres, Langfuse) [E1-03]
	@echo "Not available yet: added in E1-03 (docker-compose)." && exit 1

down: ## Stop local services [E1-03]
	@echo "Not available yet: added in E1-03 (docker-compose)." && exit 1

test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint, check formatting and type-check
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

fmt: ## Auto-format and apply safe lint fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## Type-check only (mypy --strict on src/)
	$(UV) run mypy

migrate: ## Apply database migrations [E4-01]
	@echo "Not available yet: added in E4-01 (Alembic)." && exit 1

API_PORT ?= 8000

api: ## Run the API with reload on API_PORT (default 8000)
	$(UV) run uvicorn mdx_assessment.app:app --reload --host 127.0.0.1 --port $(API_PORT)

sim: ## Generate a simulated MDX day, e.g. make sim DATE=2026-10-15 [SP2, E2-01]
	@echo "Not available yet: added in SP2 (E2-01)." && exit 1

smoke-llm: ## One structured-output call to Azure OpenAI [E5-01]
	@echo "Not available yet: added in E5-01." && exit 1
