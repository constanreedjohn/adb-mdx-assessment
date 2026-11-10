# MDX Daily Validation Agent

A supervised multi-agent proof of concept for daily market data validation. Code gathers the facts of the day and selects a playbook, a Master Agent plans the run within that playbook, and specialist agents carry out the steps. Analysts make the decisions.

## Run it in Docker

Docker runs the API, MLflow and PostgreSQL. MLflow is self-hosted, so run traces stay on your machine.

### Before you start

You need Docker Desktop running and `make`. Then create your settings file:

```bash
cp .env.example .env
```

Fill in `.env`:

| Setting | Needed for |
| --- | --- |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` | Required: the Master Agent's planning call |
| `AZURE_OPENAI_API_VERSION` | Required: must be a version your deployment supports |

`.env` is git-ignored and is never copied into the image.

### Start a run

```bash
# 1. Build and start the API, MLflow and Postgres. Migrations run on start.
make up

# 2. Generate a simulated MDX day (writes to data/sim/2026-10-15/)
make sim-docker DATE=2026-10-15

# 3. Start a session. It runs to the end, then returns session_id and trace_url.
curl -s -X POST http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"business_date":"2026-10-15","data_type":"linear"}'

# 4. Read the full result, using the session_id from step 3
curl -s http://127.0.0.1:8000/sessions/<session_id> | python3 -m json.tool
```

The interactive API docs are at <http://127.0.0.1:8000/docs>.

### See what happened

| Where | What it shows |
| --- | --- |
| `make logs` | The API's log, live |
| `log/app-<UTC start time>.log` | The same log, one file per app run. Files survive rebuilds and are not cleaned up automatically |
| MLflow (`trace_url` from step 3, or <http://127.0.0.1:5001>, experiment `mdx-sessions`, Traces tab) | One trace per session: every graph node, tool call and model call, with model and token usage |
| `GET /sessions/<session_id>` | Facts, selected playbook, plan and its validation, execution log |

### Session status

| Status | Meaning |
| --- | --- |
| `completed` | The Master Agent's plan passed validation and every step ran |
| `completed_degraded` | The Master Agent's plan was used, but at least one step failed |
| `default_complete` | The playbook's default plan ran instead of the Master Agent's: the plan was rejected twice, the model could not be reached, there were no facts to plan from, or the run was started with `"llm_enabled": false`. `analyst_notice` says which |
| `failed` | The run stopped because the day's facts could not be read from the source system. No checks ran. `analyst_notice` says so, and the log has the full reason |

### Stop or reset

```bash
make down       # stop everything and keep the data
make dev-reset  # stop everything and delete the database volume
```

After a code change, run `make up` again to rebuild the image.

If port 8000 is taken, use another one, and change the port in the `curl` commands to match:

```bash
API_PORT=8001 make up
```

To change the host's Postgres or MLflow port, set `POSTGRES_PORT` or `MLFLOW_PORT` in `.env`.

## Make commands

Run `make help` to print this list from the Makefile.

### Docker stack

| Command | What it does |
| --- | --- |
| `make up` | Builds and starts the API, MLflow and Postgres, and waits until they are healthy |
| `make logs` | Follows the API's log |
| `make sim-docker DATE=2026-10-15 [SEED=7]` | Generates a simulated MDX day from inside the API image |
| `make dev` | Starts Postgres and MLflow only, for running the API or tests on your machine |
| `make down` | Stops the containers and keeps the data |
| `make dev-reset` | Stops the containers and deletes the database volume |

### Running on your machine

These need [uv](https://docs.astral.sh/uv/) 0.13 or newer. Python 3.12 is used for the project.

| Command | What it does |
| --- | --- |
| `make install` | Installs dependencies and the pre-commit hooks |
| `make api` | Runs the API with auto-reload on `API_PORT` (default 8000). Needs Postgres (`make dev`) and `make migrate` |
| `make sim DATE=2026-10-15 [SEED=7]` | Generates a simulated MDX day into `data/sim/` |

### Quality checks

| Command | What it does |
| --- | --- |
| `make test` | Runs the test suite. Database tests are skipped if Postgres isn't running |
| `make test-live` | Runs only the tests that call Azure OpenAI and MLflow. Uses your real credentials, and needs MLflow running |
| `make lint` | Runs ruff (lint and format check) and `mypy --strict` |
| `make fmt` | Formats the code and applies safe lint fixes |
| `make typecheck` | Runs `mypy --strict` only |

### Database

| Command | What it does |
| --- | --- |
| `make migrate` | Applies database migrations. `make up` already does this inside Docker |
| `make migration MSG="add decisions"` | Creates a migration from model changes |

### Connection checks

| Command | What it does |
| --- | --- |
| `make smoke-llm` | Makes one structured-output call to your Azure OpenAI deployment |
| `make smoke-trace` | Sends a test trace to MLflow and confirms it was stored |
