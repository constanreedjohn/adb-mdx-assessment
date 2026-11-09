"""Start the MLflow server on its own database in the stack's PostgreSQL.

The database is created here, not by an init script, because Postgres only runs init
scripts on an empty volume and this has to work on an existing one too. The connection
URL goes to MLflow through the environment, so the password stays out of the process list.
"""

import os
from urllib.parse import quote_plus

import psycopg
from psycopg import sql

DATABASE = "mlflow"
HOST = os.environ["POSTGRES_HOST"]
PORT = os.environ.get("POSTGRES_PORT", "5432")
USER = os.environ["POSTGRES_USER"]
PASSWORD = os.environ["POSTGRES_PASSWORD"]

with psycopg.connect(
    host=HOST,
    port=PORT,
    user=USER,
    password=PASSWORD,
    dbname=os.environ["POSTGRES_DB"],
    autocommit=True,  # CREATE DATABASE can't run in a transaction
) as connection:
    exists = connection.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)
    ).fetchone()
    if exists is None:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DATABASE)))

os.environ["MLFLOW_BACKEND_STORE_URI"] = (
    f"postgresql+psycopg://{quote_plus(USER)}:{quote_plus(PASSWORD)}@{HOST}:{PORT}/{DATABASE}"
)
os.execvp(
    "mlflow",
    [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--serve-artifacts",
        "--artifacts-destination",
        "/mlflow/artifacts",
        "--workers",
        "1",
        # The API reaches this server as `mlflow`, which isn't in the default allow-list.
        "--allowed-hosts",
        "mlflow:*,localhost:*,127.0.0.1:*",
    ],
)
