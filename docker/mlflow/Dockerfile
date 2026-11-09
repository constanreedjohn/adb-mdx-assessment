# The MLflow tracking server. The official image has no PostgreSQL driver.
FROM ghcr.io/mlflow/mlflow:v3.18.0

RUN pip install --no-cache-dir "psycopg[binary]>=3.2,<4"
COPY start.py /start.py

EXPOSE 5000
CMD ["python", "/start.py"]
