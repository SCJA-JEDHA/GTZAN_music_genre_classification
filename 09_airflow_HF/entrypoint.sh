#!/bin/bash
set -e

export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="${NEON_DATABASE_URL}"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=7860
export AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth

airflow db migrate

airflow users create \
    --username "${_AIRFLOW_WWW_USER_USERNAME:-airflow}" \
    --password "${_AIRFLOW_WWW_USER_PASSWORD:-airflow}" \
    --firstname Admin --lastname User \
    --role Admin --email admin@example.com || true

airflow scheduler &
exec airflow webserver