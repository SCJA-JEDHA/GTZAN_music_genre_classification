FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dags/ /opt/airflow/dags/
COPY entrypoint.sh /entrypoint-hf.sh

USER root
RUN chmod +x /entrypoint-hf.sh
USER airflow

EXPOSE 7860

ENTRYPOINT ["/entrypoint-hf.sh"]