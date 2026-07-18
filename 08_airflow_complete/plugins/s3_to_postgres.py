"""
Opérateur custom : S3ToPostgresOperator
=======================================
Exemple de création d'un opérateur Airflow personnalisé.

Il télécharge un fichier CSV depuis S3 et le charge dans une table
PostgreSQL via pandas + SQLAlchemy.

À utiliser dans un DAG comme n'importe quel opérateur :

    from s3_to_postgres import S3ToPostgresOperator

    transfer = S3ToPostgresOperator(
        task_id="transfer_to_postgres",
        table="ma_table",
        bucket="{{ var.value.S3BucketName }}",
        key="mon_fichier.csv",
        postgres_conn_id="postgres_default",
        aws_conn_id="aws_default",
    )

Prérequis : les connexions `aws_default` et `postgres_default` doivent
être configurées dans l'UI Airflow (voir README, section Connexions).
"""

from typing import Sequence

import pandas as pd
from airflow.models.baseoperator import BaseOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook


class S3ToPostgresOperator(BaseOperator):
    # Ces champs acceptent le templating Jinja ({{ var.value.xxx }}, XCom...)
    template_fields: Sequence[str] = (
        "bucket",
        "key",
        "table",
        "postgres_conn_id",
        "aws_conn_id",
    )

    def __init__(
        self,
        bucket,
        key,
        table,
        postgres_conn_id="postgres_default",
        aws_conn_id="aws_default",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bucket: str = bucket
        self.key: str = key
        self.table: str = table
        self.postgres_conn_id: str = postgres_conn_id
        self.aws_conn_id: str = aws_conn_id

    def execute(self, context):
        # 1. Télécharger le fichier depuis S3
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        returned_filename = s3_hook.download_file(
            self.key, bucket_name=self.bucket, local_path="/tmp"
        )
        # 2. L'ouvrir avec pandas
        df_file = pd.read_csv(returned_filename, header=None)
        # 3. Se connecter à Postgres via le hook
        postgres_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        engine = postgres_hook.get_sqlalchemy_engine()
        # 4. Écrire dans la table avec to_sql
        df_file.to_sql(self.table, engine, if_exists="replace", index=False)
