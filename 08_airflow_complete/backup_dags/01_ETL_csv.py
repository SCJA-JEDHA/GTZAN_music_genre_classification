# dags/etl_concat_csv.py
from airflow.sdk import dag, task
from datetime import datetime
import pandas as pd
import boto3
import io
import os

S3_BUCKET = os.getenv("AWS_BUCKET")
INPUT_PREFIX_FEATURES = "MUSIC_USER/features_test"
INPUT_PREFIX_SPECTRO = "MUSIC_USER/spectro_test"
OUTPUT_KEY = "etl/output/concatenated.csv"

@dag(
    dag_id="etl_concat1_csv",
    schedule="@daily",
    catchup=False,
    start_date=datetime(2026, 1, 1),
)
def etl_concat_csv_dag():

    @task
    def list_csv_files_features() -> list[str]:
        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=INPUT_PREFIX_FEATURES)
        return [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".csv")]

    @task
    def concat_files(keys: list[str]):
        if not keys:
            print("Aucun fichier CSV trouvé, rien à faire")
            return

        s3 = boto3.client("s3")
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read())))

        result = pd.concat(dfs, ignore_index=True)

        buffer = io.StringIO()
        result.to_csv(buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY, Body=buffer.getvalue())
        print(f"{len(keys)} fichiers concaténés → {result.shape[0]} lignes → s3://{S3_BUCKET}/{OUTPUT_KEY}")

    concat_files(list_csv_files_features())

etl_concat_csv_dag()