# dags/etl_concat_csv.py
import io
import os
import re
from datetime import datetime, timezone

import boto3
import pandas as pd
from airflow.decorators import dag, task   # API TaskFlow Airflow 2.x — pas airflow.sdk (Airflow 3 uniquement)

S3_BUCKET = os.getenv("AWS_BUCKET")
INPUT_PREFIX_FEATURES = "MUSIC_USER/features_test"
INPUT_PREFIX_SPECTRO = "MUSIC_USER/spectro_test"
OUTPUT_KEY_FEATURES = "MUSIC_USER/output/concatenated_features.csv"
OUTPUT_KEY_SPECTRO = "MUSIC_USER/output/concatenated_spectro.csv"

# Ne traiter que les fichiers déposés à partir de cette date (cf. demande explicite)
PROCESS_SINCE = datetime(2026, 7, 19, tzinfo=timezone.utc)

FEATURES_FILENAME_RE = re.compile(r"^features_.+\.csv$")
SPECTRO_FILENAME_RE = re.compile(r"^spectro_.+\.csv$")
SESSION_ID_RE = re.compile(r"^(?:features|spectro)_(?P<session_id>.+)\.csv$")


def _list_matching_files(s3, prefix: str, name_re: re.Pattern) -> list:
    """Liste les objets sous `prefix` dont le NOM DE FICHIER matche `name_re` (features_*.csv
    ou spectro_*.csv) et dont la date de dépôt (LastModified) est >= PROCESS_SINCE."""
    paginator = s3.get_paginator("list_objects_v2")
    matches = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            filename = obj["Key"].rsplit("/", 1)[-1]
            if not name_re.match(filename):
                continue
            if obj["LastModified"] < PROCESS_SINCE:
                continue
            matches.append({"key": obj["Key"], "filename": filename, "last_modified": obj["LastModified"]})
    return matches


def _concat_with_dedup(s3, files: list, dedup_key: str):
    """Concatène les CSV en ajoutant 'date' (date du fichier) et 'session_id' (extrait du nom),
    puis ne garde que la ligne la plus récente par `dedup_key` en cas de doublon (ex: un même
    morceau re-tagué dans une session ultérieure — seul le label change, on prend le plus récent)."""
    if not files:
        return None

    dfs = []
    for f in files:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f["key"])
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        m = SESSION_ID_RE.match(f["filename"])
        df["session_id"] = m.group("session_id") if m else None
        df["date"] = f["last_modified"]
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    if dedup_key not in result.columns:
        print(f"⚠️ Colonne de dédoublonnage '{dedup_key}' absente — pas de déduplication appliquée")
        return result

    return (
        result.sort_values("date")
        .drop_duplicates(subset=[dedup_key], keep="last")
        .reset_index(drop=True)
    )


@dag(
    dag_id="etl_concat2_csv",
    schedule="*/5 * * * *",   # toutes les 5 minutes — cron classique, natif Airflow 2.x
    catchup=False,
    start_date=datetime(2026, 1, 1),
)
# Note : le déclenchement événementiel "à chaque nouveau fichier" (Asset + AssetWatcher +
# MessageQueueTrigger/SQS) est réservé à Airflow 3.0+ — inutilisable en 2.10.4. Le cron 5 min
# ci-dessus est le seul mécanisme disponible sur cette version. À réévaluer si vous migrez
# vers Airflow 3 un jour.
def etl_concat_csv_dag():

    @task
    def concat_features():
        s3 = boto3.client("s3")
        files = _list_matching_files(s3, INPUT_PREFIX_FEATURES, FEATURES_FILENAME_RE)
        result = _concat_with_dedup(s3, files, dedup_key="filename")
        if result is None:
            print("Aucun fichier features_*.csv trouvé depuis le 19/07 — rien à faire")
            return
        buffer = io.StringIO()
        result.to_csv(buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY_FEATURES, Body=buffer.getvalue())
        print(f"{len(files)} fichiers → {result.shape[0]} lignes (après dédup) → s3://{S3_BUCKET}/{OUTPUT_KEY_FEATURES}")

    @task
    def concat_spectro():
        s3 = boto3.client("s3")
        files = _list_matching_files(s3, INPUT_PREFIX_SPECTRO, SPECTRO_FILENAME_RE)
        result = _concat_with_dedup(s3, files, dedup_key="filename_wav")
        if result is None:
            print("Aucun fichier spectro_*.csv trouvé depuis le 19/07 — rien à faire")
            return
        buffer = io.StringIO()
        result.to_csv(buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY_SPECTRO, Body=buffer.getvalue())
        print(f"{len(files)} fichiers → {result.shape[0]} lignes (après dédup) → s3://{S3_BUCKET}/{OUTPUT_KEY_SPECTRO}")

    concat_features()
    concat_spectro()


etl_concat_csv_dag()