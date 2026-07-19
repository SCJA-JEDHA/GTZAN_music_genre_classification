# dags/etl_concat_csv.py
import io
import os
import re
from datetime import datetime, timezone

import boto3
import pandas as pd
from airflow import DAG
from airflow.providers.amazon.aws.triggers.sqs import SqsSensorTrigger
from airflow.providers.common.messaging.triggers.msg_queue import MessageQueueTrigger
from airflow.timetables.assets import AssetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.sdk import Asset, AssetWatcher, dag, task

S3_BUCKET = os.getenv("AWS_BUCKET")
INPUT_PREFIX_FEATURES = "MUSIC_USER/features_test"
INPUT_PREFIX_SPECTRO = "MUSIC_USER/spectro_test"
OUTPUT_KEY_FEATURES = "MUSIC_USER/output/concatenated_features.csv"
OUTPUT_KEY_SPECTRO = "MUSIC_USER/output/concatenated_spectro.csv"

# Ne traiter que les fichiers déposés à partir de cette date (cf. demande explicite)
PROCESS_SINCE = datetime(2026, 7, 19, tzinfo=timezone.utc)

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")  # ex: https://sqs.eu-west-3.amazonaws.com/<account_id>/musicai-s3-events

# --- Déclenchement événementiel : dès qu'un message arrive sur la queue SQS (donc dès
# qu'un fichier est déposé sur les préfixes surveillés), le DAG se relance.
_trigger = MessageQueueTrigger(scheme="sqs", sqs_queue=SQS_QUEUE_URL, aws_conn_id="aws_default")
s3_asset = Asset("music_user_bucket_asset", watchers=[AssetWatcher(name="s3_watcher", trigger=_trigger)])

FEATURES_FILENAME_RE = re.compile(r"^features_.+\.csv$")
SPECTRO_FILENAME_RE = re.compile(r"^spectro_.+\.csv$")
SESSION_ID_RE = re.compile(r"^(?:features|spectro)_(?P<session_id>.+)\.csv$")


def _list_matching_files(s3, prefix: str, name_re: re.Pattern) -> list[dict]:
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


def _concat_with_dedup(s3, files: list[dict], dedup_key: str) -> pd.DataFrame | None:
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

    result = (
        result.sort_values("date")
        .drop_duplicates(subset=[dedup_key], keep="last")
        .reset_index(drop=True)
    )
    return result


@dag(
    dag_id="etl_concat2_csv",
    schedule="*/5 * * * *",   # toutes les 5 minutes, cron classique
    catchup=False,
    start_date=datetime(2026, 1, 1),
)

# for later : corn 5 min et at each new file :
    # schedule=AssetOrTimeSchedule(
    #     timetable=CronTriggerTimetable("*/5 * * * *", timezone="UTC"),
    #     assets=[s3_asset],
    # ),
    
    
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