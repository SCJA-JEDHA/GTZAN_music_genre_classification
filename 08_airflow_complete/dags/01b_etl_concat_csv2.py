# dags/01_etl_concat_csv.py
import io
import os
import re
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

S3_BUCKET = os.getenv("AWS_BUCKET")
INPUT_PREFIX_FEATURES = "MUSIC_USER/features_test"
INPUT_PREFIX_SPECTRO = "MUSIC_USER/spectro_test"
OUTPUT_KEY_FEATURES = "MUSIC_USER/output/concatenated_features.csv"
OUTPUT_KEY_SPECTRO = "MUSIC_USER/output/concatenated_spectro.csv"

# Ne traiter que les fichiers déposés à partir de cette date
PROCESS_SINCE = datetime(2026, 7, 19, tzinfo=timezone.utc)

FEATURES_FILENAME_RE = re.compile(r"^features_.+\.csv$")
SPECTRO_FILENAME_RE = re.compile(r"^spectro_.+\.csv$")
SESSION_ID_RE = re.compile(r"^(?:features|spectro)_(?P<session_id>.+)\.csv$")


def _list_matching_files(s3, prefix: str, name_re: re.Pattern) -> list:
    """Liste les objets sous `prefix` dont le nom matche `name_re` et dont LastModified >= PROCESS_SINCE."""
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


def _load_existing_output(s3, output_key: str):
    """Charge le CSV de sortie déjà présent sur S3, ou None s'il n'existe pas encore (1er run)."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=output_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def _filter_unprocessed(files: list, existing_df) -> list:
    """Ne garde que les fichiers sources absents de la sortie existante OU modifiés depuis
    (comparaison LastModified S3 vs 'date' déjà connue pour ce session_id) — évite de
    retélécharger un fichier de session déjà intégré ET inchangé depuis le dernier run,
    tout en captant les sessions toujours en cours d'écriture côté app."""
    if existing_df is None or "session_id" not in existing_df.columns or "date" not in existing_df.columns:
        return files

    existing_df = existing_df.copy()
    existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True, errors="coerce")
    last_seen = existing_df.groupby("session_id")["date"].max().to_dict()

    unprocessed = []
    for f in files:
        m = SESSION_ID_RE.match(f["filename"])
        session_id = m.group("session_id") if m else None
        if session_id is None or session_id not in last_seen or f["last_modified"] > last_seen[session_id]:
            unprocessed.append(f)
    return unprocessed


def _merge_and_dedup(existing_df, new_files: list, s3, dedup_key: str):
    """Lit les nouveaux fichiers, ajoute 'date'/'session_id', fusionne avec l'existant,
    puis ne garde que la ligne la plus récente par `dedup_key` en cas de doublon."""
    new_dfs = []
    for f in new_files:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f["key"])
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        m = SESSION_ID_RE.match(f["filename"])
        df["session_id"] = m.group("session_id") if m else None
        df["date"] = f["last_modified"]
        new_dfs.append(df)

    if not new_dfs:
        return existing_df  # rien de neuf, on garde l'existant tel quel

    parts = ([existing_df] if existing_df is not None else []) + new_dfs
    combined = pd.concat(parts, ignore_index=True)

    if dedup_key not in combined.columns:
        print(f"⚠️ Colonne de dédoublonnage '{dedup_key}' absente — pas de déduplication appliquée")
        return combined

    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    return (
        combined.sort_values("date")
        .drop_duplicates(subset=[dedup_key], keep="last")
        .reset_index(drop=True)
    )


def _run_incremental_concat(input_prefix: str, name_re: re.Pattern, output_key: str, dedup_key: str, label: str):
    s3 = boto3.client("s3")

    existing_df = _load_existing_output(s3, output_key)
    all_files = _list_matching_files(s3, input_prefix, name_re)
    new_files = _filter_unprocessed(all_files, existing_df)

    if not new_files:
        print(f"[{label}] Aucun fichier nouveau ou modifié — sortie inchangée")
        return

    result = _merge_and_dedup(existing_df, new_files, s3, dedup_key)
    if result is None or result.empty:
        print(f"[{label}] Aucune donnée à écrire")
        return

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=output_key, Body=buffer.getvalue())
    print(f"[{label}] {len(new_files)} fichier(s) nouveau(x)/modifié(s) intégré(s) "
          f"→ {result.shape[0]} lignes au total (après dédup) → s3://{S3_BUCKET}/{output_key}")


def _concat_features():
    _run_incremental_concat(INPUT_PREFIX_FEATURES, FEATURES_FILENAME_RE,
                             OUTPUT_KEY_FEATURES, dedup_key="filename", label="features")


def _concat_spectro():
    _run_incremental_concat(INPUT_PREFIX_SPECTRO, SPECTRO_FILENAME_RE,
                             OUTPUT_KEY_SPECTRO, dedup_key="filename_wav", label="spectro")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="01b_etl_csv_musicAI2",
    default_args=default_args,
    schedule="*/5 * * * *",
    start_date=datetime(2026, 7, 19),
    catchup=False,
    description="ETL MusicAI concat csv tables (incrémental)",
    tags=["MusicAI", "etl"],
) as dag:

    start = EmptyOperator(task_id="start")

    concat_features = PythonOperator(task_id="concat_features", python_callable=_concat_features)
    concat_spectro = PythonOperator(task_id="concat_spectro", python_callable=_concat_spectro)

    end = EmptyOperator(task_id="end")

    start >> [concat_features, concat_spectro] >> end