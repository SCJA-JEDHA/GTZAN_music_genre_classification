# dags/monitoring_drift_musicai.py
from datetime import datetime
import pandas as pd
import boto3
import io
import os
# evidently version 0.7.21 
from evidently import Report
# from evidently.test_preset import DataQualityPreset  # ou un preset adapté
# from evidently.test_preset import DataDriftPreset
# from evidently.tests import ShareOfDriftedColumns  # test individuel
#from evidently.tests import ColumnDriftTest  # test valide dans 0.7.x
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

# from evidently.test_suite import TestSuite
# from evidently.tests import TestShareOfDriftedColumns
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator


S3_BUCKET = os.getenv("AWS_BUCKET")
REFERENCE_CSV_PATH = "music-database/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv"
OUTPUT_KEY_FEATURES = "MUSIC_USER/output/concatenated_features2.csv"
OUTPUT_KEY_SPECTRO = "MUSIC_USER/output/concatenated_spectro2.csv"


OUTPUT_MONITORING = "MUSIC_USER/monitoring"
FEATURE_COLUMNS = [    # tes 57 colonnes numériques GTZAN
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean', 'rms_var', 
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
    'rolloff_mean','rolloff_var',
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var', 
    'perceptr_mean', 'perceptr_var',
    'tempo',
    'mfcc1_mean', 'mfcc1_var', 
    'mfcc2_mean', 'mfcc2_var', 
    'mfcc3_mean','mfcc3_var',
    'mfcc4_mean', 'mfcc4_var',
    'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var',
    'mfcc7_mean', 'mfcc7_var',
    'mfcc8_mean','mfcc8_var',
    'mfcc9_mean', 'mfcc9_var',
    'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var',
    'mfcc12_mean', 'mfcc12_var',
    'mfcc13_mean','mfcc13_var',
    'mfcc14_mean', 'mfcc14_var', 
    'mfcc15_mean', 'mfcc15_var',
    'mfcc16_mean', 'mfcc16_var', 
    'mfcc17_mean', 'mfcc17_var',
    'mfcc18_mean','mfcc18_var',
    'mfcc19_mean', 'mfcc19_var',
    'mfcc20_mean', 'mfcc20_var',
]

def _load_reference(s3):
    #s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=REFERENCE_CSV_PATH)
    return pd.read_csv(io.BytesIO(obj["Body"].read())).to_dict()

def _load_recent_batches(s3):
    """Agrège les features loguées par Streamlit sur les dernières 24h."""
    #s3 = boto3.client("s3")
    cutoff = datetime.now().strftime("%Y%m%d")
    obj = s3.get_object(Bucket=S3_BUCKET, Key= OUTPUT_KEY_FEATURES)
    df_features = pd.read_csv(io.BytesIO(obj["Body"].read()) )
    # filter last recent period of last 100 lines
    
    df_last100 = df_features.tail(100)
    # objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"monitoring/batches/{cutoff}")
    # dfs = []
    # for o in objs.get("Contents", []):
    #     data = s3.get_object(Bucket=S3_BUCKET, Key=o["Key"])["Body"].read()
    #     dfs.append(pd.read_parquet(io.BytesIO(data)))
    # return pd.concat(dfs, ignore_index=True).to_dict() if dfs else {}
    return df_last100.to_dict() if not df_last100.empty else {}

def _run_drift_check(s3,reference: dict, current: dict) -> dict:
    reference_df = pd.DataFrame(reference)[FEATURE_COLUMNS]
    current_df = pd.DataFrame(current)[FEATURE_COLUMNS]

    if current_df.empty:
        return {"all_passed": True, "skipped": True}

    # ancien code evidently 0.4.2 :
    # tests = TestSuite(tests=[TestShareOfDriftedColumns(lt=0.3)])
    # tests.run(reference_data=reference_df, current_data=current_df)
    # result = tests.as_dict()
    # tests.save_html("/tmp/drift_report.html")
    
    # Créer un rapport avec un test spécifique
    #report = Report(tests=[ShareOfDriftedColumns(lt=0.3)])
    # Evidently 0.7.21 :
    # map columns types :
    schema = DataDefinition(numerical_columns = FEATURE_COLUMNS)
    
    # Create Evidently Datasets to work with:

    eval_ref_data = Dataset.from_pandas(reference_df,data_definition=schema)
    eval_current_data = Dataset.from_pandas(current_df,data_definition=schema)

    report = Report([DataDriftPreset() ])

    
    # Exécuter le rapport sur les données de référence et courantes
    # (current, reference)
    snapshot = report.run(eval_current_data, eval_ref_data)
    
    #report.run(reference_data=reference_df, current_data=current_df)

    # Récupérer le résultat sous forme de dictionnaire
    #result = my_eval.as_dict()

    # Sauvegarder le rapport au format HTML
    snapshot.save_html("/tmp/drift_report.html")


    s3.upload_file(
        "/tmp/drift_report.html", S3_BUCKET,
        f"{OUTPUT_MONITORING}/reports/drift_{datetime.now():%Y%m%d_%H%M%S}.html",
    )
    
    data = snapshot.dict()   # <--- c’est ça qu’il faut utiliser

    # # df = report.as_snapshot().to_pandas()
    # global_drift = data["metrics"][0]["result"]["dataset_drift"]

    # print("=== DEBUG EVIDENTLY ===")
    # print(data)
    

    # Chercher la métrique DriftedColumnsCount
    drift_metric = next(
        m for m in data["metrics"]
        if m["metric_name"].startswith("DriftedColumnsCount")
    )

    count = drift_metric["value"]["count"]
    share = drift_metric["value"]["share"]

    # Le threshold est dans metric_name → on l'extrait
    # Exemple: "DriftedColumnsCount(drift_share=0.5)"
    name = drift_metric["metric_name"]
    threshold = float(name.split("drift_share=")[1].rstrip(")"))

    global_drift = share > threshold

    return {
        "all_passed": not global_drift,
        "drifted_columns": count,
        "drift_share": share,
        "threshold": threshold,
    }


    

def _log_to_mlflow(summary: dict):
    """Cohérent with tracking MLflow existing on HF Spaces 
    (client mlflow-skinny).
    """
    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_URI"])
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("all_tests_passed", int(summary.get("all_passed", True)))
        mlflow.set_tag("trigger_source", "airflow_monitoring")

def _decide(summary: dict) -> str:
    if summary.get("skipped"):
        return "no_action"
    return "trigger_retrain" if not summary["all_passed"] else "no_action"

def _trigger_retrain():
    print("Drift détecté sur les features GTZAN → retrain nécessaire")

def _no_action():
    print("Pas de drift significatif")


def _monitoring_drift_and_decide():
    s3 = boto3.client("s3")
    ref = _load_reference(s3)
    cur = _load_recent_batches(s3)
    summary = _run_drift_check(s3,ref, cur)
    _log_to_mlflow(summary)
    
    branch_id = _decide(summary)
    return branch_id
        

## ======================== DAG ===================================

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="02_monitoring_drift_musicAI_cyril",
    default_args=default_args,
    schedule="*/5 * * * *",
    start_date=datetime(2026, 7, 19),
    catchup=False,
    description="DAG 2 MusicAI Monitoring drift",
    tags=["MusicAI", "etl"],
) as dag:

    start = EmptyOperator(task_id="start")
    
    monitoring_drift_and_decide = BranchPythonOperator(
        task_id="monitoring_drift_and_decide", python_callable=_monitoring_drift_and_decide)
    
    trigger_retrain = PythonOperator(
        task_id="trigger_retrain", python_callable=_trigger_retrain)
        
    
    no_action = PythonOperator(
        task_id = "no_action",python_callable = _no_action)
    
    end = EmptyOperator(task_id="end")

    

    start >> monitoring_drift_and_decide >> [trigger_retrain, no_action] >> end