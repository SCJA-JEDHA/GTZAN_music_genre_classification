# dags/monitoring_drift_musicai.py
from airflow.sdk import dag, task
from datetime import datetime
import pandas as pd
import boto3
import io

S3_BUCKET = "ton-bucket"
FEATURE_COLUMNS = [...]  # tes 57 colonnes numériques GTZAN

@dag(
    dag_id="monitoring_drift_musicai",
    schedule="@daily",
    catchup=False,
    start_date=datetime(2026, 1, 1),
)
def monitoring_drift_musicai_dag():

    @task
    def load_reference():
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key="reference/gtzan_train_features.parquet")
        return pd.read_parquet(io.BytesIO(obj["Body"].read())).to_dict()

    @task
    def load_recent_batches():
        """Agrège les features loguées par Streamlit sur les dernières 24h."""
        s3 = boto3.client("s3")
        cutoff = datetime.now().strftime("%Y%m%d")
        objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"monitoring/batches/{cutoff}")
        dfs = []
        for o in objs.get("Contents", []):
            data = s3.get_object(Bucket=S3_BUCKET, Key=o["Key"])["Body"].read()
            dfs.append(pd.read_parquet(io.BytesIO(data)))
        return pd.concat(dfs, ignore_index=True).to_dict() if dfs else {}

    @task
    def run_drift_check(reference: dict, current: dict) -> dict:
        from evidently.test_suite import TestSuite
        from evidently.tests import TestShareOfDriftedColumns

        reference_df = pd.DataFrame(reference)[FEATURE_COLUMNS]
        current_df = pd.DataFrame(current)[FEATURE_COLUMNS]

        if current_df.empty:
            return {"all_passed": True, "skipped": True}

        tests = TestSuite(tests=[TestShareOfDriftedColumns(lt=0.3)])
        tests.run(reference_data=reference_df, current_data=current_df)
        result = tests.as_dict()

        tests.save_html("/tmp/drift_report.html")
        boto3.client("s3").upload_file(
            "/tmp/drift_report.html", S3_BUCKET,
            f"monitoring/reports/drift_{datetime.now():%Y%m%d_%H%M%S}.html",
        )
        return result["summary"]

    @task
    def log_to_mlflow(summary: dict):
        """Cohérent avec ton tracking MLflow existant sur HF Spaces."""
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # celui déjà configuré
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metric("all_tests_passed", int(summary.get("all_passed", True)))
            mlflow.set_tag("trigger_source", "airflow_monitoring")

    @task.branch
    def decide(summary: dict) -> str:
        if summary.get("skipped"):
            return "no_action"
        return "trigger_retrain" if not summary["all_passed"] else "no_action"

    @task
    def trigger_retrain():
        print("Drift détecté sur les features GTZAN → retrain nécessaire")

    @task
    def no_action():
        print("Pas de drift significatif")

    ref = load_reference()
    cur = load_recent_batches()
    summary = run_drift_check(ref, cur)
    log_to_mlflow(summary)
    branch = decide(summary)
    branch >> [trigger_retrain(), no_action()]

monitoring_drift_musicai_dag()