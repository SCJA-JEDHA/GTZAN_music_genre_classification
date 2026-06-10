import argparse
import time
import os

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC



load_dotenv()

# Tracking server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default='rbf')
    parser.add_argument("--C", type=int, default=10)
    parser.add_argument("--gamma", type=str, default='scale')
    parser.add_argument("--probability", type=bool, default=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    experiment_name = "audio_classifier" # fixé dans le terminal au moment du run
    registered_model_name = "MGC_features_SVM_baseline"
    alias_name = "challenger"

    mlflow.set_experiment(experiment_name) # fixé dans le terminal au moment du run
    client = MlflowClient()

    print("Training model...")
    start_time = time.time()

    # Keep autolog basic, but log model manually
    mlflow.sklearn.autolog(log_models=False)

    # ------------------------------------------------------------------
    # Dataset: GTZAN Dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "s3://music-classification-project2/music-database/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv"
    )

    X = df.drop(columns=["filename", "label", "length"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    mapping = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

    # ------------------------------------------------------------------
    # Model pipeline
    # ------------------------------------------------------------------
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel=args.kernel,
                    C=args.C,
                    gamma=args.gamma,
                    probability=args.probability,
                    random_state=args.random_state,
                )
            )
        ]
    )

    with mlflow.start_run() as run:

        mlflow.set_tag("mlflow.user", "srachdi")

        model.fit(X_train, y_train_enc)

        predictions = model.predict(X_test)

        test_accuracy_score = accuracy_score(y_test_enc, predictions) 
        test_F1 = f1_score(y_test_enc, predictions, average='weighted')

        mlflow.log_metric("test_accuracy_score", test_accuracy_score)
        mlflow.log_metric("test_F1", test_F1)
        mlflow.log_param("dataset", "GTZAN Dataset")
        mlflow.log_dict(mapping, "label_mapping.json")

        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)

        # MLflow 3.x: prefer `name=` instead of deprecated `artifact_path=`
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        model_version = model_info.registered_model_version
        print(f"[INFO] Model logged as version {model_version}")

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=model_version,
        )
        print(f"[INFO] Alias '{alias_name}' now points to version {model_version}")

        # Optional: handy tags for the registry/UI
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="dataset",
            value="GTZAN Dataset",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="metric:test_accuracy_score",
            value=f"{test_accuracy_score:.4f}",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="metric:test_F1",
            value=f"{test_F1:.4f}",
        )

        print(f"[INFO] Run ID: {run.info.run_id}")
        print(f"[INFO] Test Accuracy Score: {test_accuracy_score:.4f}")
        print(f"[INFO] Test F1: {test_F1:.4f}")

        class_names = list(mapping.keys())

        fig, ax = plt.subplots(figsize=(8, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_true=y_test_enc,
            y_pred=predictions,
            display_labels=class_names,
            cmap='Blues',
            xticks_rotation='vertical',
            ax=ax
        )

        ax.set_title("Matrice de confusion test")
        # Envoi à MLflow
        mlflow.log_figure(fig, "confusion_matrix_audio_classifier_baseline_TEST.png")

    print("...Done!")
    print(f"--- Total training time: {time.time() - start_time:.2f} seconds")