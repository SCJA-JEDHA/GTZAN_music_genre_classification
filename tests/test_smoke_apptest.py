"""
Smoke test de bout en bout : charge le script complet via le framework officiel
streamlit.testing.v1.AppTest (aucun navigateur requis, exécute le script comme
le ferait `streamlit run`). Sert de garde-fou générique en plus des tests
unitaires ciblés : si l'app plante au chargement (erreur d'import, exception
dans le rendu initial...), ce test l'attrape.
"""
from pathlib import Path
import os

import boto3
import pytest
from moto import mock_aws
from streamlit.testing.v1 import AppTest

APP_PATH = os.environ.get(
    "STREAMLIT_APP_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "streamlit" / "streamlit_music_app4.py"),
)
TEST_BUCKET = "test-bucket"
AWS_REGION = "eu-west-3"

@pytest.mark.timeout(60)
def test_app_loads_without_exception():
    if not Path(APP_PATH).exists():
        pytest.skip(f"Fichier app introuvable : {APP_PATH} (ajuste STREAMLIT_APP_PATH)")

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=TEST_BUCKET)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception, f"Exception au chargement de l'app : {at.exception}"


@pytest.mark.timeout(60)
def test_volume_slider_widget_absent():
    """Régression bug #1 : le widget slider volume ne doit plus apparaître dans l'UI."""
    with mock_aws():
        boto3.client("s3", region_name=AWS_REGION).create_bucket(Bucket=TEST_BUCKET,
                             CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        slider_labels = [s.label for s in at.slider]
        assert "🔊 Volume" not in slider_labels
