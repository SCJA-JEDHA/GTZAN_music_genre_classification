"""
Fixtures partagées pour les tests de l'app Streamlit MusicAI.

Le script app n'est pas structuré en package (code d'exécution au niveau module,
comme la plupart des apps Streamlit) : l'importer exécute donc tout le rendu de
la page. On mocke S3 (via moto) et les variables d'environnement AVANT chaque
import pour que ça ne casse jamais en CI (pas de vrai AWS/API nécessaire).
"""
import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from moto import mock_aws
import boto3

# ── Chemin vers le script de l'app — À ADAPTER si l'arborescence du repo diffère ──
APP_PATH = os.environ.get(
    "STREAMLIT_APP_PATH",
    str(Path(__file__).resolve().parent.parent / "streamlit" / "streamlit_music_app4.py"),
)

TEST_BUCKET = "test-bucket"
TEST_ENV = {
    "AWS_BUCKET": TEST_BUCKET,
    "MLFLOW_URI": "http://fake-mlflow.local",
    "API_URL": "http://fake-api.local",
    "API_MODEL_CNN_URL": "http://fake-api.local/cnn",
    "API_CALCUL_URL": "http://fake-api.local/calcul",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "eu-west-3",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Applique des variables d'environnement factices à chaque test."""
    for k, v in TEST_ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def s3_mock():
    """Bucket S3 en mémoire (moto) avec le bucket de test déjà créé."""
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-3")
        client.create_bucket(Bucket=TEST_BUCKET)
        yield client


@pytest.fixture
def app(s3_mock, _env):
    """
    Importe (ou réimporte) le module de l'app dans un environnement S3 mocké.
    Un import frais par test évite les fuites de session_state entre tests.
    """
    if not Path(APP_PATH).exists():
        pytest.skip(f"Fichier app introuvable : {APP_PATH} (ajuste STREAMLIT_APP_PATH)")

    spec = importlib.util.spec_from_file_location("streamlit_music_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["streamlit_music_app"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("streamlit_music_app", None)


@pytest.fixture
def sample_audio(app):
    """Signal audio synthétique (2s, sinus) — évite de dépendre d'un vrai fichier .wav."""
    sr = app.TARGET_SR
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2)).astype(np.float32)
    return y, sr


@pytest.fixture
def batch_session(app, tmp_path):
    """
    Prépare un session_state minimal représentant UNE ligne taguée prête à être
    sauvegardée (utilisé par les tests de régression sur _save_tagged_rows).
    """
    st = app.st
    spectros_dir = tmp_path / "spectros"
    spectros_dir.mkdir()

    name = "song1.wav"
    sr = app.TARGET_SR
    y = np.sin(2 * np.pi * 220 * np.linspace(0, 2, sr * 2)).astype(np.float32)

    percu_path = tmp_path / "song1_percu.png"
    harmo_path = tmp_path / "song1_harmo.png"
    percu_path.write_bytes(b"fake-png")
    harmo_path.write_bytes(b"fake-png")

    feats = app.compute_features(y, sr, filename=name)
    feats["user_name"] = "tester"
    feats["date_heure"] = "2026-07-17"
    feats["session_id"] = "sess123"

    st.session_state.batch_temp_dir = str(tmp_path)
    st.session_state.session_id = "sess123"
    st.session_state.user_name = "tester"
    st.session_state.df_user_music_temp = pd.DataFrame([{
        "name": name, "genre_pred_feat": "rock", "genre_pred_CNN": "rock", "genre_user": "rock",
    }])
    st.session_state.features_user_temp = feats
    st.session_state.spectro_user = pd.DataFrame([{
        "name": name, "spectro_percu": str(percu_path), "spectro_harmo": str(harmo_path),
    }])
    st.session_state.batch_audio_cache = {name: (y, sr)}
    st.session_state.batch_features_cache = {name: feats}
    st.session_state.batch_display_spectro_cache = {name: b"x"}
    st.session_state.fig_my_wave = b"x"
    st.session_state.fig_my_spect = b"x"
    st.session_state.fig_batch_wave = b"x"
    st.session_state.fig_batch_spect = b"x"
    st.session_state.my_show_visu = True
    st.session_state.my_y = y
    st.session_state.my_sr = sr
    st.session_state.my_features = feats
    st.session_state.predicted_genre = "rock"

    return {"name": name, "y": y, "sr": sr}
