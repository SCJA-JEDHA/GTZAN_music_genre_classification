"""
MusicAI — Analyse & Classification de Genre Musical
Streamlit app : sélection base GTZAN, upload audio, prédiction SVM + CNN,
recommandations PCA et visualisations waveform / spectrogramme.
"""
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import streamlit as st
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.pyfunc
import boto3
import json
import pickle
from PIL import Image
import base64
import os
from dotenv import load_dotenv
from typing import Tuple, Any

load_dotenv()

# CONFIG
MLFLOW_URI        = os.getenv("MLFLOW_URI")
API_URL           = os.getenv("API_URL")
API_MODEL_CNN_URL = os.getenv("API_MODEL_CNN_URL")
API_CALCUL_URL    = os.getenv("API_CALCUL_URL")

s3 = boto3.client('s3')

BUCKET               = os.getenv("AWS_BUCKET")
MUSIC_DATABASE_PREFIX = "music-database/gtzan-dataset-music-genre-classification/Data/"
GENRES_PREFIX        = MUSIC_DATABASE_PREFIX + "genres_original/"
PCA_PREFIX           = MUSIC_DATABASE_PREFIX + "PCA/"
PCA_PIPELINE         = "pca_pipeline.pkl"
PCA_X_PCA            = "X_pca.csv"

TARGET_SR      = 22050
CLIP_DURATION  = 30
N_FFT          = 2048
HOP            = 512
IMAGE_NX       = 432
IMAGE_NY       = 288

LIST_GENRES = [
    'blues','classical','country','disco','hiphop',
    'jazz','metal','pop','reggae','rock'
]


# HELPERS — AUDIO
from botocore.exceptions import ClientError

@st.cache_data(show_spinner=False)
def s3_key_exists(bucket: str, key: str) -> bool:
    s3c = boto3.client('s3')
    try:
        s3c.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

@st.cache_data(show_spinner=False)
def list_s3_subdirectories_sorted(bucket_name: str, prefix: str):
    s3c = boto3.client('s3')
    paginator = s3c.get_paginator('list_objects_v2')
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    subdirs = []
    for page in result:
        if 'CommonPrefixes' in page:
            for cp in page['CommonPrefixes']:
                full_path = cp['Prefix']
                subdir_name = full_path.rstrip('/').split('/')[-1]
                subdirs.append(subdir_name)
    return sorted(subdirs)

@st.cache_data(show_spinner=False)
def list_genres() -> list:
    return list_s3_subdirectories_sorted(BUCKET, GENRES_PREFIX)

@st.cache_data(show_spinner=False)
def list_tracks(genre: str) -> list:
    s3c = boto3.client('s3')
    prefix = f"{GENRES_PREFIX}{genre}/"
    paginator = s3c.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
    tracks = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.wav'):
                    tracks.append(key.split('/')[-1].rsplit('.', 1)[0])
    return sorted(tracks)

@st.cache_data(show_spinner=False)
def load_audio_s3(s3_key: str) -> tuple:
    s3c = boto3.client('s3')
    obj = s3c.get_object(Bucket=BUCKET, Key=s3_key)
    audio_bytes_data = obj['Body'].read()
    audio_buffer = io.BytesIO(audio_bytes_data)
    y, sr = librosa.load(audio_buffer, sr=None)
    return y, sr

def trim_audio(y, sr) -> np.ndarray:
    audio_trimmed, _ = librosa.effects.trim(y)
    return audio_trimmed

def preprocess_signal(y, sr) -> tuple:
    y_trimmed, _ = librosa.effects.trim(y)
    n_samples = int(CLIP_DURATION * TARGET_SR)
    y_resampled = librosa.resample(y_trimmed, orig_sr=sr, target_sr=TARGET_SR)
    y_clip = y_resampled[:n_samples] if len(y_resampled) >= n_samples else np.pad(
        y_resampled, (0, n_samples - len(y_resampled))
    )
    return y_clip, TARGET_SR

def compute_features(y, sr) -> pd.DataFrame:
    features, column_names = [], []
    features.append('user.file'); column_names.append('filename')
    features.append(len(y));      column_names.append('length')

    chroma             = librosa.feature.chroma_stft(y=y, sr=sr)
    rms                = librosa.feature.rms(y=y)
    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff            = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr  = librosa.effects.hpss(y)
    tempo, _           = librosa.beat.beat_track(y=y, sr=sr)
    mfccs              = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    feature_dict = {
        'chroma_stft': chroma, 'rms': rms,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'rolloff': rolloff, 'zero_crossing_rate': zero_crossing_rate,
        'harmony': harmony, 'perceptr': perceptr,
    }
    for name, data in feature_dict.items():
        features.extend([data.mean(), data.var()])
        column_names.extend([f'{name}_mean', f'{name}_var'])

    features.append(float(tempo) if np.isscalar(tempo) else tempo.mean())
    column_names.append('tempo')

    for idx, x in enumerate(mfccs):
        features.extend([np.mean(x), np.var(x)])
        column_names.extend([f"mfcc{idx+1}_mean", f"mfcc{idx+1}_var"])

    features.append('user'); column_names.append('label')

    df_features = pd.DataFrame(columns=column_names)
    df_features.loc[0] = features
    return df_features

def compute_melspectrogram(y, sr) -> np.ndarray:
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect.resize(IMAGE_NY, IMAGE_NX, refcheck=False)
    return spect

def audio_bytes_s3(s3_key: str) -> bytes:
    s3c = boto3.client('s3')
    obj = s3c.get_object(Bucket=BUCKET, Key=s3_key)
    return obj['Body'].read()

def image_to_base64(img):
    i_bytes = io.BytesIO()
    img.save(i_bytes, format="PNG")
    i_bytes.seek(0)
    return base64.b64encode(i_bytes.getvalue()).decode("utf-8")


# HELPERS — VISUALISATION
def fig_waveform(y_bytes: bytes, sr: int, title: str) -> plt.Figure:
    y = np.frombuffer(y_bytes, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y=y, sr=sr, color="royalblue", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig

def fig_spectrogram(y_bytes: bytes, sr: int, n_fft: int, hop_length: int, title: str) -> plt.Figure:
    y = np.frombuffer(y_bytes, dtype=np.float32)
    D  = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        DB, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="log",
        cmap="inferno", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig

def _y_bytes(y: np.ndarray) -> bytes:
    """Convertit un array float32 en bytes pour le cache."""
    return y.astype(np.float32).tobytes()

def _fig_to_png(fig: plt.Figure) -> bytes:
    """Convertit une figure matplotlib en bytes PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _show_png(png_bytes: bytes) -> None:
    """Affiche un PNG bytes via HTML base64 — aucun re-render Streamlit."""
    b64 = base64.b64encode(png_bytes).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block;margin-bottom:0.4rem">',
        unsafe_allow_html=True
    )


# HELPERS — MLFLOW
@st.cache_data(show_spinner=False)
def list_mlflow_models(tracking_uri: str) -> list:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        models = [m.name for m in client.search_registered_models()]
        return models if models else ["(aucun modèle trouvé)"]
    except Exception as e:
        return [f"Erreur MLflow : {e}"]

def revert_pred(prediction):
    unique_labels = LIST_GENRES
    mapping_LI = {l: unique_labels.index(l) for l in unique_labels}
    reverse_LI = {v: k for v, k in enumerate(mapping_LI)}
    return reverse_LI[prediction]


# HELPERS — PCA & RECOMMANDATIONS

@st.cache_data(show_spinner=False)
def load_pca_df_s3(PCA_PREFIX: str) -> Tuple[pd.DataFrame, Any]:
    s3c = boto3.client('s3')
    x_pca_key        = PCA_PREFIX + PCA_X_PCA
    pca_pipeline_key = PCA_PREFIX + PCA_PIPELINE

    X_pca_df     = None
    pca_pipeline = None

    try:
        obj      = s3c.get_object(Bucket=BUCKET, Key=x_pca_key)
        X_pca_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except s3c.exceptions.NoSuchKey:
        X_pca_df = None
    except Exception:
        X_pca_df = None

    try:
        obj          = s3c.get_object(Bucket=BUCKET, Key=pca_pipeline_key)
        pca_pipeline = pickle.load(obj['Body'])
    except s3c.exceptions.NoSuchKey:
        pca_pipeline = None
    except Exception:
        pca_pipeline = None

    return X_pca_df, pca_pipeline

def get_recommendations(df_pca, track_name, genre, n_neighbors=4):
    keywords = ["principal component", "princ_comp"]
    pc_cols = [c for c in df_pca.columns if any(k in c.lower() for k in keywords)]
    if not pc_cols or "label" not in df_pca.columns:
        return pd.DataFrame()

    genre_df = df_pca[df_pca["label"] == genre].copy()
    if genre_df.empty or len(genre_df) <= n_neighbors:
        return genre_df

    name_col_list = [c for c in df_pca.columns if c.lower() in ("filename","name","track","file")]
    if not name_col_list:
        return genre_df.head(n_neighbors)
    name_col = name_col_list[0]

    source = genre_df[genre_df[name_col].str.contains(track_name, na=False)]
    if source.empty:
        return genre_df.head(n_neighbors)

    X = genre_df[pc_cols].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
    src_idx = source.index[0]
    loc_idx = genre_df.index.get_loc(src_idx)
    _, indices = nbrs.kneighbors(X[loc_idx:loc_idx+1])
    return genre_df.iloc[indices[0][1:]]

def get_recommendations_my_music(df_pca, pca_pipeline, my_features_clean, genre, n_neighbors=4):
    keywords = ["principal component", "princ_comp"]
    pc_cols = [c for c in df_pca.columns if any(k in c.lower() for k in keywords)]
    if not pc_cols or "label" not in df_pca.columns:
        return pd.DataFrame()

    genre_df = df_pca[df_pca["label"] == genre].copy()
    if genre_df.empty or len(genre_df) <= n_neighbors:
        return genre_df

    name_col_list = [c for c in df_pca.columns if c.lower() in ("filename","name","track","file")]
    if not name_col_list:
        return genre_df.head(n_neighbors)

    X_my_music = pca_pipeline.transform(my_features_clean)
    X = genre_df[pc_cols].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X_my_music)
    return genre_df.iloc[indices[0]]

def project_new_point(pca_pipeline, features):
    return pca_pipeline.transform(features)[0]


# HELPERS — API PRÉDICTION
def calcul_image_pour_CNN(y, sr):
    n_fft, hop_length, n_mels = N_FFT, HOP, 128
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    def _make_img(signal):
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = np.flipud(S_db)
        norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(
            (256, 128), Image.Resampling.LANCZOS
        )

    img_h = _make_img(y_harmonic)
    img_p = _make_img(y_percussive)

    S_base = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB   = np.flipud(librosa.power_to_db(S_base, ref=np.max))

    payload_spectro = {"harmo_file": image_to_base64(img_h), "percu_file": image_to_base64(img_p)}
    return payload_spectro, S_DB

def call_predict_api_CNN(payload_img: dict) -> int:
    try:
        resp = requests.post(API_MODEL_CNN_URL + 'predict-cnn', json=payload_img)
        resp.raise_for_status()
        return resp.json()["prediction"]
    except Exception as e:
        return f"Erreur API : {e}"

def call_predict_api(list_features_obj: list) -> str:
    features_df  = list_features_obj[0]
    num_features = features_df.iloc[0].to_dict()
    payload_f    = {"model_name": "model_dummy", "num_features": num_features}
    try:
        resp = requests.post(API_URL + 'predict_f', json=payload_f, timeout=65)
        resp.raise_for_status()
        return resp.json()["prediction"][0]
    except Exception as e:
        return f"Erreur API : {e}"


# PAGE CONFIG & STYLE
st.set_page_config(
    page_title="MusicAI — Genre Classification",
    page_icon="🎵",
    layout="wide",
)

# Invalider les figures corrompues en session state (guard)
for _fk in ['fig_db_wave','fig_db_spect','fig_my_wave','fig_my_spect',
            'fig_dbrec_wave','fig_dbrec_spect','fig_myrec_wave','fig_myrec_spect']:
    if _fk in st.session_state and not isinstance(st.session_state.get(_fk), (bytes, type(None))):
        del st.session_state[_fk]

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  code, .stCode { font-family: 'DM Mono', monospace; }

  /* ── Espaces globaux réduits ── */
  .block-container {
      padding-top: 0.4rem !important;
      padding-bottom: 0.4rem !important;
  }
  h1 { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; }
  h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.2rem !important; }

  /* Réduit l'espace autour des dividers */
  hr { margin: 0.4rem 0 !important; }

  /* Réduit l'espace entre widgets Streamlit */
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
      gap: 0.3rem !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
      padding: 0 !important;
  }

  /* Paragraphes intro */
  [data-testid="stMarkdownContainer"] p { margin-bottom: 0.3rem !important; }

  /* ── Labels de section ── */
  .section-title {
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #7c6af7;
      margin-bottom: 0.25rem !important;
      margin-top: 0.1rem !important;
  }

  /* ── Badges genre prédit — centrés, même ligne ── */
  .badge-row {
      display: flex;
      flex-direction: row;
      gap: 0.6rem;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 0.3rem;
      margin-bottom: 0.3rem;
  }
  /* Aligne le bouton Prédire avec les badges */
  [data-testid='stColumns'] [data-testid='stVerticalBlock'] {
      display: flex;
      flex-direction: column;
      justify-content: center;
  }
  .predicted-badge {
      font-size: 1rem;
      font-weight: 800;
      color: #a78bfa;
      background: #1a1730;
      border: 2px solid #7c6af7;
      border-radius: 8px;
      padding: 0.2rem 0.5rem;
      display: inline-block;
      white-space: nowrap;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
  }
  .badge-label {
      font-size: 0.6rem;
      color: #7c6af7;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      display: block;
      text-align: center;
      margin-bottom: 0.1rem;
  }

  /* ── Séparateur ancre visualisation ── */
  .visu-anchor {
      border-top: 1px solid #2a2a3a;
      margin-top: 1rem;
      margin-bottom: 0.4rem;
  }

  /* ── Section box ── */
  .section-box {
      background: #0f0f14;
      border: 1px solid #2a2a3a;
      border-radius: 10px;
      padding: 0.7rem 1rem;
      margin-bottom: 0.6rem;
  }

  /* Figures matplotlib : moins de marge */
  [data-testid="stImage"] { margin-top: 0 !important; margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)


# EN-TÊTE
st.markdown("# 🎵 MusicAI — Analyse & Classification de Genre Musical")
st.markdown("🎧 Déposez un son. Découvrez son genre. Explorez ce qui lui ressemble.")
st.divider()


# SESSION STATE
_defaults = {
    "db_audio_bytes": None, "my_audio_bytes": None, "rec_audio_bytes": None,
    "my_payload_spectro": None,
    "predicted_genre": "—", "predicted_genre_feat": "—", "predicted_genre_CNN": "—",
    "my_y": None, "my_sr": None, "my_user_S_DB": None,
    "db_y": None, "db_sr": None,
    "rec_y": None, "rec_sr": None,
    "my_features": None, "my_coords_pca": None,
    "db_show_visu": False,
    "rec_show_visu": False,
    "db_rec_show_visu": False,
    "db_rec_audio_bytes": None,
    "db_rec_y": None, "db_rec_sr": None,
    "my_rec_show_visu": False,
    "my_rec_audio_bytes": None,
    "my_rec_y": None, "my_rec_sr": None,
    "fig_db_wave": None, "fig_db_spect": None,
    "fig_my_wave": None, "fig_my_spect": None,
    "fig_dbrec_wave": None, "fig_dbrec_spect": None,
    "fig_myrec_wave": None, "fig_myrec_spect": None,
    "my_show_visu": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Paramètres spectrogramme")
    n_fft = st.selectbox("n_fft", [512, 1024, 2048, 4096, 8192], index=2)
    hop_length = st.selectbox("hop_length", [128, 256, 512, 1024, 2048, 4096], index=2)
    st.markdown("---")
    st.markdown("### 🔗 MLflow")
    mlflow_uri = st.text_input("Adresse MLflow", value=MLFLOW_URI)


# CHARGEMENT PCA (une seule fois par session)
X_pca_df, pca_pipeline = load_pca_df_s3(PCA_PREFIX)
df_pca = X_pca_df if X_pca_df is not None else pd.DataFrame()


def _run_prediction():
    if st.session_state.my_y is None:
        return
    feats = st.session_state.my_features
    spect = compute_melspectrogram(st.session_state.my_y, st.session_state.my_sr)
    pred1 = call_predict_api([feats, spect])
    st.session_state.predicted_genre_feat = revert_pred(pred1)
    pred_CNN = call_predict_api_CNN(st.session_state.my_payload_spectro)
    st.session_state.predicted_genre_CNN  = revert_pred(pred_CNN)
    st.session_state.predicted_genre      = st.session_state.predicted_genre_CNN
    st.session_state.my_show_visu         = True
    _y = st.session_state.my_y
    _sr = st.session_state.my_sr
    st.session_state.fig_my_wave  = _fig_to_png(fig_waveform(_y_bytes(_y), _sr, "Forme d'onde — Ma musique"))
    st.session_state.fig_my_spect = _fig_to_png(fig_spectrogram(_y_bytes(_y), _sr, N_FFT, HOP, "Spectrogramme — Ma musique"))


# RANGÉE HAUTE — Sélection / Upload / Recommandations
row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1], gap="small")

# RANGÉE HAUTE COL 1 — Base de données
def _ctrl_db():
    st.markdown('<div class="section-title">📂 Sélection — Base de données</div>', unsafe_allow_html=True)

    genres = list_genres()
    genre_sel = st.selectbox("Genre", genres or ["(vide)"], key="db_genre") if genres else None
    st.session_state['_genre_sel'] = genre_sel
    tracks = list_tracks(genre_sel) if genre_sel else []
    track_sel = st.selectbox("Morceau", tracks, key="db_track") if tracks else None
    st.session_state['_track_sel'] = track_sel
    s3_key = f"{GENRES_PREFIX}{genre_sel}/{track_sel}.wav" if genre_sel and track_sel else None

    if st.button("▶ Play", key="btn_play_db", disabled=(s3_key is None), width='stretch'):
        if s3_key and s3_key_exists(BUCKET, s3_key):
            st.session_state.db_audio_bytes = audio_bytes_s3(s3_key)
            y_db, sr_db = load_audio_s3(s3_key)
            st.session_state.db_y           = trim_audio(y_db, sr_db)
            st.session_state.db_sr          = sr_db
            st.session_state.db_show_visu   = True
            _ydb = st.session_state.db_y
            st.session_state.fig_db_wave  = _fig_to_png(fig_waveform(_y_bytes(_ydb), sr_db, f"Forme d'onde — {track_sel}"))
            st.session_state.fig_db_spect = _fig_to_png(fig_spectrogram(_y_bytes(_ydb), sr_db, N_FFT, HOP, f"Spectrogramme — {track_sel}"))
        else:
            st.error(f"Fichier introuvable : {s3_key}")

    if st.session_state.db_audio_bytes:
        st.audio(st.session_state.db_audio_bytes, format="audio/wav")


# RANGÉE HAUTE COL 2 — Upload & Prédiction
def _ctrl_my():
    st.markdown('<div class="section-title">🎙 Votre musique</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Fichier audio (.wav / .mp3 / .ogg)",
        type=["wav", "mp3", "ogg"],
        key="uploader",
        label_visibility="collapsed"
    )
    if uploaded is not None:
        file_id = uploaded.name + str(uploaded.size)
        if st.session_state.get("_last_uploaded") != file_id:
            st.session_state["_last_uploaded"] = file_id
            raw_bytes = uploaded.read()
            with st.spinner("Prétraitement…"):
                y_raw, sr_raw = librosa.load(io.BytesIO(raw_bytes), sr=None)
                y_proc, sr_proc = preprocess_signal(y_raw, sr_raw)
                st.session_state.my_y           = y_proc
                st.session_state.my_sr          = sr_proc
                st.session_state.my_audio_bytes = raw_bytes
                st.session_state.my_features    = compute_features(y_proc, sr_proc)
                st.session_state.my_payload_spectro, st.session_state.my_user_S_DB = (
                    calcul_image_pour_CNN(y_proc, sr_proc)
                )

    if st.session_state.my_audio_bytes:
        st.audio(st.session_state.my_audio_bytes)

    cpred, cbadge1, cbadge2 = st.columns([1.2, 1, 1])
    with cbadge1:
        st.markdown(f"""
        <div style='text-align:center;margin-top:0.6rem'>
          <span class='badge-label'>Features</span><br>
          <span class='predicted-badge'>{st.session_state.predicted_genre_feat}</span>
        </div>""", unsafe_allow_html=True)
    with cbadge2:
        st.markdown(f"""
        <div style='text-align:center;margin-top:0.6rem'>
          <span class='badge-label'>CNN</span><br>
          <span class='predicted-badge'>{st.session_state.predicted_genre_CNN}</span>
        </div>""", unsafe_allow_html=True)
    with cpred:
        st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
        st.button("🔍 Prédire", width='stretch', key="btn_predict",
                  on_click=_run_prediction,
                  disabled=(st.session_state.my_y is None))


# RANGÉE HAUTE COL 3 — Recommandations
def _ctrl_rec():
    st.markdown('<div class="section-title">🔀 Recommandations</div>', unsafe_allow_html=True)

    source_choice = st.radio("Source", ["database choice", "my music"],
                             horizontal=True, key="rec_source")
    st.session_state['_source_choice'] = source_choice

    my_pca_coords     = None
    my_features_clean = None
    if st.session_state.my_features is not None and pca_pipeline is not None:
        cols_to_drop = ["filename", "length", "label"]
        my_features_clean = st.session_state.my_features.drop(
            columns=[c for c in cols_to_drop if c in st.session_state.my_features.columns]
        )
        my_pca_coords = project_new_point(pca_pipeline, my_features_clean)
        st.session_state.my_coords_pca = my_pca_coords
        st.session_state['_my_pca_coords'] = my_pca_coords

    name_col_candidates = (
        [c for c in df_pca.columns if c.lower() in ("filename","name","track","file")]
        if not df_pca.empty else []
    )
    name_col = name_col_candidates[0] if name_col_candidates else None
    st.session_state['_name_col'] = name_col

    _track_sel = st.session_state.get('_track_sel') or ""
    _genre_sel = st.session_state.get('_genre_sel') or ""
    if source_choice == "database choice":
        src_track = _track_sel
        src_genre = _genre_sel
    else:
        src_track = "ma_musique"
        src_genre = st.session_state.predicted_genre if st.session_state.predicted_genre != "—" else ""
        if st.session_state.predicted_genre == "—":
            st.session_state.my_rec_show_visu   = False
            st.session_state.my_rec_audio_bytes = None
    st.session_state['_src_track'] = src_track

    rec_options = []
    if not df_pca.empty and src_genre:
        if source_choice == "database choice":
            rec_df = get_recommendations(df_pca, src_track, src_genre)
        elif my_features_clean is not None:
            rec_df = get_recommendations_my_music(df_pca, pca_pipeline, my_features_clean, src_genre)
        else:
            rec_df = pd.DataFrame()
        if name_col and not rec_df.empty:
            rec_options = rec_df[name_col].tolist()

    rec_sel = st.selectbox(
        "Recommandation",
        rec_options if rec_options else ["(aucune recommandation disponible)"],
        key="rec_track"
    )

    if st.button("▶ Play", key="btn_play_rec",
                 disabled=(rec_sel == "(aucune recommandation disponible)" or not src_genre),
                 width='stretch'):
        s3_key_rec = f"{GENRES_PREFIX}{src_genre}/{rec_sel}"
        if s3_key_exists(BUCKET, s3_key_rec):
            _ab = audio_bytes_s3(s3_key_rec)
            _y, _sr = load_audio_s3(s3_key_rec)
            _yt = trim_audio(_y, _sr)
            if source_choice == "database choice":
                st.session_state.db_rec_audio_bytes = _ab
                st.session_state.db_rec_y           = _yt
                st.session_state.db_rec_sr          = _sr
                st.session_state.db_rec_show_visu   = True
                st.session_state.fig_dbrec_wave  = _fig_to_png(fig_waveform(_y_bytes(_yt), _sr, f"Forme d'onde — {rec_sel}"))
                st.session_state.fig_dbrec_spect = _fig_to_png(fig_spectrogram(_y_bytes(_yt), _sr, N_FFT, HOP, f"Spectrogramme — {rec_sel}"))
            else:
                st.session_state.my_rec_audio_bytes = _ab
                st.session_state.my_rec_y           = _yt
                st.session_state.my_rec_sr          = _sr
                st.session_state.my_rec_show_visu   = True
                st.session_state.fig_myrec_wave  = _fig_to_png(fig_waveform(_y_bytes(_yt), _sr, f"Forme d'onde — {rec_sel}"))
                st.session_state.fig_myrec_spect = _fig_to_png(fig_spectrogram(_y_bytes(_yt), _sr, N_FFT, HOP, f"Spectrogramme — {rec_sel}"))
        else:
            st.error(f"Fichier introuvable : {s3_key_rec}")

    _cur_ab = st.session_state.db_rec_audio_bytes if source_choice == "database choice" else st.session_state.my_rec_audio_bytes
    if _cur_ab:
        st.audio(_cur_ab, format="audio/wav")


# Appels rangée haute
with row1_col1: _ctrl_db()
with row1_col2: _ctrl_my()
with row1_col3: _ctrl_rec()

# SÉPARATEUR
st.divider()

# RANGÉE BASSE — Visualisations (fragments indépendants)
row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1], gap="small")

@st.fragment
def _visu_db():
    st.markdown('<div class="section-title">📈 Visualisation — Base</div>', unsafe_allow_html=True)
    if st.session_state.db_show_visu and isinstance(st.session_state.get("fig_db_wave"), bytes):
        _show_png(st.session_state.fig_db_wave)
        _show_png(st.session_state.fig_db_spect)
    else:
        st.info("Sélectionnez un morceau et cliquez sur ▶ Play.")

@st.fragment
def _visu_my():
    st.markdown('<div class="section-title">📈 Visualisation — Ma musique</div>', unsafe_allow_html=True)
    if st.session_state.my_show_visu and isinstance(st.session_state.get("fig_my_wave"), bytes):
        _show_png(st.session_state.fig_my_wave)
        _show_png(st.session_state.fig_my_spect)
    else:
        st.info("Cliquez sur Prédire pour afficher la visualisation.")

with row2_col1:
    _visu_db()

with row2_col2:
    _visu_my()

# RANGÉE BASSE COL 3 — Plot & PCA
@st.fragment
def _visu_rec(src_choice):
    tab_plot, tab_pca = st.tabs(["📊 Plot", "🔵 Composantes principales"])
    _src = src_choice
    name_col    = st.session_state.get('_name_col')
    src_track   = st.session_state.get('_src_track', '')
    my_pca_coords = st.session_state.get('_my_pca_coords')
    track_sel   = st.session_state.get('_track_sel', '')

    with tab_plot:
        if _src == "my music" and st.session_state.predicted_genre == "—":
            st.info("Chargez et prédisez votre musique (colonne centrale) pour obtenir des recommandations.")
        elif _src == "database choice" and st.session_state.db_rec_show_visu and isinstance(st.session_state.get("fig_dbrec_wave"), bytes):
            _show_png(st.session_state.fig_dbrec_wave)
            _show_png(st.session_state.fig_dbrec_spect)
        elif _src == "my music" and st.session_state.my_rec_show_visu and isinstance(st.session_state.get("fig_myrec_wave"), bytes):
            _show_png(st.session_state.fig_myrec_wave)
            _show_png(st.session_state.fig_myrec_spect)
        else:
            st.info("Lancez une recommandation pour visualiser le signal.")

    with tab_pca:
        if df_pca.empty:
            st.warning(f"Aucun fichier PCA trouvé dans `{PCA_PREFIX}`.")
        else:
            keywords = ["principal component", "princ_comp"]
            pc_cols  = [c for c in df_pca.columns if any(k in c.lower() for k in keywords)]

            color_map = {
                "base": "#4a4a6a",
                "sélection DB": "#de2626",
                "ma musique": "#033688",
            }

            if len(pc_cols) < 3:
                if len(pc_cols) >= 2:
                    pc1, pc2 = pc_cols[0], pc_cols[1]
                    hover_data = {}
                    if name_col: hover_data[name_col] = True
                    if "label" in df_pca.columns: hover_data["label"] = True
                    fig_pca = px.scatter(
                        df_pca, x=pc1, y=pc2, color="label",
                        opacity=0.8, hover_data=hover_data,
                        title="Espace PCA — 2 composantes",
                    )
                    st.plotly_chart(fig_pca, width='stretch')
                st.warning("Le DataFrame PCA doit contenir au moins 3 composantes.")
            else:
                pc1, pc2, pc3 = pc_cols[0], pc_cols[1], pc_cols[2]
                df_plot = df_pca.copy()
                df_plot["_highlight"] = "base"
                if name_col and src_track in df_plot[name_col].values:
                    df_plot.loc[df_plot[name_col] == src_track, "_highlight"] = "sélection DB"

                hover_data = {}
                if name_col: hover_data[name_col] = True
                if "label" in df_plot.columns: hover_data["label"] = True

                fig_pca = px.scatter_3d(
                    df_plot, x=pc1, y=pc2, z=pc3,
                    color="label", opacity=0.8,
                    hover_data=hover_data,
                    title="Espace PCA — 3 composantes",
                )
                fig_pca.update_traces(marker=dict(size=4))

                if my_pca_coords is not None and len(my_pca_coords) >= 3:
                    fig_pca.add_trace(go.Scatter3d(
                        x=[my_pca_coords[0]], y=[my_pca_coords[1]], z=[my_pca_coords[2]],
                        mode="markers+text",
                        marker=dict(size=10, color="#3b82f6", symbol="diamond"),
                        text=["Ma musique"], textposition="top center",
                        name="ma musique",
                    ))

                if name_col and track_sel:
                    sel_filename = f"{track_sel}.wav"
                    idx_matches  = df_pca.index[df_pca[name_col] == sel_filename]
                    if len(idx_matches) > 0:
                        idx_sel = idx_matches[0]
                        fig_pca.add_trace(go.Scatter3d(
                            x=[df_pca[pc1][idx_sel]],
                            y=[df_pca[pc2][idx_sel]],
                            z=[df_pca[pc3][idx_sel]],
                            mode="markers+text",
                            marker=dict(size=10, color="#a01a10", symbol="diamond"),
                            text=["Database sel."], textposition="top center",
                            name="Database sel.",
                        ))

                fig_pca.update_layout(
                    legend_title_text="Catégorie",
                    scene=dict(xaxis_title=pc1, yaxis_title=pc2, zaxis_title=pc3),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=480,
                )
                st.plotly_chart(fig_pca, width='stretch', key="pca_chart")

                if name_col:
                    all_tracks = df_pca[name_col].dropna().tolist()
                    clicked_track = st.selectbox(
                        "Sélectionner un morceau sur le graphique PCA",
                        ["(aucun)"] + all_tracks,
                        key="pca_click_track"
                    )
                    if clicked_track and clicked_track != "(aucun)":
                        if "label" in df_pca.columns:
                            row = df_pca[df_pca[name_col] == clicked_track]
                            if not row.empty:
                                new_genre = row["label"].values[0]
                                st.info(
                                    f"Sélection : **{clicked_track}** ({new_genre}). "
                                    "Revenez en colonne gauche pour charger ce morceau."
                                )
                                st.query_params["db_genre"] = new_genre
                                st.query_params["db_track"] = clicked_track

with row2_col3:
    _visu_rec(st.session_state.get('_source_choice', 'database choice'))

