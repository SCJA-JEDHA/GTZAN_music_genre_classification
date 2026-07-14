"""
MusicAI — Analyse & Classification de Genre Musical
Streamlit app : sélection base GTZAN, upload audio, prédiction SVM + CNN,
recommandations PCA et visualisations waveform / spectrogramme.
"""
# developped in streamlit==1.58.0
# to launch the streamlit in local : 
# streamlit run .\streamlit_music_app3.py --server.runOnSave true --logger.level=debug

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
import tempfile
import uuid
from datetime import datetime
import soundfile as sf
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

# CONFIG — MODE BATCH (colonne 2)
N_MUSIC_FILES   = 20                          # nb max de fichiers en file d'attente
M_DISPLAY_ROWS  = 12                          # nb de lignes affichées dans liste_music
TIME_THRESHOLD  = 15                          # début du segment analysé (s)
TIME_SEGMENT    = 30                          # durée du segment analysé (s)
MUSIC_USER_PREFIX      = "MUSIC_USER/"                       # préfixe S3 des musiques taguées (dans le bucket existant)
SPECTRO_USER_PREFIX    = MUSIC_USER_PREFIX + "spectrograms/"  # PNG harmo/percu (CNN)
FEATURES_USER_PREFIX   = MUSIC_USER_PREFIX + "features/"      # 1 CSV features par session
SPECTRO_CSV_PREFIX     = MUSIC_USER_PREFIX + "spectro/"       # 1 CSV catalogue spectro par session
FEATURES_USER_MERGED_CSV = MUSIC_USER_PREFIX + "features_music_user.csv"  # fusion de tous les CSV /features
SPECTRO_USER_MERGED_CSV  = MUSIC_USER_PREFIX + "spectro_music_user.csv"   # fusion de tous les CSV /spectro


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
        return resp.json()["prediction"][0]     # take first element of list 
    except Exception as e:
        return f"Erreur API : {e}"


# HELPERS — MODE BATCH (colonne 2)
def extract_segment(y: np.ndarray, sr: int,
                     t_start: float = TIME_THRESHOLD,
                     t_dur: float = TIME_SEGMENT) -> np.ndarray:
    """Extrait un segment de t_dur secondes à partir de t_start (pad si trop court)."""
    i0 = int(t_start * sr)
    i1 = i0 + int(t_dur * sr)
    if i0 >= len(y):
        i0, i1 = 0, min(len(y), int(t_dur * sr))
    seg = y[i0:i1]
    n_target = int(t_dur * sr)
    if len(seg) < n_target:
        seg = np.pad(seg, (0, n_target - len(seg)))
    return seg.astype(np.float32)

def preprocess_and_extract_segment(y: np.ndarray, sr: int,
                                    t_start: float = TIME_THRESHOLD,
                                    t_dur: float = TIME_SEGMENT) -> tuple:
    """Trim + resample à TARGET_SR (preprocess_signal), puis extrait le segment [t_start, t_start+t_dur]."""
    y_trimmed, _ = librosa.effects.trim(y)
    y_resampled  = librosa.resample(y_trimmed, orig_sr=sr, target_sr=TARGET_SR)
    seg = extract_segment(y_resampled, TARGET_SR, t_start, t_dur)
    return seg, TARGET_SR

def _save_b64_png(b64_str: str, path: Path) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def _remaining_slots() -> int:
    return N_MUSIC_FILES - len(st.session_state.df_user_music_temp)

def _process_loaded_files(files: list) -> None:
    """Traite les fichiers reçus via [load] : sauvegarde disque, features + CNN, MAJ liste_music."""
    st.session_state.load_locked = True
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())
    st.session_state.date_heure_load = datetime.now()

    tmp_dir = Path(st.session_state.batch_temp_dir) if st.session_state.batch_temp_dir \
        else Path(tempfile.mkdtemp(prefix="musicai_"))
    st.session_state.batch_temp_dir = str(tmp_dir)

    new_names = []
    for f in files:
        raw = f.read()
        (tmp_dir / f.name).write_bytes(raw)
        new_names.append(f.name)

    was_empty_before_load = st.session_state.df_user_music_temp.empty

    new_rows = pd.DataFrame({
        "name": new_names, "genre_pred_feat": None,
        "genre_pred_CNN": None, "genre_user": None,
    })
    st.session_state.df_user_music_temp = pd.concat(
        [st.session_state.df_user_music_temp, new_rows], ignore_index=True
    )
    # la ligne active n'est fixée à 0 qu'une fois la 1ère ligne effectivement calculée (cf. boucle ci-dessous)
    if st.session_state.active_row_idx is None:
        st.session_state.active_row_idx = 0
    first_row_highlighted = not was_empty_before_load

    liste_ph = st.empty()
    liste_ph.dataframe(st.session_state.df_user_music_temp, hide_index=True, width='stretch')

    out_dir = tmp_dir / "spectros"
    out_dir.mkdir(exist_ok=True)

    for name in new_names:
        with st.spinner(f"Analyse de {name}…"):
            try:
                y_raw, sr_raw = librosa.load(str(tmp_dir / name), sr=None)
                y_seg, sr_seg = preprocess_and_extract_segment(y_raw, sr_raw)

                feats_df = compute_features(y_seg, sr_seg)
                pred_feat = call_predict_api([feats_df])
                genre_feat = revert_pred(pred_feat) if isinstance(pred_feat, (int, np.integer)) else pred_feat

                payload_spectro, _ = calcul_image_pour_CNN(y_seg, sr_seg)
                pred_cnn = call_predict_api_CNN(payload_spectro)
                genre_cnn = revert_pred(pred_cnn) if isinstance(pred_cnn, (int, np.integer)) else pred_cnn

                idx = st.session_state.df_user_music_temp.index[
                    st.session_state.df_user_music_temp["name"] == name][0]
                st.session_state.df_user_music_temp.loc[idx, "genre_pred_feat"] = genre_feat
                st.session_state.df_user_music_temp.loc[idx, "genre_pred_CNN"]  = genre_cnn

                feats_row = feats_df.copy()
                feats_row["user_name"]  = st.session_state.user_name
                feats_row["date_heure"] = st.session_state.date_heure_load
                feats_row["session_id"] = st.session_state.session_id
                st.session_state.features_user_temp = pd.concat(
                    [st.session_state.features_user_temp, feats_row], ignore_index=True
                )

                stem = Path(name).stem
                harmo_path = out_dir / f"{stem}_harmo.png"
                percu_path = out_dir / f"{stem}_percu.png"
                _save_b64_png(payload_spectro["harmo_file"], harmo_path)
                _save_b64_png(payload_spectro["percu_file"], percu_path)
                st.session_state.spectro_user = pd.concat([
                    st.session_state.spectro_user,
                    pd.DataFrame([{"name": name, "spectro_percu": str(percu_path),
                                    "spectro_harmo": str(harmo_path)}])
                ], ignore_index=True)

                # spectrogramme destiné à l'affichage (rejoué au clic sur [Play])
                disp_spect_png = _fig_to_png(
                    fig_spectrogram(_y_bytes(y_seg), sr_seg, N_FFT, HOP, f"Spectrogramme — {name}")
                )
                disp_path = out_dir / f"{stem}_spectrog_affichage.png"
                disp_path.write_bytes(disp_spect_png)
                st.session_state.batch_display_spectro_cache[name] = disp_spect_png

                st.session_state.batch_audio_cache[name] = (y_seg, sr_seg)
                st.session_state.batch_features_cache[name] = feats_df

                if not first_row_highlighted:
                    st.session_state.active_row_idx = idx
                    first_row_highlighted = True
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de {name} : {e}")

            liste_ph.dataframe(st.session_state.df_user_music_temp, hide_index=True, width='stretch')

def _play_active_row() -> None:
    idx = st.session_state.active_row_idx
    df = st.session_state.df_user_music_temp
    if idx is None or idx >= len(df):
        return
    name = df.iloc[idx]["name"]
    y_seg, sr = st.session_state.batch_audio_cache.get(name, (None, None))
    if y_seg is None:
        fpath = Path(st.session_state.batch_temp_dir) / name
        y_raw, sr_raw = librosa.load(str(fpath), sr=None)
        y_seg, sr = preprocess_and_extract_segment(y_raw, sr_raw)

    buf = io.BytesIO()
    sf.write(buf, y_seg, sr, format="WAV")
    st.session_state.batch_play_audio = buf.getvalue()
    st.session_state.fig_batch_wave = _fig_to_png(fig_waveform(_y_bytes(y_seg), sr, f"Forme d'onde — {name}"))

    disp_png = st.session_state.batch_display_spectro_cache.get(name)
    if disp_png is None:
        disp_path = Path(st.session_state.batch_temp_dir) / "spectros" / f"{Path(name).stem}_spectrog_affichage.png"
        if disp_path.exists():
            disp_png = disp_path.read_bytes()
        else:
            disp_png = _fig_to_png(fig_spectrogram(_y_bytes(y_seg), sr, N_FFT, HOP, f"Spectrogramme — {name}"))
        st.session_state.batch_display_spectro_cache[name] = disp_png
    st.session_state.fig_batch_spect = disp_png
    st.session_state["_active_played"] = True

    # Réutilisation par le module Recommandations (colonne 3) qui s'appuie sur
    # my_features / predicted_genre / fig_my_* — on les repointe vers la ligne active.
    st.session_state.fig_my_wave  = st.session_state.fig_batch_wave
    st.session_state.fig_my_spect = st.session_state.fig_batch_spect
    st.session_state.my_show_visu = True
    st.session_state.my_y  = y_seg
    st.session_state.my_sr = sr
    st.session_state.my_features = st.session_state.batch_features_cache.get(name)
    genre_cnn_row = df.iloc[idx]["genre_pred_CNN"]
    st.session_state.predicted_genre = genre_cnn_row if genre_cnn_row in LIST_GENRES else "—"

def _save_tagged_rows() -> None:
    """Envoie les fichiers taggués vers S3 (segment 30s + features + spectrogrammes + catalogues CSV par session), purge la liste."""
    df = st.session_state.df_user_music_temp
    tagged_mask = df["genre_user"].notna() & (df["genre_user"] != "")
    if not tagged_mask.any():
        return

    df_transfer = df[tagged_mask].copy()
    names_transfer = df_transfer["name"].tolist()
    feats = st.session_state.features_user_temp
    spectro = st.session_state.spectro_user
    tmp_dir = Path(st.session_state.batch_temp_dir)
    session_id = st.session_state.session_id

    try:
        s3c = boto3.client('s3')

        # 1) upload — uniquement le segment de 30s analysé (pas le fichier complet)
        for name in names_transfer:
            y_seg, sr_seg = st.session_state.batch_audio_cache.get(name, (None, None))
            if y_seg is None:
                y_raw, sr_raw = librosa.load(str(tmp_dir / name), sr=None)
                y_seg, sr_seg = preprocess_and_extract_segment(y_raw, sr_raw)
            wav_buf = io.BytesIO()
            sf.write(wav_buf, y_seg, sr_seg, format="WAV")
            s3_audio_key = f"{MUSIC_USER_PREFIX}{Path(name).stem}.wav"
            s3c.put_object(Bucket=BUCKET, Key=s3_audio_key, Body=wav_buf.getvalue())

        # 2) features_user_temp_transfer = left-join avec df_user_music_temp_transfer (sans re-dupliquer 'filename')
        feats_transfer = feats[feats["filename"].isin(names_transfer)].copy()
        feats_transfer = feats_transfer.merge(
            df_transfer.rename(columns={"name": "filename"}),
            on="filename", how="left"
        )

        # 3) append dans le CSV features de LA session (1 fichier par session_id -> pas de conflit inter-utilisateurs)
        feats_csv_key = f"{FEATURES_USER_PREFIX}features_{session_id}.csv"
        try:
            obj = s3c.get_object(Bucket=BUCKET, Key=feats_csv_key)
            df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
            df_out = pd.concat([df_existing, feats_transfer], ignore_index=True)
        except s3c.exceptions.NoSuchKey:
            df_out = feats_transfer
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        s3c.put_object(Bucket=BUCKET, Key=feats_csv_key, Body=csv_buf.getvalue())

        # 4) upload des spectrogrammes (fichiers image) correspondants
        spectro_transfer = spectro[spectro["name"].isin(names_transfer)].copy()
        spectro_transfer["spectro_percu_key"] = spectro_transfer["spectro_percu"].apply(
            lambda p: f"{SPECTRO_USER_PREFIX}{Path(p).name}")
        spectro_transfer["spectro_harmo_key"] = spectro_transfer["spectro_harmo"].apply(
            lambda p: f"{SPECTRO_USER_PREFIX}{Path(p).name}")
        for _, r in spectro_transfer.iterrows():
            s3c.upload_file(r["spectro_percu"], BUCKET, r["spectro_percu_key"])
            s3c.upload_file(r["spectro_harmo"], BUCKET, r["spectro_harmo_key"])

        # 5) spectro_user_temp_transfer = left-join avec df_user_music_temp_transfer (clé nom_fichier)
        #    -> on stocke la CLÉ S3 (pas le chemin local, détruit après la session)
        spectro_user_temp_transfer = spectro_transfer[["name", "spectro_percu_key", "spectro_harmo_key"]].rename(
            columns={"name": "nom_fichier", "spectro_percu_key": "spectro_percu", "spectro_harmo_key": "spectro_harmo"}
        ).merge(
            df_transfer.rename(columns={"name": "nom_fichier"}),
            on="nom_fichier", how="left"
        )

        # 6) append dans le CSV spectro de LA session
        spectro_csv_key = f"{SPECTRO_CSV_PREFIX}spectro_{session_id}.csv"
        try:
            obj = s3c.get_object(Bucket=BUCKET, Key=spectro_csv_key)
            df_spectro_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
            df_spectro_out = pd.concat([df_spectro_existing, spectro_user_temp_transfer], ignore_index=True)
        except s3c.exceptions.NoSuchKey:
            df_spectro_out = spectro_user_temp_transfer
        spectro_csv_buf = io.StringIO()
        df_spectro_out.to_csv(spectro_csv_buf, index=False)
        s3c.put_object(Bucket=BUCKET, Key=spectro_csv_key, Body=spectro_csv_buf.getvalue())

    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde S3 : {e}")
        return

    # 5) succès -> purge des lignes sauvegardées (mémoire + disque)
    spectro_transfer_paths = spectro[spectro["name"].isin(names_transfer)]

    st.session_state.df_user_music_temp = df[~tagged_mask].reset_index(drop=True)
    st.session_state.features_user_temp = feats[~feats["filename"].isin(names_transfer)].reset_index(drop=True)
    st.session_state.spectro_user = spectro[~spectro["name"].isin(names_transfer)].reset_index(drop=True)

    for name in names_transfer:
        # fichier musique temporaire
        (tmp_dir / name).unlink(missing_ok=True)
        # caches mémoire
        st.session_state.batch_audio_cache.pop(name, None)
        st.session_state.batch_features_cache.pop(name, None)
        st.session_state.batch_display_spectro_cache.pop(name, None)
        # spectrogramme d'affichage sur disque
        (tmp_dir / "spectros" / f"{Path(name).stem}_spectrog_affichage.png").unlink(missing_ok=True)

    # fichiers spectrogrammes CNN (percussif + harmonique) sur disque
    for _, r in spectro_transfer_paths.iterrows():
        Path(r["spectro_percu"]).unlink(missing_ok=True)
        Path(r["spectro_harmo"]).unlink(missing_ok=True)

    st.session_state.active_row_idx  = 0
    st.session_state["_active_played"] = False
    st.session_state.load_locked     = False   # au moins un fichier sauvé -> [load] réactivable
    st.session_state.batch_play_audio = None
    st.success(f"{len(names_transfer)} fichier(s) sauvegardé(s) sur S3.")


def merge_session_csvs(s3_client, prefix: str, merged_key: str) -> int:
    """
    Fusionne a posteriori tous les CSV par session stockés sous `prefix` (un fichier
    par session_id, écrit par _save_tagged_rows) en un seul catalogue `merged_key`.
    Sans risque de conflit d'écriture concurrente : chaque session n'écrit que dans
    son propre fichier ; seule cette fusion réécrit le fichier consolidé, à exécuter
    ponctuellement (bouton d'admin ci-dessous, tâche planifiée, etc.).
    Retourne le nombre de fichiers de session fusionnés.
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"])
    if not keys:
        return 0

    dfs = []
    for key in keys:
        obj = s3_client.get_object(Bucket=BUCKET, Key=key)
        dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read())))
    df_merged = pd.concat(dfs, ignore_index=True)

    buf = io.StringIO()
    df_merged.to_csv(buf, index=False)
    s3_client.put_object(Bucket=BUCKET, Key=merged_key, Body=buf.getvalue())
    return len(keys)


def _render_admin_sidebar():
    """Fusion manuelle des catalogues CSV par session (features + spectro) en un fichier unique."""
    with st.sidebar.expander("🛠 Administration — fusion des catalogues"):
        st.caption("Fusionne tous les CSV par session en un seul fichier consolidé sur S3.")
        if st.button("🔀 Fusionner features_music_user.csv", key="btn_merge_features"):
            try:
                n = merge_session_csvs(boto3.client('s3'), FEATURES_USER_PREFIX, FEATURES_USER_MERGED_CSV)
                st.success(f"{n} fichier(s) de session fusionné(s) dans {FEATURES_USER_MERGED_CSV}.")
            except Exception as e:
                st.error(f"Erreur lors de la fusion : {e}")
        if st.button("🔀 Fusionner spectro_music_user.csv", key="btn_merge_spectro"):
            try:
                n = merge_session_csvs(boto3.client('s3'), SPECTRO_CSV_PREFIX, SPECTRO_USER_MERGED_CSV)
                st.success(f"{n} fichier(s) de session fusionné(s) dans {SPECTRO_USER_MERGED_CSV}.")
            except Exception as e:
                st.error(f"Erreur lors de la fusion : {e}")


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
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("# 🎵 MusicAI — Analyse & Classification de Genre Musical")
    st.markdown("🎧 Déposez un son. Découvrez son genre. Explorez ce qui lui ressemble.")
with head_col2:
    st.text_input(
        "Utilisateur",
        key="user_name",
        placeholder="please write your user name",
        help="please write your user name",
    )
st.divider()
_render_admin_sidebar()



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
    # — mode batch colonne 2 —
    "user_name": "",
    "session_id": None,
    "date_heure_load": None,
    "df_user_music_temp": pd.DataFrame(columns=["name", "genre_pred_feat", "genre_pred_CNN", "genre_user"]),
    "features_user_temp": pd.DataFrame(),
    "spectro_user": pd.DataFrame(columns=["name", "spectro_percu", "spectro_harmo"]),
    "batch_temp_dir": None,
    "load_locked": False,
    "show_uploader": False,
    "active_row_idx": 0,
    "batch_audio_cache": {},
    "batch_features_cache": {},
    "batch_display_spectro_cache": {},
    "batch_play_audio": None,
    "fig_batch_wave": None,
    "fig_batch_spect": None,
    "_active_played": False,
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


# RANGÉE HAUTE COL 2 — Batch load / up / down / play / save + liste_music
def _render_liste_music():
    df = st.session_state.df_user_music_temp.reset_index(drop=True)
    if df.empty:
        st.info("Aucun fichier chargé. Renseignez votre nom puis cliquez sur [Load].")
        return

    active_idx = st.session_state.active_row_idx

    def _highlight(row):
        if row.name == active_idx:
            return ['background-color:#3b2f6b'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1),
        height=int((min(len(df), M_DISPLAY_ROWS) + 1) * 35 + 3),
        width='stretch', hide_index=True,
    )

    if 0 <= active_idx < len(df):
        row = df.iloc[active_idx]
        played = st.session_state.get("_active_played", False)
        default_genre = row["genre_pred_CNN"] if row["genre_pred_CNN"] in LIST_GENRES else LIST_GENRES[0]

        c_select, c_confirm = st.columns([3, 1])
        with c_select:
            genre_choice = st.selectbox(
                f"Genre corrigé — {row['name']}",
                LIST_GENRES,
                index=LIST_GENRES.index(default_genre),
                key=f"genre_user_select_{active_idx}_{st.session_state.session_id}",
                disabled=not played,
            )
        with c_confirm:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("✔ Valider", key=f"btn_valider_genre_{active_idx}_{st.session_state.session_id}",
                          disabled=not played, width='stretch'):
                st.session_state.df_user_music_temp.loc[active_idx, "genre_user"] = genre_choice
                st.rerun()

        current_tag = st.session_state.df_user_music_temp.loc[active_idx, "genre_user"]
        if pd.notna(current_tag) and current_tag != "":
            st.caption(f"Genre validé pour cette ligne : **{current_tag}**")

def _ctrl_my():
    st.markdown('<div class="section-title">🎙 Vos musiques (batch)</div>', unsafe_allow_html=True)

    load_disabled = (
        not st.session_state.user_name.strip()
        or st.session_state.load_locked
        or _remaining_slots() <= 0
    )
    df = st.session_state.df_user_music_temp
    n_rows = len(df)
    can_up   = st.session_state.active_row_idx > 0
    next_idx = st.session_state.active_row_idx + 1
    can_down = next_idx < n_rows and pd.notna(df.iloc[next_idx]["genre_pred_CNN"])
    can_play = 0 <= st.session_state.active_row_idx < n_rows
    can_save = df["genre_user"].notna().any() and (df["genre_user"] != "").any()

    b_load, b_up, b_down, b_play, b_save = st.columns(5)
    with b_load:
        if st.button("📥", key="btn_load", disabled=load_disabled, width='stretch',
                      help="Load"):
            st.session_state.show_uploader = True
    with b_up:
        if st.button("⬆", key="btn_up", disabled=not can_up, width='stretch', help="Up"):
            st.session_state.active_row_idx -= 1
            st.session_state["_active_played"] = False
    with b_down:
        if st.button("⬇", key="btn_down", disabled=not can_down, width='stretch', help="Down"):
            st.session_state.active_row_idx += 1
            st.session_state["_active_played"] = False
    with b_play:
        if st.button("▶", key="btn_play_batch", disabled=not can_play, width='stretch', help="Play"):
            _play_active_row()
    with b_save:
        if st.button("💾", key="btn_save", disabled=not can_save, width='stretch', help="Save"):
            _save_tagged_rows()

    if st.session_state.show_uploader:
        k = _remaining_slots()
        files = st.file_uploader(
            f"Sélectionnez jusqu'à {k} fichier(s) audio (.wav / .mp3 / .ogg)",
            type=["wav", "mp3", "ogg"], accept_multiple_files=True, key="batch_uploader",
        )
        if files:
            _process_loaded_files(files[:k])
            st.session_state.show_uploader = False
            st.rerun()

    _render_liste_music()

    if st.session_state.batch_play_audio:
        st.audio(st.session_state.batch_play_audio, format="audio/wav")


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

