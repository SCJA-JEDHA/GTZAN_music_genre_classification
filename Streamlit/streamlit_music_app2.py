"""
Application Streamlit - Analyse & Classification Musicale
Prédiction de genre musical via MLflow + recommandations PCA
"""

import os
import io
import time
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

# st.write("CWD :", os.getcwd())
# Chemin absolu basé sur l'emplacement du script (indépendant du CWD)
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
GENERAL_PATH = BASE_DIR / "gtzan-dataset-music-genre-classification" / "Data"
GENRES_PATH  = GENERAL_PATH / "genres_original"
PCA_PATH     = GENERAL_PATH / "PCA"

# START_PATH        = Path(r"C:/Users/Cyril/Documents/python/jedha/M11_projet/explo/music_genre_classification/GTZAN_music_genre_classification"
# GENERAL_PATH      = START_PATH / "Data"
# GENRES_PATH       = f"{GENERAL_PATH}/genres_original"
# PCA_PATH          = f"{GENERAL_PATH}/PCA"
MLFLOW_URI        = "http://localhost:5000"   # ← à ajuster
API_URL           = "http://localhost:8000/api_music_predict"  # ← à ajuster
TARGET_SR         = 22050
CLIP_DURATION     = 30       # secondes conservées après silence initial
N_FFT_DEFAULT     = 2048
HOP_DEFAULT       = 512
IMAGE_NX          = 432
IMAGE_NY          = 288 

# st.write("GENRES_PATH existe :", Path(GENRES_PATH), Path(GENRES_PATH).exists())
# st.write("GENRES_PATH absolu :", Path(GENRES_PATH).resolve())

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — AUDIO
# ─────────────────────────────────────────────────────────────────────────────

def list_genres() -> list[str]:
    """Retourne la liste des genres (sous-répertoires de genres_original)."""
    p = Path(GENRES_PATH)
    if not p.exists():
        st.error(f"Chemin introuvable : {p.resolve()}")
        return []
    
    genres = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not genres:
        st.warning(f"Répertoire vide : {p.resolve()}")
    return genres

def list_tracks(genre: str) -> list[str]:
    """Retourne la liste des fichiers .wav pour un genre donné."""
    p = Path(GENRES_PATH) / genre
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.wav")])


def load_audio(path: str) -> tuple:
    """Charge un fichier audio → (y, sr) avec librosa."""
    y, sr = librosa.load(path, sr=None)
    return y, sr


def trim_audio(y, sr) -> np.ndarray:
    """Supprime les silences au début/fin."""
    audio_trimmed, _ = librosa.effects.trim(y)
    return audio_trimmed


def preprocess_signal(y, sr) -> tuple:
    """
    Prétraitement pour la prédiction :
      1. Supprime le silence initial
      2. Garde les 3 premières secondes
      3. Rééchantillonne à TARGET_SR
      4. Normalise en amplitude RMS
    """
    y_trimmed, _ = librosa.effects.trim(y)
    n_samples = int(CLIP_DURATION * TARGET_SR)
    y_resampled = librosa.resample(y_trimmed, orig_sr=sr, target_sr=TARGET_SR)
    y_clip = y_resampled[:n_samples] if len(y_resampled) >= n_samples else np.pad(
        y_resampled, (0, n_samples - len(y_resampled))
    )
    # Normalisation RMS
    rms = np.sqrt(np.mean(y_clip ** 2))
    if rms > 0:
        y_clip = y_clip / rms
    return y_clip, TARGET_SR


def compute_features(y, sr) -> np.ndarray:
    """Calcule un vecteur de features (MFCCs, chroma, spectral centroid, etc.).
       TO update"""
    features = []
    column_names = []
    features.append('user.file')
    column_names.append('filename')
    
    length = len(y)
    features.append(length)
    column_names.append('length')
    
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms   = librosa.feature.rms(y=y)
    spectral_centroid= librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate= librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
    
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # for x in mfccs:
        
    #     features.append(np.mean(x))
    
    feature_dict = {
    'chroma': chroma,
    'rms': rms,
    'spectral_centroid':spectral_centroid,
    'spectral_bandwidth':spectral_bandwidth,
    'rolloff':rolloff,
    'zero_crossing_rate':zero_crossing_rate,
    'harmony':harmony,
    'perceptr':perceptr,
    'tempo':tempo,
    }
    # add features values and columns_names: 
    for name, data in feature_dict.items():
        features.extend([data.mean(), data.var()])
        column_names.extend([f'{name}_mean', f'{name}_var'])
        
    # add features mfcc1 to mfcc_20 _mean and _var :
    for idx,x in enumerate(mfccs):
        features.extend([np.mean(x),np.var(x)])
        column_names.extend([f"mfcc{idx+1}_mean",f"mfcc{idx+1}_var"])

             
    # add label
    features.append('user')
    column_names.append('label')
    
    
    # columns needed : 
    """Index(['filename', 'length', 
    
    'chroma_stft_mean', 'chroma_stft_var',
    'rms_mean','rms_var', 
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
    'rolloff_mean','rolloff_var', 
    'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 
    'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 
       'mfcc3_mean','mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 
       'mfcc5_mean', 'mfcc5_var','mfcc6_mean', 'mfcc6_var', 
       'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var',
       'label'],
    """
    # ONLY 'filename', 'length' at begin, and label at end will be missing
    df_features = pd.DataFrame(columns=column_names)
    df_features.loc[0] = features
    return features


def compute_melspectrogram(y, sr) -> np.ndarray:
    """Calcule le mel-spectrogramme normalisé (128×660) pour la prédiction."""
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect.resize(128, 660, refcheck=False)
    return spect


def audio_bytes(path: str) -> bytes:
    """Lit un fichier audio en bytes pour st.audio."""
    with open(path, "rb") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def fig_waveform(y, sr, title: str) -> plt.Figure:
    """Génère la figure de la forme d'onde."""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y=y, sr=sr, color="royalblue", ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def fig_spectrogram(y, sr, n_fft: int, hop_length: int, title: str) -> plt.Figure:
    """Génère la figure du spectrogramme (log-scale, colormap inferno)."""
    D  = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        DB, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="log",
        cmap="inferno", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_mlflow_models(tracking_uri: str) -> list[str]:
    """Récupère la liste des modèles enregistrés dans MLflow."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        models = [m.name for m in client.search_registered_models()]
        return models if models else ["(aucun modèle trouvé)"]
    except Exception as e:
        return [f"Erreur MLflow : {e}"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PCA & RECOMMANDATIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_pca_df(pca_path: str) -> pd.DataFrame:
    """Charge le DataFrame PCA depuis le disque (CSV attendu)."""
    candidates = list(Path(pca_path).glob("*.csv"))
    if not candidates:
        return pd.DataFrame()
    df = pd.read_csv(candidates[0])
    return df


def get_recommendations(
    df_pca: pd.DataFrame,
    track_name: str,
    genre: str,
    n_neighbors: int = 4
) -> pd.DataFrame:
    """
    Retourne les n_neighbors voisins les plus proches dans l'espace PCA
    pour le genre donné.
    """
    pc_cols = [c for c in df_pca.columns if "principal component" in c.lower()]
    if not pc_cols or "label" not in df_pca.columns:
        return pd.DataFrame()

    genre_df = df_pca[df_pca["label"] == genre].copy()
    if genre_df.empty or len(genre_df) <= n_neighbors:
        return genre_df

    # Recherche du morceau source
    name_col = [c for c in df_pca.columns if c.lower() in ("filename", "name", "track", "file")]
    if not name_col:
        return genre_df.head(n_neighbors)
    name_col = name_col[0]

    source = genre_df[genre_df[name_col].str.contains(track_name, na=False)]
    if source.empty:
        return genre_df.head(n_neighbors)

    X = genre_df[pc_cols].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
    src_idx = source.index[0]
    loc_idx = genre_df.index.get_loc(src_idx)
    distances, indices = nbrs.kneighbors(X[loc_idx:loc_idx+1])
    neighbor_locs = indices[0][1:]  # exclure le morceau lui-même
    return genre_df.iloc[neighbor_locs]


def project_new_point(df_pca: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    """
    Projette un nouveau vecteur de features dans l'espace PCA existant.
    Retourne les coordonnées PCA (3 composantes).
    """
    # for next update if columns are important
    # features = df_features.iloc[0].to_numpy()
    # column_names = df_features.columns
    # To check : same columns as PCA matrix !! 
    pc_cols = [c for c in df_pca.columns if "principal component" in c.lower()]
    if not pc_cols:
        return np.zeros(3)
    # On ré-entraîne un PCA sur les données existantes — approximation acceptable
    # pour la visualisation uniquement.
    feat_cols = [c for c in df_pca.columns if c not in pc_cols + ["label"] + 
                 [c for c in df_pca.columns if c.lower() in ("filename","name","track","file")]]
    if not feat_cols:
        return np.zeros(3)
    X_all = df_pca[feat_cols].values
    pca   = PCA(n_components=min(3, X_all.shape[1]))
    pca.fit(X_all)
    coords = pca.transform(features.reshape(1, -1))
    return coords[0]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — API PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────

def call_predict_api(model_name: str, features: np.ndarray, spect: np.ndarray) -> str:
    """Appelle l'API de prédiction et retourne le genre prédit."""
    payload = {
        "model_name": model_name,
        "features":   features.tolist(),
        "spectrogram": spect.tolist(),
    }
    try:
        resp = requests.get(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("predicted_genre", "—")
    except Exception as e:
        return f"Erreur API : {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MusicAI — Genre Classification",
    page_icon="🎵",
    layout="wide",
)

# ── Style ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  code, .stCode { font-family: 'DM Mono', monospace; }
  .block-container { padding-top: 1.5rem; }
  .section-box {
      background: #0f0f14;
      border: 1px solid #2a2a3a;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-bottom: 1rem;
  }
  .section-title {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #7c6af7;
      margin-bottom: 0.5rem;
  }
  .predicted-badge {
      font-size: 1.6rem;
      font-weight: 800;
      color: #a78bfa;
      background: #1a1730;
      border: 2px solid #7c6af7;
      border-radius: 8px;
      padding: 0.4rem 1.2rem;
      display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── En-tête ──────────────────────────────────────────────────────────────────
st.markdown("# 🎵 MusicAI — Analyse & Classification de Genre Musical")
st.markdown("""
Cette application permet d'**analyser**, **classifier** et **explorer** des morceaux audio.  
Sélectionnez un morceau de la base (colonne gauche), chargez votre propre musique (colonne centrale),  
et découvrez des recommandations contextualisées dans l'espace PCA (colonne droite).
""")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "db_audio_bytes":    None,
    "my_audio_bytes":    None,
    "rec_audio_bytes":   None,
    "predicted_genre":   "—",
    "my_y":              None,
    "my_sr":             None,
    "db_y":              None,
    "db_sr":             None,
    "rec_y":             None,
    "rec_sr":            None,
    "my_features":       None,
    "my_coords_pca":     None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Paramètres partagés (spectrogramme) — sidebar ────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres spectrogramme")
    n_fft = st.selectbox(
        "n_fft", options=[512, 1024, 2048, 4096, 8192], index=2,
        help="Taille de la FFT"
    )
    hop_length = st.selectbox(
        "hop_length", options=[128, 256, 512, 1024, 2048, 4096], index=2,
        help="Pas entre trames"
    )
    st.markdown("---")
    st.markdown("### 🔗 MLflow")
    mlflow_uri = st.text_input("Adresse MLflow", value=MLFLOW_URI)

# ─────────────────────────────────────────────────────────────────────────────
# COLONNES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

# ══════════════════════════════════════════════════════════════════════════════
# COLONNE 1 — Base de données
# ══════════════════════════════════════════════════════════════════════════════
with col1:
    st.markdown('<div class="section-title">📂 Sélection — Base de données</div>', unsafe_allow_html=True)

    genres = list_genres()
    if not genres:
        st.warning(f"Aucun genre trouvé dans `{GENRES_PATH}`")
        genre_sel = None
    else:
        genre_sel = st.selectbox("Genre", genres, key="db_genre")

    tracks = list_tracks(genre_sel) if genre_sel else []
    track_sel = st.selectbox("Morceau", tracks, key="db_track") if tracks else None

    db_path = None
    if genre_sel and track_sel:
        db_path = f"{GENRES_PATH}/{genre_sel}/{track_sel}.wav"

    if st.button("▶ Play — Base", use_container_width=True, key="btn_play_db"):
        if db_path and Path(db_path).exists():
            st.session_state.db_audio_bytes = audio_bytes(db_path)
            y_db, sr_db = load_audio(db_path)
            st.session_state.db_y  = trim_audio(y_db, sr_db)
            st.session_state.db_sr = sr_db
        else:
            st.error("Fichier introuvable.")

    if st.session_state.db_audio_bytes:
        st.audio(st.session_state.db_audio_bytes, format="audio/wav")

    st.markdown("---")
    st.markdown('<div class="section-title">📈 Visualisation — Base</div>', unsafe_allow_html=True)

    if st.session_state.db_y is not None:
        y_v, sr_v = st.session_state.db_y, st.session_state.db_sr
        st.pyplot(fig_waveform(y_v, sr_v, f"Forme d'onde — {track_sel}"))
        st.pyplot(fig_spectrogram(y_v, sr_v, n_fft, hop_length, f"Spectrogramme — {track_sel}"))
    else:
        st.info("Chargez un morceau via le bouton Play.")


# ══════════════════════════════════════════════════════════════════════════════
# COLONNE 2 — Prédiction
# ══════════════════════════════════════════════════════════════════════════════
with col2:
    # ── Sélection modèle MLflow ───────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 Modèle MLflow</div>', unsafe_allow_html=True)
    model_names = list_mlflow_models(mlflow_uri)
    model_sel   = st.selectbox("Modèle", model_names, key="mlflow_model")

    st.markdown("---")

    # ── Chargement fichier utilisateur ───────────────────────────────────────
    st.markdown('<div class="section-title">🎙 Votre musique</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Sélectionner un fichier audio (.wav / .mp3 / .ogg)",
        type=["wav", "mp3", "ogg"],
        key="uploader"
    )

    if uploaded is not None:
        raw_bytes = uploaded.read()
        if st.button("⬆ Charger & Prétraiter", use_container_width=True, key="btn_load"):
            with st.spinner("Prétraitement en cours…"):
                y_raw, sr_raw = librosa.load(io.BytesIO(raw_bytes), sr=None)
                y_proc, sr_proc = preprocess_signal(y_raw, sr_raw)
                st.session_state.my_y         = y_proc
                st.session_state.my_sr        = sr_proc
                st.session_state.my_audio_bytes = raw_bytes
                st.session_state.my_features  = compute_features(y_proc, sr_proc)
            st.success("Signal prétraité.")

    if st.session_state.my_audio_bytes:
        if st.button("▶ Play — Ma musique", use_container_width=True, key="btn_play_my"):
            pass  # audio déjà en session
        st.audio(st.session_state.my_audio_bytes)

    # ── Prédiction ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Prédire le genre", use_container_width=True, key="btn_predict",
                 disabled=(st.session_state.my_y is None)):
        with st.spinner("Appel API prédiction…"):
            y_p, sr_p = st.session_state.my_y, st.session_state.my_sr
            feats = st.session_state.my_features  # is a dataframe
            spect = compute_melspectrogram(y_p, sr_p)
            pred  = call_predict_api(model_sel, feats, spect)
            st.session_state.predicted_genre = pred

    st.markdown('<div class="section-title">🏷 Genre prédit</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="predicted-badge">{st.session_state.predicted_genre}</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    # ── Visualisation signal utilisateur ─────────────────────────────────────
    st.markdown('<div class="section-title">📈 Visualisation — Ma musique</div>', unsafe_allow_html=True)
    if st.session_state.my_y is not None:
        y_m, sr_m = st.session_state.my_y, st.session_state.my_sr
        st.pyplot(fig_waveform(y_m, sr_m, "Forme d'onde — Ma musique"))
        st.pyplot(fig_spectrogram(y_m, sr_m, n_fft, hop_length, "Spectrogramme — Ma musique"))
    else:
        st.info("Chargez un fichier audio pour voir les visualisations.")


# ══════════════════════════════════════════════════════════════════════════════
# COLONNE 3 — Recommandations & PCA
# ══════════════════════════════════════════════════════════════════════════════
with col3:
    st.markdown('<div class="section-title">🔀 Recommandations</div>', unsafe_allow_html=True)

    source_choice = st.radio(
        "Source", ["database choice", "my music"],
        horizontal=True, key="rec_source"
    )

    df_pca = load_pca_df(PCA_PATH)
    name_col_candidates = [c for c in df_pca.columns
                           if c.lower() in ("filename", "name", "track", "file")] if not df_pca.empty else []
    name_col = name_col_candidates[0] if name_col_candidates else None

    # Détermine le morceau/genre source pour les recommandations
    if source_choice == "database choice":
        src_track = track_sel or ""
        src_genre = genre_sel or ""
        st.markdown(f"*Similaire à :* **{src_track}** ({src_genre})")
    else:
        src_track = "ma_musique"
        src_genre = st.session_state.predicted_genre if st.session_state.predicted_genre != "—" else ""
        st.markdown(f"*Similaire à :* **ma musique** (genre prédit : {src_genre})")

    # Liste déroulante des recommandations
    rec_options = []
    if not df_pca.empty and src_genre:
        rec_df = get_recommendations(df_pca, src_track, src_genre)
        if name_col and not rec_df.empty:
            rec_options = rec_df[name_col].tolist()

    rec_sel = st.selectbox(
        "Recommandation",
        rec_options if rec_options else ["(aucune recommandation disponible)"],
        key="rec_track"
    )

    # Bouton Play recommandation
    if st.button("▶ Play — Recommandation", use_container_width=True, key="btn_play_rec"):
        if rec_sel and rec_sel != "(aucune recommandation disponible)" and src_genre:
            rec_path = f"{GENRES_PATH}/{src_genre}/{rec_sel}.wav"
            if Path(rec_path).exists():
                st.session_state.rec_audio_bytes = audio_bytes(rec_path)
                y_r, sr_r = load_audio(rec_path)
                st.session_state.rec_y  = trim_audio(y_r, sr_r)
                st.session_state.rec_sr = sr_r
            else:
                st.error("Fichier recommandation introuvable.")

    if st.session_state.rec_audio_bytes:
        st.audio(st.session_state.rec_audio_bytes, format="audio/wav")

    st.markdown("---")

    # ── Onglets visualisation ─────────────────────────────────────────────────
    tab_plot, tab_pca = st.tabs(["📊 Plot", "🔵 Composantes principales"])

    with tab_plot:
        if st.session_state.rec_y is not None:
            y_r, sr_r = st.session_state.rec_y, st.session_state.rec_sr
            st.pyplot(fig_waveform(y_r, sr_r, f"Forme d'onde — {rec_sel}"))
            st.pyplot(fig_spectrogram(y_r, sr_r, n_fft, hop_length, f"Spectrogramme — {rec_sel}"))
        else:
            st.info("Lancez une recommandation pour visualiser le signal.")

    with tab_pca:
        if df_pca.empty:
            st.warning(f"Aucun fichier PCA trouvé dans `{PCA_PATH}`.")
        else:
            pc_cols = [c for c in df_pca.columns if "principal component" in c.lower()]

            if len(pc_cols) < 3:
                pc1, pc2 = pc_cols[0], pc_cols[1]
                hover_data = {}
                if name_col:
                    hover_data[name_col] = True
                df_plot = df_pca.copy()
                if "label" in df_plot.columns:
                    hover_data["label"] = True

                fig_pca = px.scatter(
                    df_plot,
                    x=pc1, y=pc2, 
                    color="label", #"_highlight",
                    color_discrete_map=color_map,
                    opacity=0.8,
                    hover_data=hover_data,
                    title="Espace PCA — 3 composantes",
                )
                st.warning("Le DataFrame PCA doit contenir au moins 3 composantes.")
            else:
                pc1, pc2, pc3 = pc_cols[0], pc_cols[1], pc_cols[2]

                # Couleur : highlight sélection DB (rouge) et ma musique (bleu)
                df_plot = df_pca.copy()
                df_plot["_highlight"] = "base"

                if name_col and src_track in df_plot[name_col].values:
                    df_plot.loc[df_plot[name_col] == src_track, "_highlight"] = "sélection DB"

                # Projection de ma musique dans l'espace PCA
                my_pca_coords = None
                if st.session_state.my_features is not None:
                    my_pca_coords = project_new_point(df_pca, st.session_state.my_features)
                    st.session_state.my_coords_pca = my_pca_coords

                color_map = {
                    "base":         "#4a4a6a",
                    "sélection DB": "#ef4444",
                    "ma musique":   "#3b82f6",
                }

                hover_data = {}
                if name_col:
                    hover_data[name_col] = True
                if "label" in df_plot.columns:
                    hover_data["label"] = True

                fig_pca = px.scatter_3d(
                    df_plot,
                    x=pc1, y=pc2, z=pc3,
                    color="label", #"_highlight",
                    color_discrete_map=color_map,
                    opacity=0.8,
                    hover_data=hover_data,
                    title="Espace PCA — 3 composantes",
                )
                fig_pca.update_traces(marker=dict(size=4))

                # Ajoute le point "ma musique"
                if my_pca_coords is not None and len(my_pca_coords) >= 3:
                    fig_pca.add_trace(go.Scatter3d(
                        x=[my_pca_coords[0]],
                        y=[my_pca_coords[1]],
                        z=[my_pca_coords[2]],
                        mode="markers+text",
                        marker=dict(size=10, color="#3b82f6", symbol="diamond"),
                        text=["Ma musique"],
                        textposition="top center",
                        name="ma musique",
                    ))

                fig_pca.update_layout(
                    legend_title_text="Catégorie",
                    scene=dict(
                        xaxis_title=pc1,
                        yaxis_title=pc2,
                        zaxis_title=pc3,
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    height=500,
                )

                # Gestion du clic → remplacer la sélection DB
                st.plotly_chart(fig_pca, use_container_width=True,
                                key="pca_chart")

                # Sélection manuelle via liste (workaround Streamlit/Plotly click)
                if name_col:
                    all_tracks = df_pca[name_col].dropna().tolist()
                    clicked_track = st.selectbox(
                        "Sélectionner un morceau sur le graphique PCA",
                        ["(aucun)"] + all_tracks,
                        key="pca_click_track"
                    )
                    if clicked_track and clicked_track != "(aucun)":
                        # Retrouve le genre et met à jour la sélection colonne 1
                        if "label" in df_pca.columns:
                            row = df_pca[df_pca[name_col] == clicked_track]
                            if not row.empty:
                                new_genre = row["label"].values[0]
                                st.info(
                                    f"Sélection mise à jour : **{clicked_track}** ({new_genre}). "
                                    "Revenez en colonne gauche pour charger ce morceau."
                                )
                                # Mise à jour des selectbox via query params (workaround)
                                st.query_params["db_genre"] = new_genre
                                st.query_params["db_track"] = clicked_track
