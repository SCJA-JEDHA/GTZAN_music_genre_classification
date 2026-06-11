"""
Application Streamlit - Analyse & Classification Musicale
Prédiction de genre musical via MLflow + recommandations PCA
"""
# to launch the streamlit in local : 
# streamlit run .\streamlit_music_app3.py --server.runOnSave true --logger.level=debug

import io
from regex import P
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
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.pyfunc
import boto3
import json
import pickle
from PIL import Image
import base64

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os

# st.write("CWD :", os.getcwd())
# Chemin absolu basé sur l'emplacement du script (indépendant du CWD)
#BASE_DIR     = Path(__file__).resolve().parent.parent.parent
# GENERAL_PATH = BASE_DIR / "gtzan-dataset-music-genre-classification" / "Data"
# GENRES_PATH  = GENERAL_PATH / "genres_original"
# PCA_PATH     = GENERAL_PATH / "PCA"

from urllib.parse import urljoin

# Env vars : 
from dotenv import load_dotenv
import os
import boto3

load_dotenv()  # charge les variables du fichier .env

### ENV   ### 

#MLFLOW_URI        = "https://cyrilbrg-mlflow-music.hf.space/"   # ← à jour
MLFLOW_URI = os.getenv("MLFLOW_URI")
#API_URL           = "http://localhost:8000/"  # ← for local use 
# API_URL = "https://cyrilbrg-api-music-model-f.hf.space/" # ← for network config
API_URL = os.getenv("API_URL")
API_MODEL_CNN_URL = os.getenv("API_MODEL_CNN_URL")
API_CALCUL_URL = os.getenv("API_CALCUL_URL")


#### PATHES  ### 

s3 = boto3.client('s3')

DATA_URL = "https://music-classification-project2.s3.eu-west-3.amazonaws.com/music-database/"  # to remove 

GENERAL_PATH = urljoin(DATA_URL, "gtzan-dataset-music-genre-classification/Data") # to remove
GENRES_PATH  = urljoin(GENERAL_PATH, "genres_original")

BUCKET = os.getenv("AWS_BUCKET")
#BUCKET = 'music-classification-project2'
MUSIC_DATABASE_PREFIX = "music-database/gtzan-dataset-music-genre-classification/Data/"
#GENRES_PREFIX = 'music-database/gtzan-dataset-music-genre-classification/Data/genres_original/'
GENRES_PREFIX = MUSIC_DATABASE_PREFIX + "genres_original/"

#PCA_DIR     = "PCA"
PCA_PREFIX = MUSIC_DATABASE_PREFIX + "PCA/"
PCA_PIPELINE = "pca_pipeline.pkl"  # pkl containing complete pca pipeline
PCA_X_PCA = "X_pca.csv"  # coordinates in PCA axes 

# START_PATH        = Path(r"C:/Users/Cyril/Documents/python/jedha/M11_projet/explo/music_genre_classification/GTZAN_music_genre_classification"
# GENERAL_PATH      = START_PATH / "Data"
# GENRES_PATH       = f"{GENERAL_PATH}/genres_original"
# PCA_PATH          = f"{GENERAL_PATH}/PCA"


TARGET_SR         = 22050
CLIP_DURATION     = 30       # secondes conservées après silence initial
N_FFT             = 2048
HOP               = 512
IMAGE_NX          = 432
IMAGE_NY          = 288 
LIST_GENRES = [
    'blues', 
    'classical', 
    'country', 
    'disco', 
    'hiphop', 
    'jazz', 
    'metal', 
    'pop', 
    'reggae', 
    'rock']

dict_genres =  {
  "blues": 0,
  "classical": 1,
  "country": 2,
  "disco": 3,
  "hiphop": 4,
  "jazz": 5,
  "metal": 6,
  "pop": 7,
  "reggae": 8,
  "rock": 9
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — AUDIO
# ─────────────────────────────────────────────────────────────────────────────

def url_exists(url: str) -> bool:
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

import boto3
from botocore.exceptions import ClientError

@st.cache_data(show_spinner=False) 
def s3_key_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise  # autre erreur, on la remonte


@st.cache_data(show_spinner=False)
def list_s3_subdirectories_sorted(bucket_name: str, prefix: str):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
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
    
    genres = sorted(subdirs)
    if not genres:
        st.warning(f"Répertoire vide : s3://{bucket_name}/{prefix}")
    return genres


@st.cache_data(show_spinner=False)
def list_genres() -> list[str]:
    """Retourne la liste des genres (sous-répertoires de genres_original)."""
    
    list_genres = list_s3_subdirectories_sorted(BUCKET, GENRES_PREFIX)
    return list_genres
    
    """def list_tracks(genre: str) -> list[str]:
        """"""Retourne la liste des fichiers .wav pour un genre donné.""""""
        p = Path(GENRES_PATH) / genre
        if not p.exists():
            return []
        return sorted([f.stem for f in p.glob("*.wav")])
    """

@st.cache_data(show_spinner=False)
def list_tracks(genre: str) -> list[str]:
    """Retourne la liste des fichiers .wav pour un genre donné dans S3."""
    s3 = boto3.client('s3')
    prefix = f"{GENRES_PREFIX}{genre}/"
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
    
    tracks = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.wav'):
                    # Extraire le nom du fichier sans extension
                    filename = key.split('/')[-1].rsplit('.', 1)[0]
                    tracks.append(filename)
    
    return sorted(tracks)

@st.cache_data(show_spinner=False)
def load_audio(path: str) -> tuple:
    """Charge un fichier audio → (y, sr) avec librosa."""
    y, sr = librosa.load(path, sr=None)
    return y, sr

@st.cache_data(show_spinner=False)
def load_audio_s3(s3_key: str) -> tuple:
    """
    Charge un fichier audio depuis S3 → (y, sr) avec librosa.
    
    Args:
        s3_key (str): Chemin complet de l'objet dans le bucket S3.
        
    Returns:
        tuple: (y, sr) signal audio et fréquence d'échantillonnage.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
    audio_bytes = obj['Body'].read()
    
    # Charger le fichier audio depuis un buffer mémoire
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=None)
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
    # # Normalisation RMS
    # rms = np.sqrt(np.mean(y_clip ** 2))
    # if rms > 0:
    #     y_clip = y_clip / rms
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
    'chroma_stft': chroma,
    'rms': rms,
    'spectral_centroid':spectral_centroid,
    'spectral_bandwidth':spectral_bandwidth,
    'rolloff':rolloff,
    'zero_crossing_rate':zero_crossing_rate,
    'harmony':harmony,
    'perceptr':perceptr,
    }
    # add features values and columns_names: 
    for name, data in feature_dict.items():
        features.extend([data.mean(), data.var()])
        column_names.extend([f'{name}_mean', f'{name}_var'])
    
    tempo_val = tempo.mean()
    features.append(tempo_val)
    column_names.append('tempo')
        
        
    # add features mfcc1 to mfcc_20 _mean and _var :
    for idx,x in enumerate(mfccs):
        features.extend([np.mean(x),np.var(x)])
        column_names.extend([f"mfcc{idx+1}_mean",f"mfcc{idx+1}_var"])

             
    # add label
    features.append('user')
    column_names.append('label')
    
    
    # columns needed : 
    # """Index(['filename', 'length', 
    # 'chroma_stft_mean', 'chroma_stft_var',
    # 'rms_mean','rms_var', 
    # 'spectral_centroid_mean', 'spectral_centroid_var',
    # 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
    # 'rolloff_mean','rolloff_var', 
    # 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    # 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 
    # 'tempo',
    #    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 
    #    'mfcc3_mean','mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 
    #    'mfcc5_mean', 'mfcc5_var','mfcc6_mean', 'mfcc6_var', 
    #    'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
    #    'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
    #    'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
    #    'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
    #    'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
    #    'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var',
    #    'label'],
    # """
    # ONLY 'filename', 'length' at begin, and label at end will be missing
    df_features = pd.DataFrame(columns=column_names)
    df_features.loc[0] = features
    return df_features


def compute_melspectrogram(y, sr) -> np.ndarray:
    """Calcule le mel-spectrogramme normalisé (IMAGE_NY x IMAGE_NX) pour la prédiction."""
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect.resize(IMAGE_NY, IMAGE_NX, refcheck=False)
    
    return spect


def audio_bytes(path: str) -> bytes:
    """Lit un fichier audio en bytes pour st.audio."""
    with open(path, "rb") as f:
        return f.read()

def audio_bytes_s3(s3_key: str) -> bytes:
    """
    Lit un fichier audio depuis S3 et retourne son contenu en bytes pour st.audio.
    
    Args:
        s3_key (str): Chemin complet de l'objet dans le bucket S3.
        
    Returns:
        bytes: Contenu du fichier audio.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
    audio_bytes = obj['Body'].read()
    return audio_bytes

def image_to_base64(img):
    i_bytes = io.BytesIO()
    img.save(i_bytes, format = "PNG")
    i_bytes.seek(0)
    return base64.b64encode(i_bytes.getvalue()).decode("utf-8")

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

def revert_pred(prediction):
    """ revert prediction from number to genre label"""    
    unique_labels = LIST_GENRES
    mapping_LI = {l : unique_labels.index(l) for l in unique_labels}
    # Creating reverse mapping
    reverse_LI = {v : k for v, k in enumerate(mapping_LI)}
    return reverse_LI[prediction]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PCA & RECOMMANDATIONS
# ─────────────────────────────────────────────────────────────────────────────
from typing import Tuple, Any

@st.cache_data(show_spinner=False)
def load_pca_df(pca_path: str) -> Tuple[pd.DataFrame, Any]:
    """Charge les donnees PCA : 
    - X_pca_df : DataFrame X_PCA depuis le disque (CSV attendu).
    - PCA_pipeline : model pipeline (transform + pca) depuis in .pkl """
    
    
    candidates = list(Path(pca_path).glob("*.csv"))
    if not candidates:
        X_pca_df = pd.DataFrame()
    else:
        X_pca_df = pd.read_csv(candidates[0]) # coordinates 
    
    candidates = list(Path(pca_path).glob("*.pkl"))
    if not candidates:
        print('no .pkl found in PCA directory')
        pca_pipeline = None
    else: 
        with open(candidates[0], 'rb') as f:
            pca_pipeline = pickle.load(f) # coordinates 
    return (X_pca_df,pca_pipeline)

@st.cache_data(show_spinner=False)
def load_pca_df_s3(PCA_PREFIX: str) -> Tuple[pd.DataFrame, Any]:
    """
    Charge les donnees PCA depuis un bucket S3 (CSV attendu).
    X_pca_df, pca_pipeline = load_pca_df_s3(pca_prefix) 
        
    Args:
        pca_prefix (str): Préfixe S3 où chercher les fichiers PCA (ex: 'path/to/pca/')
        
    Returns:
        X_pca_df : pd.DataFrame X_PCA depuis le disque (CSV attendu).
        PCA_pipeline : model pipeline (transform + pca) depuis in .pkl
        pd.DataFrame: DataFrame chargé depuis le premier fichier CSV trouvé, ou DataFrame vide si aucun fichier.
    """
    
    # PCA_PREFIX = MUSIC_DATABASE_PREFIX + "PCA/"
    # PCA_PIPELINE = "pca_pipeline.pkl"  # pkl containing complete pca pipeline
    # PCA_X_PCA = "X_pca.csv"
    
    s3 = boto3.client('s3')
    x_pca_key = PCA_PREFIX + PCA_X_PCA
    pca_pipeline_key = PCA_PREFIX + PCA_PIPELINE
    # Initialisation des résultats
    X_pca_df = None
    pca_pipeline = None
    
    # Lister les fichiers CSV sous le préfixe donné
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PCA_PREFIX)
    # Charger le CSV
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=x_pca_key)
        data = obj['Body'].read()
        X_pca_df = pd.read_csv(io.BytesIO(data))
        print(f"fichier {x_pca_key} chargé")
    except s3.exceptions.NoSuchKey:
        print(f"Fichier CSV '{x_pca_key}' non trouvé dans le bucket {BUCKET}.")
        X_pca_df= None
    except Exception as e:
        print(f"Erreur lors du chargement du CSV '{x_pca_key}': {e}")
        X_pca_df= None
            
    # Filtrer les fichiers .pkl
    
    try:
        # Charger le premier fichier pkl trouvé
        obj = s3.get_object(Bucket=BUCKET, Key=pca_pipeline_key)
        pca_pipeline = pickle.load(obj['Body'])
        print(f"file {pca_pipeline_key} loaded")
    except s3.exceptions.NoSuchKey:
        print(f"Fichier pickle '{pca_pipeline_key}' non trouvé dans le bucket {BUCKET}.")
    except Exception as e:
        print(f"Erreur lors du chargement du pickle '{pca_pipeline_key}': {e}")

    return X_pca_df, pca_pipeline

def get_recommendations(
    df_pca: pd.DataFrame,
    track_name: str,
    genre: str,
    n_neighbors: int = 4
) -> pd.DataFrame:
    """
    Retourne les n_neighbors voisins les plus proches dans l'espace PCA
    pour le genre donné.
    Note : ajouter la coordonnee pca de "ma musique" si necessaire : 
    """
    keywords = ["principal component" ,"princ_comp"]
    pc_cols = [c for c in df_pca.columns if any(keyword in c.lower() for keyword in keywords)]
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


def get_recommendations_my_music(
    df_pca: pd.DataFrame,
    pca_pipeline: Pipeline,
    my_features_clean: pd.DataFrame,
    genre: str,
    n_neighbors: int = 4
) -> pd.DataFrame:
    """
    Retourne les n_neighbors voisins les plus proches de my_music dans l'espace PCA
    pour le genre donné.
    specific verwion for my_music with pca_pipeline & coordonnee pca de "ma musique" si necessaire : 
    """
    
    
    keywords = ["principal component" ,"princ_comp"]
    pc_cols = [c for c in df_pca.columns if any(keyword in c.lower() for keyword in keywords)]
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

    X_my_music = pca_pipeline.transform(my_features_clean)
    X = genre_df[pc_cols].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    # src_idx = source.index[0]
    # loc_idx = genre_df.index.get_loc(src_idx)
    #distances, indices = nbrs.kneighbors(X[loc_idx:loc_idx+1])
    distances, indices = nbrs.kneighbors(X_my_music)
    neighbor_locs = indices[0][:]  # on prend tous les morceaux car my_music n'est pas dans la base !
    return genre_df.iloc[neighbor_locs]

def project_new_point(pca_pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """
    Projette un nouveau vecteur de features dans l'espace PCA existant.
    Retourne les coordonnées PCA (3 composantes).
    input:
        pca_pipeline : pipeline of PCA transform : 
            - transformer
            - pca 
        features : coord of all features used for pca transformation 
        cols to remove are already removed 
        
    """
    # for next update if columns are important
    # features = df_features.iloc[0].to_numpy()
    # column_names = df_features.columns
    # To check : same columns as PCA matrix !! 
    
    
    X_test_pca = pca_pipeline.transform(features)
    return X_test_pca[0]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — API PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────


def calcul_image_pour_CNN(y, sr) -> Tuple[dict[2],np.ndarray[Any]]:
    """
    Compute spectrogram image(s) from one music "file" 
    (payload_spectro,S_DB) = calcul_image_pour_CNN(y,sr)
    
    input
        y, sr from librosa load of user music file
    output 
        payload_spectro : dict of 2 librosa spectro images (harmonic & percusive) 
         for plotting in streamlit
        payload_img : dict of 2 images dedicated for API_predict_CNN
    """
    #y, _ = librosa.effects.trim(y)

    n_fft = N_FFT #2048        # Précision des détails du timbre
    hop_length = HOP #512    # Résolution temporelle
    n_mels = 128        # Hauteur de la fréquence

    # Séparation harmonique/percussive
    y_full = y
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # for visu :
    S_base = librosa.feature.melspectrogram(y=y_full, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels)
    
    S_h = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels)
    S_p = librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = n_mels)

    S_DB = librosa.power_to_db(S_base, ref=np.max) # nous avons remplacé amplitude_to_db
    
    S_DB_h = librosa.power_to_db(S_h, ref=np.max)
    S_DB_p = librosa.power_to_db(S_p, ref=np.max)

    S_DB = np.flipud(S_DB)
    
    S_DB_h = np.flipud(S_DB_h)
    S_DB_p = np.flipud(S_DB_p)

    norm_S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
    
    norm_S_DB_h = (S_DB_h - S_DB_h.min()) / (S_DB_h.max() - S_DB_h.min())
    norm_S_DB_p = (S_DB_p - S_DB_p.min()) / (S_DB_p.max() - S_DB_p.min())

    #img_f = Image.fromarray((norm_S_DB * 255).astype(np.uint8))
    img_h = Image.fromarray((norm_S_DB_h * 255).astype(np.uint8), mode = "L")
    img_p = Image.fromarray((norm_S_DB_p * 255).astype(np.uint8), mode = "L")

    #img_f = img_f.resize((512, 256), Image.Resampling.LANCZOS)
    img_h = img_h.resize((256, 128), Image.Resampling.LANCZOS)
    img_p = img_p.resize((256, 128), Image.Resampling.LANCZOS)

    b_harmo = image_to_base64(img_h)
    b_percu = image_to_base64(img_p)

    #spectro_at_predict = {"harmo_file" : ("harmo.png", h_bytes, "image/png"), "percu_file" : ("percu.png", p_bytes, "image/png")}
    payload_spectro = {"harmo_file" : b_harmo, "percu_file" : b_percu}

    #resp = requests.post(url = "https://dareindodo-api-dl-predict.hf.space/predict-cnn", json = spectro_at_predict)
    return (payload_spectro,S_DB)
    

  
def call_predict_api_CNN(payload_img: dict ) -> int:
    """ call api predict of CNN model 
        input :
            dict of 2 img converted in UTF8
            payload_img = {"img1":[ utf8_1, utf8_2, ... ],"img2":[] }
    """
    try:
        predict_url = API_MODEL_CNN_URL + 'predict-cnn'
        resp = requests.post(url = predict_url, json = payload_img)
        
        resp.raise_for_status()
        data = resp.json()
        # Adapter la clé de retour selon votre API (exemple ici : 'prediction')
        return data["prediction"]
        # return resp.json()
    except Exception as e:
        return f"Erreur API : {e}"
    

def call_predict_api(list_features_obj: list) -> str:
    """
    Appelle l'API de prédiction et retourne la prédiction.

    Args:
        model_name (str): Nom du modèle à utiliser (clé dans model_dict).
        num_features_obj (dict): Dictionnaire conforme à la classe NumFeatures.
        image_coords_list (list): Liste de listes, chaque sous-liste correspond à un vecteur coord de taille IMAGE_N.

    Returns:
        str: Résultat de la prédiction ou message d'erreur.
    """
    features_df = list_features_obj[0]    # is a dataframe
    num_features = features_df.iloc[0].to_dict()
    
    image_vector = list_features_obj[1]   # is format of image vector : [[],[],[]] perhaps
    image_vect_list = image_vector.tolist()
    coords_2 = [{"coord":[0,0]} ]
    # Sérialiser en JSON
    json_img_str = json.dumps(image_vect_list)
    
    payload_f = {
        "model_name": "model_dummy",
        "num_features": num_features, # [num_features_obj],  # liste d'un seul élément NumFeatures
    }
    # payload_i = {
    #     "model_name": model_name_selected,
    #     "image_coords": coords_2 # [{"coord": coords} for coords in image_coords_list]  # liste d'objets ImageCoord
    # }
    #st.write(payload_f)

    
    # #####
    
    #######
    
    try:
        predict_url = API_URL+'predict_f'
        resp = requests.post(predict_url, json=payload_f, timeout=65)
        
        resp.raise_for_status()
        data = resp.json()
        #  clé de retour API 
        return data["prediction"][0]  # correction : ce n'est plus une liste
    except Exception as e:
        return f"Erreur API : {e}"

import requests

def push_models(api_url: str, model_dict: dict) -> dict:
    """
    Envoie le dictionnaire model_dict à l'API via une requête POST sur /update-model.

    Args:
        api_url (str): URL de base de l'API (ex: "http://localhost:8000")
        model_dict (dict): Dictionnaire des modèles à envoyer

    Returns:
        dict: Réponse JSON de l'API
    """
    url = f"{api_url}/update-model"
    payload = {"data": model_dict}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

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
  .block-container { padding-top: 0.5rem; }
  
  /* Réduction espace titre principal */
  h1 {
      margin-top: 0.5rem !important;
      margin-bottom: 0.5rem !important;
  }
  
  /* Réduction espace sous texte d'intro */
  [data-testid="stMarkdownContainer"] > div > p {
      margin-bottom: 0.5rem !important;
  }
  /* Réduction espace entre 'modèle MLFlow' et 'visualisation' */
  .section-title {
      margin-bottom: 0.3rem !important;
  }
  /* Remplacez .visualisation-zone-class par la classe CSS réelle de la zone visualisation */
  .visualisation-zone-class {
      margin-top: 0.3rem !important;
  }

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
    "db_audio_bytes":       None,
    "my_audio_bytes":       None,
    "rec_audio_bytes":      None,
    "my_payload_spectro":   None,
    "predicted_genre":   "—",
    "predicted_genre_feat":   "—",
    "predicted_genre_CNN":   "—",
    "my_y":                 None,
    "my_sr":                None,
    "my_user_S_DB"      :   None,
    "db_y":                 None,
    "db_sr":                None,
    "rec_y":                None,
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
col1, col2, col3 = st.columns([1, 1, 1], gap="small")

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
        #db_path = f"{GENRES_PATH}/{genre_sel}/{track_sel}.wav"
        s3_key = f"{GENRES_PREFIX}{genre_sel}/{track_sel}.wav"

    exist_key = s3_key_exists(BUCKET, s3_key)
    
    if st.button("▶ Play — Base", width='stretch', key="btn_play_db"):
        if s3_key and  exist_key:
            st.session_state.db_audio_bytes = audio_bytes_s3(s3_key)
            y_db, sr_db = load_audio_s3(s3_key)
            #y_db, sr_db = load_audio(db_path)
            st.session_state.db_y  = trim_audio(y_db, sr_db)
            st.session_state.db_sr = sr_db
        else:
            print(s3_key)
            print(f"key exist?: {exist_key}")
            st.error(f"Fichier introuvable. key ?: {s3_key}")

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
        
    # ── Chargement fichier utilisateur ───────────────────────────────────────
    st.markdown('<div class="section-title">🎙 Votre musique</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Sélectionner un fichier audio (.wav / .mp3 / .ogg)",
        type=["wav", "mp3", "ogg"],
        key="uploader"
    )

    if uploaded is not None:
        raw_bytes = uploaded.read()
        if st.button("⬆ Charger & Prétraiter", width='stretch', key="btn_load"):
            with st.spinner("Prétraitement en cours…"):
                y_raw, sr_raw = librosa.load(io.BytesIO(raw_bytes), sr=None)
                y_proc, sr_proc = preprocess_signal(y_raw, sr_raw)
                st.session_state.my_y         = y_proc
                st.session_state.my_sr        = sr_proc
                st.session_state.my_audio_bytes = raw_bytes
                st.session_state.my_features  = compute_features(y_proc, sr_proc)
                (st.session_state.my_payload_spectro,st.session_state.my_user_S_DB) = (
                    calcul_image_pour_CNN(y_proc, sr_proc)
                )
                 
            st.success("Signal prétraité.")

    if st.session_state.my_audio_bytes:
        if st.button("▶ Play — Ma musique", width='stretch', key="btn_play_my"):
            pass  # audio déjà en session
        st.audio(st.session_state.my_audio_bytes)
    
    
    
    # ── Prédiction ────────────────────────────────────────────────────────────
    st.markdown("---")

    if st.button("🔍 Prédire le genre", width='stretch', key="btn_predict",
                 disabled=(st.session_state.my_y is None)):
        #push models in API 
        # st.session_state.push_result = push_models(API_URL, MODEL_DICT)
        # if "error" in st.session_state.push_result:
        #     st.error(f"Erreur lors de la mise à jour : {st.session_state.push_result['error']}")
        # else:
        #     st.success("Dictionnaire mis à jour avec succès")
        #     st.json(st.session_state.push_result.get("stored_data", {}))
        # predict
        with st.spinner("Appel API prédiction…"):
            y_p, sr_p = st.session_state.my_y, st.session_state.my_sr
            feats = st.session_state.my_features  # is a dataframe
            spect = compute_melspectrogram(y_p, sr_p)  # to remove
            list_features = [feats,spect]
            
            # prediction from features model 
            pred1  = call_predict_api(list_features)
            st.session_state.predicted_genre_feat = revert_pred(pred1)
            
            # prediction from CNN model :
            payload_spectro = st.session_state.my_payload_spectro
            pred_CNN = call_predict_api_CNN(payload_spectro)
            st.session_state.predicted_genre_CNN = revert_pred(pred_CNN)
            st.session_state.predicted_genre = st.session_state.predicted_genre_CNN
    st.markdown('<div class="section-title">🏷 Genre prédit</div>', unsafe_allow_html=True)
    
    st.markdown(
        f'<span class="predicted-badge">{st.session_state.predicted_genre_feat}</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<span class="predicted-badge">{st.session_state.predicted_genre_CNN}</span>',
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

    #df_pca = load_pca_df_s3(PCA_PREFIX)
    X_pca_df, pca_pipeline = load_pca_df_s3(PCA_PREFIX)
    
    # Projection de ma musique dans l'espace PCA
    my_pca_coords = None
    if st.session_state.my_features is not None:
        cols_to_drop = ["filename", "length","label"]
        my_features_clean = (
            st.session_state.my_features.drop(
                columns=[col for col in cols_to_drop if col in st.session_state.my_features.columns])
        )
        print(f"my_features_clean :{my_features_clean}")
        X_test_pca = pca_pipeline.transform(my_features_clean)
        print(f"X__test_pca: {X_test_pca[0]}")

        my_pca_coords = project_new_point(pca_pipeline, my_features_clean)
        print(f"PCA_coords: {my_pca_coords}")
        st.session_state.my_coords_pca = my_pca_coords
    df_pca = X_pca_df
    print(df_pca.head())
    print(pca_pipeline)
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
        if source_choice == "database choice":
            rec_df = get_recommendations(df_pca, src_track, src_genre)
        else:
            rec_df = get_recommendations_my_music(df_pca,pca_pipeline,my_features_clean,src_genre) 
        if name_col and not rec_df.empty:
            rec_options = rec_df[name_col].tolist()

    rec_sel = st.selectbox(
        "Recommandation",
        rec_options if rec_options else ["(aucune recommandation disponible)"],
        key="rec_track"
    )

    # Bouton Play recommandation
    if st.button("▶ Play — Recommandation", width='stretch', key="btn_play_rec"):
        if rec_sel and rec_sel != "(aucune recommandation disponible)" and src_genre:
            #rec_path = f"{GENRES_PATH}/{src_genre}/{rec_sel}"
            s3_key = f"{GENRES_PREFIX}{src_genre}/{rec_sel}"
            print(f"reco path: {s3_key}")
            exist_key = s3_key_exists(BUCKET, s3_key)
            print("exist_key: {exist_key}")
            if s3_key and exist_key:
                st.session_state.rec_audio_bytes = audio_bytes_s3(s3_key)
                y_r, sr_r = load_audio_s3(s3_key)
                st.session_state.rec_y  = trim_audio(y_r, sr_r)
                st.session_state.rec_sr = sr_r
            else:
                print(s3_key)
                print(f"key exist?: {exist_key}")
                st.error(f"Fichier introuvable. key ?: {s3_key}")

    if st.session_state.rec_audio_bytes:
        st.audio(st.session_state.rec_audio_bytes, format="audio/wav")

    st.markdown("---")

    # ── Onglets visualisation ─────────────────────────────────────────────────
    tab_plot, tab_pca = st.tabs(["📊 Plot", "🔵 Composantes principales"])

    # plot : 
    with tab_plot:
        if st.session_state.rec_y is not None:
            y_r, sr_r = st.session_state.rec_y, st.session_state.rec_sr
            st.pyplot(fig_waveform(y_r, sr_r, f"Forme d'onde — {rec_sel}"))
            st.pyplot(fig_spectrogram(y_r, sr_r, n_fft, hop_length, f"Spectrogramme — {rec_sel}"))
        else:
            st.info("Lancez une recommandation pour visualiser le signal.")

    # composantes principales:
    with tab_pca:
        if df_pca.empty:
            st.warning(f"Aucun fichier PCA trouvé dans `{PCA_PREFIX}`.")
        else:
            keywords = ["principal component", "princ_comp"]
            pc_cols = [c for c in df_pca.columns if any (keyword in c.lower() for keyword in keywords) ]

            
            color_map = {
                "base":         "#4a4a6a",
                "sélection DB": "#de2626",
                "ma musique":   "#033688",
            }
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

                

                hover_data = {}
                if name_col:
                    hover_data[name_col] = True
                if "label" in df_plot.columns:
                    hover_data["label"] = True

                # plot database points:
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
                # Ajoute le point "database selection"
                idx_sel = df_pca.index[df_pca['filename']==f"{track_sel}.wav"][0]
                print(f"idx_sel: {idx_sel}")
                if my_pca_coords is not None and len(my_pca_coords) >= 3:
                    fig_pca.add_trace(go.Scatter3d(
                        x=[df_pca[pc1][idx_sel]],
                        y=[df_pca[pc2][idx_sel]],
                        z=[df_pca[pc3][idx_sel]],
                        mode="markers+text",
                        marker=dict(size=10, color="#a01a10", symbol="diamond"),
                        text=["Database sel."],
                        textposition="top center",
                        name="Database sel.",
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
                st.plotly_chart(fig_pca, width='stretch',
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
