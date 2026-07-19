"""
audio_pipeline.py
------------------
Fonctions PURES de traitement audio (chargement/preprocess/features/spectrogrammes),
extraites de streamlit_music_app4.py pour être importables/testables sans déclencher
le rendu Streamlit ni d'appels réseau (S3/MLflow) au moment de l'import.

streamlit_music_app4.py importe désormais ces fonctions depuis ce module plutôt que
de les redéfinir — comportement strictement identique, juste déplacé.
"""
from __future__ import annotations

import base64
import io

import librosa
import numpy as np
import pandas as pd
from PIL import Image

# --- Constantes partagées avec l'app Streamlit --------------------------------
TARGET_SR      = 22050
CLIP_DURATION  = 30
N_FFT          = 2048
HOP            = 512
IMAGE_NX       = 432
IMAGE_NY       = 288
TIME_THRESHOLD = 15   # début du segment analysé (s)
TIME_SEGMENT   = 30   # durée du segment analysé (s)

MIN_AUDIO_DURATION_S = 30.0  # durée minimale acceptée pour l'analyse


# --- Exceptions dédiées (message explicite, distinguables en test) ------------
class AudioValidationError(ValueError):
    """Classe de base pour les erreurs de validation d'un fichier audio chargé."""


class EmptyAudioError(AudioValidationError):
    """Levée quand le fichier audio ne contient aucun échantillon exploitable."""


class AudioTooShortError(AudioValidationError):
    """Levée quand le fichier audio est plus court que MIN_AUDIO_DURATION_S."""


def validate_audio_duration(y: np.ndarray, sr: int,
                             min_duration: float = MIN_AUDIO_DURATION_S) -> None:
    """
    Vérifie qu'un signal chargé est exploitable : ni vide, ni trop court.
    Lève EmptyAudioError / AudioTooShortError avec un message explicite sinon.
    À appeler juste après le chargement (librosa.load), avant tout traitement.
    """
    if y is None or len(y) == 0 or not np.any(np.abs(y) > 1e-9):
        raise EmptyAudioError("Fichier vide : aucun signal audio exploitable.")
    duration_s = len(y) / float(sr)
    if duration_s < min_duration:
        raise AudioTooShortError(
            f"Fichier trop court : {duration_s:.1f}s (minimum requis : {min_duration:.0f}s)."
        )


def load_and_validate_audio(path_or_buffer, min_duration: float = MIN_AUDIO_DURATION_S):
    """
    Charge un fichier audio (chemin ou buffer) via librosa (wav/mp3/ogg/flac supportés
    nativement par le backend soundfile/audioread) puis valide sa durée.
    Retourne (y, sr). Lève EmptyAudioError / AudioTooShortError si invalide
    (y compris si le fichier est corrompu ou vide au point de ne pas être décodable).
    """
    try:
        y, sr = librosa.load(path_or_buffer, sr=None)
    except Exception as e:
        raise EmptyAudioError(f"Fichier vide ou illisible : impossible de décoder l'audio ({e}).") from e
    validate_audio_duration(y, sr, min_duration=min_duration)
    return y, sr


def preprocess_signal(y, sr) -> tuple:
    y_trimmed, _ = librosa.effects.trim(y)
    n_samples = int(CLIP_DURATION * TARGET_SR)
    y_resampled = librosa.resample(y_trimmed, orig_sr=sr, target_sr=TARGET_SR)
    y_clip = y_resampled[:n_samples] if len(y_resampled) >= n_samples else np.pad(
        y_resampled, (0, n_samples - len(y_resampled))
    )
    return y_clip, TARGET_SR


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


def compute_features(y, sr, filename: str = 'user.file') -> pd.DataFrame:
    features, column_names = [], []
    features.append(filename); column_names.append('filename')
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


def image_to_base64(img: Image.Image) -> str:
    i_bytes = io.BytesIO()
    img.save(i_bytes, format="PNG")
    i_bytes.seek(0)
    return base64.b64encode(i_bytes.getvalue()).decode("utf-8")


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
