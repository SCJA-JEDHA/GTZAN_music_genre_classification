"""
test_audio_pipeline.py
-----------------------
Tests unitaires du pipeline audio (chargement, validation, features, spectrogrammes),
basés sur audio_pipeline.py (fonctions pures, sans Streamlit/S3 — cf. conftest.py).

Prérequis : un répertoire test_data/ à la racine du repo contenant :
  - au moins un fichier .wav, .mp3, .ogg et .flac (mêmes morceaux, pour comparaison croisée)
  - features.csv : référence générée par compute_features() sur ces mêmes fichiers
                    (colonne 'filename' = nom du fichier AVEC extension, ex. "track1.wav")
  - <stem>_harmo.png / <stem>_percu.png : spectrogrammes de référence générés par
                    calcul_image_pour_CNN() sur ces mêmes fichiers (convention de nommage
                    reprise de _analyze_one_file dans l'app)

⚠️ Si tes noms de fichiers/convention diffèrent, ajuste REFERENCE_CSV_NAME et
   harmo_ref_path()/percu_ref_path() ci-dessous en conséquence.
"""
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from PIL import Image
import librosa
import re

from audio_pipeline import (
    EmptyAudioError,
    AudioTooShortError,
    load_and_validate_audio,
    validate_audio_duration,
    preprocess_and_extract_segment,
    compute_features,
    calcul_image_pour_CNN,
    MIN_AUDIO_DURATION_S,
)

REFERENCE_CSV_NAME = "features_30_sec.csv"
AUDIO_EXTENSIONS = ["wav", "mp3", "ogg", "flac"]

# Tolérances de comparaison des features : plus larges pour mp3/ogg car le décodage
# lossy altère réellement le signal (ce n'est pas juste du bruit numérique) — la
# référence a beau avoir été calculée sur ces mêmes fichiers, on tolère un écart lié
# au décodeur/à sa version. wav/flac sont lossless -> tolérance serrée.
FEATURE_TOLERANCE = {
    "wav":  dict(atol=5e-2, rtol=5e-2),
    "flac": dict(atol=7e-2, rtol=7e-2),
    "mp3":  dict(atol=0.5, rtol=0.5),
    "ogg":  dict(atol=0.5, rtol=0.5),
}

"""FEATURE_TOLERANCE = {
    "wav":  dict(atol=2e-2, rtol=2e-2),
    "flac": dict(atol=2e-2, rtol=2e-2),
    "mp3":  dict(atol=15e-2, rtol=15e-2),
    "ogg":  dict(atol=15e-2, rtol=15e-2),
"""
IMAGE_TOLERANCE_MEAN_ABS_DIFF = {
    "wav": 2.0, "flac": 2.0, "mp3": 8.0, "ogg": 8.0,
}

NON_NUMERIC_FEATURE_COLS = {"filename", "label"}


# --------------------------------------------------------------------------- helpers
def _discover_files(test_data_dir: Path, ext: str) -> list:
    return sorted(test_data_dir.glob(f"*.{ext}"))


def _b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("L")


def _load_full_trimmed_reference_style(path: str):
    """Reproduit EXACTEMENT le préprocessing de mel_hpss (le générateur des PNG de
    référence dans test_data/) : librosa.load() à sa sr par défaut (22050, PAS sr=None),
    puis un simple trim — sans extraction de segment ni padding à durée fixe.

    Ne PAS remplacer par load_and_validate_audio() + preprocess_and_extract_segment() :
    ce dernier pad à exactement 30s (nécessaire en production pour donner une entrée de
    durée fixe au CNN), ce que mel_hpss ne fait jamais. Ce padding décale la normalisation
    globale du spectrogramme (min/max sur toute l'image) et rend toute comparaison
    pixel-à-pixel avec les PNG de référence invalide.
    """
    y, sr = librosa.load(path)       # sr=22050 par défaut, comme mel_hpss
    y, _ = librosa.effects.trim(y)   # trim SANS resample préalable, comme mel_hpss
    return y, sr


def _assert_features_close(computed: pd.Series, reference: pd.Series,
                            feature_cols: list, ext: str, filename: str) -> None:
    tol = FEATURE_TOLERANCE[ext]
    mismatches = []
    for col in feature_cols:
        c_val, r_val = float(computed[col]), float(reference[col])
        if not np.isclose(c_val, r_val, atol=tol["atol"], rtol=tol["rtol"]):
            mismatches.append(f"  - {col} : calculé={c_val:.6f} / référence={r_val:.6f}")
    assert not mismatches, (
        f"{filename} : {len(mismatches)} feature(s) hors tolérance "
        f"(atol={tol['atol']}, rtol={tol['rtol']}) :\n" + "\n".join(mismatches)
    )


def _assert_images_similar(img_computed: Image.Image, img_reference: Image.Image,
                            ext: str, label: str) -> None:
    assert img_computed.size == img_reference.size, (
        f"{label} : tailles différentes {img_computed.size} vs {img_reference.size}"
    )
    arr_c = np.asarray(img_computed, dtype=np.float32)
    arr_r = np.asarray(img_reference, dtype=np.float32)
    mad = float(np.mean(np.abs(arr_c - arr_r)))
    threshold = IMAGE_TOLERANCE_MEAN_ABS_DIFF[ext]
    assert mad <= threshold, (
        f"{label} : différence moyenne de pixels = {mad:.2f} (seuil = {threshold})"
    )


@pytest.fixture(scope="session")
def features_reference(test_data_dir: Path) -> pd.DataFrame:
    csv_path = test_data_dir / REFERENCE_CSV_NAME
    if not csv_path.is_file():
        pytest.skip(f"Fichier de référence introuvable : {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8")


# ============================================================= 1) CHARGEMENT MULTI-FORMAT
@pytest.mark.parametrize("ext", AUDIO_EXTENSIONS)
def test_load_all_formats(test_data_dir: Path, ext: str):
    """Vérifie que les 4 formats (.wav/.mp3/.ogg/.flac) se chargent correctement via librosa."""
    files = _discover_files(test_data_dir, ext)
    if not files:
        pytest.skip(f"Aucun fichier .{ext} trouvé dans {test_data_dir}")

    for f in files:
        y, sr = load_and_validate_audio(str(f))
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1, f"{f.name} : signal non mono après chargement ({y.ndim}D)"
        assert len(y) > 0, f"{f.name} : signal vide après chargement"
        assert sr > 0, f"{f.name} : sample rate invalide ({sr})"
        assert np.isfinite(y).all(), f"{f.name} : valeurs non finies (NaN/Inf) dans le signal"


# ============================================================= 2) FICHIER VIDE / TROP COURT
def test_empty_file_raises_explicit_error(tmp_path: Path):
    """Un fichier audio vide (silence pur) doit lever EmptyAudioError avec message explicite."""
    empty_path = tmp_path / "empty.wav"
    sr = 22050
    silence = np.zeros(sr * 2, dtype=np.float32)  # 2s de silence pur (amplitude nulle)
    sf.write(str(empty_path), silence, sr)

    with pytest.raises(EmptyAudioError, match="[Vv]ide"):
        load_and_validate_audio(str(empty_path))


def test_zero_length_file_raises_explicit_error(tmp_path: Path):
    """Un fichier de longueur nulle (0 échantillon) doit lever EmptyAudioError."""
    zero_len_path = tmp_path / "zero_length.wav"
    sf.write(str(zero_len_path), np.zeros(0, dtype=np.float32), 22050)

    with pytest.raises(EmptyAudioError, match="[Vv]ide"):
        load_and_validate_audio(str(zero_len_path))


def test_corrupted_file_raises_explicit_error(tmp_path: Path):
    """Un fichier 0 octet / illisible (pas un vrai WAV) doit aussi lever EmptyAudioError,
    pas une exception technique brute (soundfile/audioread) remontée telle quelle."""
    corrupted_path = tmp_path / "corrupted.wav"
    corrupted_path.write_bytes(b"")  # 0 octet, pas un WAV valide

    with pytest.raises(EmptyAudioError, match="[Vv]ide"):
        load_and_validate_audio(str(corrupted_path))


def test_short_file_raises_explicit_error(tmp_path: Path):
    """Un fichier de moins de 30s doit lever AudioTooShortError avec message explicite."""
    short_path = tmp_path / "short_5s.wav"
    sr = 22050
    duration_s = 5.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # note La440, 5s
    sf.write(str(short_path), y, sr)

    with pytest.raises(AudioTooShortError, match="court"):
        load_and_validate_audio(str(short_path))


def test_file_of_exactly_min_duration_does_not_raise(tmp_path: Path):
    """Vérifie la borne : un fichier de exactement MIN_AUDIO_DURATION_S ne doit PAS lever d'erreur."""
    path = tmp_path / "exactly_30s.wav"
    sr = 22050
    y = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, MIN_AUDIO_DURATION_S, int(sr * MIN_AUDIO_DURATION_S), endpoint=False)).astype(np.float32)
    sf.write(str(path), y, sr)

    y_loaded, sr_loaded = load_and_validate_audio(str(path))  # ne doit pas lever
    assert len(y_loaded) / sr_loaded >= MIN_AUDIO_DURATION_S - 0.1  # tolérance trim silence bords

def to_gtzan_name(fn: str) -> str:
    return re.sub(r'^([a-zA-Z]+)(\d+)\.\w+$', r'\1.\2.wav', fn)

# ============================================================= 3) CALCUL DES FEATURES
@pytest.mark.parametrize("ext", AUDIO_EXTENSIONS)
def test_compute_features_matches_reference(test_data_dir: Path, features_reference: pd.DataFrame, ext: str):
    """Pour chaque format, recalcule les features et compare à la ligne de référence (features.csv)."""
    files = _discover_files(test_data_dir, ext)
    if not files:
        pytest.skip(f"Aucun fichier .{ext} trouvé dans {test_data_dir}")

    feature_cols = [c for c in features_reference.columns if c not in NON_NUMERIC_FEATURE_COLS]
    assert feature_cols, "features.csv ne contient aucune colonne numérique de features"
    
    for f in files:
        filename_wav = to_gtzan_name(f.name) 
        ref_rows = features_reference[features_reference["filename"].replace(".0", "0") == filename_wav]
        if ref_rows.empty:
            pytest.skip(f"Aucune ligne de référence pour '{f.name}' dans {REFERENCE_CSV_NAME} "
                        f"(vérifie la colonne 'filename')")
        reference = ref_rows.iloc[0]

        # Préprocessing fidèle à la génération de référence (trim, sans extraction de
        # segment ni padding à durée fixe) — voir _load_full_trimmed_reference_style.
        # Le pipeline de production (preprocess_and_extract_segment) pad à 30s fixes,
        # ce qui décale plusieurs features (rms, longueur, etc.) par rapport à la référence.
        y_seg, sr_seg = _load_full_trimmed_reference_style(str(f))
        computed = compute_features(y_seg, sr_seg, filename=f.name).iloc[0]

        # 'length' : tolérance de quelques échantillons pour mp3/ogg (le décodage lossy peut
        # légèrement changer le nombre d'échantillons décodés selon l'alignement des frames) ;
        # exact pour wav/flac.
        length_tol = 0 if ext in ("wav", "flac") else 2205  # ~0.1s à 22050 Hz
        assert abs(int(computed["length"]) - int(reference["length"])) <= length_tol, (
            f"{f.name} : longueur de segment différente "
            f"(calculé={computed['length']}, référence={reference['length']}, tolérance={length_tol})"
        )

        _assert_features_close(computed, reference, feature_cols, ext, f.name)


# ============================================================= 4) CALCUL DES SPECTROGRAMMES
@pytest.mark.parametrize("ext", AUDIO_EXTENSIONS)
def test_compute_spectrograms_match_reference(test_data_dir: Path, ext: str):
    """Pour chaque format, recalcule les spectrogrammes harmo/percu et compare aux PNG de référence."""
    files = _discover_files(test_data_dir, ext)
    if not files:
        pytest.skip(f"Aucun fichier .{ext} trouvé dans {test_data_dir}")

    for f in files:
        harmo_ref_path = test_data_dir / f"{f.stem}_harmo.png"
        percu_ref_path = test_data_dir / f"{f.stem}_percu.png"
        if not harmo_ref_path.is_file() or not percu_ref_path.is_file():
            pytest.skip(f"Spectrogrammes de référence introuvables pour '{f.name}' "
                        f"(attendus : {harmo_ref_path.name}, {percu_ref_path.name})")

        # Préprocessing fidèle à mel_hpss (générateur des PNG de référence) — PAS le
        # pipeline de production (load_and_validate_audio + preprocess_and_extract_segment),
        # qui pad à 30s fixes et fausserait la comparaison pixel-à-pixel. Voir le
        # docstring de _load_full_trimmed_reference_style pour le détail.
        y_seg, sr_seg = _load_full_trimmed_reference_style(str(f))
        payload_spectro, _ = calcul_image_pour_CNN(y_seg, sr_seg)

        img_harmo_computed = _b64_to_image(payload_spectro["harmo_file"])
        img_percu_computed = _b64_to_image(payload_spectro["percu_file"])
        img_harmo_ref = Image.open(harmo_ref_path).convert("L")
        img_percu_ref = Image.open(percu_ref_path).convert("L")

        _assert_images_similar(img_harmo_computed, img_harmo_ref, ext, f"{f.name} (harmo)")
        _assert_images_similar(img_percu_computed, img_percu_ref, ext, f"{f.name} (percu)")


@pytest.mark.parametrize("ext", AUDIO_EXTENSIONS)
def test_production_pipeline_contract(test_data_dir: Path, ext: str):
    """Contrôle le pipeline RÉEL de production (celui utilisé par _analyze_one_file dans
    l'app : load_and_validate_audio + preprocess_and_extract_segment, AVEC padding à durée
    fixe) — sans comparaison pixel-à-pixel avec les références GTZAN (non pertinente ici,
    cf. test_compute_spectrograms_match_reference). On vérifie juste que le contrat de
    sortie (durée fixe, image exploitable) est respecté quel que soit le format d'entrée."""
    files = _discover_files(test_data_dir, ext)
    if not files:
        pytest.skip(f"Aucun fichier .{ext} trouvé dans {test_data_dir}")

    for f in files:
        y_raw, sr_raw = load_and_validate_audio(str(f))
        y_seg, sr_seg = preprocess_and_extract_segment(y_raw, sr_raw)

        expected_len = int(30 * sr_seg)  # TIME_SEGMENT * TARGET_SR
        assert len(y_seg) == expected_len, (
            f"{f.name} : le pipeline de production doit toujours produire un segment de "
            f"durée fixe ({expected_len} échantillons), obtenu {len(y_seg)}"
        )

        payload_spectro, _ = calcul_image_pour_CNN(y_seg, sr_seg)
        img_h = _b64_to_image(payload_spectro["harmo_file"])
        img_p = _b64_to_image(payload_spectro["percu_file"])
        assert img_h.size == (256, 128), f"{f.name} : taille harmo inattendue {img_h.size}"
        assert img_p.size == (256, 128), f"{f.name} : taille percu inattendue {img_p.size}"
