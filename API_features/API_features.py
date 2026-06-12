import io
import logging
from contextlib import asynccontextmanager

import librosa
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
N_FFT_DEFAULT     = 2048
HOP_DEFAULT       = 512
IMAGE_NX          = 432
IMAGE_NY          = 288
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
MAX_FILE_SIZE_MB  = 50
MAX_DURATION_S    = 300

description = """
## 🎵 MusicAI Features API

Envoie un fichier audio, reçois les features GTZAN-compatibles + le mel-spectrogramme.

### Endpoints
- `POST /extract` — features + spectrogramme depuis un fichier audio
- `GET  /health`  — statut de l'API
"""


# ── Feature extraction ────────────────────────────────────────────────────────
def compute_features(y, sr) -> dict:
    """Calcule le vecteur de features GTZAN-compatible (58 colonnes)."""
    chroma             = librosa.feature.chroma_stft(y=y, sr=sr)
    rms                = librosa.feature.rms(y=y)
    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff            = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr  = librosa.effects.hpss(y)
    tempo, _           = librosa.beat.beat_track(y=y, sr=sr)
    mfccs              = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    row = {}

    for name, data in {
        "chroma_stft":        chroma,
        "rms":                rms,
        "spectral_centroid":  spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "rolloff":            rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "harmony":            harmony,
        "perceptr":           perceptr,
    }.items():
        row[f"{name}_mean"] = float(np.mean(data))
        row[f"{name}_var"]  = float(np.var(data))

    row["tempo"] = float(np.atleast_1d(tempo)[0])

    for idx, coef in enumerate(mfccs, start=1):
        row[f"mfcc{idx}_mean"] = float(np.mean(coef))
        row[f"mfcc{idx}_var"]  = float(np.var(coef))

    return row


def compute_melspectrogram(y, sr) -> dict:
    """Calcule le mel-spectrogramme (128×660) + PNG base64."""
    import base64
    from PIL import Image

    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect.resize(128, 660, refcheck=False)

    s_min, s_max = spect.min(), spect.max()
    if s_max > s_min:
        img_arr = ((spect - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        img_arr = np.zeros_like(spect, dtype=np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "matrix":  spect.tolist(),
        "png_b64": png_b64,
        "shape":   list(spect.shape),
    }


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MusicAI Features API — démarrage")
    yield
    logger.info("MusicAI Features API — arrêt")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎵 MusicAI Features API",
    description=description,
    version="0.1",
    contact={
        "name": "Adrien, John, Cyril et Sandra",
        "url": "https://jedha.co",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "version": "0.1"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


# ── Extraction ────────────────────────────────────────────────────────────────
@app.post("/extract", tags=["features"])
async def extract(
    file: UploadFile = File(..., description="Fichier audio (wav, mp3, ogg, flac…)"),
    target_sr: int        = Query(22050, description="Sample rate cible (Hz)"),
    include_spectrogram: bool = Query(True, description="Inclure le mel-spectrogramme"),
    include_png: bool         = Query(True, description="Inclure l'image PNG base64"),
):
    """
    Retourne un JSON avec :
    - **features** : dict GTZAN-compatible (58 colonnes : filename → label)
    - **spectrogram** : matrice 128×660 + PNG base64 (si `include_spectrogram=true`)
    """
    filename = file.filename or "audio"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté : '{ext}'. Acceptés : {sorted(SUPPORTED_FORMATS)}",
        )

    raw = await file.read()
    if len(raw) / (1024 ** 2) > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"Fichier trop grand (max {MAX_FILE_SIZE_MB} MB)")

    try:
        y, sr = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
    except Exception as e:
        raise HTTPException(422, f"Impossible de décoder le fichier audio : {e}")

    duration = len(y) / sr
    if duration > MAX_DURATION_S:
        raise HTTPException(422, f"Audio trop long : {duration:.1f}s (max {MAX_DURATION_S}s)")
    if duration < 0.1:
        raise HTTPException(422, "Audio trop court (< 100 ms)")

    try:
        features = compute_features(y, sr)
    except Exception as e:
        logger.exception("Erreur compute_features")
        raise HTTPException(500, f"Erreur extraction features : {e}")

    response: dict = {"features": features}

    if include_spectrogram:
        try:
            spect = compute_melspectrogram(y, sr)
            if not include_png:
                spect["png_b64"] = None
            response["spectrogram"] = spect
        except Exception as e:
            logger.exception("Erreur compute_melspectrogram")
            raise HTTPException(500, f"Erreur extraction spectrogramme : {e}")

    logger.info(f"OK — '{filename}' ({duration:.1f}s, {sr} Hz)")
    return JSONResponse(content=response)


# ── Lancement local ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
