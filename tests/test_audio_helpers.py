import numpy as np


def test_preprocess_signal_output_length(app, sample_audio):
    y, sr = sample_audio
    y_clip, sr_out = app.preprocess_signal(y, sr)
    assert sr_out == app.TARGET_SR
    assert len(y_clip) == app.CLIP_DURATION * app.TARGET_SR


def test_preprocess_signal_pads_short_audio(app):
    sr = app.TARGET_SR
    y_short = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    y_clip, _ = app.preprocess_signal(y_short, sr)
    assert len(y_clip) == app.CLIP_DURATION * app.TARGET_SR


def test_trim_audio_removes_silence(app):
    sr = app.TARGET_SR
    silence = np.zeros(sr, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    y = np.concatenate([silence, tone, silence])
    trimmed = app.trim_audio(y, sr)
    assert len(trimmed) < len(y)


def test_compute_features_returns_expected_columns(app, sample_audio):
    y, sr = sample_audio
    df = app.compute_features(y, sr, filename="x.wav")
    assert "filename" in df.columns
    assert "tempo" in df.columns
    assert "label" in df.columns
    assert len(df) == 1


def test_compute_melspectrogram_shape(app, sample_audio):
    y, sr = sample_audio
    spect = app.compute_melspectrogram(y, sr)
    assert spect.shape == (app.IMAGE_NY, app.IMAGE_NX)
