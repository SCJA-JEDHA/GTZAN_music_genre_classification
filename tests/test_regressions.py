"""
Tests de non-régression pour les bugs corrigés le 17/07 :
1. Réglette volume ajoutée -> supprimée, volume fixé à 50%
2. Save laissait les visus (waveform/spectrogramme) affichées -> doivent être effacées
3. Plus de son au 2ème morceau -> conséquence du slider, couvert par le test #1
4. CSV vide après save -> compute_features écrivait 'user.file' en dur
5. Écrasement du CSV à chaque save -> doit ajouter les lignes, pas écraser
"""
import io
import pandas as pd
import pytest


# ── Bug #1/#3 — volume fixé, plus de widget slider ────────────────────────
def test_no_volume_slider_in_session_state(app):
    """La clé playback_volume_pct ne doit plus exister : le slider a été retiré."""
    assert "playback_volume_pct" not in app.st.session_state


def test_play_active_row_uses_fixed_volume(app, batch_session):
    """_play_active_row() doit produire un signal audio non silencieux, basé
    sur DEFAULT_VOLUME_PCT, sans dépendre d'un widget volume."""
    app.st.session_state.active_row_idx = 0
    app.st.session_state.batch_play_audio = None

    app._play_active_row()

    audio_bytes = app.st.session_state.batch_play_audio
    assert audio_bytes is not None and len(audio_bytes) > 0
    assert app.st.session_state["_active_played"] is True


# ── Bug #2 — save doit effacer les visualisations ──────────────────────────
def test_save_clears_visualizations(app, batch_session):
    app._save_tagged_rows()

    st = app.st.session_state
    assert st.fig_my_wave is None
    assert st.fig_my_spect is None
    assert st.fig_batch_wave is None
    assert st.fig_batch_spect is None
    assert st.my_show_visu is False
    assert st.my_y is None
    assert st.my_sr is None
    assert st.my_features is None
    assert st.predicted_genre == "—"


def test_save_clears_the_list(app, batch_session):
    app._save_tagged_rows()
    assert app.st.session_state.df_user_music_temp.empty


# ── Bug #4 — le nom réel du fichier doit être écrit dans les features ─────
def test_compute_features_uses_real_filename(app, sample_audio):
    y, sr = sample_audio
    df = app.compute_features(y, sr, filename="track42.wav")
    assert df.loc[0, "filename"] == "track42.wav"


def test_compute_features_default_filename_unchanged(app, sample_audio):
    """Compat ascendante : sans argument, le comportement historique est conservé."""
    y, sr = sample_audio
    df = app.compute_features(y, sr)
    assert df.loc[0, "filename"] == "user.file"


def test_save_writes_data_to_csv(app, batch_session, s3_mock):
    app._save_tagged_rows()

    obj = s3_mock.get_object(
        Bucket="test-bucket",
        Key=f"{app.FEATURES_USER_PREFIX}features_sess123.csv",
    )
    df_csv = pd.read_csv(io.BytesIO(obj["Body"].read()))
    assert len(df_csv) == 1
    assert df_csv.loc[0, "filename"] == "song1.wav"


# ── Bug #5 — plusieurs saves de la même session s'ajoutent, n'écrasent pas ─
def test_second_save_appends_not_overwrites(app, batch_session, s3_mock, tmp_path):
    app._save_tagged_rows()

    # on retagge une 2ème ligne pour la même session_id et on resauvegarde
    name2 = "song2.wav"
    y2, sr2 = batch_session["y"], batch_session["sr"]
    feats2 = app.compute_features(y2, sr2, filename=name2)
    feats2["user_name"] = "tester"
    feats2["date_heure"] = "2026-07-17"
    feats2["session_id"] = "sess123"

    percu2 = tmp_path / "song2_percu.png"
    harmo2 = tmp_path / "song2_harmo.png"
    percu2.write_bytes(b"fake-png")
    harmo2.write_bytes(b"fake-png")

    st = app.st.session_state
    st.df_user_music_temp = pd.DataFrame([{
        "name": name2, "genre_pred_feat": "jazz", "genre_pred_CNN": "jazz", "genre_user": "jazz",
    }])
    st.features_user_temp = feats2
    st.spectro_user = pd.DataFrame([{
        "name": name2, "spectro_percu": str(percu2), "spectro_harmo": str(harmo2),
    }])
    st.batch_audio_cache = {name2: (y2, sr2)}

    app._save_tagged_rows()

    obj = s3_mock.get_object(
        Bucket="test-bucket",
        Key=f"{app.FEATURES_USER_PREFIX}features_sess123.csv",
    )
    df_csv = pd.read_csv(io.BytesIO(obj["Body"].read()))
    # les 2 lignes (song1 ET song2) doivent être présentes -> pas d'écrasement
    assert len(df_csv) == 2
    assert set(df_csv["filename"]) == {"song1.wav", "song2.wav"}


def test_append_helper_creates_file_when_absent(app, s3_mock):
    df = pd.DataFrame([{"a": 1, "b": 2}])
    key = "some/prefix/new_file.csv"
    app._append_df_to_s3_csv(s3_mock, "test-bucket", key, df)

    obj = s3_mock.get_object(Bucket="test-bucket", Key=key)
    out = pd.read_csv(io.BytesIO(obj["Body"].read()))
    assert len(out) == 1


def test_append_helper_appends_to_existing_file(app, s3_mock):
    key = "some/prefix/existing.csv"
    df1 = pd.DataFrame([{"a": 1}])
    df2 = pd.DataFrame([{"a": 2}])

    app._append_df_to_s3_csv(s3_mock, "test-bucket", key, df1)
    app._append_df_to_s3_csv(s3_mock, "test-bucket", key, df2)

    obj = s3_mock.get_object(Bucket="test-bucket", Key=key)
    out = pd.read_csv(io.BytesIO(obj["Body"].read()))
    assert sorted(out["a"].tolist()) == [1, 2]


def test_append_helper_noop_on_empty_dataframe(app, s3_mock):
    key = "some/prefix/should_not_exist.csv"
    app._append_df_to_s3_csv(s3_mock, "test-bucket", key, pd.DataFrame())

    with pytest.raises(Exception):
        s3_mock.get_object(Bucket="test-bucket", Key=key)
