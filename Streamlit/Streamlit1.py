import os
import io
import requests
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import streamlit as st

# =========================
# CONFIG GLOBALE
# =========================
st.set_page_config(layout="wide")

general_path = "dataset/Data"
genres_path = f"{general_path}/genres_original"
pca_path = f"{general_path}/PCA"  # à adapter
my_music_path = "my_music"        # répertoire local pour la colonne 2

# =========================
# TITRE + DESCRIPTION
# =========================
st.title("🎧 Music Explorer & Recommender")

st.write("""
Application d'exploration audio, de prédiction de genre et de recommandations.  
Sélectionne un genre, visualise le signal et le spectrogramme,  
charge ta propre musique et découvre des morceaux similaires.
""")

# =========================
# PARAMÈTRES PARTAGÉS (n_fft, hop_length)
# =========================
if "n_fft" not in st.session_state:
    st.session_state.n_fft = 2048
if "hop_length" not in st.session_state:
    st.session_state.hop_length = 512

# =========================
# LAYOUT 3 COLONNES
# =========================
col1, col2, col3 = st.columns([1.3, 1.2, 1.5])

# ============================================================
# COLONNE 1 : DATABASE AUDIO (GENRES_ORIGINAL)
# ============================================================
with col1:
    st.subheader("🎼 Database selection")

    # --- Cadre select genre + morceau ---
    with st.container():
        st.markdown("**Sélection du genre et du morceau**")

        genres = sorted([g for g in os.listdir(genres_path) if os.path.isdir(os.path.join(genres_path, g))])
        genre = st.selectbox("Genre :", genres)

        genre_dir = os.path.join(genres_path, genre)
        files = sorted([f.replace(".wav", "") for f in os.listdir(genre_dir) if f.endswith(".wav")])
        audio_file_name = st.selectbox("Morceau :", files)

        db_audio_path = os.path.join(genre_dir, f"{audio_file_name}.wav")

        if st.button("▶️ Play (database)"):
            st.audio(db_audio_path)

    # --- Chargement audio + trim silence ---
    y_db, sr_db = librosa.load(db_audio_path)
    audio_db, _ = librosa.effects.trim(y_db)

    # --- Cadre visu temporelle + spectrogramme ---
    st.markdown("**Visualisation (database)**")

    # Choix n_fft / hop_length (partagés)
    n_fft_options = [512, 1024, 2048, 4096, 8192]
    hop_options = [128, 256, 512, 1024, 2048, 4096]

    st.session_state.n_fft = st.selectbox("n_fft :", n_fft_options, index=n_fft_options.index(st.session_state.n_fft))
    st.session_state.hop_length = st.selectbox("hop_length :", hop_options, index=hop_options.index(st.session_state.hop_length))

    n_fft = st.session_state.n_fft
    hop_length = st.session_state.hop_length

    # Visu temporelle
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y=audio_db, sr=sr_db, color="blue", ax=ax1)
    ax1.set_title(f"Sound Waves in {audio_file_name}", fontsize=16)
    st.pyplot(fig1)

    # Spectrogramme
    D_db = np.abs(librosa.stft(audio_db, n_fft=n_fft, hop_length=hop_length))
    DB_db = librosa.amplitude_to_db(D_db, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(DB_db, sr=sr_db, hop_length=hop_length,
                                   x_axis='time', y_axis='log', cmap='inferno', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogram (database)")
    st.pyplot(fig2)

# ============================================================
# COLONNE 2 : MY MUSIC + PREDICTION
# ============================================================
with col2:
    st.subheader("🤖 Model & My Music")

    # --- Cadre select model (MLflow placeholder) ---
    with st.container():
        st.markdown("**Model selection (MLflow)**")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.write("Model :")
        with col_m2:
            # Placeholder : à remplacer par une requête MLflow
            model_name = st.selectbox(" ", ["model_A", "model_B", "model_C"])

    # --- Cadre load your music ---
    with st.container():
        st.markdown("**Load your music**")

        if not os.path.exists(my_music_path):
            os.makedirs(my_music_path, exist_ok=True)

        my_files = [f for f in os.listdir(my_music_path) if f.lower().endswith((".wav", ".mp3", ".flac"))]
        my_file = st.selectbox("Fichier local :", my_files if my_files else ["<aucun fichier>"])

        load_btn = st.button("📥 Load my music")
        play_my_btn = st.button("▶️ Play (my music)")

        y_my, sr_my = None, None
        audio_my = None

        if my_files and my_file != "<aucun fichier>":
            my_audio_path = os.path.join(my_music_path, my_file)

            if load_btn:
                # Chargement
                y_my, sr_my = librosa.load(my_audio_path, sr=None)

                # Prétraitement :
                # 1) trim silence début
                audio_my, _ = librosa.effects.trim(y_my)

                # 2) garder 3 premières secondes
                duration_sec = 3
                audio_my = audio_my[: int(duration_sec * sr_my)]

                # 3) resample à 22500 Hz
                target_sr = 22500
                audio_my = librosa.resample(audio_my, orig_sr=sr_my, target_sr=target_sr)
                sr_my = target_sr

                # 4) normalisation RMS
                rms = np.sqrt(np.mean(audio_my**2) + 1e-12)
                audio_my = audio_my / rms

                st.session_state["my_audio"] = audio_my
                st.session_state["my_sr"] = sr_my
                st.success("Signal chargé et prétraité.")

            if play_my_btn and os.path.exists(os.path.join(my_music_path, my_file)):
                st.audio(os.path.join(my_music_path, my_file))

    # --- Features + spectrogramme + appel API ---
    with st.container():
        st.markdown("**Prediction**")

        if "my_audio" in st.session_state and "my_sr" in st.session_state:
            audio_my = st.session_state["my_audio"]
            sr_my = st.session_state["my_sr"]

            # Exemple de features (à adapter)
            # Ici : MFCC moyens
            mfcc = librosa.feature.mfcc(y=audio_my, sr=sr_my, n_mfcc=20)
            features = mfcc.mean(axis=1).tolist()

            # Spectrogramme Mel
            spect = librosa.feature.melspectrogram(y=audio_my, sr=sr_my, n_fft=2048, hop_length=512)
            spect = librosa.power_to_db(spect, ref=np.max)
            spect.resize(128, 660, refcheck=False)

            # Appel API (placeholder)
            api_url = "http://localhost:8000/api_music_predict"  # à adapter
            payload = {
                "model_name": model_name,
                "features": features,
                "spect": spect.tolist(),
            }

            if st.button("🔮 Predict genre"):
                try:
                    r = requests.get(api_url, json=payload, timeout=5)
                    if r.status_code == 200:
                        predicted_genre = r.json().get("predicted_genre", "unknown")
                    else:
                        predicted_genre = f"error ({r.status_code})"
                except Exception as e:
                    predicted_genre = f"error: {e}"

                st.session_state["predicted_genre"] = predicted_genre

    # --- Cadre predicted genre ---
    with st.container():
        st.markdown("**Predicted genre**")
        col_pg1, col_pg2 = st.columns([1, 2])
        with col_pg1:
            st.write("Predicted genre :")
        with col_pg2:
            st.success(st.session_state.get("predicted_genre", "—"))

    # --- Visu my music (temporel + spectrogramme avec n_fft/hop de col1) ---
    with st.container():
        st.markdown("**Visualisation (my music)**")

        if "my_audio" in st.session_state and "my_sr" in st.session_state:
            audio_my = st.session_state["my_audio"]
            sr_my = st.session_state["my_sr"]

            # Temporel
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            librosa.display.waveshow(y=audio_my, sr=sr_my, color="green", ax=ax3)
            ax3.set_title("My music - waveform", fontsize=16)
            st.pyplot(fig3)

            # Spectrogramme avec n_fft/hop de col1
            D_my = np.abs(librosa.stft(audio_my, n_fft=st.session_state.n_fft,
                                       hop_length=st.session_state.hop_length))
            DB_my = librosa.amplitude_to_db(D_my, ref=np.max)

            fig4, ax4 = plt.subplots(figsize=(8, 3))
            img2 = librosa.display.specshow(DB_my, sr=sr_my,
                                            hop_length=st.session_state.hop_length,
                                            x_axis='time', y_axis='log',
                                            cmap='inferno', ax=ax4)
            fig4.colorbar(img2, ax=ax4, format="%+2.0f dB")
            ax4.set_title("My music - spectrogram")
            st.pyplot(fig4)
        else:
            st.info("Charge d'abord ta musique pour voir les visualisations.")

# ============================================================
# COLONNE 3 : RECOMMANDATIONS + PCA
# ============================================================
with col3:
    st.subheader("🎯 Recommendations & PCA")

    # --- Cadre recommandations ---
    with st.container():
        st.markdown("**Recommendations**")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.write("Similar to :")
        with col_r2:
            source_choice = st.selectbox("Source :", ["database choice", "my music"])

        # Placeholder : liste de 4 voisins
        recommend_options = [f"neighbor_{i}" for i in range(1, 5)]
        recommend = st.selectbox("Recommend :", recommend_options)

        if st.button("▶️ Play (recommend)"):
            # TODO : jouer le fichier correspondant à recommend
            st.info("Lecture du morceau recommandé (à implémenter).")

    # --- Cadre visu avec onglets ---
    with st.container():
        tab_plot, tab_pca = st.tabs(["Plot", "Principal components"])

        # --------- Onglet Plot ---------
        with tab_plot:
            st.markdown("**Signal & spectrogram (recommend)**")
            # TODO : charger le signal 'recommend' (database ou my_music selon source_choice)
            # Placeholder : on réutilise audio_db
            if audio_db is not None:
                # Temporel
                fig5, ax5 = plt.subplots(figsize=(8, 3))
                librosa.display.waveshow(y=audio_db, sr=sr_db, color="purple", ax=ax5)
                ax5.set_title(f"Recommend - waveform ({recommend})", fontsize=16)
                st.pyplot(fig5)

                # Spectrogramme avec n_fft/hop de col1
                D_rec = np.abs(librosa.stft(audio_db, n_fft=st.session_state.n_fft,
                                            hop_length=st.session_state.hop_length))
                DB_rec = librosa.amplitude_to_db(D_rec, ref=np.max)

                fig6, ax6 = plt.subplots(figsize=(8, 3))
                img3 = librosa.display.specshow(DB_rec, sr=sr_db,
                                                hop_length=st.session_state.hop_length,
                                                x_axis='time', y_axis='log',
                                                cmap='inferno', ax=ax6)
                fig6.colorbar(img3, ax=ax6, format="%+2.0f dB")
                ax6.set_title("Recommend - spectrogram")
                st.pyplot(fig6)
            else:
                st.info("Aucun signal recommandé chargé pour l'instant.")

        # --------- Onglet PCA ---------
        with tab_pca:
            st.markdown("**PCA space (3D)**")

            # TODO : charger finalDf depuis /dataset/Data/PCA
            # finalDf doit contenir : 'principal component 1', 'principal component 2', 'principal component 3', 'label', 'track_name', 'genre'
            # Placeholder : petit DataFrame factice
            np.random.seed(0)
            finalDf = pd.DataFrame({
                "principal component 1": np.random.randn(50),
                "principal component 2": np.random.randn(50),
                "principal component 3": np.random.randn(50),
                "label": np.random.choice(["rock", "jazz", "classical"], size=50),
                "track_name": [f"track_{i}" for i in range(50)],
                "genre": np.random.choice(["rock", "jazz", "classical"], size=50),
            })

            # Calcul PCA pour my_music (placeholder)
            # En vrai : projeter les features de my_music dans l'espace PCA appris
            fig7 = plt.figure(figsize=(6, 5))
            ax7 = fig7.add_subplot(111, projection='3d')

            scatter = ax7.scatter(
                finalDf["principal component 1"],
                finalDf["principal component 2"],
                finalDf["principal component 3"],
                c=pd.Categorical(finalDf["label"]).codes,
                cmap="tab10",
                alpha=0.7,
                s=60,
            )
            ax7.set_xlabel("PC1")
            ax7.set_ylabel("PC2")
            ax7.set_zlabel("PC3")
            ax7.set_title("PCA space (placeholder)")

            # Surbrillance de la sélection database (rouge) et my_music (bleu)
            # TODO : identifier les points correspondants
            # Exemple : point rouge
            ax7.scatter([0], [0], [0], c="red", s=120, label="database selection")
            # Exemple : point bleu
            ax7.scatter([1], [1], [1], c="blue", s=120, label="my_music")

            ax7.legend()
            st.pyplot(fig7)

            st.info("Pour les infobulles et la sélection interactive, utiliser Plotly ou Altair (à implémenter).")
