y, sr = librosa.load(in_path)
    y, _ = librosa.effects.trim(y)

    n_fft = 2048        # Précision des détails du timbre
    hop_length = 512    # Résolution temporelle
    n_mels = 128        # Hauteur de la fréquence

    # Séparation harmonique/percussive
    y_full = y
    y_harmonic, y_percussive = librosa.effects.hpss(y)

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

    img_f = Image.fromarray((norm_S_DB * 255).astype(np.uint8))
    img_h = Image.fromarray((norm_S_DB_h * 255).astype(np.uint8), mode = "L")
    img_p = Image.fromarray((norm_S_DB_p * 255).astype(np.uint8), mode = "L")

    img_f = img_f.resize((512, 256), Image.Resampling.LANCZOS)
    img_h = img_h.resize((256, 128), Image.Resampling.LANCZOS)
    img_p = img_p.resize((256, 128), Image.Resampling.LANCZOS)

    """h_bytes = io.BytesIO()
    img_h.save(h_bytes, format="PNG")
    h_bytes.seek(0)
    p_bytes = io.BytesIO()
    img_h.save(p_bytes, format="PNG")
    p_bytes.seek(0)"""

    b_harmo = image_to_base64(img_h)
    b_percu = image_to_base64(img_p)

    #spectro_at_predict = {"harmo_file" : ("harmo.png", h_bytes, "image/png"), "percu_file" : ("percu.png", p_bytes, "image/png")}
    payload_spectro = {{"harmo_file" : b_harmo, "percu_file" : b_percu}

    resp = requests.post(url = "https://dareindodo-api-dl-predict.hf.space/predict-cnn", json = spectro_at_predict)

    return resp.json()
