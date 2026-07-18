def test_s3_key_exists_true(app, s3_mock):
    s3_mock.put_object(Bucket="test-bucket", Key="some/key.txt", Body=b"data")
    assert app.s3_key_exists.__wrapped__("test-bucket", "some/key.txt") is True


def test_s3_key_exists_false(app, s3_mock):
    assert app.s3_key_exists.__wrapped__("test-bucket", "missing/key.txt") is False


def test_list_genres_empty_bucket(app, s3_mock):
    # NB : list_genres() est mise en cache (st.cache_data) dès l'import du module
    # (le rendu initial de la page appelle _ctrl_db() -> list_genres()), donc ce
    # test vérifie le comportement à bucket vide. Pour tester le contenu réel
    # après ajout d'objets, voir test_list_s3_subdirectories_sorted ci-dessous
    # qui appelle directement la fonction bas niveau non affectée par ce cache.
    assert app.list_genres.__wrapped__() == []


def test_list_s3_subdirectories_sorted(app, s3_mock):
    prefix = app.GENRES_PREFIX
    for genre in ["rock", "blues", "jazz"]:
        s3_mock.put_object(Bucket="test-bucket", Key=f"{prefix}{genre}/track1.wav", Body=b"x")
    result = app.list_s3_subdirectories_sorted.__wrapped__("test-bucket", prefix)
    assert result == ["blues", "jazz", "rock"]
